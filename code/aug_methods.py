"""The augmentations compared in the paper.  All act on one training image before mosaic (see augment.py).

graft     cross-source instance transplantation (the proposed method).  Animals of the instance bank (SAM 2.1 masks) are pasted into
          cruise training images.  Its parts can be switched off one by one:
            sampling   'scarcity' (taxon drawn with probability ~ (n_k + 1)^-alpha, n_k = cruise training boxes) or 'instance'
                       (every bank instance equally likely, i.e. taxa in proportion to their bank counts)
            placement  'geom' (box bottom drawn from the cruise boxes, height from the row-conditional size model of the cruise
                       boxes plus a taxon offset; benthic taxa only on textured seafloor and, with avoid_gear, off the vehicle
                       equipment masks in <bank>/gear/masks.npz) or 'uniform' (anywhere, native size x U(scale))
            relight    'local_range' (the relighting of GRAFT, Eq. (4) of the paper: each channel of the instance is normalized
                       with the 5th and 95th percentiles of its own pixels and scaled to the 5th and 95th percentiles of the
                       background ring at the paste location), 'affine_q' (per-channel affine map that moves the 5th and 95th
                       percentiles of the background ring around the instance in its source image onto those of the ring at the
                       paste location: under artificial light the local image is a_c J + b_c, so the map transfers the local lamp
                       gain and backscatter), 'affine_m' (same with mean and standard deviation), 'chroma' (transfer of the median
                       colour cast of the ring with a mild change of brightness), 'median_gain' (one gain from the luminance
                       spread of the two rings and per-channel medians matched), 'poisson' (cv2.seamlessClone) or 'none'
            strength   range of the relighting strength lambda, drawn per paste
            degrade    match blur and sensor noise of the paste location, then re-encode the patch as JPEG
            mask       'sam' (alpha from the SAM mask, feathered) or 'rect' (the whole box crop)
            sources    'oc,cruise', 'oc' or 'cruise' ('oc' is the archive set); cruise instances only from images of the run's
                       training list (no leakage across cross-validation folds)
            select     'random' or 'context' (instance whose source background colour is closest to the paste location, the
                       context-matched collage of McEver et al.)
            scale      range of the size factor of uniform placement (native size times U(scale))
            rot        largest in-plane rotation of an instance in degrees (0 = none)
            p, extra, max_n, max_tries
                       probability that a frame receives pastes, mean of the Poisson number of animals added to the first,
                       largest number of animals per frame, and positions tried per animal before it is skipped
          The configuration of the paper: sampling='scarcity', placement='uniform', relight='local_range', degrade=False,
          max_tries=30, p=0.6, other arguments at their defaults.
photo     generic photometric augmentation (colour jitter, CLAHE, gamma, blur, noise)
ocenv     habitat presets chosen by the taxa in the frame (gamma, blue boost and red loss, haze, particles, noise)
ampmix    AmpMix, phase-preserving Fourier amplitude mixing with another training image
lumiaug   LumiAug, illumination-field randomization (lamp field, backscatter veil, channel gains, marine snow)
snow      real marine-snow layers: bright particles high-pass filtered from water-column frames of the training dives, added on top
          (reads <bank>/snow/snow.json and its layers)
gear      equipment distractors: unlabelled cut-outs of vehicle equipment from the training frames (reads <bank>/gear/gear.json)
graft+snow, and any 'a+b': applied in order
"""
import csv
import json
import math
import os

import cv2
import numpy as np

from augment import register

TAXA = ['fish', 'seastar', 'brittlestar', 'urchin', 'cucumber', 'crinoid', 'coral', 'anemone', 'seapen', 'sponge', 'crustacean', 'worm',
        'otherbio']
NAMED = list(range(12))
PELAGIC = {TAXA.index('fish')}


def _instances(label):
    inst = label['instances']
    inst.convert_bbox('xywh')
    return inst.bboxes.copy(), label['cls'].reshape(-1).copy()


def _set_instances(label, boxes, cls):
    from ultralytics.utils.instance import Instances
    label['instances'] = Instances(boxes.astype(np.float32).reshape(-1, 4), np.zeros((0, 1000, 2), np.float32), None,
                                   bbox_format='xywh', normalized=True)
    label['cls'] = cls.astype(np.float32).reshape(-1, 1)
    return label


def _names(root, lst):
    return [os.path.basename(l.strip())[:-4] for l in open(os.path.join(root, 'lists', lst + '.txt')) if l.strip()]


def _ioa(a, b):
    """Intersection of box a with each box of b over the area of a; boxes as x0 y0 x1 y1."""
    if len(b) == 0:
        return np.zeros(0)
    ix = np.clip(np.minimum(a[2], b[:, 2]) - np.maximum(a[0], b[:, 0]), 0, None)
    iy = np.clip(np.minimum(a[3], b[:, 3]) - np.maximum(a[1], b[:, 1]), 0, None)
    return ix * iy / max(1e-9, (a[2] - a[0]) * (a[3] - a[1]))


def _ring(mask, width):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * width + 1, 2 * width + 1))
    return (cv2.dilate(mask, k) > 0) & (mask == 0)


@register('graft')
class Graft:
    def __init__(self, bank, root, train_lists, p=0.5, extra=1.0, max_n=5, sampling='scarcity', alpha=0.5, placement='geom',
                 relight='affine_q', strength=(0.5, 1.0), degrade=True, mask='sam', sources='oc,cruise', select='random', targets='cruise',
                 empty_targets=True, max_up=1.5, avoid_gear=True, max_tries=10, scale=(0.5, 1.5), rot=0.0, **kw):
        self.p, self.extra, self.max_n = p, extra, max_n
        self.scale, self.rot = tuple(scale), float(rot)
        self.sampling, self.placement, self.relight, self.strength = sampling, placement, relight, tuple(strength)
        self.degrade, self.mask, self.select, self.targets, self.empty_targets, self.max_up = degrade, mask, select, targets, empty_targets, max_up
        self.max_tries = max_tries
        self.crops = os.path.join(bank, 'crops')
        self.gear_file = os.path.join(bank, 'gear', 'masks.npz') if avoid_gear else None
        self.gear = None
        train_imgs = set()
        for lst in train_lists.split(','):
            train_imgs |= set(_names(root, lst))
        srcs = set(sources.split(','))
        rows = []
        for r in csv.DictReader(open(os.path.join(bank, 'bank.csv'))):
            if r['ok'] != '1' or int(r['cls']) not in NAMED or r['source'] not in srcs:
                continue
            if r['source'] == 'cruise' and r['image'] not in train_imgs:
                continue
            rows.append(r)
        self.rows = rows
        self.by_cls = {c: [i for i, r in enumerate(rows) if int(r['cls']) == c] for c in NAMED}
        self.ring_q = np.array([[[float(r[f'{ch}_q05']), float(r[f'{ch}_q95'])] for ch in 'bgr'] for r in rows], np.float32)
        self.ring_m = np.array([[[float(r[f'{ch}_mean']), float(r[f'{ch}_std'])] for ch in 'bgr'] for r in rows], np.float32)
        self.ring_med = np.array([[float(r[f'{ch}_q50']) for ch in 'bgr'] for r in rows], np.float32) if rows and 'b_q50' in rows[0] else None
        # cruise training boxes: taxon counts (scarcity weights) and the size model of the oblique seafloor view
        n_k = np.zeros(len(TAXA))
        yb, lh, kk = [], [], []
        for n in sorted(train_imgs):  # sorted: set order follows the per-process string hash
            if n.startswith('S_'):
                continue
            for l in open(os.path.join(root, 'labels', n + '.txt')):
                if not l.strip():
                    continue
                c, cx, cy, w, h = l.split()
                c, cy, h = int(c), float(cy), float(h)
                n_k[c] += 1
                yb.append(min(1.0, cy + h / 2))
                lh.append(math.log(max(h, 1e-3)))
                kk.append(c)
        self.n_k = n_k
        avail = np.array([len(self.by_cls[c]) > 0 for c in NAMED])
        if sampling == 'scarcity':
            w = (n_k[NAMED] + 1.0) ** (-alpha) * avail
        else:
            w = np.array([len(self.by_cls[c]) for c in NAMED], float)
        self.cls_p = w / w.sum()
        yb, lh, kk = np.array(yb), np.array(lh), np.array(kk)
        self.yb = yb
        self.edges = np.quantile(yb, np.linspace(0, 1, 9))
        b = np.clip(np.searchsorted(self.edges, yb, side='right') - 1, 0, 7)
        self.m_bin = np.array([np.median(lh[b == i]) if (b == i).sum() >= 5 else np.median(lh) for i in range(8)])
        res = lh - self.m_bin[b]
        self.sd = float(np.std(res))
        self.delta = np.zeros(len(TAXA))
        for c in NAMED:
            rc = res[kk == c]
            if len(rc):
                self.delta[c] = float(np.median(rc)) * len(rc) / (len(rc) + 10.0)

    def describe(self):
        return dict(instances=len(self.rows), per_taxon={TAXA[c]: len(self.by_cls[c]) for c in NAMED},
                    taxon_p={TAXA[c]: round(float(p), 4) for c, p in zip(NAMED, self.cls_p)}, size_sd=round(self.sd, 3),
                    delta={TAXA[c]: round(float(self.delta[c]), 3) for c in NAMED})

    # ----------------------------------------------------------------------------------------------------------------
    def __call__(self, label, rng, dataset=None):
        name = os.path.basename(label['im_file'])
        if self.targets == 'cruise' and name.startswith('S_'):
            return label
        boxes, cls = _instances(label)
        if not self.empty_targets and len(boxes) == 0:
            return label
        if rng.random() >= self.p:
            return label
        img = label['img'].copy()
        H, W = img.shape[:2]
        n = int(min(self.max_n, 1 + rng.poisson(self.extra)))
        xyxy = np.concatenate([boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2], 1) if len(boxes) else np.zeros((0, 4))
        valid = None
        new_b, new_c = [], []
        for _ in range(n):
            c = int(rng.choice(NAMED, p=self.cls_p)) if self.sampling == 'scarcity' else None
            for _try in range(self.max_tries):
                if self.placement == 'geom' and valid is None:
                    valid = self._valid_map(img, name[:-4])
                out = self._paste_one(img, c, xyxy, valid, rng)
                if out is not None:
                    img, b, k = out
                    xyxy = np.concatenate([xyxy, b[None]], 0)
                    new_b.append(b)
                    new_c.append(k)
                    break
        if not new_b:
            return label
        label['img'] = img
        nb = np.array(new_b)
        nb = np.concatenate([(nb[:, :2] + nb[:, 2:]) / 2, nb[:, 2:] - nb[:, :2]], 1)
        return _set_instances(label, np.concatenate([boxes, nb], 0), np.concatenate([cls, np.array(new_c, float)]))

    def _valid_map(self, img, name):
        """Seafloor mask at 1/4 resolution: local texture above that of open water, and no vehicle equipment."""
        g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (img.shape[1] // 4, img.shape[0] // 4), interpolation=cv2.INTER_AREA)
        g = g.astype(np.float32)
        mu = cv2.blur(g, (5, 5))
        sd = np.sqrt(np.clip(cv2.blur(g * g, (5, 5)) - mu * mu, 0, None))
        valid = sd > 6.0
        self._gear_now = np.zeros_like(valid)
        if self.gear_file:
            if self.gear is None:
                self.gear = np.load(self.gear_file)
            if name in self.gear.files:
                gm = cv2.resize(self.gear[name].astype(np.uint8), (valid.shape[1], valid.shape[0]), interpolation=cv2.INTER_NEAREST)
                self._gear_now = gm > 0
                valid &= gm == 0
        return valid

    def _pick(self, c, img, cx, cy, rng):
        if self.sampling == 'instance' and c is None:
            i = int(rng.integers(len(self.rows)))
            return i
        pool = self.by_cls[c]
        if self.select == 'context':
            H, W = img.shape[:2]
            x0, y0 = int(max(0, cx * W - 20)), int(max(0, cy * H - 20))
            patch = img[y0:y0 + 40, x0:x0 + 40].reshape(-1, 3).astype(np.float32)
            t = patch.mean(0)
            cand = rng.choice(pool, size=min(len(pool), 64), replace=False)
            d = np.abs(self.ring_m[cand, :, 0] - t[None]).sum(1)
            return int(cand[np.argmin(d)])
        return int(pool[int(rng.integers(len(pool)))])

    def _paste_one(self, img, c, xyxy, valid, rng):
        H, W = img.shape[:2]
        # position of the box bottom centre
        if self.placement == 'geom':
            ybot = float(np.clip(self.yb[int(rng.integers(len(self.yb)))] + rng.normal(0, 0.03), 0.05, 1.0))
        else:
            ybot = float(rng.uniform(0.05, 1.0))
        xc = float(rng.uniform(0.02, 0.98))
        if c is None:
            i = self._pick(None, img, xc, ybot, rng)
            c = int(self.rows[i]['cls'])
        else:
            i = self._pick(c, img, xc, ybot, rng)
        r = self.rows[i]
        if self.placement == 'geom' and c not in PELAGIC:
            vy = min(valid.shape[0] - 1, int(ybot * valid.shape[0]) - 1)
            vx = min(valid.shape[1] - 1, int(xc * valid.shape[1]))
            if not valid[max(0, vy), vx]:
                return None
        crop = cv2.imread(os.path.join(self.crops, r.get('crop_file') or r['id'] + '.png'), cv2.IMREAD_UNCHANGED)
        if crop is None:
            return None
        cw, ch = crop.shape[1], crop.shape[0]
        sx = cw / (float(r['crop_x1']) - float(r['crop_x0']))
        sy = ch / (float(r['crop_y1']) - float(r['crop_y0']))
        bx0 = (float(r['box_x0']) - float(r['crop_x0'])) * sx
        by0 = (float(r['box_y0']) - float(r['crop_y0'])) * sy
        bx1 = (float(r['box_x1']) - float(r['crop_x0'])) * sx
        by1 = (float(r['box_y1']) - float(r['crop_y0'])) * sy
        bh_crop = max(1.0, by1 - by0)
        native_h = bh_crop * (640.0 / float(r.get('crop_long') or 1280)) / H  # box height as a fraction of a 640-px image
        if self.placement == 'geom':
            b = min(7, max(0, int(np.searchsorted(self.edges, ybot, side='right')) - 1))
            h = math.exp(self.m_bin[b] + self.delta[c] + rng.normal(0, 0.6 * self.sd))
            h = min(h, self.max_up * native_h)
        else:
            h = native_h * rng.uniform(*self.scale)
        h = float(np.clip(h, 8.0 / H, 0.7))
        s = h * H / bh_crop
        if self.rot > 0:
            # in-plane rotation about the crop centre; the box becomes the bound of its rotated corners, as in the rotation of
            # Ultralytics without segments, and the scale stays that of the unrotated animal
            ang = float(rng.uniform(-self.rot, self.rot))
            R = cv2.getRotationMatrix2D((cw / 2, ch / 2), ang, 1.0)
            nw = int(math.ceil(abs(R[0, 0]) * cw + abs(R[0, 1]) * ch))
            nh = int(math.ceil(abs(R[0, 1]) * cw + abs(R[0, 0]) * ch))
            R[0, 2] += nw / 2 - cw / 2
            R[1, 2] += nh / 2 - ch / 2
            crop = cv2.warpAffine(crop, R, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
            corners = np.array([[bx0, by0, 1], [bx1, by0, 1], [bx0, by1, 1], [bx1, by1, 1]]) @ R.T
            bx0, by0 = corners.min(0)
            bx1, by1 = corners.max(0)
            cw, ch = nw, nh
        pw, ph = max(1, int(round(cw * s))), max(1, int(round(ch * s)))
        if pw < 4 or ph < 4:
            return None
        patch = cv2.resize(crop, (pw, ph), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        # place so that the scaled box has its bottom centre at (xc, ybot)
        box_w, box_h = (bx1 - bx0) * s, (by1 - by0) * s
        tx0 = xc * W - box_w / 2 - bx0 * s
        ty0 = ybot * H - box_h - by0 * s
        nb = np.array([(tx0 + bx0 * s) / W, (ty0 + by0 * s) / H, (tx0 + bx1 * s) / W, (ty0 + by1 * s) / H])
        if nb[0] < 0 or nb[1] < 0 or nb[2] > 1 or nb[3] > 1:
            return None
        if len(xyxy) and (_ioa(nb, xyxy).max() > 0.3 or max(_ioa(e, nb[None])[0] for e in xyxy) > 0.3):
            return None
        if self.placement == 'geom' and self.gear_file:
            gh, gw = self._gear_now.shape
            gx0, gy0 = int(nb[0] * gw), int(nb[1] * gh)
            gx1, gy1 = max(gx0 + 1, int(math.ceil(nb[2] * gw))), max(gy0 + 1, int(math.ceil(nb[3] * gh)))
            if self._gear_now[gy0:gy1, gx0:gx1].mean() > 0.25:
                return None
        X0, Y0 = int(round(tx0)), int(round(ty0))
        x0, y0, x1, y1 = max(0, X0), max(0, Y0), min(W, X0 + pw), min(H, Y0 + ph)
        if x1 - x0 < 2 or y1 - y0 < 2:
            return None
        patch = patch[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0]
        rgb = patch[..., :3].astype(np.float32)
        if self.mask == 'rect':
            a = np.ones(rgb.shape[:2], np.float32)
        else:
            a = patch[..., 3].astype(np.float32) / 255.0
            a = cv2.GaussianBlur(a, (0, 0), 0.8)
        bg = img[y0:y1, x0:x1].astype(np.float32)
        m = (a > 0.5).astype(np.uint8)
        if m.sum() < 6:
            return None
        if self.relight == 'local_range':
            ring = _ring(m, max(2, int(0.15 * min(m.shape))))
            if ring.sum() < 20:
                ring = m == 0
            if ring.sum() >= 20:
                lam = rng.uniform(*self.strength)
                px = bg[ring]
                inside = rgb[m > 0]
                for k in range(3):
                    lo_i, hi_i = np.percentile(inside[:, k], [5, 95])
                    lo_t, hi_t = np.percentile(px[:, k], [5, 95])
                    # the instance keeps its internal light and dark structure and takes the local range and colour cast
                    mapped = lo_t + (rgb[..., k] - lo_i) * (hi_t - lo_t) / max(4.0, hi_i - lo_i)
                    rgb[..., k] = lam * mapped + (1 - lam) * rgb[..., k]
        if self.relight == 'chroma':
            ring = _ring(m, max(2, int(0.15 * min(m.shape))))
            if ring.sum() < 20:
                ring = m == 0
            if ring.sum() >= 20:
                lam = rng.uniform(*self.strength)
                w = np.array([0.114, 0.587, 0.299], np.float32)
                med_t = np.median(bg[ring], axis=0) + 1.0
                med_s = (self.ring_med[i] if self.ring_med is not None else self.ring_m[i, :, 0]) + 1.0
                # von Kries adaptation to the colour cast of the place, and a mild brightness change (square root of the ratio of
                # background luminances, clipped); purely multiplicative, so dark animals stay dark and bright ones bright
                cast = (med_t / float(med_t @ w)) / (med_s / float(med_s @ w))
                gl = float(np.clip(np.sqrt(float(med_t @ w) / float(med_s @ w)), 0.67, 1.5))
                gain = np.clip(cast * gl, 0.3, 3.0)
                rgb = lam * rgb * gain[None, None, :] + (1 - lam) * rgb
        if self.relight == 'median_gain':
            ring = _ring(m, max(2, int(0.15 * min(m.shape))))
            if ring.sum() < 20:
                ring = m == 0
            if ring.sum() >= 20:
                px = bg[ring]
                lam = rng.uniform(*self.strength)
                # one gain for all channels from the luminance spread of the two rings, clipped, and per-channel medians matched:
                # the animal keeps its contrast to the background (scaled by g) and takes the brightness and colour cast of the place
                lum_t = px @ np.array([0.114, 0.587, 0.299], np.float32)
                spread_s = float(np.dot(self.ring_q[i, :, 1] - self.ring_q[i, :, 0], [0.114, 0.587, 0.299]))
                spread_t = float(np.percentile(lum_t, 95) - np.percentile(lum_t, 5))
                g = float(np.clip(spread_t / max(4.0, spread_s), 0.5, 2.0))
                for k in range(3):
                    med_s = self.ring_m[i, k, 0] if self.ring_med is None else self.ring_med[i, k]
                    med_t = float(np.median(px[:, k]))
                    rgb[..., k] = lam * (g * (rgb[..., k] - med_s) + med_t) + (1 - lam) * rgb[..., k]
        if self.relight in ('affine_q', 'affine_m'):
            ring = _ring(m, max(2, int(0.15 * min(m.shape))))
            if ring.sum() < 20:
                ring = m == 0
            if ring.sum() >= 20:
                px = bg[ring]
                lam = rng.uniform(*self.strength)
                for k in range(3):
                    if self.relight == 'affine_q':
                        s05, s95 = self.ring_q[i, k]
                        t05, t95 = np.percentile(px[:, k], [5, 95])
                        g = np.clip((t95 - t05) / max(4.0, s95 - s05), 0.3, 3.0)
                        off = t05 - g * s05
                    else:
                        sm, ss = self.ring_m[i, k]
                        tm, ts = px[:, k].mean(), px[:, k].std()
                        g = np.clip(ts / max(2.0, ss), 0.3, 3.0)
                        off = tm - g * sm
                    rgb[..., k] = lam * (g * rgb[..., k] + off) + (1 - lam) * rgb[..., k]
        if self.degrade:
            tl = cv2.Laplacian(cv2.cvtColor(np.clip(bg, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY), cv2.CV_32F).var()
            best, out = None, rgb
            for sig in (0.0, 0.6, 1.0, 1.5, 2.2):
                cand = cv2.GaussianBlur(rgb, (0, 0), sig) if sig > 0 else rgb
                pl = cv2.Laplacian(cv2.cvtColor(np.clip(cand, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY), cv2.CV_32F)[m > 0].var()
                d = abs(math.log((pl + 1) / (tl + 1)))
                if best is None or d < best:
                    best, out = d, cand
            rgb = out
            hi_t = np.median(np.abs(bg - cv2.GaussianBlur(bg, (0, 0), 1.0)))
            hi_s = np.median(np.abs(rgb - cv2.GaussianBlur(rgb, (0, 0), 1.0))[m > 0])
            if hi_t > hi_s:
                rgb = rgb + rng.normal(0, 1.25 * (hi_t - hi_s), rgb.shape)
        rgb = np.clip(rgb, 0, 255)
        if self.relight == 'poisson':
            src = img.copy()
            src[y0:y1, x0:x1] = rgb.astype(np.uint8)
            mm = np.zeros(img.shape[:2], np.uint8)
            mm[y0:y1, x0:x1] = (m * 255).astype(np.uint8)
            mm[0, :] = mm[-1, :] = mm[:, 0] = mm[:, -1] = 0
            if mm.sum() == 0:
                return None
            ys, xs = np.nonzero(mm)
            center = (int((xs.min() + xs.max()) / 2), int((ys.min() + ys.max()) / 2))
            try:
                img = cv2.seamlessClone(src, img, mm, center, cv2.NORMAL_CLONE)
            except cv2.error:
                return None
        else:
            comp = a[..., None] * rgb + (1 - a[..., None]) * bg
            if self.degrade:
                q = int(rng.integers(60, 96))
                ok, enc = cv2.imencode('.jpg', np.clip(comp, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, q])
                comp = cv2.imdecode(enc, cv2.IMREAD_COLOR).astype(np.float32) if ok else comp
            img = img.copy()
            img[y0:y1, x0:x1] = np.clip(comp, 0, 255).astype(np.uint8)
        return img, nb, c


@register('photo')
class Photo:
    def __init__(self, p=0.8, **kw):
        self.p = p

    def __call__(self, label, rng, dataset=None):
        if rng.random() >= self.p:
            return label
        img = label['img'].astype(np.float32)
        img *= rng.uniform(0.7, 1.3, 3)[None, None]                                    # colour gains
        img = np.clip(img + rng.uniform(-25, 25), 0, 255)                              # brightness
        if rng.random() < 0.3:
            lab = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2LAB)
            lab[..., 0] = cv2.createCLAHE(clipLimit=float(rng.uniform(1.5, 3.0)), tileGridSize=(8, 8)).apply(lab[..., 0])
            img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR).astype(np.float32)
        if rng.random() < 0.5:
            img = 255 * (img / 255) ** rng.uniform(0.7, 1.4)
        if rng.random() < 0.3:
            img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.5, 1.5))
        if rng.random() < 0.3:
            img = img + rng.normal(0, rng.uniform(2, 8), img.shape)
        label['img'] = np.clip(img, 0, 255).astype(np.uint8)
        return label


# habitat presets, particle size scaled from 1600-px to 640-px images
OC_PRESETS = {
    'DEEP_DARK': dict(density=120, gamma=1.25, blue=1.1, red=0.75, haze=0.12, noise=6.0, size=3.0),
    'BENTHIC_SILT': dict(density=250, gamma=1.1, blue=1.05, red=0.85, haze=0.2, noise=8.0, size=2.5),
    'REEF_CLEAR': dict(density=60, gamma=1.0, blue=1.0, red=0.95, haze=0.05, noise=2.0, size=3.0),
    'PELAGIC': dict(density=100, gamma=1.1, blue=1.15, red=0.8, haze=0.12, noise=5.0, size=3.0),
}
OC_ENV = {'fish': 'PELAGIC', 'seastar': 'BENTHIC_SILT', 'brittlestar': 'BENTHIC_SILT', 'urchin': 'BENTHIC_SILT', 'cucumber': 'BENTHIC_SILT',
          'crinoid': 'REEF_CLEAR', 'coral': 'REEF_CLEAR', 'anemone': 'REEF_CLEAR', 'seapen': 'REEF_CLEAR', 'sponge': 'REEF_CLEAR',
          'crustacean': 'BENTHIC_SILT', 'worm': 'BENTHIC_SILT', 'otherbio': 'PELAGIC'}


@register('ocenv')
class OCEnv:
    def __init__(self, p=0.5, **kw):
        self.p = p

    def __call__(self, label, rng, dataset=None):
        if rng.random() >= self.p:
            return label
        _, cls = _instances(label)
        env = OC_ENV[TAXA[int(rng.choice(cls))]] if len(cls) else str(rng.choice(list(OC_PRESETS)))
        q = OC_PRESETS[env]
        img = label['img'].astype(np.float32) / 255.0
        img = img ** (1.0 / q['gamma'])
        img[..., 0] *= q['blue']
        img[..., 2] *= q['red']
        img = np.clip(img, 0, 1) * 255
        img = img * (1 - q['haze']) + np.array([200, 200, 50], np.float32) * q['haze']
        H, W = img.shape[:2]
        ov = np.zeros((H, W), np.float32)
        for _ in range(q['density']):
            sz = int(np.clip(rng.normal(q['size'], 1.0) * 0.4, 1, 3))
            cv2.circle(ov, (int(rng.integers(W)), int(rng.integers(H))), sz, float(255 * np.clip(rng.normal(0.5, 0.1), 0.2, 0.9)), -1)
        img = img + 0.4 * cv2.GaussianBlur(ov, (3, 3), 0)[..., None]
        img = img + rng.normal(0, q['noise'], img.shape)
        label['img'] = np.clip(img, 0, 255).astype(np.uint8)
        return label


@register('ampmix')
class AmpMixAug:
    def __init__(self, p=0.5, lam_max=1.0, **kw):
        self.p, self.lam_max = p, lam_max

    def __call__(self, label, rng, dataset=None):
        if dataset is None or rng.random() >= self.p:
            return label
        img = label['img'].astype(np.float32) / 255.0
        other, _, _ = dataset.load_image(int(rng.integers(len(dataset))))
        other = cv2.resize(other, (img.shape[1], img.shape[0])).astype(np.float32) / 255.0
        f, g = np.fft.rfft2(img, axes=(0, 1)), np.fft.rfft2(other, axes=(0, 1))
        lam = rng.uniform(0, self.lam_max)
        amp = (1 - lam) * np.abs(f) + lam * np.abs(g)
        out = np.fft.irfft2(amp * np.exp(1j * np.angle(f)), s=img.shape[:2], axes=(0, 1))
        label['img'] = np.clip(out * 255, 0, 255).astype(np.uint8)
        return label


@register('lumiaug')
class LumiAugNP:
    """Per-image LumiAug; images here are BGR."""

    def __init__(self, p=0.5, emin=(0.15, 0.8), bmax=0.25, sigma=(0.25, 0.9), gain_sd=0.15, red=(0.6, 1.0), snow=(0, 300), **kw):
        self.p, self.emin, self.bmax, self.sigma, self.gain_sd, self.red, self.snow = p, emin, bmax, sigma, gain_sd, red, snow

    def __call__(self, label, rng, dataset=None):
        if rng.random() >= self.p:
            return label
        x = label['img'].astype(np.float32) / 255.0
        H, W = x.shape[:2]
        yy = np.linspace(-1, 1, H)[:, None]
        xx = np.linspace(-1, 1, W)[None, :] * (W / H)
        E = np.zeros((H, W), np.float32)
        for j in range(2 if rng.random() < 0.3 else 1):
            cy, cx = rng.uniform(-0.6, 0.6), rng.uniform(-0.6, 0.6) * (W / H)
            sg = rng.uniform(*self.sigma)
            E += (1.0 if j == 0 else rng.uniform(0.4, 1.0)) * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sg * sg))
        E /= max(E.max(), 1e-6)
        em = rng.uniform(*self.emin)
        E = (em + (1 - em) * E)[..., None]
        gam = np.exp(rng.normal(0, self.gain_sd, 3))
        gam[2] *= rng.uniform(*self.red)                                    # red is channel 2 in BGR
        Es = cv2.blur(E[..., 0], (31, 31))[..., None]
        A = np.array([rng.uniform(0.4, 0.9), rng.uniform(0.4, 0.8), rng.uniform(0.05, 0.3)], np.float32)  # B, G, R
        y = gam * E * x + rng.uniform(0, self.bmax) * A * Es
        n = rng.poisson(rng.uniform(*self.snow))
        seeds = np.zeros((H, W), np.float32)
        seeds[rng.integers(0, H, n), rng.integers(0, W, n)] = rng.uniform(0.3, 1.0, n)
        k = np.array([1., 4., 6., 4., 1.])
        snow = np.clip(cv2.filter2D(seeds, -1, np.outer(k, k) / 256.0 * 7.0), 0, 1)[..., None] * E
        label['img'] = np.clip((y + snow) * 255, 0, 255).astype(np.uint8)
        return label


@register('snow')
class RealSnow:
    """Adds a marine-snow layer taken from a real water-column frame of the training dives."""

    def __init__(self, bank, root, train_lists, p=0.5, gain=(0.5, 1.3), **kw):
        self.p, self.gain = p, tuple(gain)
        allowed = set()
        for lst in train_lists.split(','):
            allowed |= set(_names(root, lst))
        meta = json.load(open(os.path.join(bank, 'snow', 'snow.json')))
        self.files = [os.path.join(bank, 'snow', m['file']) for m in meta if m['image'] in allowed]

    def describe(self):
        return dict(layers=len(self.files))

    def __call__(self, label, rng, dataset=None):
        if not self.files or rng.random() >= self.p:
            return label
        lay = cv2.imread(self.files[int(rng.integers(len(self.files)))], cv2.IMREAD_GRAYSCALE)
        img = label['img']
        H, W = img.shape[:2]
        if rng.random() < 0.5:
            lay = lay[:, ::-1]
        s = rng.uniform(1.0, 1.6)
        lay = cv2.resize(lay, (int(W * s), int(H * s)))
        y0, x0 = int(rng.integers(0, lay.shape[0] - H + 1)), int(rng.integers(0, lay.shape[1] - W + 1))
        lay = lay[y0:y0 + H, x0:x0 + W].astype(np.float32)
        out = img.astype(np.float32) + rng.uniform(*self.gain) * lay[..., None]
        label['img'] = np.clip(out, 0, 255).astype(np.uint8)
        return label


@register('gear')
class GearDistractor:
    """Pastes 1-2 unlabeled cut-outs of vehicle equipment from training frames; animal boxes more than 60 % covered are dropped."""

    def __init__(self, bank, root, train_lists, p=0.3, **kw):
        self.p = p
        allowed = set()
        for lst in train_lists.split(','):
            allowed |= set(_names(root, lst))
        self.dir = os.path.join(bank, 'gear', 'crops')
        self.items = [g for g in json.load(open(os.path.join(bank, 'gear', 'gear.json'))) if g['image'] in allowed and not g['touches_border']]

    def describe(self):
        return dict(cutouts=len(self.items))

    def __call__(self, label, rng, dataset=None):
        if not self.items or rng.random() >= self.p:
            return label
        boxes, cls = _instances(label)
        img = label['img'].copy()
        H, W = img.shape[:2]
        cover = np.zeros((H, W), np.uint8)
        for _ in range(int(rng.integers(1, 3))):
            g = self.items[int(rng.integers(len(self.items)))]
            crop = cv2.imread(os.path.join(self.dir, g['file']), cv2.IMREAD_UNCHANGED)
            if crop is None:
                continue
            s = rng.uniform(0.5, 1.2)
            if rng.random() < 0.5:
                crop = crop[:, ::-1]
            pw, ph = max(2, int(crop.shape[1] * s)), max(2, int(crop.shape[0] * s))
            if pw >= W or ph >= H:
                continue
            crop = cv2.resize(crop, (pw, ph))
            x0, y0 = int(rng.integers(0, W - pw)), int(rng.integers(0, H - ph))
            a = cv2.GaussianBlur(crop[..., 3].astype(np.float32) / 255.0, (0, 0), 0.8)[..., None]
            img[y0:y0 + ph, x0:x0 + pw] = (a * crop[..., :3] + (1 - a) * img[y0:y0 + ph, x0:x0 + pw]).astype(np.uint8)
            cover[y0:y0 + ph, x0:x0 + pw] |= (a[..., 0] > 0.5).astype(np.uint8)
        label['img'] = img
        if len(boxes):
            keep = []
            for b in boxes:
                X0, Y0 = int(max(0, (b[0] - b[2] / 2) * W)), int(max(0, (b[1] - b[3] / 2) * H))
                X1, Y1 = int(min(W, (b[0] + b[2] / 2) * W)) + 1, int(min(H, (b[1] + b[3] / 2) * H)) + 1
                keep.append(cover[Y0:Y1, X0:X1].mean() <= 0.6 if Y1 > Y0 and X1 > X0 else True)
            keep = np.array(keep, bool)
            label = _set_instances(label, boxes[keep], cls[keep])
        return label


class Chain:
    def __init__(self, parts):
        self.parts = parts

    def describe(self):
        return [p.describe() if hasattr(p, 'describe') else type(p).__name__ for p in self.parts]

    def __call__(self, label, rng, dataset=None):
        for p in self.parts:
            label = p(label, rng, dataset)
        return label
