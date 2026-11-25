"""Score predictions of train.py against the evaluation labels (labels14/: the 13 taxa and class 13, bio_tbd, an animal of
unknown taxon).

    python code/evaluate.py --root data --pred <run>/pred_test_cruise.json --list test_cruise [--by-split] [--boot 1000]
    python code/evaluate.py --root data --pred a.json,b.json,c.json,d.json --list cruise_trainval   (pooled cross-validation)

Matching follows COCO: per image and class, detections in descending confidence take the unmatched ground-truth box with the
highest IoU at or above the threshold.  A detection that matches nothing and lies for at least half of its area inside a bio_tbd
box (an animal whose taxon is unknown) is ignored.  AP is the area under the monotone precision envelope sampled at 101 recall
points, as in Ultralytics.  mAP50 averages AP50 over the main classes (taxa with at least 20 boxes in the pooled cruise test
sets, fixed in eval_classes.json of the data folder, else in data/eval_classes.json of this repository).  The rare group pools the
detections and boxes of the remaining named taxa into one precision-recall curve; a detection counts only when its taxon is right.
Class-agnostic AP50 treats all 13 classes as one.
"""
import argparse
import csv
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TAXA = ['fish', 'seastar', 'brittlestar', 'urchin', 'cucumber', 'crinoid', 'coral', 'anemone', 'seapen', 'sponge', 'crustacean', 'worm',
        'otherbio']
IGNORE = 13
IOUS = np.linspace(0.5, 0.95, 10)


def load_classes(root=None):
    # a data folder may fix its own classes (<root>/eval_classes.json); otherwise data/eval_classes.json of this repository
    f = os.path.join(root, 'eval_classes.json') if root else None
    c = json.load(open(f if f and os.path.exists(f) else os.path.join(HERE, '..', 'data', 'eval_classes.json')))
    return [TAXA.index(x) for x in c['main']], [TAXA.index(x) for x in c['rare']]


def xywh2xyxy(b):
    b = np.asarray(b, float).reshape(-1, 4)
    return np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], 1)


def load_gt(root, names, gt_dir='labels14'):
    gt = {}
    for n in names:
        rows = [l.split() for l in open(os.path.join(root, gt_dir, n + '.txt')) if l.strip()]
        c = np.array([int(r[0]) for r in rows], int)
        b = xywh2xyxy([[float(v) for v in r[1:5]] for r in rows]) if rows else np.zeros((0, 4))
        gt[n] = (b[c != IGNORE], c[c != IGNORE], b[c == IGNORE])
    return gt


def iou(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(2)
    aa = (a[:, 2:] - a[:, :2]).prod(1)
    bb = (b[:, 2:] - b[:, :2]).prod(1)
    return inter / (aa[:, None] + bb[None, :] - inter + 1e-12)


def ioa(a, b):
    """Intersection over the area of each box in a."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(2)
    return inter / ((a[:, 2:] - a[:, :2]).prod(1)[:, None] + 1e-12)


def match(db, ds, gb, ign):
    """TP flags (n_det x 10 IoU thresholds) and a keep mask (False = ignored) for detections of one class in one image."""
    order = np.argsort(-ds, kind='stable')
    tp = np.zeros((len(db), len(IOUS)), bool)
    keep = np.ones((len(db), len(IOUS)), bool)
    m = iou(db, gb)
    near_ign = (ioa(db, ign).max(1) >= 0.5) if len(ign) else np.zeros(len(db), bool)
    for t, thr in enumerate(IOUS):
        used = np.zeros(len(gb), bool)
        for i in order:
            if len(gb):
                cand = np.where(~used & (m[i] >= thr))[0]
                if len(cand):
                    j = cand[np.argmax(m[i, cand])]
                    used[j] = True
                    tp[i, t] = True
                    continue
            if near_ign[i]:
                keep[i, t] = False
    return tp, keep


def ap_from(scores, tp, n_gt, w=None):
    """AP of one curve; w (optional) are bootstrap weights per detection, n_gt then the weighted box count."""
    if n_gt <= 0:
        return np.nan
    o = np.argsort(-scores, kind='stable')
    t = tp[o].astype(float)
    ww = np.ones_like(t) if w is None else w[o]
    ctp = np.cumsum(t * ww)
    cfp = np.cumsum((1 - t) * ww)
    rec = ctp / n_gt
    prec = ctp / np.maximum(ctp + cfp, 1e-12)
    mrec = np.concatenate([[0.0], rec, [rec[-1] if len(rec) else 1.0], [1.0]])  # precision is 0 beyond the highest recall
    mpre = np.concatenate([[1.0], prec, [0.0], [0.0]])
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    return float(np.trapezoid(np.interp(x, mrec, mpre), x))


def collect(pred, gt, names, conf_thr):
    """Per-class arrays of (score, tp[10], image index, keep[10]), the box count per class and image, and false positives."""
    per = {c: [] for c in range(len(TAXA))}
    agn = []
    n_gt = np.zeros((len(names), len(TAXA)), int)
    empty_fp = []
    for k, n in enumerate(names):
        gb, gc, ign = gt[n]
        for c in range(len(TAXA)):
            n_gt[k, c] = int((gc == c).sum())
        p = pred.get(n, dict(xyxyn=[], conf=[], cls=[]))
        db = np.asarray(p['xyxyn'], float).reshape(-1, 4)
        ds = np.asarray(p['conf'], float)
        dc = np.asarray(p['cls'], int)
        for c in range(len(TAXA)):
            sel = dc == c
            if not sel.any():
                continue
            tp, keep = match(db[sel], ds[sel], gb[gc == c], ign)
            per[c].append((ds[sel], tp, np.full(sel.sum(), k), keep))
        tp, keep = match(db, ds, gb, ign)
        agn.append((ds, tp, np.full(len(ds), k), keep))
        if len(gb) == 0 and len(ign) == 0:
            empty_fp.append(int((ds >= conf_thr).sum()))
    return {c: concat(v) for c, v in per.items()}, concat(agn), n_gt, empty_fp


def concat(parts):
    if not parts:
        return np.zeros(0), np.zeros((0, len(IOUS)), bool), np.zeros(0, int), np.zeros((0, len(IOUS)), bool)
    return tuple(np.concatenate([p[i] for p in parts]) for i in range(4))


def score_curves(per, agn, n_gt, main, rare, w_img=None):
    w_all = np.ones(len(n_gt)) if w_img is None else w_img

    def one(s, tp, img, keep, ngt_c, t):
        m = keep[:, t]
        w = None if w_img is None else w_img[img[m]]
        return ap_from(s[m], tp[m, t], ngt_c, w)

    def wsum(cols):
        return float((n_gt[:, np.atleast_1d(cols)].sum(1) * w_all).sum())

    out = {'AP50': {}, 'AP50-95': {}}
    for c in range(len(TAXA)):
        s, tp, img, keep = per[c]
        g = wsum(c)
        out['AP50'][TAXA[c]] = one(s, tp, img, keep, g, 0)
        out['AP50-95'][TAXA[c]] = float(np.nanmean([one(s, tp, img, keep, g, t) for t in range(10)])) if g > 0 else np.nan
    out['mAP50'] = float(np.nanmean([out['AP50'][TAXA[c]] for c in main]))
    out['mAP50-95'] = float(np.nanmean([out['AP50-95'][TAXA[c]] for c in main]))
    rs = [per[c] for c in rare]
    s, tp, img, keep = (np.concatenate([r[i] for r in rs]) for i in range(4))
    out['rare_AP50'] = one(s, tp, img, keep, wsum(rare), 0)
    s, tp, img, keep = agn
    out['agnostic_AP50'] = one(s, tp, img, keep, wsum(list(range(len(TAXA)))), 0)
    return out


def evaluate(root, pred, names, conf_thr=0.25, boot=0, seed=0, gt_dir='labels14', keep_boot=False):
    main, rare = load_classes(root)
    gt = load_gt(root, names, gt_dir)
    per, agn, n_gt, empty_fp = collect(pred, gt, names, conf_thr)
    res = score_curves(per, agn, n_gt, main, rare)
    res['n_images'] = len(names)
    res['n_boxes'] = {TAXA[c]: int(n_gt[:, c].sum()) for c in range(len(TAXA))}
    res['empty_images'] = len(empty_fp)
    res['fp_per_empty_image'] = float(np.mean(empty_fp)) if empty_fp else np.nan
    if boot:
        rng = np.random.default_rng(seed)
        vals = {'mAP50': [], 'rare_AP50': [], 'agnostic_AP50': []}
        for _ in range(boot):
            w = np.bincount(rng.integers(0, len(names), len(names)), minlength=len(names)).astype(float)
            r = score_curves(per, agn, n_gt, main, rare, w_img=w)
            for k in vals:
                vals[k].append(r[k])
        res['ci95'] = {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k, v in vals.items()}
        if keep_boot:  # the same seed and image list give the same resamples, so runs can be averaged per resample
            res['boot_values'] = {k: [float(x) for x in v] for k, v in vals.items()}
    return res


def names_of(root, lst):
    return [os.path.basename(l.strip())[:-4] for l in open(os.path.join(root, 'lists', lst + '.txt')) if l.strip()]


def main_cli():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--pred', required=True, help='one json, or several (comma) that are merged, e.g. the folds of a CV run')
    ap.add_argument('--list', required=True)
    ap.add_argument('--by-split', action='store_true', help='also score each cruise test set T1..T5')
    ap.add_argument('--boot', type=int, default=0)
    ap.add_argument('--conf-thr', type=float, default=0.25)
    ap.add_argument('--out', default=None)
    ap.add_argument('--gt-dir', default='labels14', help='labels (13 classes, no ignore region) reproduces Ultralytics val')
    a = ap.parse_args()
    pred = {}
    for p in a.pred.split(','):
        pred.update(json.load(open(p)))
    names = names_of(a.root, a.list)
    res = {'all': evaluate(a.root, pred, names, a.conf_thr, a.boot, gt_dir=a.gt_dir)}
    if a.by_split:
        split = {r['name']: r['split'] for r in csv.DictReader(open(os.path.join(a.root, 'images.csv')))}
        for s in ['T1', 'T2', 'T3', 'T4', 'T5']:
            sub = [n for n in names if split.get(n) == s]
            if sub:
                res[s] = evaluate(a.root, pred, sub, a.conf_thr, a.boot, gt_dir=a.gt_dir)
    if a.out:
        json.dump(res, open(a.out, 'w'), indent=1)
    for k, r in res.items():
        print(k, 'mAP50 %.4f mAP50-95 %.4f rare %.4f agnostic %.4f fp/empty %.3f' % (r['mAP50'], r['mAP50-95'], r['rare_AP50'],
                                                                                    r['agnostic_AP50'], r['fp_per_empty_image']))


if __name__ == '__main__':
    main_cli()
