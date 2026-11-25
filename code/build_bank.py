"""Instance bank for transplantation: every annotated animal of the training sources, cut out with a SAM 2.1 mask.

    python code/build_bank.py --mix data --fullres <folder with images/ at full resolution> --out bank_full \
        --lists oc_train,cruise_trainval

--mix is the data folder (lists/, labels14/ and images.csv); the frames in <fullres>/images/ carry the same names.

For each box, SAM 2.1 (base) is prompted with the box on the full-resolution image.  The crop around the box (10 % margin) is
stored as RGBA PNG at the scale of a 1280-px image (so up to twice the training scale), with the mask as alpha channel.
bank.csv holds one row per instance with its class, source, dive, box geometry, mask quality and the statistics of a ring of
background around the mask (per-channel quantiles, mean and standard deviation, sharpness and noise), which the transplantation
uses to relight the instance at its new place.  Rows with ok=0 failed the quality rules and are not pasted.
"""
import argparse
import csv
import os

import cv2
import numpy as np

TAXA = ['fish', 'seastar', 'brittlestar', 'urchin', 'cucumber', 'crinoid', 'coral', 'anemone', 'seapen', 'sponge', 'crustacean', 'worm',
        'otherbio']


def ring_stats(img, mask, width):
    """Statistics of the pixels in a ring of `width` px around the mask (inside the crop)."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * width + 1, 2 * width + 1))
    ring = (cv2.dilate(mask, k) > 0) & (mask == 0)
    if ring.sum() < 30:
        ring = mask == 0
    if ring.sum() < 30:  # the mask fills the crop: use all pixels
        ring = np.ones_like(mask, bool)
    px = img[ring].astype(np.float32)
    lap = cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_32F)
    hi = img.astype(np.float32) - cv2.GaussianBlur(img, (0, 0), 1.0).astype(np.float32)
    out = {}
    for c, ch in enumerate('bgr'):
        q = np.percentile(px[:, c], [5, 50, 95])
        out.update({f'{ch}_q05': q[0], f'{ch}_q50': q[1], f'{ch}_q95': q[2], f'{ch}_mean': px[:, c].mean(), f'{ch}_std': px[:, c].std()})
    out['sharp'] = float(lap[ring].var())
    out['noise'] = float(np.median(np.abs(hi[ring])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mix', required=True)
    ap.add_argument('--fullres', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--lists', default='oc_train,cruise_trainval')
    ap.add_argument('--sam', default='sam2.1_b.pt', help='SAM 2.1 weights; Ultralytics downloads sam2.1_b.pt when it is missing')
    ap.add_argument('--scale-long', type=int, default=1280)
    a = ap.parse_args()
    from ultralytics import SAM
    sam = SAM(a.sam)
    os.makedirs(os.path.join(a.out, 'crops'), exist_ok=True)
    meta = {r['name']: r for r in csv.DictReader(open(os.path.join(a.mix, 'images.csv')))}
    names = []
    for lst in a.lists.split(','):
        names += [os.path.basename(l.strip())[:-4] for l in open(os.path.join(a.mix, 'lists', lst + '.txt')) if l.strip()]
    csv_path = os.path.join(a.out, 'bank.csv')
    done_imgs, k, writer, fcsv = set(), 0, None, None
    if os.path.exists(csv_path):
        old = list(csv.DictReader(open(csv_path)))
        done_imgs = {r['image'] for r in old}
        k = len(old)
    rows = []
    for i, n in enumerate(names):
        if n in done_imgs:
            continue
        lab = [l.split() for l in open(os.path.join(a.mix, 'labels14', n + '.txt')) if l.strip()]
        lab = [l for l in lab if int(l[0]) < len(TAXA)]  # no crops of bio_tbd boxes
        if not lab:
            continue
        img = cv2.imread(os.path.join(a.fullres, 'images', n + '.jpg'))
        H, W = img.shape[:2]
        boxes = []
        for l in lab:
            cx, cy, bw, bh = (float(v) for v in l[1:5])
            boxes.append([(cx - bw / 2) * W, (cy - bh / 2) * H, (cx + bw / 2) * W, (cy + bh / 2) * H])
        res = sam(img, bboxes=boxes, verbose=False)[0]
        masks = res.masks.data.cpu().numpy().astype(np.uint8) if res.masks is not None else np.zeros((0, H, W), np.uint8)
        s = a.scale_long / max(H, W)
        for j, (l, b) in enumerate(zip(lab, boxes)):
            m = masks[j] if j < len(masks) else np.zeros((H, W), np.uint8)
            x0, y0, x1, y1 = b
            bw, bh = x1 - x0, y1 - y0
            mx0, my0 = int(max(0, x0 - 0.1 * bw)), int(max(0, y0 - 0.1 * bh))
            mx1, my1 = int(min(W, x1 + 0.1 * bw + 1)), int(min(H, y1 + 0.1 * bh + 1))
            crop, cm = img[my0:my1, mx0:mx1], m[my0:my1, mx0:mx1]
            # mask quality: IoU of the mask's bounding box with the annotated box, largest component share, fill ratio
            ys, xs = np.nonzero(m)
            if len(xs):
                mb = [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]
                inter = max(0, min(mb[2], x1) - max(mb[0], x0)) * max(0, min(mb[3], y1) - max(mb[1], y0))
                box_iou = inter / ((mb[2] - mb[0]) * (mb[3] - mb[1]) + bw * bh - inter)
                nlab, cc, st, _ = cv2.connectedComponentsWithStats(cm, connectivity=8)
                main_share = st[1:, cv2.CC_STAT_AREA].max() / max(1, cm.sum()) if nlab > 1 else 0.0
                fill = m[int(y0):int(y1) + 1, int(x0):int(x1) + 1].mean()
            else:
                box_iou, main_share, fill = 0.0, 0.0, 0.0
            border = x0 < 2 or y0 < 2 or x1 > W - 2 or y1 > H - 2
            short_640 = min(bw, bh) * 640 / max(H, W)
            ok = int(box_iou >= 0.6 and main_share >= 0.7 and fill >= 0.08 and not border and short_640 >= 8)
            cs = cv2.resize(crop, (max(1, round(crop.shape[1] * s)), max(1, round(crop.shape[0] * s))), interpolation=cv2.INTER_AREA)
            ms = cv2.resize(cm * 255, (cs.shape[1], cs.shape[0]), interpolation=cv2.INTER_LINEAR)
            iid = f'{k:06d}'
            cv2.imwrite(os.path.join(a.out, 'crops', iid + '.png'), np.dstack([cs, ms]))
            st = ring_stats(cs, (ms > 127).astype(np.uint8), width=max(3, round(0.15 * min(cs.shape[:2]))))
            r = meta[n]
            rows.append(dict(id=iid, image=n, source=r['source'], split=r['split'], dive=r['dive'], camera=r['camera'], cls=int(l[0]),
                             taxon=TAXA[int(l[0])], img_w=W, img_h=H, box_x0=x0 / W, box_y0=y0 / H, box_x1=x1 / W, box_y1=y1 / H,
                             crop_x0=mx0 / W, crop_y0=my0 / H, crop_x1=mx1 / W, crop_y1=my1 / H, crop_w=cs.shape[1], crop_h=cs.shape[0],
                             box_iou=round(float(box_iou), 3), main_share=round(float(main_share), 3), fill=round(float(fill), 3),
                             border=int(border), ok=ok, **{kk: round(float(v), 3) for kk, v in st.items()}))
            k += 1
        if rows:
            if writer is None:
                new = not os.path.exists(csv_path)
                fcsv = open(csv_path, 'a', newline='')
                writer = csv.DictWriter(fcsv, fieldnames=list(rows[0]))
                if new:
                    writer.writeheader()
            writer.writerows(rows)
            fcsv.flush()
            rows = []
        if i % 200 == 0:
            print(i, len(names), k, flush=True)
    allrows = list(csv.DictReader(open(csv_path)))
    print(len(allrows), 'instances,', sum(r['ok'] == '1' for r in allrows), 'pass the quality rules')


if __name__ == '__main__':
    main()
