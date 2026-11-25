"""Copy of the instance bank at the training scale: crops of the instances that pass the quality rules, resized to the scale of a
640-px image and stored as lossy WebP with alpha (quality 92); the background-ring statistics are recomputed at that scale.

    python code/shrink_bank.py --src bank_full --dst bank

The bank used for training in the paper is this copy.  Snow layers and equipment files, if present, are copied unchanged.
"""
import argparse
import csv
import os
import shutil
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_bank import ring_stats  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--dst', required=True)
    ap.add_argument('--long', type=int, default=640)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.dst, 'crops'), exist_ok=True)
    rows = [r for r in csv.DictReader(open(os.path.join(a.src, 'bank.csv'))) if r['ok'] == '1']
    out = []
    for r in rows:
        im = cv2.imread(os.path.join(a.src, 'crops', r['id'] + '.png'), cv2.IMREAD_UNCHANGED)
        s = a.long / 1280.0
        im = cv2.resize(im, (max(1, round(im.shape[1] * s)), max(1, round(im.shape[0] * s))), interpolation=cv2.INTER_AREA)
        f = r['id'] + '.webp'
        cv2.imwrite(os.path.join(a.dst, 'crops', f), im, [cv2.IMWRITE_WEBP_QUALITY, 92])
        st = ring_stats(im[..., :3], (im[..., 3] > 127).astype(np.uint8), width=max(2, round(0.15 * min(im.shape[:2]))))
        r = dict(r)
        r.update({k: round(float(v), 3) for k, v in st.items()})
        r.update(crop_file=f, crop_long=a.long, crop_w=im.shape[1], crop_h=im.shape[0])
        out.append(r)
    with open(os.path.join(a.dst, 'bank.csv'), 'w', newline='') as fo:
        w = csv.DictWriter(fo, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    for d in ['snow', 'gear']:
        if not os.path.isdir(os.path.join(a.src, d)):
            continue
        if os.path.exists(os.path.join(a.dst, d)):
            shutil.rmtree(os.path.join(a.dst, d))
        shutil.copytree(os.path.join(a.src, d), os.path.join(a.dst, d))
    print(len(out), 'instances written')


if __name__ == '__main__':
    main()
