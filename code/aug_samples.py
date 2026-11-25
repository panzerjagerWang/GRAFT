"""Sample sheets of an augmentation: 16 cruise training images after the augmentation, with original boxes in green and added
boxes in magenta (taxon written next to them).

    python code/aug_samples.py --root data --aug graft --aug-args '{"bank": "bank"}' --out sheet.jpg [--list cruise_train] [--seed 0]
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import augment  # noqa: E402
from ultralytics.utils.instance import Instances  # noqa: E402

TAXA = ['fish', 'seastar', 'brittlestar', 'urchin', 'cucumber', 'crinoid', 'coral', 'anemone', 'seapen', 'sponge', 'crustacean', 'worm',
        'otherbio']


class ListDataset:
    def __init__(self, root, names):
        self.root, self.names = root, names

    def __len__(self):
        return len(self.names)

    def load_image(self, i):
        im = cv2.imread(os.path.join(self.root, 'images', self.names[i] + '.jpg'))
        return im, im.shape[:2], im.shape[:2]

    def label(self, i):
        n = self.names[i]
        rows = [l.split() for l in open(os.path.join(self.root, 'labels', n + '.txt')) if l.strip()]
        b = np.array([[float(v) for v in r[1:5]] for r in rows], np.float32).reshape(-1, 4)
        c = np.array([int(r[0]) for r in rows], np.float32).reshape(-1, 1)
        im, _, _ = self.load_image(i)
        return dict(img=im, im_file=os.path.join(self.root, 'images', n + '.jpg'), cls=c,
                    instances=Instances(b, np.zeros((0, 1000, 2), np.float32), None, bbox_format='xywh', normalized=True))


def draw(label, n_orig):
    img = label['img'].copy()
    H, W = img.shape[:2]
    inst = label['instances']
    inst.convert_bbox('xyxy')
    for j, (b, c) in enumerate(zip(inst.bboxes, label['cls'].reshape(-1))):
        x0, y0, x1, y1 = (int(v) for v in (b[0] * W, b[1] * H, b[2] * W, b[3] * H))
        col = (0, 220, 0) if j < n_orig else (255, 0, 255)
        cv2.rectangle(img, (x0, y0), (x1, y1), col, 1)
        if j >= n_orig:
            cv2.putText(img, TAXA[int(c)], (x0, max(10, y0 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--aug', required=True)
    ap.add_argument('--aug-args', default='{}')
    ap.add_argument('--list', default='cruise_train')
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n', type=int, default=16)
    a = ap.parse_args()
    names = [os.path.basename(l.strip())[:-4] for l in open(os.path.join(a.root, 'lists', a.list + '.txt')) if l.strip()]
    args = json.loads(a.aug_args)
    args.setdefault('root', a.root)
    args.setdefault('train_lists', a.list)
    aug = augment.build(a.aug, args)
    ds = ListDataset(a.root, names)
    rng = np.random.default_rng(a.seed)
    tiles = []
    for i in rng.choice(len(names), a.n, replace=False):
        lab = ds.label(int(i))
        n0 = len(lab['cls'])
        for _ in range(5):  # retry a few times so that most tiles show the augmentation acting
            out = aug(ds.label(int(i)), rng, ds)
            if len(out['cls']) > n0 or a.aug != 'graft':
                break
        tiles.append(draw(out, n0))
    h = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
    rows = [np.hstack(tiles[k:k + 4]) for k in range(0, len(tiles), 4)]
    cv2.imwrite(a.out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
    if hasattr(aug, 'describe'):
        print(json.dumps(aug.describe()))


if __name__ == '__main__':
    main()
