"""Train YOLO26 on the mixed benthic-fauna set with one training-time augmentation, then predict the evaluation lists.

    python code/train.py --root data --train cruise_trainval,oc_train --val oc_val --aug base --seed 0 --name t_B0_base \
        --predict test_cruise,oc_test [--model yolo26n] [--epochs 100] [--init weights.pt]

--root is the data folder with lists/, images/ and labels/ (see the README).  A list name may carry a repeat factor, e.g.
--train cruise_trainval=2,oc_train.

Every run uses the same recipe: COCO-pretrained weights, 640 px, batch 32, MuSGD with lr0 0.01 and momentum 0.9 (pinned, so
that the optimizer does not change with the size of the training set), Ultralytics default augmentation (mosaic closed in the
last 10 epochs, HSV jitter, horizontal flip, translation, scale), a fixed number of epochs and the weights of the last epoch.
The augmentation under study is inserted per image before mosaic (augment.py).  Predictions on the lists given with --predict
are written to <run>/pred_<list>.json and scored by evaluate.py.
"""
import argparse
import json
import os
import shutil
import sys
import time

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
TAXA = ['fish', 'seastar', 'brittlestar', 'urchin', 'cucumber', 'crinoid', 'coral', 'anemone', 'seapen', 'sponge', 'crustacean', 'worm',
        'otherbio']


def names_of(root, lst):
    return [os.path.basename(l.strip())[:-4] for l in open(os.path.join(root, 'lists', lst + '.txt')) if l.strip()]


def rfs_repeats(root, names, t, seed):
    """Repeat-factor sampling (Gupta et al., LVIS): r_i = max over the classes c in image i of max(1, sqrt(t / f_c)), f_c the share
    of training images that contain c; fractional repeats are rounded stochastically once per run."""
    import math
    import random
    cls_of = []
    for n in names:
        cls_of.append({int(l.split()[0]) for l in open(os.path.join(root, 'labels', n + '.txt')) if l.strip()})
    f = {}
    for cs in cls_of:
        for c in cs:
            f[c] = f.get(c, 0) + 1
    rnd = random.Random(seed)
    reps = []
    for cs in cls_of:
        r = max([1.0] + [math.sqrt(t / (f[c] / len(names))) for c in cs])
        reps.append(int(r) + (1 if rnd.random() < r - int(r) else 0))
    return reps


def write_list(root, run_dir, spec, tag, rfs=0.0, seed=0):
    """spec 'a,b' or 'a=2,b' (list name and repeat factor) -> one txt list of absolute image paths."""
    paths, names = [], []
    for item in spec.split(','):
        name, _, rep = item.partition('=')
        names += names_of(root, name) * int(rep or 1)
    reps = rfs_repeats(root, names, rfs, seed) if rfs > 0 else [1] * len(names)
    for n, r in zip(names, reps):
        paths += [os.path.join(root, 'images', n + '.jpg')] * r
    out = os.path.join(run_dir, f'list_{tag}.txt')
    with open(out, 'w') as f:
        f.writelines(p + '\n' for p in paths)
    return out, len(paths)


def letterbox_val(im):
    """Pad without scaling to the shape Ultralytics validation uses for a one-image batch (rect, pad 0.5, stride 32)."""
    import cv2
    import math
    h, w = im.shape[:2]
    H, W = math.ceil(h / 32 + 0.5) * 32, math.ceil(w / 32 + 0.5) * 32
    dy, dx = (H - h) // 2, (W - w) // 2
    out = cv2.copyMakeBorder(im, dy, H - h - dy, dx, W - w - dx, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return out, dx, dy


def predict(model, root, lst, out, device, batch=32):
    """Predictions with the input geometry of Ultralytics validation, boxes normalized to the image.  Images are batched by padded
    shape because the test sets hold more than one camera; each image is padded on its own, so the grouping changes no prediction."""
    import cv2
    res = {}
    pending = {}  # padded shape -> [(name, image, (w, h, dx, dy))]

    def flush(shape):
        chunk = pending.pop(shape)
        for (n, _, (w, h, dx, dy)), r in zip(chunk, model.predict([x[1] for x in chunk], imgsz=list(shape), conf=0.001, max_det=300,
                                                                   device=device, verbose=False)):
            b = r.boxes.xyxy.cpu().numpy().copy()
            b[:, [0, 2]] = ((b[:, [0, 2]] - dx) / w).clip(0, 1)
            b[:, [1, 3]] = ((b[:, [1, 3]] - dy) / h).clip(0, 1)
            res[n] = dict(xyxyn=[[round(float(v), 5) for v in x] for x in b], conf=[round(float(v), 5) for v in r.boxes.conf.tolist()],
                          cls=[int(v) for v in r.boxes.cls.tolist()])

    for n in names_of(root, lst):
        im = cv2.imread(os.path.join(root, 'images', n + '.jpg'))
        pim, dx, dy = letterbox_val(im)
        pending.setdefault(pim.shape[:2], []).append((n, pim, (im.shape[1], im.shape[0], dx, dy)))
        if len(pending[pim.shape[:2]]) == batch:
            flush(pim.shape[:2])
    for shape in list(pending):
        flush(shape)
    json.dump(res, open(out, 'w'))
    return len(res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--train', required=True)
    ap.add_argument('--val', default='cruise_val')
    ap.add_argument('--predict', default='')
    ap.add_argument('--aug', default='base')
    ap.add_argument('--aug-args', default='{}', help='json dict passed to the augmentation')
    ap.add_argument('--hyp', default='{}', help='json dict of extra Ultralytics training arguments, e.g. {"mixup": 0.15}')
    ap.add_argument('--rfs', type=float, default=0.0, help='repeat-factor sampling threshold t (0 = off)')
    ap.add_argument('--model', default='yolo26n')
    ap.add_argument('--weights-dir', default='', help='folder with the COCO weights <model>.pt; empty: Ultralytics downloads them')
    ap.add_argument('--init', default=None, help='start from these weights instead of COCO (fine-tuning)')
    ap.add_argument('--epochs', type=int, default=150)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--name', required=True)
    ap.add_argument('--project', default='runs')
    # fewer data-loader processes suit a machine short of memory; the runs are not deterministic anyway
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--device', default='0')
    a = ap.parse_args()
    # absolute paths, because Ultralytics resolves the relative paths of data.yaml against the data root or its own data folder
    a.root, a.project = os.path.abspath(a.root), os.path.abspath(a.project)

    run = f'{a.name}_s{a.seed}'
    run_dir = os.path.join(a.project, run)
    if os.path.exists(os.path.join(run_dir, 'done.json')):
        print('already done', run_dir)
        return
    os.makedirs(run_dir, exist_ok=True)
    train_txt, n_train = write_list(a.root, run_dir, a.train, 'train', rfs=a.rfs, seed=a.seed)
    val_txt, n_val = write_list(a.root, run_dir, a.val, 'val')
    data = dict(path=a.root, train=train_txt, val=val_txt, names={i: n for i, n in enumerate(TAXA)})
    data_yaml = os.path.join(run_dir, 'data.yaml')
    yaml.safe_dump(data, open(data_yaml, 'w'))

    import augment  # noqa: E402  (patches the Ultralytics dataset before the dataloaders are built)
    aug_args = json.loads(a.aug_args)
    if 'bank' in aug_args:
        aug_args['bank'] = os.path.abspath(aug_args['bank'])
    aug_args.setdefault('root', a.root)
    aug_args.setdefault('train_lists', ','.join(x.partition('=')[0] for x in a.train.split(',') if not x.startswith('oc_')))
    aug = augment.install(a.aug, aug_args, seed=a.seed)
    if aug is not None and hasattr(aug, 'describe'):
        json.dump(aug.describe(), open(os.path.join(run_dir, 'aug_describe.json'), 'w'), indent=1)
    hyp = json.loads(a.hyp)
    from ultralytics import YOLO  # noqa: E402
    import ultralytics.models.yolo.detect.train as detect_train  # noqa: E402

    # Ultralytics gives the validation loader twice the training workers; the held-out lists are small, and the extra processes
    # take about 2 GB of memory per run.  Validation is unaugmented and only logged.
    build = detect_train.build_dataloader
    detect_train.build_dataloader = lambda dataset, batch, workers, shuffle=True, **kw: build(
        dataset, batch, workers if shuffle else min(workers, a.workers), shuffle=shuffle, **kw)

    init = a.init or os.path.join(a.weights_dir, f'{a.model}.pt')
    wdir = os.path.join(run_dir, 'weights')
    csv = os.path.join(run_dir, 'results.csv')
    rows = open(csv).read().strip().splitlines() if os.path.exists(csv) else []
    if os.path.exists(os.path.join(wdir, 'last.pt')) and len(rows) > 1 and int(float(rows[-1].split(',')[0])) == a.epochs:
        # an earlier attempt finished training and failed afterwards (while predicting), so only the predictions are redone
        train_min = float(rows[-1].split(',')[1]) / 60
        print('training already finished, predicting with', os.path.join(wdir, 'last.pt'))
    else:
        model = YOLO(init)
        t0 = time.time()
        model.train(data=data_yaml, epochs=a.epochs, patience=a.epochs, batch=a.batch, imgsz=640, seed=a.seed, deterministic=False,
                    optimizer='MuSGD', lr0=0.01, momentum=0.9, project=a.project, name=run, exist_ok=True, workers=a.workers,
                    device=a.device, plots=False, val=True, save_period=-1, amp=True, cache=False, **hyp)
        train_min = (time.time() - t0) / 60
    if os.path.exists(os.path.join(wdir, 'best.pt')):
        os.remove(os.path.join(wdir, 'best.pt'))
    last = YOLO(os.path.join(wdir, 'last.pt'))
    preds = {}
    for lst in [x for x in a.predict.split(',') if x]:
        preds[lst] = predict(last, a.root, lst, os.path.join(run_dir, f'pred_{lst}.json'), a.device)
    done = dict(name=a.name, seed=a.seed, aug=a.aug, aug_args=json.loads(a.aug_args), hyp=hyp, rfs=a.rfs, model=a.model, init=init, train=a.train,
                val=a.val, n_train=n_train, n_val=n_val, epochs=a.epochs, train_min=round(train_min, 1), predicted=preds,
                host=os.uname().nodename, predict_geometry='ultralytics-val')
    json.dump(done, open(os.path.join(run_dir, 'done.json'), 'w'), indent=1)
    print(json.dumps(done))


if __name__ == '__main__':
    main()
