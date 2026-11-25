"""Collect finished runs and score them.

    python code/aggregate.py --root data --runs runs/cv --out results/cv.csv                (cross-validation runs)
    python code/aggregate.py --root data --runs runs/test --test --out results/test.csv [--boot 1000]

Cross-validation runs are named cv_<method>_f<k>_s<seed>; the predictions of the four fold models on their held-out dives are merged
and scored as one set (all 1785 cruise training and validation frames).  Test runs are named t_<method>_s<seed> and scored on the
pooled cruise test sets, on each of T1 to T5 and on the archive test set.  One csv row per method and seed; a cached score json is kept
next to each prediction file.
"""
import argparse
import csv
import glob
import json
import os
import re
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from evaluate import evaluate, names_of, TAXA  # noqa: E402


def flat(prefix, r):
    out = {f'{prefix}mAP50': r['mAP50'], f'{prefix}mAP50-95': r['mAP50-95'], f'{prefix}rare_AP50': r['rare_AP50'],
           f'{prefix}agnostic_AP50': r['agnostic_AP50'], f'{prefix}fp_per_empty': r['fp_per_empty_image']}
    for t in TAXA:
        out[f'{prefix}AP50_{t}'] = r['AP50'][t]
    if 'ci95' in r:
        for k, (lo, hi) in r['ci95'].items():
            out[f'{prefix}{k}_lo'], out[f'{prefix}{k}_hi'] = lo, hi
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--runs', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--test', action='store_true')
    ap.add_argument('--boot', type=int, default=0)
    a = ap.parse_args()
    dirs = []
    for d in a.runs.split(','):
        dirs += [x for x in glob.glob(os.path.join(d, '*')) if os.path.exists(os.path.join(x, 'done.json'))]
    rows = []
    if not a.test:
        groups = defaultdict(dict)
        for d in dirs:
            m = re.match(r'cv_(.+)_f(\d)_s(\d+)$', os.path.basename(d))
            if m:
                groups[(m.group(1), int(m.group(3)))][int(m.group(2))] = d
        names = names_of(a.root, 'cruise_trainval')
        for (method, seed), folds in sorted(groups.items()):
            if len(folds) < 4:
                continue
            pred = {}
            for k, d in folds.items():
                pred.update(json.load(open(os.path.join(d, f'pred_fold{k}_heldout.json'))))
            r = evaluate(a.root, pred, names, boot=a.boot)
            row = dict(method=method, seed=seed, n_images=r['n_images'],
                       train_min=sum(json.load(open(os.path.join(d, 'done.json')))['train_min'] for d in folds.values()))
            row.update(flat('cv_', r))
            rows.append(row)
    else:
        for d in dirs:
            m = re.match(r't_(.+)_s(\d+)$', os.path.basename(d))
            if not m:
                continue
            row = dict(method=m.group(1), seed=int(m.group(2)), train_min=json.load(open(os.path.join(d, 'done.json')))['train_min'])
            split = {r['name']: r['split'] for r in csv.DictReader(open(os.path.join(a.root, 'images.csv')))}
            cache = os.path.join(d, f'scores_boot{a.boot}.json')
            if os.path.exists(cache) and os.path.getmtime(cache) > os.path.getmtime(os.path.join(d, 'pred_test_cruise.json')):
                sc = json.load(open(cache))
            else:
                pred = json.load(open(os.path.join(d, 'pred_test_cruise.json')))
                names = names_of(a.root, 'test_cruise')
                sc = {'test': evaluate(a.root, pred, names, boot=a.boot)}
                for s in ['T1', 'T2', 'T3', 'T4', 'T5']:
                    sc[s] = evaluate(a.root, pred, [n for n in names if split[n] == s], boot=a.boot)
                if os.path.exists(os.path.join(d, 'pred_oc_test.json')):
                    sc['oc'] = evaluate(a.root, json.load(open(os.path.join(d, 'pred_oc_test.json'))), names_of(a.root, 'oc_test'))
                json.dump(sc, open(cache, 'w'))
            for k, r in sc.items():
                row.update(flat(f'{k}_', r))
            rows.append(row)
    rows.sort(key=lambda r: (r['method'], r['seed']))
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    prefix = 'test_' if a.test else 'cv_'
    for r in rows:
        print(f"{r['method']:28s} s{r['seed']}  mAP50 {r[prefix + 'mAP50']:.4f}  rare {r[prefix + 'rare_AP50']:.4f}  "
              f"agnostic {r[prefix + 'agnostic_AP50']:.4f}")


if __name__ == '__main__':
    main()
