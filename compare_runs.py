"""Paired MAE difference between two finished evaluation runs on the same images (report Bảng 6, ∆MAE ghép cặp).

Reads predictions.csv of both runs; images must match one-to-one. CI = block bootstrap within scene (5 images,
2000 resamples, seed 0) applied to the same resampled images for both runs. No image is processed, so this does not
re-run the test set; it only combines two runs already logged in outputs/test_runs.log.

    uv run python compare_runs.py --a outputs/experiments/<baseline_run> --b outputs/experiments/<final_run>
"""
import argparse
from object_counting.io import read_csv, resolve, write_json
from object_counting.metrics import paired_mae_difference


def load(run):
    rows = [r for r in read_csv(resolve(run)/'predictions.csv') if r['gt'] != '']
    return {r['image_id']: r for r in rows}, [r['image_id'] for r in rows]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--a', required=True, help='reference run (e.g. baseline)')
    parser.add_argument('--b', required=True, help='compared run (e.g. final)')
    parser.add_argument('--out', help='optional JSON output path')
    args = parser.parse_args(argv)
    a, order = load(args.a); b, _ = load(args.b)
    if set(a) != set(b):
        raise SystemExit('Runs do not cover the same labelled images.')
    gt = [int(a[i]['gt']) for i in order]
    pa, pb = [int(a[i]['pred_cv']) for i in order], [int(b[i]['pred_cv']) for i in order]
    groups = [a[i]['scene_id'] for i in order]
    d = paired_mae_difference(pa, pb, gt, groups)
    result = dict(a=str(args.a), b=str(args.b), N=len(order), **d, meaning='MAE(b) - MAE(a); negative = b better')
    print(f"N={len(order)}  MAE(b)−MAE(a) = {d['MAE_diff']:+.3f}  KTC 95% [{d['CI95'][0]:.2f}, {d['CI95'][1]:.2f}]")
    if args.out:
        write_json(resolve(args.out), result)


if __name__ == '__main__':
    main()
