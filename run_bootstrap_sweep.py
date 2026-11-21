"""Choice of bootstrap block length (report §3, "Chỉ số và khoảng tin cậy"): 95% MAE interval of a finished DEV run
for block = 1, 3, 5, 10 consecutive images. Reads predictions.csv of an existing run; no image is processed.

    uv run python run_bootstrap_sweep.py --run outputs/experiments/<run_id>
"""
import argparse
from object_counting.io import ROOT, read_csv, resolve, write_json
from object_counting.metrics import metrics_with_ci
from object_counting.experiments import TEST_SPLITS


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, help='outputs/experiments/<run_id> of run_eval.py')
    parser.add_argument('--blocks', type=int, nargs='+', default=[1, 3, 5, 10])
    args = parser.parse_args(argv)
    run = resolve(args.run)
    rows = [r for r in read_csv(run/'predictions.csv') if r['gt'] != '' and r['kind'] in ('', 'none')]
    if any(r['split'] in TEST_SPLITS for r in rows):
        raise SystemExit('Block-length choice is made on DEV only.')
    # predictions.csv keeps manifest order, which is temporal within each scene.
    pred, gt, groups = [int(r['pred_cv']) for r in rows], [int(r['gt']) for r in rows], [r['scene_id'] for r in rows]
    out = {}
    for b in args.blocks:
        m = metrics_with_ci(pred, gt, groups, block=b)
        out[b] = dict(MAE=m['MAE'], MAE_CI95=m['MAE_CI95'], width=m['MAE_CI95'][1]-m['MAE_CI95'][0])
        print(f"block {b:>2}: MAE {m['MAE']:.3f}  KTC 95% [{m['MAE_CI95'][0]:.2f}, {m['MAE_CI95'][1]:.2f}]  rộng {out[b]['width']:.2f}")
    write_json(run/'bootstrap_block_sweep.json', dict(N=len(rows), blocks=out, samples=2000, seed=0))


if __name__ == '__main__':
    main()
