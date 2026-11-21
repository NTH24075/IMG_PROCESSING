"""Out-of-sample check of C3 (report §3.4, Bảng 6): K-fold cross-validation over contiguous TIME BLOCKS, DEV only.

For each scene, dev images are ordered by source_frame and split into K contiguous blocks. For every block,
p,q (k,c in the scene config) are refitted with the same rule as calibrate_scene.py — single-person blobs of the
BASELINE mask (exactly one point in the bbox grown 3 px) — using only the other K-1 blocks; the configured
pipeline (default final.toml) then runs on the held-out block with that model. Nothing is written to the scene
configs. Output: outputs/experiments/<run_id>/{cv_predictions.csv, cv_folds.csv, metrics.json}.
"""
import argparse
import json
import numpy as np
from object_counting.io import ROOT, read_csv, write_csv, write_json, load_params, load_scene, read_image, resolve, new_run, load_points
from object_counting.pipeline import count_image
from object_counting.counting import fit_perspective
from object_counting.experiments import REVIEWED, touches_test, git_state
from object_counting.metrics import metrics_with_ci, paired_mae_difference
from calibrate_scene import single_person_blobs


def time_blocks(n, k):
    """Indices 0..n-1 cut into k contiguous, nearly equal blocks (np.array_split)."""
    return [list(b) for b in np.array_split(np.arange(n), k) if len(b)]


def cross_validate(k, rows_by_scene, images, backgrounds, base_pairs, final_params, ratio=.3):
    """Returns {image_id: prediction} and one fold record per (scene, block)."""
    preds, folds = {}, []
    for scene_id, rows in rows_by_scene.items():
        scene, bg, roi = backgrounds[scene_id]
        for b, held in enumerate(time_blocks(len(rows), k)):
            held_ids = {rows[i]['image_id'] for i in held}
            pairs = [p for r in rows if r['image_id'] not in held_ids for p in base_pairs[r['image_id']]]
            kk, cc = fit_perspective(pairs)
            model = dict(k=kk, c=cc, min_area_ratio=ratio, head_alpha=scene.get('perspective', {}).get('head_alpha', .1))
            for i in held:
                r = rows[i]
                preds[r['image_id']] = count_image(images[r['image_id']], bg, final_params, roi, model).count
            folds.append(dict(K=k, scene_id=scene_id, block=b, held_out=len(held),
                              first_frame=rows[held[0]]['source_frame'], last_frame=rows[held[-1]]['source_frame'],
                              single_person_blobs=len(pairs), k=round(kk, 4), c=round(cc, 3)))
    return preds, folds


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/final.toml', help='pipeline evaluated on held-out blocks')
    parser.add_argument('--base', default='configs/baseline.toml', help='mask used to find single-person blobs (as calibrate_scene)')
    parser.add_argument('--k', type=int, nargs='+', default=[2, 5, 10])
    args = parser.parse_args(argv)

    rows = read_csv(ROOT/'data/week1/manifests/dev.csv')
    if touches_test(rows):
        raise SystemExit('Cross-validation is DEV only.')
    labels = {r['image_id']: r for r in read_csv(ROOT/'data/week1/annotations/counts.csv')}
    rows = [r for r in rows if labels.get(r['image_id'], {}).get('review_status') in REVIEWED]
    rows_by_scene = {}
    for r in sorted(rows, key=lambda r: (r['scene_id'], int(r['source_frame']))):
        rows_by_scene.setdefault(r['scene_id'], []).append(r)
    order = [r for s in rows_by_scene.values() for r in s]
    gt = [int(labels[r['image_id']]['count']) for r in order]
    groups = [r['scene_id'] for r in order]

    base_params, _ = load_params(args.base); final_params, _ = load_params(args.config)
    backgrounds = {s: load_scene(s) for s in rows_by_scene}
    points = load_points(ROOT)
    images = {r['image_id']: read_image(resolve(r['image'])) for r in order}
    base_pairs, baseline, insample = {}, [], []
    for r in order:
        scene, bg, roi = backgrounds[r['scene_id']]
        base = count_image(images[r['image_id']], bg, base_params, roi)
        base_pairs[r['image_id']] = single_person_blobs(base, points.get(r['image_id'], []))
        baseline.append(base.count)
        insample.append(count_image(images[r['image_id']], bg, final_params, roi, scene.get('perspective')).count)

    path = new_run('experiments')
    table, fold_rows = [], []
    summary = dict(protocol='K contiguous time blocks per scene; p,q refit on the other K-1 blocks from single-person '
                            'blobs of the baseline mask (margin 3 px); DEV only; CI = paired block bootstrap (5 images, 2000, seed 0)',
                   config=args.config, base=args.base, git=git_state(),
                   baseline=metrics_with_ci(baseline, gt, groups), in_sample=metrics_with_ci(insample, gt, groups), by_K={})
    for k in args.k:
        preds, folds = cross_validate(k, rows_by_scene, images, backgrounds, base_pairs, final_params)
        cv = [preds[r['image_id']] for r in order]
        res = dict(overall=metrics_with_ci(cv, gt, groups),
                   minus_in_sample=paired_mae_difference(insample, cv, gt, groups),
                   baseline_minus_cv=paired_mae_difference(cv, baseline, gt, groups), by_scene={})
        for s in rows_by_scene:
            idx = [i for i, g in enumerate(groups) if g == s]
            sub = lambda a: [a[i] for i in idx]
            f = [x for x in folds if x['scene_id'] == s]
            res['by_scene'][s] = dict(cv=metrics_with_ci(sub(cv), sub(gt), [s]*len(idx)),
                                      in_sample_MAE=float(np.mean(np.abs(np.subtract(sub(insample), sub(gt))))),
                                      baseline_MAE=float(np.mean(np.abs(np.subtract(sub(baseline), sub(gt))))),
                                      baseline_minus_cv=paired_mae_difference(sub(cv), sub(baseline), sub(gt), [s]*len(idx)),
                                      k_range=[min(x['k'] for x in f), max(x['k'] for x in f)],
                                      blobs_range=[min(x['single_person_blobs'] for x in f), max(x['single_person_blobs'] for x in f)])
        summary['by_K'][str(k)] = res
        fold_rows += folds
        table += [dict(K=k, image_id=r['image_id'], scene_id=r['scene_id'], gt=g, baseline=b, in_sample=i, cv=c)
                  for r, g, b, i, c in zip(order, gt, baseline, insample, cv)]
    write_csv(path/'cv_predictions.csv', table, table[0].keys())
    write_csv(path/'cv_folds.csv', fold_rows, fold_rows[0].keys())
    write_json(path/'metrics.json', summary)

    f = lambda m: f"{m['MAE']:.3f} [{m['MAE_CI95'][0]:.2f}, {m['MAE_CI95'][1]:.2f}]"
    d = lambda x: f"{x['MAE_diff']:+.3f} [{x['CI95'][0]:.2f}, {x['CI95'][1]:.2f}]"
    print('Baseline', f(summary['baseline']), '| in-sample', f(summary['in_sample']))
    for k, res in summary['by_K'].items():
        print(f"K={k}: CV {f(res['overall'])}  CV−in-sample {d(res['minus_in_sample'])}  baseline−CV {d(res['baseline_minus_cv'])}")
        for s, x in res['by_scene'].items():
            print(f"   {s}: CV {x['cv']['MAE']:.3f} (in-sample {x['in_sample_MAE']:.3f}, baseline {x['baseline_MAE']:.3f}) "
                  f"baseline−CV {d(x['baseline_minus_cv'])}  k∈{x['k_range']}  blobs∈{x['blobs_range']}")
    print('Output:', path)


if __name__ == '__main__':
    main()
