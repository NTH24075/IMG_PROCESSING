import numpy as np


def counting_metrics(predicted, actual):
    p, g = np.asarray(predicted, float), np.asarray(actual, float)
    if p.ndim != 1 or g.shape != p.shape or not len(p):
        raise ValueError('Metrics require equal nonempty one-dimensional arrays')
    if not np.isfinite(p).all() or not np.isfinite(g).all() or (g < 0).any() or (p < 0).any():
        raise ValueError('Counts must be finite and nonnegative')
    e = p-g
    return dict(N=len(p), MAE=float(abs(e).mean()), RMSE=float(np.sqrt((e*e).mean())),
                MAPE_modified=float((abs(e)/np.maximum(g,1)).mean()*100),
                Acc=float((e == 0).mean()), Bias=float(e.mean()))


BOOTSTRAP_BLOCK = 5      # consecutive images per block: ~1.8 s of scene_02, 7.5 s of vtest
BOOTSTRAP_SAMPLES = 2000


def block_bootstrap(groups, block=BOOTSTRAP_BLOCK, samples=BOOTSTRAP_SAMPLES, seed=0):
    """Index matrix (samples x N) resampling whole blocks of consecutive images within each group (scene).

    Neighbouring frames are near-duplicates, so resampling single images would give too-narrow intervals.
    Rows must be in temporal order within each group.
    """
    groups = list(groups)
    if not groups:
        raise ValueError('Bootstrap needs at least one image')
    blocks = []
    for g in dict.fromkeys(groups):
        idx = [i for i, x in enumerate(groups) if x == g]
        blocks += [idx[j:j+block] for j in range(0, len(idx), block)]
    rng = np.random.default_rng(seed)
    out = np.empty((samples, len(groups)), int)
    for s in range(samples):
        pick = []
        while len(pick) < len(groups):
            pick += blocks[rng.integers(len(blocks))]
        out[s] = pick[:len(groups)]
    return out


def ci95(values):
    return [float(v) for v in np.percentile(values, [2.5, 97.5])]


def metrics_with_ci(predicted, actual, groups, block=BOOTSTRAP_BLOCK, samples=BOOTSTRAP_SAMPLES, seed=0):
    """counting_metrics plus 95% block-bootstrap intervals for MAE, RMSE and Bias."""
    m = counting_metrics(predicted, actual)
    e = np.asarray(predicted, float)-np.asarray(actual, float)
    idx = block_bootstrap(groups, block, samples, seed)
    m.update(MAE_CI95=ci95(np.abs(e)[idx].mean(1)), RMSE_CI95=ci95(np.sqrt((e*e)[idx].mean(1))),
             Bias_CI95=ci95(e[idx].mean(1)),
             ci_method=f'block bootstrap, {block} consecutive images/block within scene, {samples} resamples, seed {seed}')
    return m


def paired_mae_difference(pred_a, pred_b, actual, groups, block=BOOTSTRAP_BLOCK, samples=BOOTSTRAP_SAMPLES, seed=0):
    """MAE(b) - MAE(a) with a 95% interval from the SAME resampled images for both methods."""
    g = np.asarray(actual, float)
    da, db = np.abs(np.asarray(pred_a, float)-g), np.abs(np.asarray(pred_b, float)-g)
    idx = block_bootstrap(groups, block, samples, seed)
    return dict(MAE_diff=float(db.mean()-da.mean()), CI95=ci95(db[idx].mean(1)-da[idx].mean(1)))


def error_types(regions, points, margin=3):
    """Explain count errors with clicked points (v3 §3.6). A point belongs to a blob when inside its
    bbox grown by `margin` px. shortcut: bbox not pixel mask, upgrade if blobs overlap heavily."""
    inside = [[i for i, r in enumerate(regions)
               if r['bbox'][0]-margin <= x < r['bbox'][0]+r['bbox'][2]+margin and r['bbox'][1]-margin <= y < r['bbox'][1]+r['bbox'][3]+margin]
              for x, y in points]
    per_blob = [0]*len(regions)
    for owners in inside:
        for i in owners:
            per_blob[i] += 1
    n = [r.get('n', 1) for r in regions]
    # over_assign: C3 gives a blob more people than the points it holds (m >= 1). With merge, split,
    # false_blob and miss this makes the signed error  ≈ -merge + over_assign + split + false_blob - miss.
    return dict(merge=sum(max(0, k-c) for k, c in zip(per_blob, n) if k >= 2),
                over_assign=sum(max(0, c-k) for k, c in zip(per_blob, n) if k >= 1),
                split=sum(1 for owners in inside if len(owners) >= 2),
                false_blob=sum(c for k, c in zip(per_blob, n) if k == 0),
                miss=sum(1 for owners in inside if not owners))
