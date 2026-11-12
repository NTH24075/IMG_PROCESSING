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
    return dict(merge=sum(max(0, k-c) for k, c in zip(per_blob, n) if k >= 2),
                split=sum(1 for owners in inside if len(owners) >= 2),
                false_blob=sum(c for k, c in zip(per_blob, n) if k == 0),
                miss=sum(1 for owners in inside if not owners))
