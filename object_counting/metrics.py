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
