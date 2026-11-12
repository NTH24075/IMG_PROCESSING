"""Benchmark B: controlled degradation of the current image only; background stays unchanged."""
import hashlib
import numpy as np

KINDS = ('none', 'gamma', 'contrast', 'gain', 'gauss', 'saltpepper')
GRID = dict(gamma=[.5, .7, 1.5, 2.0, 2.5], contrast=[.7, .5, .3], gain=[.6, .8, 1.2],
            gauss=[5, 10, 20], saltpepper=[.01, .03, .05])


def seed_for(image_id):
    """Stable across runs/machines, unlike hash()."""
    return int.from_bytes(hashlib.sha256(image_id.encode()).digest()[:4], 'little')


def degrade(img, kind, level, seed=0):
    if kind not in KINDS:
        raise ValueError('Unknown degradation: '+str(kind))
    rng = np.random.default_rng(seed)
    x = img.astype(np.float32)
    if kind == 'gamma':            # level >1: darker, <1: brighter
        x = 255.0 * (x / 255.0) ** level
    elif kind == 'contrast':       # level in (0,1]: compress around 128
        x = (x - 128.0) * level + 128.0
    elif kind == 'gain':           # linear light change
        x = x * level
    elif kind == 'gauss':          # level = sigma
        x = x + rng.normal(0, level, x.shape)
    elif kind == 'saltpepper':     # level = fraction of noisy pixels
        u = rng.random(x.shape[:2])
        x[u < level / 2] = 0
        x[u > 1 - level / 2] = 255
    return np.clip(np.round(x), 0, 255).astype(np.uint8)
