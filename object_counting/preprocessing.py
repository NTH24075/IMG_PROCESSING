"""Lecture 2–4: histogram, LUT enhancement, spatial filters and seeded noise."""
import cv2
import numpy as np
import imgproc_scratch as scratch


def histogram(gray):
    hist = np.bincount(gray.ravel(), minlength=256)
    return hist, np.cumsum(hist) / gray.size


def describe(gray):
    hist, cdf = histogram(gray)
    low, high = np.percentile(gray, [5, 95])
    return dict(brightness=float(gray.mean()), percentile_spread=float(high-low),
                saturation_fraction=float(((gray == 0) | (gray == 255)).mean()),
                histogram=hist.tolist(), cdf=cdf.tolist())


def enhance_pair(gray, background, method='none', gamma=1.0):
    """Derive one LUT for the pair, so background subtraction stays comparable."""
    if method == 'none':
        return gray.copy(), background.copy()
    levels = np.arange(256, dtype=np.float64)
    if method == 'gamma':
        lut = np.round(255 * (levels / 255) ** gamma)
    elif method == 'percentile':
        low, high = np.percentile(background, [2, 98])
        if high <= low:
            return gray.copy(), background.copy()
        lut = (levels-low) * 255 / (high-low)
    else:
        raise ValueError('Unknown enhancement: ' + method)
    lut = np.clip(lut, 0, 255).astype(np.uint8)
    return lut[gray], lut[background]


def prepare_pair(image, background, enhancement, gamma, filter_name, blur, backend):
    gray_fn = scratch.to_gray if backend == 'scratch' else lambda a: cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    a, b = enhance_pair(gray_fn(image), gray_fn(background), enhancement, gamma)
    filters = {'gaussian': lambda x: cv2.GaussianBlur(x, (blur, blur), 0),
               'median': lambda x: cv2.medianBlur(x, blur),
               'mean': lambda x: cv2.blur(x, (blur, blur)), 'none': lambda x: x.copy()}
    if backend == 'scratch' and filter_name == 'gaussian':
        fn = lambda x: scratch.gaussian_blur(x, blur)
    else:
        fn = filters[filter_name]
    return (fn(a), fn(b)) if blur > 1 else (a, b)


def add_noise(image, kind, amount, seed):
    rng = np.random.default_rng(seed)
    if kind == 'gaussian':
        if amount < 0:
            raise ValueError('Noise sigma must be nonnegative')
        return np.clip(image.astype(float) + rng.normal(0, amount, image.shape), 0, 255).astype(np.uint8)
    if kind == 'salt_pepper':
        if not 0 <= amount <= 1:
            raise ValueError('Impulse fraction must be in [0,1]')
        result = image.copy(); r = rng.random(image.shape[:2])
        result[r < amount/2] = 0; result[r > 1-amount/2] = 255
        return result
    raise ValueError('Unknown noise type')
