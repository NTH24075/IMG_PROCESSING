"""Lecture 2–4: histogram, LUT enhancement, C1 histogram matching, spatial filters and seeded noise."""
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


def cdf(gray, mask=None):
    values = gray[mask] if mask is not None else gray.ravel()
    h = np.bincount(values, minlength=256).astype(np.float64)
    return np.cumsum(h) / max(h.sum(), 1)


def match_histogram(src, ref, mask=None):
    """C1 (L3): map the CDF of src onto the CDF of ref. Returns image and monotone LUT."""
    lut = np.searchsorted(cdf(ref, mask), cdf(src, mask), side='left')
    lut = np.clip(lut, 0, 255).astype(np.uint8)
    return lut[src], lut


def background_pixels(gray, background, minimum=20, dilate=15):
    """Pass 1 of two-pass matching: rough foreground from a 1-pass match, dilated, then inverted."""
    first, _ = match_histogram(gray, background)
    diff = np.abs(first.astype(np.int16)-background.astype(np.int16)).astype(np.uint8)
    rough = (diff > max(scratch.otsu_threshold(diff), minimum)).astype(np.uint8)
    keep = cv2.dilate(rough, np.ones((dilate, dilate), np.uint8)) == 0
    return keep if keep.mean() > .2 else None  # mostly foreground: fall back to whole image


def enhance_pair(gray, background, method='none', gamma=1.0):
    """Derive one LUT for the pair, so background subtraction stays comparable."""
    if method == 'none':
        return gray.copy(), background.copy()
    if method in ('match1', 'match2'):
        mask = background_pixels(gray, background) if method == 'match2' else None
        return match_histogram(gray, background, mask)[0], background.copy()
    if method == 'equalize_separate':  # expected to be worse: two different LUTs break subtraction
        return cv2.equalizeHist(gray), cv2.equalizeHist(background)
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
    if backend == 'scratch':
        box = np.full((blur, blur), 1/blur**2, np.float32)
        filters = dict(filters, gaussian=lambda x: scratch.gaussian_blur(x, blur), median=lambda x: scratch.median_blur(x, blur),
                       mean=lambda x: np.clip(np.round(scratch.convolve2d(x, box)), 0, 255).astype(np.uint8))
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
