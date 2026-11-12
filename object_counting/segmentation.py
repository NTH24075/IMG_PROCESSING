"""Lecture 3–5: absolute difference, fixed/Otsu threshold, C2 shadow removal and edge illustrations."""
import cv2
import numpy as np
import imgproc_scratch as scratch


def foreground(gray, background, threshold=0, minimum=20, backend='opencv'):
    diff = np.abs(gray.astype(np.int16)-background.astype(np.int16)).astype(np.uint8)
    if threshold > 0:
        value = float(threshold)
    else:
        value = scratch.otsu_threshold(diff) if backend == 'scratch' else cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[0]
        value = max(value, minimum)
    return diff, ((diff > value)*255).astype(np.uint8), value


def sobel_mag(gray, backend='opencv'):
    if backend == 'scratch':
        gx, gy = scratch.sobel(gray)
    else:
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return np.hypot(gx, gy).astype(np.float32)


def shadow_mask(gray, background, fg, lo=.45, hi=.92, t_grad=20., k=5, backend='opencv'):
    """C2: shadow = foreground pixel darker than background by a bounded ratio (L3 image division)
    whose local gradient texture still matches the background (L4 Sobel)."""
    ratio = (gray.astype(np.float32)+1)/(background.astype(np.float32)+1)
    dgrad = np.abs(sobel_mag(gray, backend)-sobel_mag(background, backend))
    dgrad = cv2.blur(dgrad, (k, k)) if k > 1 else dgrad
    shadow = (fg > 0) & (ratio > lo) & (ratio < hi) & (dgrad < t_grad)
    return shadow, ratio, dgrad


def edge_illustrations(gray):
    smooth = cv2.GaussianBlur(gray, (3, 3), 0)
    x = cv2.Sobel(smooth, cv2.CV_32F, 1, 0)
    y = cv2.Sobel(smooth, cv2.CV_32F, 0, 1)
    magnitude = cv2.normalize(cv2.magnitude(x, y), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return {'sobel': magnitude, 'canny': cv2.Canny(smooth, 50, 150)}
