"""Lecture 3–5: absolute difference, fixed/Otsu threshold and edge illustrations."""
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


def edge_illustrations(gray):
    smooth = cv2.GaussianBlur(gray, (3, 3), 0)
    x = cv2.Sobel(smooth, cv2.CV_32F, 1, 0)
    y = cv2.Sobel(smooth, cv2.CV_32F, 0, 1)
    magnitude = cv2.normalize(cv2.magnitude(x, y), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return {'sobel': magnitude, 'canny': cv2.Canny(smooth, 50, 150)}
