"""Lecture 6–7: morphology and regions. Watershed is an opt-in experiment."""
import cv2
import numpy as np
import imgproc_scratch as scratch


def se(k):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def fill_holes(mask):
    """Fill background pockets not connected to the border (flood fill from a padded corner)."""
    padded = np.pad(mask, 1); flood = padded.copy()
    cv2.floodFill(flood, None, (0, 0), 255)
    return mask | cv2.bitwise_not(flood)[1:-1, 1:-1]


def morph_gradient(mask, k=3):
    """Morphological gradient (dilation − erosion) used to draw blob boundaries."""
    return cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, se(k))


def twopass_components(mask):
    labels = scratch.label_twopass(mask)
    components = []
    for label in range(1, int(labels.max())+1):
        ys, xs = np.nonzero(labels == label)
        components.append(dict(label=label, area=len(xs), bbox=(xs.min(), ys.min(), xs.max(), ys.max())))
    return labels, components


def clean_mask(raw, open_k, close_k, backend='opencv', holes=False):
    mask = raw.copy()
    if open_k > 1:
        mask = scratch.opening(mask, se(open_k)) if backend == 'scratch' else cv2.morphologyEx(mask, cv2.MORPH_OPEN, se(open_k))
    if close_k > 1:
        mask = scratch.closing(mask, se(close_k)) if backend == 'scratch' else cv2.morphologyEx(mask, cv2.MORPH_CLOSE, se(close_k))
    return fill_holes(mask) if holes else mask


def watershed_contours(mask, ratio):
    contours = []
    _, labels = cv2.connectedComponents(mask)
    for label in range(1, int(labels.max())+1):
        component = ((labels == label)*255).astype(np.uint8)
        distance = cv2.distanceTransform(component, cv2.DIST_L2, 5)
        cores = ((distance > ratio*distance.max())*255).astype(np.uint8)
        n, markers = cv2.connectedComponents(cores)
        if n <= 2:
            cs, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours.extend(cs)
            continue
        markers += 1
        markers[(component > 0) & (cores == 0)] = 0
        relief = 255-cv2.normalize(distance, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        markers = cv2.watershed(cv2.cvtColor(relief, cv2.COLOR_GRAY2BGR), markers)
        for region in range(2, n+1):
            piece = (((markers == region) & (component > 0))*255).astype(np.uint8)
            cs, _ = cv2.findContours(piece, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours.extend(cs)
    return contours


def extract_regions(mask, minimum, maximum, backend='opencv', split_merged=False, ratio=.5, labeling='bfs'):
    if backend == 'scratch':
        if split_merged:
            raise ValueError('Watershed is an OpenCV-only experimental branch')
        _, components = (twopass_components if labeling == 'twopass' else scratch.label_components)(mask)
        keep = [c for c in components if minimum <= c['area'] <= maximum]
        regions = []
        for c in keep:
            x0, y0, x1, y1 = map(int, c['bbox'])
            regions.append(dict(area=float(c['area']), area_definition='pixel_count',
                                bbox=[x0, y0, x1-x0+1, y1-y0+1], y_bottom=y1+1))
        return regions, []
    if split_merged:
        contours = watershed_contours(mask, ratio)
    else:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if minimum <= cv2.contourArea(c) <= maximum]
    regions = []
    for c in contours:
        area = float(cv2.contourArea(c)); x,y,w,h = cv2.boundingRect(c)
        hull_area = cv2.contourArea(cv2.convexHull(c))
        regions.append(dict(area=area, area_definition='contour_area', bbox=[x,y,w,h], y_bottom=y+h,
                            aspect_ratio=w/h, extent=area/(w*h), solidity=area/hull_area if hull_area else 0))
    return regions, contours
