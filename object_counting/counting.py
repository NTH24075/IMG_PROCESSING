"""C3: count merged blobs with a perspective area model (L7) and head detection by Hough circles (L5–6)."""
import cv2
import numpy as np
import imgproc_scratch as scratch


def fit_perspective(pairs):
    """pairs: (y_bottom, area) of dev blobs holding exactly one clicked person. sqrt(area) = k*y + c."""
    y, a = np.asarray(pairs, float).T
    if len(y) < 3:
        raise ValueError('Need at least 3 single-person blobs to fit perspective')
    k, c = np.polyfit(y, np.sqrt(a), 1)
    return float(k), float(c)


def expected_area(y_bottom, k, c):
    return max((k*y_bottom+c)**2, 1.0)


def count_by_area(area, y_bottom, k, c, cap=6):
    return int(np.clip(np.round(area/expected_area(y_bottom, k, c)), 1, cap))


def hough_circles(edge, gx, gy, radii):
    """Vote along the gradient direction (L6); both signs since a head may be brighter or darker."""
    H, W = edge.shape
    ys, xs = np.nonzero(edge)
    th = np.arctan2(gy[ys, xs], gx[ys, xs])
    acc = np.zeros((len(radii), H, W), np.int32)
    for i, r in enumerate(radii):
        for s in (1, -1):
            cx = np.round(xs - s*r*np.cos(th)).astype(int)
            cy = np.round(ys - s*r*np.sin(th)).astype(int)
            ok = (cx >= 0) & (cx < W) & (cy >= 0) & (cy < H)
            np.add.at(acc[i], (cy[ok], cx[ok]), 1)
    return acc


def peaks(acc, radii, vote=.35, inside=None):
    """Threshold by fraction of circumference 2πr, then greedy 3D non-maximum suppression."""
    # Digital circles scatter votes over neighbouring cells; sum each 3x3 cell before thresholding.
    acc = np.stack([cv2.boxFilter(a.astype(np.float32), -1, (3, 3), normalize=False, borderType=cv2.BORDER_CONSTANT) for a in acc])
    score = acc/(2*np.pi*np.asarray(radii, float))[:, None, None]
    if inside is not None:
        score = score*(inside[None] > 0)
    cand = np.argwhere(score >= vote)
    cand = cand[np.argsort(-score[tuple(cand.T)], kind='stable')]
    kept = []
    for i, y, x in cand:
        if all((x-kx)**2+(y-ky)**2 > max(radii[i], kr)**2 for kx, ky, kr in kept):
            kept.append((int(x), int(y), int(radii[i])))
    return kept


def detect_heads(gray, mask, bbox, radius, vote=.35, top=.4, backend='opencv'):
    """Search circles in the top part of a blob; centres must lie on foreground."""
    x, y, w, h = bbox
    crop = gray[y:y+max(int(h*top), 1), x:x+w]; inside = mask[y:y+crop.shape[0], x:x+w]
    radii = sorted({max(2, int(round(radius*f))) for f in (.8, 1., 1.25)})
    if min(crop.shape) < 2*radii[0]:
        return []
    if backend == 'scratch':
        gx, gy = scratch.sobel(crop)
    else:
        gx = cv2.Sobel(crop, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(crop, cv2.CV_32F, 0, 1, ksize=3)
    edge = cv2.Canny(crop, 40, 100) > 0
    return [(cx+x, cy+y, r) for cx, cy, r in peaks(hough_circles(edge, gx, gy, radii), radii, vote, inside)]


def combine(n_area, n_head):
    """n_head is trusted only when it stays within ±1 of the area estimate."""
    return (n_head, 'head') if n_head >= 1 and abs(n_head-n_area) <= 1 else (n_area, 'area')


def count_regions(regions, gray, mask, perspective=None, method='blob', cap=6, vote=.35, backend='opencv'):
    """Adds n/n_area/n_head/source to each region. Without a perspective model every blob counts 1."""
    if method == 'blob' or not perspective:
        for r in regions:
            r.update(n=1, n_area=None, n_head=None, source='blob' if method == 'blob' else 'no_perspective')
        return regions, []
    k, c = perspective['k'], perspective['c']
    ratio = perspective.get('min_area_ratio', 0)
    kept, heads = [], []
    for r in regions:
        expected = expected_area(r['y_bottom'], k, c)
        if r['area'] < ratio*expected:
            continue
        n_area = count_by_area(r['area'], r['y_bottom'], k, c, cap)
        r.update(n=n_area, n_area=n_area, n_head=None, source='area')
        if method == 'area_hough' and n_area >= 2:
            found = detect_heads(gray, mask, r['bbox'], perspective['head_alpha']*np.sqrt(expected), vote, backend=backend)
            r['n'], r['source'] = combine(n_area, len(found)); r['n_head'] = len(found)
            heads.extend(found)
        kept.append(r)
    return kept, heads
