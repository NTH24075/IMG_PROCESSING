"""E6: self-implemented (NumPy) vs OpenCV on a few DEV images — agreement and time (median of 3 after warm-up)."""
from time import perf_counter
import cv2
import numpy as np
import imgproc_scratch as scratch
from object_counting.io import ROOT, read_csv, read_image, resolve, load_scene, new_run, write_csv, write_json
from object_counting.preprocessing import match_histogram
from object_counting.regions import se, twopass_components
from object_counting.counting import hough_circles, peaks


def timed(fn, repeats=3):
    fn()
    times = []
    for _ in range(repeats):
        start = perf_counter(); out = fn(); times.append((perf_counter()-start)*1000)
    return out, float(np.median(times))


def same(a, b, crop=0):
    a, b = np.asarray(a), np.asarray(b)
    if crop:
        a, b = a[crop:-crop, crop:-crop], b[crop:-crop, crop:-crop]
    return float((np.abs(a.astype(float)-b.astype(float)) <= 1).mean())


def cases(gray, bg):
    diff = scratch.abs_diff(gray, bg)
    mask = ((diff > max(scratch.otsu_threshold(diff), 20))*255).astype(np.uint8)
    s = se(3)
    np_match = lambda: np.interp(np.cumsum(np.bincount(gray.ravel(), minlength=256))/gray.size,   # independent reference
                                 np.cumsum(np.bincount(bg.ravel(), minlength=256))/bg.size, np.arange(256))
    edge = cv2.Canny(gray, 40, 100) > 0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0); gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    return [
        ('histogram', lambda: scratch.histogram(gray), lambda: cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel(), 0),
        ('gaussian_5', lambda: scratch.gaussian_blur(gray, 5), lambda: cv2.GaussianBlur(gray, (5, 5), 0), 2),
        ('median_3', lambda: scratch.median_blur(gray, 3), lambda: cv2.medianBlur(gray, 3), 1),
        ('sobel_x', lambda: scratch.sobel(gray)[0], lambda: cv2.Sobel(gray, cv2.CV_32F, 1, 0), 1),
        ('otsu', lambda: scratch.otsu_threshold(diff), lambda: cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[0], 0),
        ('opening_3', lambda: scratch.opening(mask, s), lambda: cv2.morphologyEx(mask, cv2.MORPH_OPEN, s), 2),
        ('label_bfs', lambda: len(scratch.label_components(mask)[1]), lambda: cv2.connectedComponents(mask)[0]-1, 0),
        ('label_twopass', lambda: len(twopass_components(mask)[1]), lambda: cv2.connectedComponents(mask)[0]-1, 0),
        ('match_lut', lambda: match_histogram(gray, bg)[1], lambda: np.clip(np.ceil(np_match()-1e-9), 0, 255), 0),
        ('hough_r6', lambda: len(peaks(hough_circles(edge, gx, gy, [5, 6, 7]), [5, 6, 7])),
         lambda: 0 if (c := cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 6, param1=100, param2=12, minRadius=5, maxRadius=7)) is None else c.shape[1], None),
    ]


def main():
    rows = read_csv(ROOT/'data/week1/manifests/dev.csv')
    sample = [rows[4], rows[12], rows[30], rows[42]]
    table = []
    for row in sample:
        _, background, _ = load_scene(row['scene_id'])
        gray = scratch.to_gray(read_image(resolve(row['image']))); bg = scratch.to_gray(background)
        for name, mine, ref, crop in cases(gray, bg):
            a, t_mine = timed(mine); b, t_ref = timed(ref)
            agree = '' if crop is None else same(a, b, crop) if np.ndim(a) else float(abs(float(a)-float(b)) <= 1)
            table.append(dict(image_id=row['image_id'], op=name, scratch_ms=round(t_mine, 3), opencv_ms=round(t_ref, 3),
                              agreement=agree, scratch_value=a if np.ndim(a) == 0 else '', opencv_value=b if np.ndim(b) == 0 else ''))
    path = new_run('benchmarks')
    write_csv(path/'e6_scratch_vs_opencv.csv', table, table[0].keys())
    write_json(path/'protocol.json', dict(split='dev', images=[r['image_id'] for r in sample], timing='median of 3 after 1 warm-up, ms',
        agreement='fraction of pixels within ±1 (border cropped) or equal scalar; hough: counts only (different algorithms)'))
    for op in dict.fromkeys(r['op'] for r in table):
        sub = [r for r in table if r['op'] == op]
        agree = [r['agreement'] for r in sub if r['agreement'] != '']
        print(f"{op:<14} scratch {np.median([r['scratch_ms'] for r in sub]):9.2f} ms  opencv {np.median([r['opencv_ms'] for r in sub]):8.2f} ms"
              f"  agreement {np.mean(agree) if agree else float('nan'):.3f}")
    print('Output:', path)


if __name__ == '__main__':
    main()
