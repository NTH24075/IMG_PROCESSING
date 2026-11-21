"""Comparison method: HOG + linear SVM people detector (Dalal & Triggs 2005, OpenCV pretrained, no training here).

Runs HOG and the contour pipeline on the same manifest/labels/ROI and writes one comparison run:
predictions.csv, metrics.json (overall + by_scene for both methods, ms/image) and a few HOG overlay examples.
Same freeze guard as run_eval.py for any manifest touching test images.
"""
import argparse
from time import perf_counter
import cv2
import numpy as np
from object_counting.io import ROOT, resolve, read_csv, write_csv, write_json, load_scene, load_params, read_image, new_run, digest
from object_counting.pipeline import count_image
from object_counting.metrics import metrics_with_ci, paired_mae_difference
from object_counting.experiments import REVIEWED, touches_test, check_freeze, log_test_run

# Chosen on dev with NMS (MAE 2.33 @2x, 1.78 @2.5x, 1.50 @3x, 2.08 @3.5x): HOG's 64x128 window needs people ~128 px tall.
UPSCALE = 3.0
NMS_IOU = 0.3
BASE_WIDTH = 384  # scene_01 (768 px) is first brought to the CAVIAR scale so one factor fits every scene


def np_flat(indices):
    return [int(i) for i in np.asarray(indices).ravel()]


def hog_detector():
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    return hog


def hog_count(hog, image, roi=None, upscale=UPSCALE):
    """Count HOG detections whose box centre lies inside the ROI (same rule as labels and contour). Returns (count, boxes in image coords)."""
    s = upscale*BASE_WIDTH/image.shape[1]
    boxes, weights = hog.detectMultiScale(cv2.resize(image, None, fx=s, fy=s), winStride=(4, 4), padding=(8, 8), scale=1.05)
    # Standard NMS on SVM scores: drops nested/duplicate windows on one person, as in Dalal & Triggs.
    keep = cv2.dnn.NMSBoxes([list(map(int, b)) for b in boxes], [float(x) for x in weights.ravel()], 0.0, NMS_IOU) if len(boxes) else []
    h, w = image.shape[:2]
    kept = []
    for bx, by, bw, bh in (boxes[i] for i in np_flat(keep)):
        x, y = min(int((bx+bw/2)/s), w-1), min(int((by+bh/2)/s), h-1)
        if roi is None or roi[y, x] > 0:
            kept.append([int(v/s) for v in (bx, by, bw, bh)])
    return len(kept), kept


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', default='dev'); parser.add_argument('--manifest')
    parser.add_argument('--config', default='configs/final.toml', help='contour pipeline config to compare against')
    parser.add_argument('--upscale', type=float, default=UPSCALE); parser.add_argument('--examples', type=int, default=4)
    args = parser.parse_args()
    rows = read_csv(resolve(args.manifest) if args.manifest else ROOT/'data/week1/manifests'/f'{args.split}.csv')
    config = resolve(args.config)
    freeze = None
    if touches_test(rows):
        freeze = check_freeze(config)
        log_test_run(ROOT, event='start_hog_baseline', config=str(config.relative_to(ROOT)), manifest=args.manifest or args.split, **freeze)
    params, _ = load_params(config)
    labels = {r['image_id']: r for r in read_csv(ROOT/'data/week1/annotations/counts.csv')}
    hog, scenes, out = hog_detector(), {}, []
    path = new_run('experiments')
    for i, row in enumerate(rows):
        label = labels.get(row['image_id'], {})
        if label.get('review_status') not in REVIEWED or label.get('count', '') == '':
            continue
        image_path = resolve(row['image'])
        if digest(image_path) != row['sha256']:
            raise ValueError('Ảnh đã đổi so với manifest: '+row['image_id'])
        if row['scene_id'] not in scenes:
            scenes[row['scene_id']] = load_scene(row['scene_id'])
        scene, bg, roi = scenes[row['scene_id']]
        image = read_image(image_path)
        contour = count_image(image, bg, params, roi, scene.get('perspective'))
        start = perf_counter(); n_hog, boxes = hog_count(hog, image, roi, args.upscale); hog_ms = (perf_counter()-start)*1000
        out.append(dict(image_id=row['image_id'], scene_id=row['scene_id'], gt=int(label['count']),
                        pred_contour=contour.count, pred_hog=n_hog,
                        contour_ms=round(contour.timings_ms['pipeline'], 2), hog_ms=round(hog_ms, 2)))
        if i < args.examples:
            vis = image.copy()
            for x, y, w, h in boxes:
                cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 0, 255), 2)
            (path/'examples').mkdir(exist_ok=True); cv2.imwrite(str(path/'examples'/f"{row['image_id']}_hog.png"), vis)
    if not out:
        raise ValueError('Không có ảnh nào có nhãn đã duyệt')
    write_csv(path/'predictions.csv', out, out[0].keys())

    def summarise(subset):
        g, groups = [r['gt'] for r in subset], [r['scene_id'] for r in subset]
        return {m: dict(metrics_with_ci([r['pred_'+m] for r in subset], g, groups),
                        ms_per_image=sum(r[m+'_ms'] for r in subset)/len(subset)) for m in ('contour', 'hog')}
    summary = dict(method_hog=dict(detector='OpenCV HOG + linear SVM (default people detector)', upscale=args.upscale,
                                   base_width=BASE_WIDTH, winStride=4, padding=8, scale=1.05, nms_iou=NMS_IOU, roi_rule='box centre inside ROI'),
                   contour_config=str(config.relative_to(ROOT)), overall=summarise(out),
                   by_scene={s: summarise([r for r in out if r['scene_id'] == s]) for s in sorted({r['scene_id'] for r in out})},
                   hog_minus_contour_MAE=paired_mae_difference([r['pred_contour'] for r in out], [r['pred_hog'] for r in out],
                                                               [r['gt'] for r in out], [r['scene_id'] for r in out]),
                   timing_scope='contour: count_image pipeline; hog: resize + detectMultiScale + ROI filter; both exclude I/O')
    write_json(path/'metrics.json', summary)
    if freeze:
        log_test_run(ROOT, event='done_hog_baseline', run_id=path.name, **freeze)
    print(f"{'':18}{'N':>3}{'MAE':>8}{'KTC 95% MAE':>16}{'RMSE':>8}{'Acc':>8}{'Bias':>8}{'ms/ảnh':>9}")
    for name, group in [('Tất cả', summary['overall'])]+list(summary['by_scene'].items()):
        for m in ('contour', 'hog'):
            x = group[m]
            print(f"{name+' '+m:18}"[:18]+f"{x['N']:>3}{x['MAE']:>8.3f}{'[%.2f, %.2f]' % tuple(x['MAE_CI95']):>16}{x['RMSE']:>8.3f}{x['Acc']:>8.1%}{x['Bias']:>8.3f}{x['ms_per_image']:>9.1f}")
    d = summary['hog_minus_contour_MAE']
    print(f"MAE(HOG) - MAE(contour) = {d['MAE_diff']:.3f}, KTC 95% [{d['CI95'][0]:.2f}, {d['CI95'][1]:.2f}] (bootstrap khối, cặp)")
    print('Kết quả:', path)


if __name__ == '__main__':
    main()
