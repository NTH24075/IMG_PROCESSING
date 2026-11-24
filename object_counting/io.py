"""Filesystem/config boundaries. Relative paths always resolve from project root."""
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import csv
import hashlib
import json
import tomllib
import uuid
import cv2
import numpy as np
from .pipeline import Params

ROOT = Path(__file__).resolve().parents[1]
MIN_SCENE_INLIERS = 30  # dev frames: true scene >=143 inliers, unrelated images <=6 (2026-11-15)


def resolve(path, root=ROOT):
    p = Path(path)
    return p.resolve() if p.is_absolute() else (root / p).resolve()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read_image(path):
    path = Path(path)
    if not path.is_file():
        raise ValueError('Image not found: '+str(path))
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError('Cannot read image: '+str(path))
    if image.shape[0]*image.shape[1] > 20_000_000:
        raise ValueError('Image exceeds 20 megapixels')
    return image


def read_csv(path):
    with Path(path).open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='raise')
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def load_params(path='configs/baseline.toml', root=ROOT):
    path = resolve(path, root)
    with path.open('rb') as f:
        config = tomllib.load(f)
    unknown = set(config.get('processing', {}))-set(Params.__dataclass_fields__)
    if unknown:
        raise ValueError('Unknown processing parameters: '+', '.join(sorted(unknown)))
    p = Params(**config.get('processing', {})); p.validate()
    return p, config


def load_scene(scene_id, root=ROOT):
    if not scene_id.replace('_','').isalnum():
        raise ValueError('Invalid scene ID')
    with (root/'configs/scenes'/f'{scene_id}.toml').open('rb') as f:
        scene = tomllib.load(f)['scene']
    if scene['id'] != scene_id or not scene.get('background'):
        raise ValueError('Invalid scene configuration')
    background = read_image(resolve(scene['background'], root))
    width, height = scene['width'], scene['height']
    if background.shape[:2] != (height, width):
        raise ValueError('Background dimensions differ from scene configuration')
    roi = None
    if scene.get('roi'):
        points = np.asarray(scene['roi'], dtype=np.int32)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or (points < 0).any() or (points[:,0] >= width).any() or (points[:,1] >= height).any():
            raise ValueError('Invalid ROI polygon')
        roi = np.zeros((height, width), np.uint8); cv2.fillPoly(roi, [points], 255)
    return scene, background, roi


def _orb_features(image, width=384):
    # Fixed working width so frames of any resolution yield comparable keypoints.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray = cv2.resize(gray, (width, max(1, round(gray.shape[0]*width/gray.shape[1]))), interpolation=cv2.INTER_AREA)
    return cv2.ORB_create(nfeatures=1500).detectAndCompute(gray, None)


def detect_scene(image, root=ROOT, min_inliers=MIN_SCENE_INLIERS):
    """Pick the scene whose background shares the most geometrically consistent ORB matches.

    ORB keypoints (corners, invariant to rotation/brightness) are matched to each background with
    Lowe's ratio test; RANSAC then keeps only matches agreeing on one homography, so moving people
    and chance matches are discarded. Score = inlier count (higher is better). Image size is not used.
    Returns (scene_id, scores); raises ValueError when no scene reaches min_inliers.
    """
    keypoints, descriptors = _orb_features(image)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    scores = {}
    for path in sorted((root/'configs/scenes').glob('*.toml')):
        scene, background, _ = load_scene(path.stem, root)
        bg_keypoints, bg_descriptors = _orb_features(background)
        score = 0
        if descriptors is not None and bg_descriptors is not None and len(bg_descriptors) >= 2:
            good = [m for m, n in (p for p in matcher.knnMatch(descriptors, bg_descriptors, k=2) if len(p) == 2)
                    if m.distance < 0.75*n.distance]
            if len(good) >= 4:
                src = np.float32([keypoints[m.queryIdx].pt for m in good])
                dst = np.float32([bg_keypoints[m.trainIdx].pt for m in good])
                _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
                score = int(mask.sum()) if mask is not None else 0
        scores[scene['id']] = score
    best = max(scores, key=scores.get)
    if scores[best] < min_inliers:
        raise ValueError(f'Image does not match any scene background (best {best}: {scores[best]} matched points).')
    return best, scores


def new_run(kind, root=ROOT):
    if kind not in ('predictions','experiments','illustrations','benchmarks'):
        raise ValueError('Unknown output category')
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:10]
    path = root/'outputs'/kind/run_id; path.mkdir(parents=True, exist_ok=False)
    return path


def load_points(root=ROOT):
    """Person points per image_id: vtest clicks + CAVIAR XML box centres. Diagnosis/calibration only, never inference."""
    points = {}
    clicks = root/'data/annotations.json'
    if clicks.exists():
        for name, xy in json.loads(clicks.read_text()).items():
            points['scene_01_'+Path(name).stem] = [tuple(p) for p in xy]
    boxes = root/'data/week1/annotations/xml_candidates.json'
    if boxes.exists():
        for image_id, items in json.loads(boxes.read_text()).items():
            points[image_id] = [(b['xc'], b['yc']) for b in items]
    return points
