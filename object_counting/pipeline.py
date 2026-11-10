"""Single inference pipeline used by CLI, evaluator and GUI. No label input."""
from dataclasses import dataclass, asdict
from time import perf_counter
import cv2
import numpy as np
from .preprocessing import prepare_pair, describe
from .segmentation import foreground
from .regions import clean_mask, extract_regions


@dataclass
class Params:
    mode: str = 'bgsub'
    blur: int = 3
    thresh: int = 0
    min_thresh: int = 20
    invert: bool = True
    open_k: int = 3
    close_k: int = 3
    min_area: int = 150
    max_area: int = 30000
    enhancement: str = 'none'
    gamma: float = 1.0
    filter: str = 'gaussian'
    backend: str = 'opencv'
    split_merged: bool = False
    watershed_ratio: float = .5

    def validate(self):
        for key in ('thresh','min_thresh','min_area','max_area','gamma','watershed_ratio'):
            value=getattr(self,key)
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value):
                raise ValueError(key+' must be a finite number')
        if type(self.split_merged) is not bool:
            raise ValueError('split_merged must be boolean')
        for key in ('blur', 'open_k', 'close_k'):
            value = getattr(self, key)
            if type(value) is not int or value < 1 or value % 2 == 0 or value > 31:
                raise ValueError(key+' must be an odd integer in [1,31]')
        if self.mode != 'bgsub':
            raise ValueError('Shared pipeline requires corresponding background')
        if self.filter not in ('gaussian','median','mean','none') or self.backend not in ('opencv','scratch'):
            raise ValueError('Unknown filter/backend')
        if self.backend == 'scratch' and self.filter not in ('gaussian','none'):
            raise ValueError('Scratch backend implements Gaussian/none; median/mean are OpenCV comparison branches')
        if self.enhancement not in ('none','gamma','percentile') or not np.isfinite(self.gamma) or not 0 < self.gamma <= 5:
            raise ValueError('Invalid enhancement/gamma')
        if not 0 <= self.thresh <= 255 or not 0 <= self.min_thresh <= 255 or not 0 <= self.min_area <= self.max_area:
            raise ValueError('Invalid threshold or area limits')
        if not 0 < self.watershed_ratio < 1:
            raise ValueError('watershed_ratio must be between 0 and 1')


@dataclass
class Prediction:
    count: int
    regions: list
    contours: list
    stages: dict
    statistics: dict
    params: dict
    timings_ms: dict


def count_image(image, background, params=None, roi=None):
    p = params or Params(); p.validate()
    for value in (image, background):
        if not isinstance(value, np.ndarray) or value.dtype != np.uint8 or value.ndim != 3 or value.shape[2] != 3 or not value.size:
            raise ValueError('Input/background must be nonempty uint8 BGR images')
        if value.shape[0]*value.shape[1] > 20_000_000:
            raise ValueError('Image too large; configure a smaller processing size')
    if image.shape != background.shape:
        raise ValueError('Input and background dimensions must match')
    if roi is not None and (roi.shape != image.shape[:2] or roi.dtype != np.uint8 or not np.isin(roi,[0,255]).all() or not roi.any()):
        raise ValueError('ROI must be a nonempty binary uint8 mask at image size')
    start = perf_counter()
    gray, bg_gray = prepare_pair(image, background, p.enhancement, p.gamma, p.filter, p.blur, p.backend)
    prepared = perf_counter()
    diff, raw, threshold = foreground(gray, bg_gray, p.thresh, p.min_thresh, p.backend)
    if roi is not None:
        raw = cv2.bitwise_and(raw, roi)
    segmented = perf_counter()
    mask = clean_mask(raw, p.open_k, p.close_k, p.backend)
    if roi is not None:
        mask = cv2.bitwise_and(mask, roi)
    regions, contours = extract_regions(mask, p.min_area, p.max_area, p.backend, p.split_merged, p.watershed_ratio)
    end = perf_counter()
    timings = dict(preprocess=(prepared-start)*1000, segmentation=(segmented-prepared)*1000,
                   regions=(end-segmented)*1000, pipeline=(end-start)*1000)
    statistics = describe(gray); statistics['threshold'] = float(threshold)
    return Prediction(len(regions), regions, contours,
                      dict(gray=gray, background_gray=bg_gray, diff=diff, mask_raw=raw, mask=mask),
                      statistics, asdict(p), timings)
