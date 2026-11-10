"""Compatibility imports for the shared background-subtraction pipeline.

The original implementation is preserved in the pre-week1 snapshot. New code
should import object_counting.pipeline directly. Standalone still-image mode
without a background is outside the agreed project scope.
"""
from types import SimpleNamespace
import cv2
from object_counting.pipeline import Params, count_image
from object_counting.regions import se
from object_counting.visualization import overlay


def count_objects(img, p=None, bg=None):
    result = count_image(img, bg, p)
    stages = dict(result.stages, thr=result.statistics['threshold'])
    return result.count, result.contours, stages


def segment(img, p, bg=None):
    return count_objects(img, p, bg)[2]


def draw(img, cnts):
    regions = [dict(bbox=list(cv2.boundingRect(c))) for c in cnts]
    return overlay(img, SimpleNamespace(count=len(cnts), contours=cnts, regions=regions))
