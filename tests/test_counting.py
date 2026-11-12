import unittest
import cv2
import numpy as np
from object_counting.counting import fit_perspective, count_by_area, hough_circles, peaks, combine, detect_heads
from object_counting.pipeline import Params, count_image


class CountingTests(unittest.TestCase):
    def test_fit_recovers_linear_sqrt_area(self):
        k, c = fit_perspective([(y, (0.1*y+5)**2) for y in (50, 100, 200, 300)])
        self.assertAlmostEqual(k, .1, 5); self.assertAlmostEqual(c, 5, 4)

    def test_count_by_area_rounds_and_caps(self):
        self.assertEqual(count_by_area(100, 0, 0, 10), 1)
        self.assertEqual(count_by_area(260, 0, 0, 10), 3)
        self.assertEqual(count_by_area(10**6, 0, 0, 10, cap=6), 6)
        self.assertEqual(count_by_area(5, 0, 0, 10), 1)

    def test_hough_finds_synthetic_circle_centre(self):
        image = np.zeros((60, 80), np.uint8); cv2.circle(image, (30, 25), 9, 255, -1)
        gx = cv2.Sobel(image, cv2.CV_32F, 1, 0); gy = cv2.Sobel(image, cv2.CV_32F, 0, 1)
        found = peaks(hough_circles(cv2.Canny(image, 40, 100) > 0, gx, gy, [7, 9, 11]), [7, 9, 11])
        x, y, r = found[0]
        self.assertLessEqual(abs(x-30)+abs(y-25), 2); self.assertEqual(r, 9); self.assertEqual(len(found), 1)

    def test_detect_two_heads_in_merged_blob(self):
        image = np.zeros((80, 100), np.uint8)
        for cx in (35, 60):
            cv2.circle(image, (cx, 15), 8, 255, -1); cv2.rectangle(image, (cx-10, 22), (cx+10, 75), 255, -1)
        self.assertEqual(len(detect_heads(image, image, [24, 6, 48, 70], 8, top=.4)), 2)

    def test_combine_rule(self):
        self.assertEqual(combine(2, 3), (3, 'head')); self.assertEqual(combine(2, 0), (2, 'area'))
        self.assertEqual(combine(2, 5), (2, 'area'))

    def test_pipeline_area_counts_merged_blob_and_blob_mode_unchanged(self):
        bg = np.zeros((100, 120, 3), np.uint8); image = bg.copy(); image[50:90, 10:90] = 255   # area 3200 ≈ 2 x 1600
        persp = dict(k=0., c=40., min_area_ratio=.3, head_alpha=.15)
        common = dict(blur=1, open_k=1, close_k=1, min_area=10)
        self.assertEqual(count_image(image, bg, Params(**common), perspective=persp).count, 1)
        self.assertEqual(count_image(image, bg, Params(**common, count_method='area'), perspective=persp).count, 2)
        # no perspective model: falls back to one per blob, never guesses
        result = count_image(image, bg, Params(**common, count_method='area'))
        self.assertEqual(result.count, 1); self.assertEqual(result.regions[0]['source'], 'no_perspective')


if __name__ == '__main__':
    unittest.main()
