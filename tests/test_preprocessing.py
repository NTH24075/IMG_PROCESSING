import unittest
import cv2
import numpy as np
import imgproc_scratch as scratch
from object_counting.preprocessing import cdf, match_histogram, enhance_pair


class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.bg = self.rng.integers(40, 200, (60, 80)).astype(np.uint8)

    def test_cdf_nondecreasing_and_ends_at_one(self):
        c = cdf(self.bg)
        self.assertTrue(np.all(np.diff(c) >= 0)); self.assertAlmostEqual(c[-1], 1.0)

    def test_match_lut_monotone_and_undoes_global_gain(self):
        dark = (self.bg*.5).astype(np.uint8)
        matched, lut = match_histogram(dark, self.bg)
        self.assertTrue(np.all(np.diff(lut.astype(int)) >= 0))
        self.assertLess(np.abs(matched.astype(int)-self.bg).mean(), 2)

    def test_match_keeps_background_and_two_pass_ignores_foreground(self):
        image = (self.bg*.6).astype(np.uint8); image[10:40, 20:50] = 255
        one, bg1 = enhance_pair(image, self.bg, 'match1'); two, bg2 = enhance_pair(image, self.bg, 'match2')
        self.assertTrue(np.array_equal(bg1, self.bg) and np.array_equal(bg2, self.bg))
        outside = np.ones_like(self.bg, bool); outside[5:45, 15:55] = False
        self.assertLessEqual(np.abs(two.astype(int)-self.bg)[outside].mean(), np.abs(one.astype(int)-self.bg)[outside].mean())

    def test_separable_gaussian_equals_2d(self):
        g = scratch.gaussian_kernel_1d(5)
        a = scratch.convolve2d(scratch.convolve2d(self.bg, g[None, :]), g[:, None])
        self.assertTrue(np.allclose(a, scratch.convolve2d(self.bg, np.outer(g, g)), atol=1e-3))

    def test_scratch_median_matches_opencv_inside(self):
        self.assertTrue(np.array_equal(scratch.median_blur(self.bg, 3)[1:-1, 1:-1], cv2.medianBlur(self.bg, 3)[1:-1, 1:-1]))


if __name__ == '__main__':
    unittest.main()
