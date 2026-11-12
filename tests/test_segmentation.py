import unittest
import numpy as np
import imgproc_scratch as scratch
from object_counting.segmentation import shadow_mask


class SegmentationTests(unittest.TestCase):
    def test_otsu_splits_bimodal_histogram(self):
        image = np.r_[np.full(500, 40), np.full(500, 200)].astype(np.uint8).reshape(20, 50)
        self.assertTrue(40 <= scratch.otsu_threshold(image) < 200)

    def test_shadow_removed_only_inside_foreground_and_person_kept(self):
        rng = np.random.default_rng(2)
        bg = rng.integers(100, 160, (60, 90)).astype(np.uint8)
        image = bg.copy()
        image[10:50, 10:40] = 20                                       # person: flat, very dark
        image[10:50, 50:80] = (bg[10:50, 50:80]*.7).astype(np.uint8)   # shadow: darker, same texture
        image[0:5, 50:80] = (bg[0:5, 50:80]*.7).astype(np.uint8)       # shadow-like but outside fg
        fg = np.zeros_like(bg); fg[10:50, 10:40] = 255; fg[10:50, 50:80] = 255
        shadow, ratio, dgrad = shadow_mask(image, bg, fg, t_grad=30)
        self.assertFalse(shadow[fg == 0].any())
        self.assertGreater(shadow[15:45, 55:75].mean(), .9)
        self.assertLess(shadow[15:45, 15:35].mean(), .05)
        self.assertEqual(ratio.dtype, np.float32)


if __name__ == '__main__':
    unittest.main()
