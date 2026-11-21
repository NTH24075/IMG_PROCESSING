import unittest
import numpy as np
from object_counting.io import read_image, resolve, load_scene
from run_hog_baseline import hog_detector, hog_count


class HogBaselineTest(unittest.TestCase):
    def test_counts_people_and_respects_roi(self):
        image = read_image(resolve('data/frames/dev/f0120.png'))  # labelled 8 people
        n, boxes = hog_count(hog_detector(), image, load_scene('scene_01')[2])
        self.assertEqual(n, len(boxes)); self.assertGreater(n, 0)
        empty_roi = np.zeros(image.shape[:2], np.uint8)
        self.assertEqual(hog_count(hog_detector(), image, empty_roi)[0], 0)

    def test_blank_image_has_no_people(self):
        self.assertEqual(hog_count(hog_detector(), np.full((288, 384, 3), 128, np.uint8))[0], 0)


if __name__ == '__main__':
    unittest.main()
