import unittest
import cv2
import numpy as np
from object_counting.io import detect_scene, read_image, resolve


class SceneDetectTest(unittest.TestCase):
    def test_picks_matching_background(self):
        for path, expected in [('data/frames/dev/f0120.png','scene_01'), ('data/week1/frames/scene_02/f00180.png','scene_02')]:
            image = read_image(resolve(path))
            self.assertEqual(detect_scene(image)[0], expected)
            self.assertEqual(detect_scene((image*0.5).astype(np.uint8))[0], expected)
            # size must not decide: swap each scene to the other scene's resolution
            other = (384, 288) if expected == 'scene_01' else (768, 576)
            self.assertEqual(detect_scene(cv2.resize(image, other))[0], expected)

    def test_unrelated_image_rejected(self):
        noise = np.random.default_rng(0).integers(0, 255, (288, 384, 3), np.uint8)
        with self.assertRaises(ValueError):
            detect_scene(noise)


if __name__ == '__main__':
    unittest.main()
