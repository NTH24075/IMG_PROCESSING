import unittest
import cv2
import numpy as np

from object_counting.pipeline import Params, count_image
from object_counting.preprocessing import histogram, enhance_pair, add_noise
from object_counting.metrics import counting_metrics


class CoreTests(unittest.TestCase):
    def test_histogram_and_cdf(self):
        h, cdf = histogram(np.array([[0, 0], [1, 255]], np.uint8))
        self.assertEqual(h.sum(), 4)
        self.assertEqual(h[0], 2)
        self.assertEqual(cdf[-1], 1.0)
        self.assertTrue(np.all(np.diff(cdf) >= 0))

    def test_pair_uses_same_percentile_mapping(self):
        bg = np.full((10, 10), 50, np.uint8)
        image = bg.copy(); image[0, 0] = 200
        a, b = enhance_pair(image, bg, 'percentile', 1.0)
        self.assertTrue(np.array_equal(a[1:], b[1:]))

    def test_noise_seed_and_no_mutation(self):
        image = np.full((20, 20, 3), 120, np.uint8)
        self.assertTrue(np.array_equal(add_noise(image, 'gaussian', 5, 42), add_noise(image, 'gaussian', 5, 42)))
        self.assertTrue(np.all(image == 120))

    def test_metrics_hand_calculation_and_zero(self):
        result = counting_metrics([0, 3, 1], [0, 1, 2])
        self.assertAlmostEqual(result['MAE'], 1)
        self.assertAlmostEqual(result['RMSE'], (5 / 3) ** .5)
        self.assertAlmostEqual(result['Bias'], 1 / 3)
        self.assertAlmostEqual(result['Acc'], 1 / 3)
        with self.assertRaises(ValueError):
            counting_metrics([], [])

    def test_new_image_without_labels_and_two_objects(self):
        bg = np.zeros((80, 100, 3), np.uint8)
        image = bg.copy()
        image[10:30, 10:25] = 255; image[45:70, 65:90] = 255
        result = count_image(image, bg, Params(blur=1, open_k=1, close_k=1, min_area=30))
        self.assertEqual(result.count, 2)
        self.assertEqual(len(result.regions), 2)
        self.assertTrue(set(np.unique(result.stages['mask'])) <= {0, 255})
        self.assertTrue(np.all(bg == 0))

    def test_zero_is_valid_but_invalid_input_raises(self):
        image = np.zeros((30, 30, 3), np.uint8)
        self.assertEqual(count_image(image, image, Params()).count, 0)
        with self.assertRaises(ValueError):
            count_image(image, np.zeros((31, 30, 3), np.uint8), Params())
        with self.assertRaises(ValueError):
            count_image(image, image, Params(blur=2))
        with self.assertRaises(ValueError):
            count_image(image, image, Params(gamma='bad'))

    def test_watershed_can_separate_touching_synthetic_objects(self):
        bg=np.zeros((100,140,3),np.uint8); image=bg.copy()
        cv2.circle(image,(50,50),23,(255,255,255),-1)
        cv2.circle(image,(89,50),23,(255,255,255),-1)
        common=dict(blur=1,open_k=1,close_k=1,min_area=40,watershed_ratio=.7)
        self.assertEqual(count_image(image,bg,Params(**common)).count,1)
        self.assertEqual(count_image(image,bg,Params(**common,split_merged=True)).count,2)

    def test_roi_removes_outside_object(self):
        bg = np.zeros((60, 100, 3), np.uint8); image = bg.copy()
        image[10:30, 10:30] = 255; image[10:30, 70:90] = 255
        roi = np.zeros((60, 100), np.uint8); roi[:, :50] = 255
        result = count_image(image, bg, Params(blur=1, open_k=1, close_k=1, min_area=10), roi)
        self.assertEqual(result.count, 1)


if __name__ == '__main__':
    unittest.main()
