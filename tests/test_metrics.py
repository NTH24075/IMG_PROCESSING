import unittest
from object_counting.metrics import counting_metrics, error_types


def blob(x, y, w, h, n=1):
    return dict(bbox=[x, y, w, h], n=n)


class MetricTests(unittest.TestCase):
    def test_hand_example_with_zero_label(self):
        m = counting_metrics([0, 2], [0, 1])
        self.assertEqual((m['MAE'], m['Bias'], m['Acc']), (.5, .5, .5))

    def test_four_error_kinds_on_constructed_case(self):
        regions = [blob(0, 0, 10, 10),          # merge: 2 points, counted 1
                   blob(20, 0, 10, 10),         # false blob: no point
                   blob(40, 0, 10, 10), blob(45, 0, 10, 10)]   # split: point at x=47 in both
        points = [(2, 2), (7, 7), (47, 5), (90, 90)]           # last one: miss
        self.assertEqual(error_types(regions, points, margin=0), dict(merge=1, split=1, false_blob=1, miss=1))

    def test_counted_merge_is_not_an_error(self):
        self.assertEqual(error_types([blob(0, 0, 10, 10, n=2)], [(2, 2), (7, 7)])['merge'], 0)


if __name__ == '__main__':
    unittest.main()
