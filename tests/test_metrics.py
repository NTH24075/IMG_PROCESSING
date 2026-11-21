import unittest
import numpy as np
from object_counting.metrics import counting_metrics, error_types, block_bootstrap, metrics_with_ci, paired_mae_difference


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
        self.assertEqual(error_types(regions, points, margin=0), dict(merge=1, over_assign=0, split=1, false_blob=1, miss=1))

    def test_counted_merge_is_not_an_error(self):
        self.assertEqual(error_types([blob(0, 0, 10, 10, n=2)], [(2, 2), (7, 7)])['merge'], 0)

    def test_over_assign_and_signed_error_identity(self):
        regions = [blob(0, 0, 10, 10, n=3),     # 1 point, counted 3 -> over_assign 2
                   blob(20, 0, 10, 10, n=2),    # 0 points, counted 2 -> false_blob 2 (not over_assign)
                   blob(40, 0, 10, 10, n=1)]    # 3 points, counted 1 -> merge 2
        points = [(2, 2), (42, 2), (45, 5), (48, 8), (90, 90)]
        e = error_types(regions, points, margin=0)
        self.assertEqual((e['over_assign'], e['false_blob'], e['merge'], e['miss']), (2, 2, 2, 1))
        signed = sum(r['n'] for r in regions) - len(points)
        self.assertEqual(signed, -e['merge'] + e['over_assign'] + e['split'] + e['false_blob'] - e['miss'])


class BootstrapTest(unittest.TestCase):
    def test_blocks_stay_within_scene_and_contiguous(self):
        groups = ['a']*7 + ['b']*3
        idx = block_bootstrap(groups, block=3, samples=50)
        self.assertEqual(idx.shape, (50, 10))
        allowed = [{0,1,2},{3,4,5},{6},{7,8,9}]
        for row in idx:
            i = 0
            while i < len(row):  # walk the row block by block: each block is one allowed set in order
                block = next(b for b in allowed if row[i] in b); n = min(len(block), len(row)-i)
                self.assertEqual(list(row[i:i+n]), sorted(block)[:n]); i += n

    def test_interval_contains_point_estimate(self):
        rng = np.random.default_rng(1); gt = rng.integers(0, 6, 40); pred = np.maximum(gt + rng.integers(-2, 3, 40), 0)
        m = metrics_with_ci(pred, gt, ['s']*40)
        self.assertLessEqual(m['MAE_CI95'][0], m['MAE']); self.assertGreaterEqual(m['MAE_CI95'][1], m['MAE'])

    def test_paired_difference_zero_for_identical_methods(self):
        d = paired_mae_difference([1,2,3], [1,2,3], [1,1,1], ['s']*3)
        self.assertEqual(d['MAE_diff'], 0); self.assertEqual(d['CI95'], [0.0, 0.0])


if __name__ == '__main__':
    unittest.main()
