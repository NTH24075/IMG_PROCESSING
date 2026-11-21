import unittest
from object_counting.io import ROOT, read_csv
from run_c3_cv import time_blocks


class TimeBlockTest(unittest.TestCase):
    def test_blocks_are_contiguous_disjoint_and_cover_all(self):
        for n, k in [(26, 5), (50, 5), (26, 2), (50, 10), (7, 10)]:
            blocks = time_blocks(n, k)
            self.assertEqual(sum(blocks, []), list(range(n)))           # ordered, disjoint, complete
            self.assertLessEqual(max(map(len, blocks)) - min(map(len, blocks)), 1)


class RobustnessManifestTest(unittest.TestCase):
    def test_test_benchmark_holds_only_transfer_scene(self):
        rows = read_csv(ROOT/'data/week1/manifests/robustness_test.csv')
        legacy = {r['image_id'] for r in read_csv(ROOT/'data/week1/manifests/legacy_holdout.csv')}
        self.assertEqual({r['scene_id'] for r in rows}, {'scene_03'})
        self.assertFalse({r['source_image_id'] for r in rows} & legacy)
        self.assertEqual(len(rows), 75*18)


if __name__ == '__main__':
    unittest.main()
