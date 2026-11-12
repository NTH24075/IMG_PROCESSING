import unittest
import cv2
import numpy as np
import imgproc_scratch as scratch
from object_counting.regions import fill_holes, extract_regions


class RegionTests(unittest.TestCase):
    def test_empty_single_and_diagonal_connectivity(self):
        empty = np.zeros((10, 10), np.uint8)
        self.assertEqual(scratch.label_twopass(empty).max(), 0)
        diag = np.zeros((5, 5), np.uint8); diag[1, 1] = diag[2, 2] = 255
        self.assertEqual(scratch.label_twopass(diag, 8).max(), 1)
        self.assertEqual(scratch.label_twopass(diag, 4).max(), 2)

    def test_twopass_matches_bfs_and_opencv_on_u_shape(self):
        m = np.zeros((40, 60), np.uint8)
        m[5:30, 5:10] = m[5:30, 30:35] = m[25:30, 5:35] = 255   # U: needs label equivalence
        m[2:6, 45:55] = 255; m[35, 58] = 255
        labels = scratch.label_twopass(m); bfs, _ = scratch.label_components(m)
        self.assertEqual(labels.max(), cv2.connectedComponents(m, connectivity=8)[0]-1)
        self.assertEqual(labels.max(), bfs.max())
        for a in range(1, labels.max()+1):   # same partition up to renumbering
            self.assertEqual(len(np.unique(bfs[labels == a])), 1)

    def test_scratch_twopass_regions_equal_bfs_regions(self):
        m = np.zeros((40, 60), np.uint8); m[5:20, 5:20] = 255; m[25:35, 30:50] = 255
        self.assertEqual(extract_regions(m, 1, 10**6, 'scratch', labeling='twopass')[0], extract_regions(m, 1, 10**6, 'scratch')[0])

    def test_fill_holes_keeps_border_background(self):
        m = np.zeros((20, 20), np.uint8); m[3:15, 3:15] = 255; m[7:10, 7:10] = 0
        filled = fill_holes(m)
        self.assertTrue(filled[7:10, 7:10].all() and not filled[0].any() and set(np.unique(filled)) <= {0, 255})


if __name__ == '__main__':
    unittest.main()
