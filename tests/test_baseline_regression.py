"""Golden counts captured BEFORE extracting the shared pipeline."""
import unittest
import cv2
from object_counting.io import ROOT, read_csv
from object_counting.pipeline import Params, count_image


class BaselineRegressionTests(unittest.TestCase):
    def test_both_backends_preserve_all_26_original_counts(self):
        rows=read_csv(ROOT/'tests/fixtures/baseline_vtest_expected.csv')
        background=cv2.imread(str(ROOT/'data/background.png'))
        self.assertEqual(len(rows),26)
        for backend,column in [('opencv','pred_cv'),('scratch','pred_scratch')]:
            for row in rows:
                with self.subTest(backend=backend,image=row['image']):
                    image=cv2.imread(str(ROOT/'data/frames'/row['image']))
                    prediction=count_image(image,background,Params(backend=backend))
                    self.assertEqual(prediction.count,int(row[column]))


if __name__=='__main__':
    unittest.main()
