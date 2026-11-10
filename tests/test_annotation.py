from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from object_counting.io import write_csv
from object_counting.annotation import review_label


class AnnotationTests(unittest.TestCase):
    def test_different_reviewer_approves_but_correction_resets_peer(self):
        with TemporaryDirectory() as directory:
            root=Path(directory); path=root/'data/week1/annotations/counts.csv'
            row=dict(image_id='x',count='2',provenance='xml',review_status='draft_xml',reviewer='',peer_reviewer='',notes='')
            write_csv(path,[row],row.keys())
            self.assertEqual(review_label('x',2,'Alice',root)['review_status'],'reviewed')
            self.assertEqual(review_label('x',2,'Alice',root)['review_status'],'reviewed')
            self.assertEqual(review_label('x',2,'Bob',root)['review_status'],'peer_reviewed')
            self.assertEqual(review_label('x',2,'Alice',root)['review_status'],'peer_reviewed')
            updated=review_label('x',3,'Bob',root)
            self.assertEqual(updated['review_status'],'reviewed')
            self.assertEqual(updated['peer_reviewer'],'')
            self.assertTrue(list((path.parent/'backups').glob('*.csv')))


if __name__=='__main__':
    unittest.main()
