from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

from object_counting.evaluation_guard import authorize, create_freeze
from object_counting.io import digest, write_csv


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory(); self.root=Path(self.temp.name)
        for folder in ('configs/scenes','object_counting','data/week1/manifests','data/week1/annotations'):
            (self.root/folder).mkdir(parents=True)
        (self.root/'object_counting/pipeline.py').write_text('version = 1')
        (self.root/'configs/baseline.toml').write_text('[processing]\nblur=3\n')
        (self.root/'test.png').write_bytes(b'fixture-image')
        self.row=dict(image_id='test_1',image='test.png',scene_id='scene_03',split='test',sha256=digest(self.root/'test.png'))
        write_csv(self.root/'data/week1/manifests/test.csv',[self.row],self.row.keys())
        write_csv(self.root/'data/week1/annotations/counts.csv',[dict(image_id='test_1',count=0,review_status='peer_reviewed',reviewer='a',peer_reviewer='b')],['image_id','count','review_status','reviewer','peer_reviewer'])

    def tearDown(self):
        self.temp.cleanup()

    def test_unfrozen_test_and_all_are_denied_and_audited(self):
        for split in ('test','all'):
            with self.assertRaises(ValueError):
                authorize([self.row],self.root/'configs/baseline.toml',split,self.root)
        events=[json.loads(x) for x in (self.root/'outputs/test_audit.jsonl').read_text().splitlines()]
        self.assertEqual([x['status'] for x in events],['requested','denied','requested','denied'])
        self.assertEqual(events[0]['request_id'],events[1]['request_id'])

    def test_renamed_image_and_fake_dev_are_still_test(self):
        fake=dict(self.row,split='dev',image_id='fake')
        (self.root/'renamed.png').write_bytes((self.root/'test.png').read_bytes())
        fake['image']='renamed.png'
        with self.assertRaises(ValueError):
            authorize([fake],self.root/'configs/baseline.toml','dev',self.root)

    def test_registered_baseline_passes_but_changed_code_is_denied(self):
        create_freeze([self.root/'configs/baseline.toml'],['a','b'],self.root)
        authorize([self.row],self.root/'configs/baseline.toml','test',self.root)
        (self.root/'object_counting/pipeline.py').write_text('version = 2')
        with self.assertRaises(ValueError):
            authorize([self.row],self.root/'configs/baseline.toml','test',self.root)

    def test_draft_labels_prevent_freeze(self):
        file=self.root/'data/week1/annotations/counts.csv'
        file.write_text(file.read_text().replace('peer_reviewed','draft_xml'))
        with self.assertRaises(ValueError):
            create_freeze([self.root/'configs/baseline.toml'],['a','b'],self.root)

    def test_different_data_or_unregistered_config_is_denied(self):
        create_freeze([self.root/'configs/baseline.toml'],['a','b'],self.root)
        other=self.root/'configs/other.toml'; other.write_text('[processing]\nblur=5\n')
        with self.assertRaises(ValueError):
            authorize([self.row],other,'test',self.root)
        other.unlink()
        (self.root/'test.png').write_bytes(b'changed-image')
        with self.assertRaises(ValueError):
            authorize([self.row],self.root/'configs/baseline.toml','test',self.root)

    def test_unwritable_audit_fails_closed(self):
        (self.root/'outputs/test_audit.jsonl').mkdir(parents=True)
        with self.assertRaises(OSError):
            authorize([self.row],self.root/'configs/baseline.toml','test',self.root)


if __name__=='__main__':
    unittest.main()
