"""Freeze guard on a throw-away git repo; never touches real test data."""
from pathlib import Path
import json
import subprocess
import tempfile
import unittest
from object_counting.experiments import touches_test, check_freeze, evaluate
from object_counting.io import write_csv

ROW = dict(image_id='scene_09_f0', image='x.png', scene_id='scene_09', split='test', sha256='abc')


class FreezeGuardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
        git = lambda *a: subprocess.run(['git', *a], cwd=self.root, check=True, capture_output=True)
        self.git = git
        git('init', '-q'); git('config', 'user.email', 't@t'); git('config', 'user.name', 't')
        (self.root/'configs').mkdir(); (self.root/'configs/baseline.toml').write_text('[processing]\n')
        (self.root/'.gitignore').write_text('outputs/\n')
        write_csv(self.root/'data/week1/manifests/test.csv', [ROW], ROW.keys())
        write_csv(self.root/'data/week1/annotations/counts.csv', [dict(image_id='scene_09_f0', count=1, review_status='peer_reviewed')],
                  ['image_id', 'count', 'review_status'])
        write_csv(self.root/'data/week1/manifests/dev.csv', [dict(ROW, image_id='d', split='dev', sha256='def')], ROW.keys())
        git('add', '.'); git('commit', '-qm', 'init')

    def tearDown(self):
        self.tmp.cleanup()

    def test_detects_test_by_id_hash_or_split(self):
        self.assertTrue(touches_test([dict(image_id='scene_09_f0')], self.root))
        self.assertTrue(touches_test([dict(image_id='renamed', sha256='abc')], self.root))
        self.assertTrue(touches_test([dict(image_id='r', source_image_id='scene_09_f0')], self.root))
        self.assertFalse(touches_test([dict(image_id='d', sha256='def', split='dev')], self.root))

    def test_blocked_without_tag_or_on_dirty_tree(self):
        with self.assertRaisesRegex(ValueError, 'freeze-test'):
            check_freeze('configs/baseline.toml', self.root)
        self.git('tag', 'freeze-test')
        self.assertEqual(check_freeze('configs/baseline.toml', self.root)['tag'], 'freeze-test')
        (self.root/'configs/baseline.toml').write_text('[processing]\nblur = 5\n')
        with self.assertRaisesRegex(ValueError, 'sạch'):
            check_freeze('configs/baseline.toml', self.root)

    def test_latest_tag_must_be_head_and_config_committed(self):
        self.git('tag', 'freeze-test'); (self.root/'a').write_text('x'); self.git('add', 'a'); self.git('commit', '-qm', 'fix')
        with self.assertRaisesRegex(ValueError, 'freeze-test'):
            check_freeze('configs/baseline.toml', self.root)
        self.git('tag', 'freeze-test-2')
        self.assertEqual(check_freeze('configs/baseline.toml', self.root)['tag'], 'freeze-test-2')
        (self.root/'configs/other.toml').write_text(''); (self.root/'.gitignore').write_text('outputs/\nconfigs/other.toml\n')
        self.git('commit', '-qam', 'ignore'); self.git('tag', 'freeze-test-3')
        with self.assertRaises(ValueError):
            check_freeze('configs/other.toml', self.root)

    def test_evaluate_refuses_test_before_reading_and_logs_nothing(self):
        with self.assertRaisesRegex(ValueError, 'KHÓA'):
            evaluate('configs/baseline.toml', 'test', root=self.root)
        self.assertFalse((self.root/'outputs/test_runs.log').exists())

    def test_log_written_before_run_once_frozen(self):
        self.git('tag', 'freeze-test')
        evaluate('configs/baseline.toml', 'test', root=self.root, examples=0)   # x.png missing → errors.csv, not a crash
        events = [json.loads(x)['event'] for x in (self.root/'outputs/test_runs.log').read_text().splitlines()]
        self.assertEqual(events, ['start', 'done'])


if __name__ == '__main__':
    unittest.main()
