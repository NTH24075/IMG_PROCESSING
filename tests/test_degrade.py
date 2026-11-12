import unittest
import numpy as np
from object_counting.degrade import degrade, seed_for


class DegradeTests(unittest.TestCase):
    image = np.random.default_rng(1).integers(0, 256, (20, 30, 3)).astype(np.uint8)

    def test_same_seed_same_image_and_range(self):
        for kind, level in [('gauss', 10), ('saltpepper', .05)]:
            a = degrade(self.image, kind, level, seed_for('x')); b = degrade(self.image, kind, level, seed_for('x'))
            self.assertTrue(np.array_equal(a, b)); self.assertEqual(a.dtype, np.uint8)

    def test_neutral_levels_do_not_change(self):
        for kind, level in [('none', 0), ('gamma', 1), ('contrast', 1), ('gain', 1), ('gauss', 0), ('saltpepper', 0)]:
            self.assertTrue(np.array_equal(degrade(self.image, kind, level), self.image), kind)

    def test_gamma_above_one_darkens_and_input_untouched(self):
        before = self.image.copy()
        self.assertLess(degrade(self.image, 'gamma', 2).mean(), self.image.mean())
        self.assertTrue(np.array_equal(before, self.image))
        with self.assertRaises(ValueError):
            degrade(self.image, 'blur', 1)

    def test_seed_is_stable(self):
        self.assertEqual(seed_for('scene_01_f0000'), seed_for('scene_01_f0000'))
        self.assertNotEqual(seed_for('a'), seed_for('b'))


if __name__ == '__main__':
    unittest.main()
