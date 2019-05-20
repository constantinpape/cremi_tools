import unittest
import numpy as np


class TestMetrics(unittest.TestCase):
    def test_voi(self):
        from cremi_tools.metrics import voi
        shape = (64, 64, 64)
        seg = np.random.randint(0, 100, size=shape, dtype='uint32')
        gt = np.random.randint(0, 100, size=shape, dtype='uint32')

        vis, vim = voi(seg, gt)
        self.assertGreater(vis, 0)
        self.assertGreater(vim, 0)


if __name__ == '__main__':
    unittest.main()
