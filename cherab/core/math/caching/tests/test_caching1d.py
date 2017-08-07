import unittest

import numpy as np

from cherab.core.math.caching import Caching1D


class TestCaching1D(unittest.TestCase):

    def setUp(self):
        self.function = lambda x: np.cos(10 * x)
        self.grad_f = lambda x: -10 * np.sin(10 * x)
        self.hess_f = lambda x: 100 * np.cos(10 * x)
        xmin, xmax = -10, 2
        self.space_area = xmin, xmax
        self.resolution = 0.1

    def tearDown(self):
        del self.function
        del self.grad_f
        del self.space_area
        del self.resolution

    def tolerance(self, x):
        tolerance = self.grad_f(x) * self.resolution + self.hess_f(x) * self.resolution ** 2 * 0.5
        return max(abs(tolerance), 0.1)

    def test_values(self):
        cached_func = Caching1D(self.function, self.space_area, self.resolution)

        for x in np.linspace(self.space_area[0], self.space_area[1], 100):
            self.assertAlmostEqual(cached_func(x), self.function(x), delta=self.tolerance(x),
                                   msg='Cached function at {} is too far from exact function!'.format(x))
            self.assertAlmostEqual(cached_func(x), self.function(x), delta=0.1,
                                   msg='Cached function at {} is too far from exact function!'.format(x))


if __name__ == '__main__':
    unittest.main()