import unittest
import numpy as np
from cherab.core.math.caching import Caching2D


class TestCaching2D(unittest.TestCase):

    def setUp(self):
        self.function = lambda x, y: (x - y) * y * np.cos(10 * x)
        self.grad_f = lambda x, y: np.array([y * (np.cos(10 * x) - 10 * (x - y) * np.sin(10 * x)),
                                            (x - 2 * y) * np.cos(10 * x)])
        self.hess_f = lambda x, y: np.array([[10 * y * (-2 * np.sin(10 * x) - 10 * (x - y) * np.cos(10 * x)), np.cos(10 * x) - 10 * (x - 2 * y) * np.sin(10 * x)],
                                             [np.cos(10 * x) - 10 * (x - 2 * y) * np.sin(10 * x)            , -2 * np.cos(10 * x)                               ]])
        xmin, xmax = -5, 2
        ymin, ymax = 1, 4.6
        self.space_area = xmin, xmax, ymin, ymax
        self.resolution = (0.08, 0.05)

    def tearDown(self):
        del self.function
        del self.grad_f
        del self.space_area
        del self.resolution

    def tolerance(self, x, y):
        np_res = np.array(self.resolution)
        tolerance = (self.grad_f(x, y) * np_res).sum() + 0.5 * (self.hess_f(x, y).dot(np_res) * np_res).sum()
        return max(abs(tolerance), 0.2)

    def test_values(self):
        cached_func = Caching2D(self.function, self.space_area, self.resolution)

        for x in np.linspace(self.space_area[0], self.space_area[1], 30):
            for y in np.linspace(self.space_area[2], self.space_area[3], 30):
                self.assertAlmostEqual(cached_func(x, y), self.function(x, y), delta=self.tolerance(x, y),
                                       msg='Cached function at ({}, {}) is too far from exact function!'.format(x, y))
                self.assertAlmostEqual(cached_func(x, y), self.function(x, y), delta=1.,
                                       msg='Cached function at ({}, {}) is too far from exact function!'.format(x, y))


if __name__ == '__main__':
    unittest.main()