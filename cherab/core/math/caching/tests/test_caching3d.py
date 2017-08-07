import unittest
import numpy as np
from cherab.core.math.caching import Caching3D


class TestCaching3D(unittest.TestCase):

    def setUp(self):
        self.function = lambda x, y, z: (x - y) * y * np.cos(10 * x) * np.exp(z) + z
        self.grad_f = lambda x, y, z: np.array([y * np.exp(z) * (np.cos(10 * x) - 10 * (x - y) * np.sin(10 * x)),
                                                (x - 2 * y) * np.cos(10 * x) * np.exp(z),
                                                (x - y) * y * np.cos(10 * x) * np.exp(z) + 1])
        self.hess_f = lambda x, y, z: np.array([[10 * y * (-2 * np.sin(10 * x) - 10 * (x - y) * np.cos(10 * x)) * np.exp(z), np.cos(10 * x) - 10 * (x - 2 * y) * np.sin(10 * x) * np.exp(z), y * np.exp(z) * (np.cos(10 * x) - 10 * (x - y) * np.sin(10 * x))],
                                                [np.cos(10 * x) - 10 * (x - 2 * y) * np.sin(10 * x) * np.exp(z)            , -2 * np.cos(10 * x) * np.exp(z)                               , (x - 2 * y) * np.cos(10 * x) * np.exp(z)                        ],
                                                [y * np.exp(z) * (np.cos(10 * x) - 10 * (x - y) * np.sin(10 * x))          , (x - 2 * y) * np.cos(10 * x) * np.exp(z)                      , (x - y) * y * np.cos(10 * x) * np.exp(z)                        ]])
        xmin, xmax = -5, 2
        ymin, ymax = 0.5, 4
        zmin, zmax = -8.3, -1.1
        self.space_area = xmin, xmax, ymin, ymax, zmin, zmax
        xresolution = 0.1
        yresolution = 0.05
        zresolution = 0.2
        self.resolution = xresolution, yresolution, zresolution

    def tearDown(self):
        del self.function

    def tolerance(self, x, y, z):
        np_res = np.array(self.resolution)
        tolerance = (self.grad_f(x, y, z) * np_res).sum() + 0.5 * (self.hess_f(x, y, z).dot(np_res) * np_res).sum()
        return max(abs(tolerance), 0.2)

    def test_values(self):
        cached_func = Caching3D(self.function, self.space_area, self.resolution)

        for x in np.linspace(self.space_area[0], self.space_area[1], 20):
            for y in np.linspace(self.space_area[2], self.space_area[3], 20):
                for z in np.linspace(self.space_area[4], self.space_area[5], 20):
                    self.assertAlmostEqual(cached_func(x, y, z), self.function(x, y, z), delta=self.tolerance(x, y, z),
                                           msg='Cached function at ({}, {}, {}) is too far from exact function!'.format(x, y, z))
                    self.assertAlmostEqual(cached_func(x, y, z), self.function(x, y, z), delta=1.,
                                           msg='Cached function at ({}, {}, {}) is too far from exact function!'.format(x, y, z))


if __name__ == '__main__':
    unittest.main()