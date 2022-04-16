import unittest

from cherab.core.math import OutofRangeFallback1D, OutofRangeFallback2D, OutofRangeFallback3D


class TestFallbacks(unittest.TestCase):

    def setUp(self):
        """Initialisation with functions to map."""

        def f1d(x): return x
        self.function1d = f1d
        def f2d(x, y): return x + y
        self.function2d = f2d
        def f3d(x, y, z): return x + y + z
        self.function3d = f3d

    def test_fallback1d_init(self):

        OutofRangeFallback1D(self.function1d, 0)

        with self.assertRaises(ValueError):
            OutofRangeFallback1D(self.function1d, 0, xmin=10, xmax=10)
            OutofRangeFallback1D(self.function1d, 0, xmin=10, xmax=-10)

    def test_fallback1d(self):

        # test with no limits
        fb1d_nolim = OutofRangeFallback1D(self.function1d, 0)
        self.assertEqual(fb1d_nolim(-100), -100., msg="fallback should return input")

        # test bottom limit
        fb1d_bottomlim = OutofRangeFallback1D(self.function1d, 333, xmin=-10)
        self.assertEqual(fb1d_bottomlim(-10), -10., msg="fallback should return input")
        self.assertEqual(fb1d_bottomlim(-1000), 333., msg="fallback should return fallback")

        # test upper limit
        fb1d_upperlim = OutofRangeFallback1D(self.function1d, -333, xmax=100)
        self.assertEqual(fb1d_upperlim(100), 100., msg="Should return input")
        self.assertEqual(fb1d_upperlim(1000), -333., msg="Should return fallback")

    def test_fallback2d_init(self):

        # test limits
        with self.assertRaises(ValueError):
            OutofRangeFallback2D(self.function2d, 0, xmin=10, xmax=10)
            OutofRangeFallback2D(self.function2d, 0, xmin=10, xmax=-10)
            OutofRangeFallback2D(self.function2d, 0, ymin=10, ymax=10)
            OutofRangeFallback2D(self.function2d, 0, ymin=10, ymax=-10)

    def test_fallback2d(self):

        # nolimits
        fb2d_nolim = OutofRangeFallback2D(self.function2d, 0)

        self.assertEqual(fb2d_nolim(1, 1), 2, msg="should return sum of inputs")

        # bottom limits
        fb2d_bottomx = OutofRangeFallback2D(self.function2d, 333, xmin=-10)
        self.assertEqual(fb2d_bottomx(-10, -20), -30., msg="should return sum of inputs")
        self.assertEqual(fb2d_bottomx(-11, 200), 333., msg="should return fallback")

        fb2d_bottomy = OutofRangeFallback2D(self.function2d, 333, ymin=-10)
        self.assertEqual(fb2d_bottomy(-20, -10), -30., msg="should return sum of inputs")
        self.assertEqual(fb2d_bottomy(10, -11), 333., msg="should return fallback")

        fb2d_bottomxy = OutofRangeFallback2D(self.function2d, 333, xmin=-10, ymin=-10)
        self.assertEqual(fb2d_bottomxy(-10, -10), -20., msg="should return sum of inputs")
        self.assertEqual(fb2d_bottomxy(-11, 1), 333., msg="should return fallback")
        self.assertEqual(fb2d_bottomxy(1, -11), 333., msg="should return fallback")
        self.assertEqual(fb2d_bottomxy(-200, -200), 333., msg="should return fallback")

        # upper limits
        fb2d_upperx = OutofRangeFallback2D(self.function2d, 333, xmax=10)
        self.assertEqual(fb2d_upperx(10, 20), 30., msg="should return sum of inputs")
        self.assertEqual(fb2d_upperx(11, -200), 333., msg="should return fallback")

        fb2d_uppery = OutofRangeFallback2D(self.function2d, 333, ymax=10)
        self.assertEqual(fb2d_uppery(20, 10), 30., msg="should return sum of inputs")
        self.assertEqual(fb2d_uppery(1, 11), 333., msg="should return fallback")

        fb2d_upperxy = OutofRangeFallback2D(self.function2d, 333, xmax=10, ymax=10)
        self.assertEqual(fb2d_upperxy(10, 10), 20., msg="should return sum of inputs")
        self.assertEqual(fb2d_upperxy(11, 11), 333., msg="should return fallback")
        self.assertEqual(fb2d_upperxy(1, 11), 333., msg="should return fallback")
        self.assertEqual(fb2d_upperxy(11, 1), 333., msg="should return fallback")

    def test_fallback3d_init(self):

        # test limits
        with self.assertRaises(ValueError):
            OutofRangeFallback3D(self.function3d, 0, xmin=10, xmax=10)
            OutofRangeFallback3D(self.function3d, 0, xmin=10, xmax=-10)
            OutofRangeFallback3D(self.function3d, 0, ymin=10, ymax=10)
            OutofRangeFallback3D(self.function3d, 0, ymin=10, ymax=-10)
            OutofRangeFallback3D(self.function3d, 0, zmin=10, zmax=-10)
            OutofRangeFallback3D(self.function3d, 0, zmin=10, zmax=-10)

    def test_fallback3d(self):

        # nolimits
        fb3d_nolim = OutofRangeFallback3D(self.function3d, 0)

        self.assertEqual(fb3d_nolim(1, 1, 1), 3, msg="should return sum of inputs")

        # bottom limits
        fb3d_bottomx = OutofRangeFallback3D(self.function3d, 333, xmin=-10)
        self.assertEqual(fb3d_bottomx(-10, -20, -30), -60., msg="should return sum of inputs")
        self.assertEqual(fb3d_bottomx(-11, -10, -10), 333., msg="should return sum of inputs")

        fb3d_bottomy = OutofRangeFallback3D(self.function3d, 333, ymin=-10)
        self.assertEqual(fb3d_bottomy(-20, -10, -30), -60., msg="should return sum of inputs")
        self.assertEqual(fb3d_bottomy(10, -11, 30), 333., msg="should return fallback")

        fb3d_bottomz = OutofRangeFallback3D(self.function3d, 333, zmin=-10)
        self.assertEqual(fb3d_bottomz(-20, -20, -10), -50., msg="should return sum of inputs")
        self.assertEqual(fb3d_bottomz(10, 10, -11), 333., msg="should return fallback")

        fb3d_bottomxyz = OutofRangeFallback3D(self.function3d, 333, xmin=-10, ymin=-10, zmin=-10)
        self.assertEqual(fb3d_bottomxyz(-10, -10, -10), -30., msg="should return sum of inputs")
        self.assertEqual(fb3d_bottomxyz(-11, -10, -10), 333., msg="should return fallback")
        self.assertEqual(fb3d_bottomxyz(-10, -11, -10), 333., msg="should return fallback")
        self.assertEqual(fb3d_bottomxyz(-10, -10, -11), 333., msg="should return fallback")
        self.assertEqual(fb3d_bottomxyz(-11, -11, -10), 333., msg="should return fallback")
        self.assertEqual(fb3d_bottomxyz(-11, -10, -11), 333., msg="should return fallback")
        self.assertEqual(fb3d_bottomxyz(-10, -11, -11), 333., msg="should return fallback")
        self.assertEqual(fb3d_bottomxyz(-11, -11, -11), 333., msg="should return fallback")

        # upper limits
        fb3d_upperx = OutofRangeFallback3D(self.function3d, 333, xmax=10)
        self.assertEqual(fb3d_upperx(10, 20, 20), 50., msg="should return sum of inputs")
        self.assertEqual(fb3d_upperx(11, 5, 5), 333., msg="should return fallback")

        fb3d_uppery = OutofRangeFallback3D(self.function3d, 333, ymax=10)
        self.assertEqual(fb3d_uppery(20, 10, 20), 50., msg="should return sum of inputs")
        self.assertEqual(fb3d_uppery(5, 11, 5), 333., msg="should return fallback")

        fb3d_upperz = OutofRangeFallback3D(self.function3d, 333, zmax=10)
        self.assertEqual(fb3d_upperz(20, 20, 10), 50., msg="should return sum of inputs")
        self.assertEqual(fb3d_upperz(5, 5, 11), 333., msg="should return fallback")

        fb3d_bottomxyz = OutofRangeFallback3D(self.function3d, 333, xmax=10, ymax=10, zmax=10)
        self.assertEqual(fb3d_bottomxyz(10, 10, 10), 30., msg="should return sum of inputs")
        self.assertEqual(fb3d_bottomxyz(11, 10, 10), 333., msg="should return fallback")
        self.assertEqual(fb3d_bottomxyz(10, 11, 10), 333., msg="should return fallback")
        self.assertEqual(fb3d_bottomxyz(10, 10, 11), 333., msg="should return fallback")
        self.assertEqual(fb3d_bottomxyz(11, 11, 10), 333., msg="should return fallback")
        self.assertEqual(fb3d_bottomxyz(11, 10, 11), 333., msg="should return fallback")
        self.assertEqual(fb3d_bottomxyz(10, 11, 11), 333., msg="should return fallback")
        self.assertEqual(fb3d_bottomxyz(11, 11, 11), 333., msg="should return fallback")