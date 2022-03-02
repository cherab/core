import unittest
from cherab.core.laser.models.laserspectrum_base import LaserSpectrum_base


class TestLaserSpectrum_base(unittest.TestCase):

    def test_laserspectrum_init_negative_values(self):
        # test zero min_wavelength error
        with self.assertRaises(ValueError,
                               msg="LaserSpectrum did not raise a ValueError with min_wavelength being zero."):
            LaserSpectrum_base(0., 100, 200, 50)

        # test negative min_wavelength error
        with self.assertRaises(ValueError,
                               msg="LaserSpectrum did not raise a ValueError with min_wavelength being negative."):
            LaserSpectrum_base(-1, 100, 200, 50)

        # test zero max_wavelength error
        with self.assertRaises(ValueError,
                               msg="LaserSpectrum did not raise a ValueError with max_wavelength being zero."):
            LaserSpectrum_base(10, 0, 200, 50)

        # test negative max_wavelength error
        with self.assertRaises(ValueError,
                               msg="LaserSpectrum did not raise a ValueError with max_wavelength being negative."):
            LaserSpectrum_base(10, -1, 200, 50)

        # test negative central_wavelength error
        with self.assertRaises(ValueError,
                               msg="LaserSpectrum did not raise a ValueError with central_wavelength being negative."):
            LaserSpectrum_base(10, 20, 200, -1)

        with self.assertRaises(ValueError,
                               msg="LaserSpectrum did not raise a ValueError with central_wavelength being 0."):
            LaserSpectrum_base(10, 20, 200, 0)

        with self.assertRaises(ValueError,
                               msg="LaserSpectrum did not raise a ValueError with min_wavelength < max_wavelength."):
            LaserSpectrum_base(20, 30, 200, 50)
