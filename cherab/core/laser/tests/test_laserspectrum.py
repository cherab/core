import unittest
from cherab.core.laser.laserspectrum import LaserSpectrum


class TestLaserSpectrum(unittest.TestCase):

    def test_laserspectrum_init_negative_values(self):
        # test zero min_wavelength error
        with self.assertRaises(ValueError,
                               msg="LaserSpectrum did not raise a ValueError with min_wavelength being zero."):
            LaserSpectrum(0., 100, 200)

        # test negative min_wavelength error
        with self.assertRaises(ValueError,
                               msg="LaserSpectrum did not raise a ValueError with min_wavelength being negative."):
            LaserSpectrum(-1, 100, 200)

        # test zero max_wavelength error
        with self.assertRaises(ValueError,
                               msg="LaserSpectrum did not raise a ValueError with max_wavelength being zero."):
            LaserSpectrum(10, 0, 200)

        # test negative max_wavelength error
        with self.assertRaises(ValueError,
                               msg="LaserSpectrum did not raise a ValueError with max_wavelength being negative."):
            LaserSpectrum(10, -1, 200)

        # test min_wavelength >= max_wavelength
        with self.assertRaises(ValueError,
                               msg="LaserSpectrum did not raise a ValueError with max_wavelength < min_wavelength."):
            LaserSpectrum(40, 30, 200)

        with self.assertRaises(ValueError,
                               msg="LaserSpectrum did not raise a ValueError with max_wavelength < min_wavelength."):
            LaserSpectrum(30, 30, 200)
        
        # test bins > 0
        with self.assertRaises(ValueError,
                               msg="LaserSpectrum did not raise a ValueError with max_wavelength < min_wavelength."):
            LaserSpectrum(30, 30, 0)

        with self.assertRaises(ValueError,
                               msg="LaserSpectrum did not raise a ValueError with max_wavelength < min_wavelength."):
            LaserSpectrum(30, 30, -1)