import unittest
import numpy as np

from cherab.core.laser.laserspectrum import LaserSpectrum
from raysect.optical.spectrum import Spectrum

class TestLaserSpectrum(unittest.TestCase):

    def test_laserspectrum_init(self):
        # test min_wavelength boundaries
        with self.assertRaises(ValueError,
                               msg="LaserSpectrum did not raise a ValueError with min_wavelength being zero."):
            LaserSpectrum(0., 100, 200)
            LaserSpectrum(-1, 100, 200)

        # test max_wavelength boundaries
        with self.assertRaises(ValueError,
                               msg="LaserSpectrum did not raise a ValueError with max_wavelength being zero."):
            LaserSpectrum(10, 0, 200)
            LaserSpectrum(10, -1, 200)

        # test min_wavelength >= max_wavelength
        with self.assertRaises(ValueError,
                               msg="LaserSpectrum did not raise a ValueError with max_wavelength < min_wavelength."):
            LaserSpectrum(40, 30, 200)
            LaserSpectrum(30, 30, 200)
        
        # test bins > 0
        with self.assertRaises(ValueError,
                               msg="LaserSpectrum did not raise a ValueError with max_wavelength < min_wavelength."):
            LaserSpectrum(30, 30, 0)
            LaserSpectrum(30, 30, -1)

    def test_laserspectrum_changes(self):
        laser_spectrum = LaserSpectrum(100, 200, 100)

        # change min_wavelength to be larger or equal to max_wavelength
        with self.assertRaises(ValueError,
                               msg="LaserSpectrum did not raise ValueError for min_wavelength change "
                                   "with min_wavelength >= max_wavelength."):
            laser_spectrum.min_wavelength = 300
            laser_spectrum.min_wavelength = 200

        # change max_wavelength to be smaller than max_wavelength
        with self.assertRaises(ValueError,
                               msg="LaserSpectrum did not raise ValueError for max_wavelength change "
                                   "with min_wavelength > max_wavelength."):
            laser_spectrum.max_wavelength = 50
            laser_spectrum.max_wavelength = 100

        with self.assertRaises(ValueError,
                               msg="LaserSpectrum did not raise a ValueError with max_wavelength < min_wavelength."):
            laser_spectrum.bins = -1
            laser_spectrum.bins = 0

        # laser spectrum should have same behaviour as Spectrum from raysect.optical
        spectrum = Spectrum(laser_spectrum.min_wavelength, laser_spectrum.max_wavelength, laser_spectrum.bins)

        # test caching of spectrum data, behaviour should be consistent with raysect.optical.spectrum.Spectrum
        self.assertTrue(np.array_equal(laser_spectrum.wavelengths, spectrum.wavelengths),
                        "LaserSpectrum.wavelengths values are not equal to Spectrum.wavelengths "
                        "with same boundaries and number of bins")