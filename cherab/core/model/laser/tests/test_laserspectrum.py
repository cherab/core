import unittest
from scipy.integrate import nquad

from cherab.core.model.laser.laserspectrum import GaussianSpectrum, ConstantSpectrum

class TestLaserSpectrum(unittest.TestCase):

    def test_constantspectrum(self):
        """
        Laser spectrum should be normalized, i.e. integral from minuns inf. to inf. should be one.
        :return:
        """
        min_wavelength = 1039.9
        max_wavelength = 1040.1
        bins = 10

        spectrum = ConstantSpectrum(min_wavelength, max_wavelength, bins)

        # check if the power_spectral density is normalized

        integral = spectrum.power_spectral_density.sum() * spectrum.delta_wavelength
        self.assertTrue(integral == 1, msg="Power spectral density is not normalised.")

    def test_gaussian_spectrum(self):
        """
        Laser spectrum should be normalized, i.e. integral from minuns inf. to inf. should be one.
        :return:
        """
        min_wavelength = 1035
        max_wavelength = 1045
        bins = 100
        mean = 1040
        stddev = 0.5

        spectrum = GaussianSpectrum(min_wavelength, max_wavelength, bins, mean, stddev)
        integral = nquad(spectrum, [(min_wavelength, max_wavelength)])[0]

        # check if the power_spectral density is normalized
        self.assertAlmostEqual(integral, 1., 8, msg="Power spectral density function is not normalised.")

        psd = spectrum.power_spectral_density
        integral = 0
        for index in range(0, spectrum.bins - 1):
            integral += (psd[index] + psd[index + 1]) / 2 * spectrum.delta_wavelength

        self.assertAlmostEqual(integral, 1, 8, msg="Power spectral density is not normalised.")
