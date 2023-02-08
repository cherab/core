# Copyright 2016-2021 Euratom
# Copyright 2016-2021 United Kingdom Atomic Energy Authority
# Copyright 2016-2021 Centro de Investigaciones Energéticas, Medioambientales y Tecnológicas
#
# Licensed under the EUPL, Version 1.1 or – as soon they will be approved by the
# European Commission - subsequent versions of the EUPL (the "Licence");
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at:
#
# https://joinup.ec.europa.eu/software/page/eupl5
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the Licence is distributed on an "AS IS" basis, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied.
#
# See the Licence for the specific language governing permissions and limitations
# under the Licence.

import unittest
import numpy as np

from raysect.optical import Spectrum
from raysect.optical.observer.pipeline import RadiancePipeline0D, SpectralRadiancePipeline0D
from cherab.tools.spectroscopy import TrapezoidalFilter, PolychromatorFilter, Polychromator, CzernyTurnerSpectrometer, Spectrometer


class TestPolychromatorFilter(unittest.TestCase):
    """
    Test for PolychromatorFilter class.
    """

    def test_spectrum(self):
        wavelengths = [658, 654, 656]  # unsorted
        samples = [0.5, 0.5, 1]  # non-zero at the ends
        poly_filter = PolychromatorFilter(wavelengths, samples, name='test_filter')
        wavelengths = np.linspace(653., 659., 7)
        spectrum_true = np.array([0, 0.5, 0.75, 1., 0.75, 0.5, 0])
        spectrum_test = np.array([poly_filter(wvl) for wvl in wavelengths])
        self.assertTrue(np.all(spectrum_true == spectrum_test))


class TestTrapezoidalFilter(unittest.TestCase):
    """
    Test for TrapezoidalFilter class.
    """

    def test_spectrum(self):
        wavelength = 500.
        window = 6.
        flat_top = 2.
        poly_filter = TrapezoidalFilter(wavelength, window, flat_top, 'test_filter')
        wavelengths = np.linspace(496., 504., 9)
        spectrum_true = np.array([0, 0, 0.5, 1., 1., 1., 0.5, 0, 0])
        spectrum_test = np.array([poly_filter(wvl) for wvl in wavelengths])
        self.assertTrue(np.all(spectrum_true == spectrum_test))


class TestPolychromator(unittest.TestCase):
    """
    Test cases for Polychromator class.
    """

    poly_filters_default = (TrapezoidalFilter(400., 6., 2., 'filter 1'),
                            TrapezoidalFilter(700., 8., 4., 'filter 2'))
    min_bins_per_window_default = 10

    def test_pipeline_classes(self):
        polychromator = Polychromator(self.poly_filters_default, self.min_bins_per_window_default, 'test polychromator')
        pipeline_classes_true = [RadiancePipeline0D, RadiancePipeline0D]
        self.assertSequenceEqual(pipeline_classes_true, polychromator.pipeline_classes)

    def test_pipeline_kwargs(self):
        polychromator = Polychromator(self.poly_filters_default, self.min_bins_per_window_default, 'test polychromator')
        pipeline_kwargs_true = [{'name': 'test polychromator: filter 1', 'filter': self.poly_filters_default[0]},
                                {'name': 'test polychromator: filter 2', 'filter': self.poly_filters_default[1]}]
        self.assertSequenceEqual(pipeline_kwargs_true, polychromator.pipeline_kwargs)
        
    def test_spectral_properties(self):
        polychromator = Polychromator(self.poly_filters_default, self.min_bins_per_window_default)
        min_wavelength_true = 397.
        max_wavelength_true = 704.
        spectral_bins_true = 512
        self.assertTrue(polychromator.min_wavelength == min_wavelength_true and
                        polychromator.max_wavelength == max_wavelength_true and
                        polychromator.spectral_bins == spectral_bins_true)

    def test_filter_change(self):
        """ Checks if the spectral properties are updated correctly when the filters are replaced."""
        polychromator = Polychromator(self.poly_filters_default, self.min_bins_per_window_default)
        polychromator.min_bins_per_window = 20
        polychromator.filters = [TrapezoidalFilter(500., 5., 2., 'filter 1'),
                                 TrapezoidalFilter(600., 7., 4., 'filter 2')]
        min_wavelength_true = 497.5
        max_wavelength_true = 603.5
        spectral_bins_true = 424
        self.assertTrue(polychromator.min_wavelength == min_wavelength_true and
                        polychromator.max_wavelength == max_wavelength_true and
                        polychromator.spectral_bins == spectral_bins_true)    


class TestSpectrometer(unittest.TestCase):
    """
    Test cases for Spectrometer class.
    """

    def test_pipeline_classes(self):
        wavelength_to_pixel = ([400., 400.5],)
        spectrometer = Spectrometer(wavelength_to_pixel, name='test spectrometer')
        self.assertSequenceEqual([SpectralRadiancePipeline0D], spectrometer.pipeline_classes)

    def test_pipeline_kwargs(self):
        wavelength_to_pixel = ([400., 400.5],)
        spectrometer = Spectrometer(wavelength_to_pixel, name='test spectrometer')
        self.assertSequenceEqual([{'name': 'test spectrometer'}], spectrometer.pipeline_kwargs)

    def test_spectral_properties(self):
        wavelength_to_pixel = ([400., 400.5, 401.5, 402., 404.], [600., 600.5, 601.5, 602., 604., 607.])
        spectrometer = Spectrometer(wavelength_to_pixel, min_bins_per_pixel=2, name='test spectrometer')
        min_wavelength_true = 400.
        max_wavelength_true = 607.
        spectra_bins_true = 828
        self.assertTrue(spectrometer.min_wavelength == min_wavelength_true and
                        spectrometer.max_wavelength == max_wavelength_true and
                        spectrometer.spectral_bins == spectra_bins_true)

    def test_calibration(self):
        wavelength_to_pixel = ([400., 400.5, 401.5, 402., 404.],)
        spectrometer = Spectrometer(wavelength_to_pixel, name='test spectrometer')
        spectrum = Spectrum(399, 405, 12)
        s, ds = np.linspace(0, 6., 13, retstep=True)
        spectrum.samples[:] = s[:-1] + 0.5 * ds
        calibrated_spectra = spectrometer.calibrate(spectrum)
        self.assertTrue(np.all(calibrated_spectra[0] == np.array([1.25, 2., 2.75, 4.])))


class TestCzernyTurnerSpectrometer(unittest.TestCase):
    """
    Test cases for CzernyTurnerSpectrometer class.
    """

    diffraction_order = 1
    grating = 2.e-3
    focal_length = 1.e9
    pixel_spacing = 2.e4
    diffraction_angle = 10.
    accommodated_spectra = ((400., 64), (500., 32))
    min_bins_per_pixel = 2

    def test_resolution(self):
        wavelengths = np.array([350., 550., 750.])
        resolutions_true = np.array([8.587997e-3, 7.199328e-3, 5.0599164e-3])
        spectrometer = CzernyTurnerSpectrometer(self.diffraction_order, self.grating, self.focal_length, self.pixel_spacing,
                                                self.diffraction_angle, self.accommodated_spectra, name='test spectrometer')
        resolutions = spectrometer.resolution(wavelengths)
        self.assertTrue(np.all(np.abs(resolutions / resolutions_true - 1.) < 1.e-7))

    def test_spectral_properties(self):
        min_wavelength_true = 400
        max_wavelength_true = 500.24326
        spectra_bins_true = 26377
        spectrometer = CzernyTurnerSpectrometer(self.diffraction_order, self.grating, self.focal_length, self.pixel_spacing,
                                                self.diffraction_angle, self.accommodated_spectra,
                                                min_bins_per_pixel=self.min_bins_per_pixel, name='test spectrometer')
        self.assertTrue(spectrometer.min_wavelength == min_wavelength_true and
                        spectrometer.spectral_bins == spectra_bins_true and
                        abs(spectrometer.max_wavelength - max_wavelength_true) < 1.e-5)


if __name__ == '__main__':
    unittest.main()
