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

from raysect.optical.observer.pipeline import RadiancePipeline0D, SpectralRadiancePipeline0D
from cherab.tools.spectroscopy import TrapezoidalFilter, PolychromatorFilter, Polychromator, CzernyTurnerSpectrometer, SurveySpectrometer


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

    def test_pipeline_properties(self):
        polychromator = Polychromator(self.poly_filters_default, self.min_bins_per_window_default, 'test polychromator')
        pipeline_properties_true = [(RadiancePipeline0D, 'test polychromator: filter 1', self.poly_filters_default[0]),
                                    (RadiancePipeline0D, 'test polychromator: filter 2', self.poly_filters_default[1])]
        self.assertSequenceEqual(pipeline_properties_true, polychromator.pipeline_properties)

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


class TestSurveySpectrometer(unittest.TestCase):
    """
    Test cases for SurveySpectrometer class.
    """

    def test_pipeline_properties(self):
        resolution = 0.1
        reference_wavelength = 500
        reference_bin = 50
        spectral_bins = 200
        spectrometer = SurveySpectrometer(resolution, spectral_bins, reference_wavelength, reference_bin, name='test spectrometer')
        pipeline_properties_true = [(SpectralRadiancePipeline0D, 'test spectrometer', None)]
        self.assertSequenceEqual(pipeline_properties_true, spectrometer.pipeline_properties)

    def test_spectral_properties(self):
        resolution = 0.1
        reference_wavelength = 500
        reference_bin = 50
        spectral_bins = 200
        spectrometer = SurveySpectrometer(resolution, spectral_bins, reference_wavelength, reference_bin, name='test spectrometer')
        min_wavelength_true = 494.95
        max_wavelength_true = 514.95
        self.assertTrue(spectrometer.min_wavelength == min_wavelength_true and
                        spectrometer.max_wavelength == max_wavelength_true)


class TestCzernyTurnerSpectrometer(unittest.TestCase):
    """
    Test cases for CzernyTurnerSpectrometer class.
    """

    diffraction_order = 1
    grating = 2.e-3
    focal_length = 1.e9
    pixel_spacing = 2.e4
    diffraction_angle = 10.
    spectral_bins = 512
    reference_bin = 255

    def test_resolution(self):
        wavelengths = [350., 550., 750.]
        resolutions_true = np.array([8.587997e-3, 7.199328e-3, 5.0599164e-3])
        spectrometer = CzernyTurnerSpectrometer(self.diffraction_order, self.grating, self.focal_length, self.pixel_spacing,
                                                self.diffraction_angle, self.spectral_bins, 500., self.reference_bin,
                                                name='test spectrometer')
        resolutions = []
        for wvl in wavelengths:
            spectrometer.reference_wavelength = wvl
            resolutions.append(spectrometer.resolution)
        self.assertTrue(np.all(np.abs(resolutions / resolutions_true - 1.) < 1.e-7))

    def test_spectral_properties(self):
        wavelength = 500.
        min_wavelength_true = 498.0575
        max_wavelength_true = 501.9501
        spectrometer = CzernyTurnerSpectrometer(self.diffraction_order, self.grating, self.focal_length, self.pixel_spacing,
                                                self.diffraction_angle, self.spectral_bins, wavelength, self.reference_bin,
                                                name='test spectrometer')
        self.assertTrue(abs(spectrometer.min_wavelength - min_wavelength_true) < 1.e-4 and
                        abs(spectrometer.max_wavelength - max_wavelength_true) < 1.e-4)


if __name__ == '__main__':
    unittest.main()
