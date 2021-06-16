
# Copyright 2014-2017 United Kingdom Atomic Energy Authority
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

import numpy as np
from raysect.optical.observer import SpectralRadiancePipeline0D

from .instrument import SpectroscopicInstrument


class Spectrometer(SpectroscopicInstrument):
    """
    Spectrometer base class.
    This is an abstract class.

    :param int spectral_bins: The number of spectral samples over the wavelength range.
    :param float reference_wavelength: Wavelength (in nm) corresponding to
                                       the centre of reference bin.
    :param int reference_bin: Reference bin index. Can be negative to specify the offset.
                              Default is None (spectral_bins // 2).
    :param str name: Spectrometer name.
    """

    def __init__(self, spectral_bins, reference_wavelength, reference_bin=None, name=''):
        super().__init__(name)
        self.spectral_bins = spectral_bins
        if reference_bin is None:
            self.reference_bin = self._spectral_bins // 2
        else:
            self.reference_bin = reference_bin
        self.reference_wavelength = reference_wavelength

    @property
    def spectral_bins(self):
        """
        The number of spectral samples over the wavelength range.
        """
        return self._spectral_bins

    @spectral_bins.setter
    def spectral_bins(self, value):

        value = int(value)
        if value <= 0:
            raise ValueError("Attribute 'spectral_bins' must be > 0.")

        self._spectral_bins = value
        self._clear_spectral_settings()

    @property
    def reference_wavelength(self):
        """
        Wavelength (in nm) corresponding to the centre of reference bin.
        """
        return self._reference_wavelength

    @reference_wavelength.setter
    def reference_wavelength(self, value):

        if value <= 0:
            raise ValueError("Attribute 'reference_wavelength' must be > 0.")

        self._reference_wavelength = value
        self._clear_spectral_settings()

    @property
    def reference_bin(self):
        """
        Reference bin index.
        """
        return self._reference_bin

    @reference_bin.setter
    def reference_bin(self, value):

        value = int(value)

        self._reference_bin = value
        self._clear_spectral_settings()

    def _update_pipeline_properties(self):
        self._pipeline_properties = [(SpectralRadiancePipeline0D, self._name, None)]

    def _clear_spectral_settings(self):
        self._min_wavelength = None
        self._max_wavelength = None


class SurveySpectrometer(Spectrometer):
    """
    Survey spectrometer with a constant spectral resolution.

    Note: survey spectrometers usually have non-constant spectral resolution
    in the supported wavelength range. However, Raysect does not support
    the observers with variable spectral resolution.

    :param float resolution: Spectral resolution in nm (can be negative).
    :param int spectral_bins: The number of spectral samples over the wavelength range.
    :param float reference_wavelength: Wavelength (in nm) corresponding to
                                       the centre of reference bin.
    :param int reference_bin: Reference bin index. Can be negative to specify the offset.
                              Default is None (spectral_bins // 2).
    :param str name: Spectrometer name.
    """

    def __init__(self, resolution, spectral_bins, reference_wavelength, reference_bin=None, name=''):
        super().__init__(spectral_bins, reference_wavelength, reference_bin, name)
        self.resolution = resolution

    @property
    def resolution(self):
        """
        Spectrometer resolution.
        """
        return self._resolution

    @resolution.setter
    def resolution(self, value):
        """
        Spectral resolution in nm (can be negative).
        """
        if value == 0:
            raise ValueError("Attribute 'resolution' must be non-zero.")

        self._resolution = value
        self._clear_spectral_settings()

    def _update_spectral_settings(self):

        if self._resolution > 0:
            self._min_wavelength = self._reference_wavelength - (self._reference_bin + 0.5) * self._resolution
            self._max_wavelength = self._min_wavelength + self._spectral_bins * self._resolution
        else:
            self._min_wavelength = self._reference_wavelength + (self._spectral_bins - self._reference_bin - 0.5) * self._resolution
            self._max_wavelength = self._min_wavelength - self._spectral_bins * self._resolution


class CzernyTurnerSpectrometer(Spectrometer):
    """
    Czerny-Turner high-resolution spectrometer.

    :param int diffraction_order: Diffraction order.
    :param float grating: Diffraction grating in nm-1.
    :param float focal_length: Focal length in nm.
    :param float pixel_spacing: Pixel to pixel spacing on CCD in nm.
    :param float diffraction_angle: Angle between incident and diffracted light in degrees.
    :param int spectral_bins: The number of spectral samples over the wavelength range.
    :param float reference_wavelength: Wavelength (in nm) corresponding to
                                       the centre of reference bin.
    :param int reference_bin: Reference bin index. Default is None (spectral_bins // 2).
    :param str name: Spectrometer name.
    """

    def __init__(self, diffraction_order, grating, focal_length, pixel_spacing, diffraction_angle, spectral_bins,
                 reference_wavelength, reference_bin=None, name=''):
        super().__init__(spectral_bins, reference_wavelength, reference_bin, name)
        self.diffraction_order = diffraction_order
        self.grating = grating
        self.focal_length = focal_length
        self.pixel_spacing = pixel_spacing
        self.diffraction_angle = diffraction_angle

    @property
    def diffraction_order(self):
        """ Diffraction order."""
        return self._diffraction_order

    @diffraction_order.setter
    def diffraction_order(self, value):

        value = int(value)
        if value <= 0:
            raise ValueError("Attribute 'diffraction_order' must be positive.")

        self._diffraction_order = value
        self._clear_spectral_settings()

    @property
    def grating(self):
        """ Diffraction grating in nm-1."""
        return self._grating

    @grating.setter
    def grating(self, value):

        if value <= 0:
            raise ValueError("Attribute 'grating' must be positive.")

        self._grating = value
        self._clear_spectral_settings()

    @property
    def focal_length(self):
        """ Focal length in nm."""
        return self._focal_length

    @focal_length.setter
    def focal_length(self, value):

        if value <= 0:
            raise ValueError("Attribute 'focal_length' must be positive.")

        self._focal_length = value
        self._clear_spectral_settings()

    @property
    def pixel_spacing(self):
        """ Pixel to pixel spacing on CCD in nm."""
        return self._pixel_spacing

    @pixel_spacing.setter
    def pixel_spacing(self, value):

        if value == 0:
            raise ValueError("Attribute 'pixel_spacing' must be non-zero.")

        self._pixel_spacing = value
        self._clear_spectral_settings()

    @property
    def diffraction_angle(self):
        """ Angle between incident and diffracted light in degrees."""
        return np.rad2deg(self._diffraction_angle)

    @diffraction_angle.setter
    def diffraction_angle(self, value):

        if value <= 0:
            raise ValueError("Attribute 'diffraction_angle' must be positive.")

        self._diffraction_angle = np.deg2rad(value)
        self._clear_spectral_settings()

    def _update_spectral_settings(self):

        resolution = self.resolution()

        if resolution > 0:
            self._min_wavelength = self._reference_wavelength - (self._reference_bin + 0.5) * resolution
            self._max_wavelength = self._min_wavelength + self._spectral_bins * resolution
        else:
            self._min_wavelength = self._reference_wavelength + (self._spectral_bins - self._reference_bin - 0.5) * resolution
            self._max_wavelength = self._min_wavelength - self._spectral_bins * resolution

    def resolution(self):
        """
        Calculates spectral resolution in nm.

        :return: resolution
        """
        grating = self._grating
        m = self._diffraction_order
        dxdp = self._pixel_spacing
        angle = self._diffraction_angle
        fl = self._focal_length

        p = 0.5 * m * grating * self._reference_wavelength
        resolution = dxdp * (np.sqrt(np.cos(angle)**2 - p * p) - p * np.tan(angle)) / (m * fl * grating)

        return resolution
