
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
from raysect.optical import InterpolatedSF
from raysect.optical.observer import RadiancePipeline0D

from .instrument import SpectroscopicInstrument


class PolychromatorFilter(InterpolatedSF):
    """
    Defines a symmetrical trapezoidal polychromator filter as a Raysect's InterpolatedSF.

    :param float wavelength: Central wavelength of the filter in nm.
    :param float window: Size of the filtering window in nm. Default is 3.
    :param float flat_top: Size of the flat top part of the filter in nm.
                           Default is None (equal to window).
    :param str name: Filter name (e.g. "H-alpha filter"). Default is ''.

    """

    def __init__(self, wavelength, window=3., flat_top=None, name=''):

        if wavelength <= 0:
            raise ValueError("Argument 'wavelength' must be positive.")

        if window <= 0:
            raise ValueError("Argument 'window' must be positive.")

        flat_top = flat_top or window - 1.e-15

        if flat_top <= 0:
            raise ValueError("Argument 'flat_top' must be positive.")
        if flat_top > window:
            raise ValueError("Argument 'flat_top' must be less or equal than 'window'.")
        if flat_top == window:
            flat_top = window - 1.e-15

        self._window = window
        self._flat_top = flat_top
        self._wavelength = wavelength
        self._name = str(name)

        wavelengths = [wavelength - 0.5 * window,
                       wavelength - 0.5 * flat_top,
                       wavelength + 0.5 * flat_top,
                       wavelength + 0.5 * window]
        samples = [0, 1, 1, 0]
        super().__init__(wavelengths, samples, normalise=False)

    @property
    def window(self):
        """ Size of the filtering window in nm."""
        return self._window

    @property
    def flat_top(self):
        """ Size of the flat top part of the filter in nm."""
        return self._flat_top

    @property
    def wavelength(self):
        """ Central wavelength of the filter in nm."""
        return self._wavelength

    @property
    def name(self):
        """ Filter name."""
        return self._name


class Polychromator(SpectroscopicInstrument):
    """
    A polychromator assembly with a set of different filters.

    :param list filters: List of the `PolychromatorFilter` instances.
    :param int min_bins_per_window: Minimal number of spectral bins
                                    per filtering window. Default is 10.
    """

    def __init__(self, filters, min_bins_per_window=10, name=''):
        super().__init__(name)
        self.min_bins_per_window = min_bins_per_window
        self.filters = filters

    @property
    def min_bins_per_window(self):
        """
        Minimal number of spectral bins per filtering window.
        """
        return self._min_bins_per_window

    @min_bins_per_window.setter
    def min_bins_per_window(self, value):

        value = int(value)
        if value <= 0:
            raise ValueError("Attribute 'min_bins_per_window' must be positive.")

        self._min_bins_per_window = value
        self._clear_spectral_settings()

    @property
    def filters(self):
        """
        List of the PolychromatorFilter instances.
        """
        return self._filters

    @filters.setter
    def filters(self, value):
        for poly_filter in value:
            if not isinstance(poly_filter, PolychromatorFilter):
                raise TypeError('Property filters must contain only PolychromatorFilter instances.')

        self._filters = value
        self._clear_spectral_settings()
        self._pipeline_properties = None

    def _update_pipeline_properties(self):
        self._pipeline_properties = [(RadiancePipeline0D, self._name + ': ' + poly_filter.name, poly_filter) for poly_filter in self._filters]

    def _update_spectral_settings(self):

        min_wavelength = np.inf
        max_wavelength = 0
        step = np.inf
        for poly_filter in self._filters:
            step = min(step, poly_filter.window / self._min_bins_per_window)
            min_wavelength = min(min_wavelength, poly_filter.wavelength - 0.5 * poly_filter.window)
            max_wavelength = max(max_wavelength, poly_filter.wavelength + 0.5 * poly_filter.window)

        self._min_wavelength = min_wavelength
        self._max_wavelength = max_wavelength
        self._spectral_bins = int(np.ceil((max_wavelength - min_wavelength) / step))
