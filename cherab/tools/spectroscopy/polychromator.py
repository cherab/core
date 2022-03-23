
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

import numpy as np
from raysect.optical import InterpolatedSF
from raysect.optical.observer import RadiancePipeline0D

from .instrument import SpectroscopicInstrument


class PolychromatorFilter(InterpolatedSF):
    """
    Defines a polychromator filter as a Raysect's InterpolatedSF.

    :param object wavelengths: 1D array of wavelengths in nanometers.
    :param object samples: 1D array of spectral samples.
    :param bool normalise: True/false toggle for whether to normalise the
                           spectral function so its integral equals 1.
    :param str name: Filter name (e.g. "H-alpha filter"). Default is ''.

    :ivar float min_wavelength: Lower wavelength bound of the filter's spectral range in nm.
    :ivar float max_wavelength: Upper wavelength bound of the filter's spectral range in nm.
    """

    def __init__(self, wavelengths, samples, normalise=False, name=''):

        wavelengths = np.array(wavelengths, dtype=np.float64)
        samples = np.array(samples, dtype=np.float64)

        if wavelengths.ndim != 1:
            raise ValueError("Wavelength array must be 1D.")

        if samples.shape[0] != wavelengths.shape[0]:
            raise ValueError("Wavelength and sample arrays must be the same length.")

        indices = np.argsort(wavelengths)
        wavelengths = wavelengths[indices]
        samples = samples[indices]

        self._min_wavelength = wavelengths[0]
        self._max_wavelength = wavelengths[-1]
        self._window = self._max_wavelength - self._min_wavelength
        self._central_wavelength = 0.5 * (self._max_wavelength + self._min_wavelength)

        # setting the ends of the filter to zero, if they are not
        if samples[0] != 0:
            wavelengths = np.insert(wavelengths, 0, wavelengths[0] * (1. - 1.e-15))
            samples = np.insert(samples, 0, 0)
        if samples[-1] != 0:
            wavelengths = np.append(wavelengths, wavelengths[-1] * (1. + 1.e-15))
            samples = np.append(samples, 0)

        super().__init__(wavelengths, samples, normalise)
        self._name = str(name)

    @property
    def name(self):
        # Filter name.
        return self._name

    @property
    def min_wavelength(self):
        # Lower wavelength bound of the filter's spectral range in nm.
        return self._min_wavelength

    @property
    def max_wavelength(self):
        # Upper wavelength bound of the filter's spectral range in nm.
        return self._max_wavelength

    @property
    def window(self):
        # Size of the filtering window in nm.
        return self._window

    @property
    def central_wavelength(self):
        # Central wavelength of the filter in nm.
        return self._central_wavelength


class TrapezoidalFilter(PolychromatorFilter):
    """
    Symmetrical trapezoidal polychromator filter.

    :param float wavelength: Central wavelength of the filter in nm.
    :param float window: Size of the filtering window in nm. Default is 3.
    :param float flat_top: Size of the flat top part of the filter in nm.
                           Default is None (equal to window).
    :param str name: Filter name (e.g. "H-alpha filter"). Default is ''.
    """

    def __init__(self, central_wavelength, window=3., flat_top=None, name=''):

        if central_wavelength <= 0:
            raise ValueError("Argument 'central_wavelength' must be positive.")

        if window <= 0:
            raise ValueError("Argument 'window' must be positive.")

        flat_top = flat_top or window

        if flat_top <= 0:
            raise ValueError("Argument 'flat_top' must be positive.")
        if flat_top > window:
            raise ValueError("Argument 'flat_top' must be less or equal than 'window'.")

        self._flat_top = flat_top

        if flat_top == window:
            flat_top -= flat_top * 1.e-15

        wavelengths = [central_wavelength - 0.5 * window,
                       central_wavelength - 0.5 * flat_top,
                       central_wavelength + 0.5 * flat_top,
                       central_wavelength + 0.5 * window]
        samples = [0, 1, 1, 0]
        super().__init__(wavelengths, samples, normalise=False, name=name)

    @property
    def flat_top(self):
        # Size of the flat top part of the filter in nm.
        return self._flat_top


class Polychromator(SpectroscopicInstrument):
    """
    A polychromator assembly with a set of different filters.

    :param list filters: List of the `PolychromatorFilter` instances.
    :param int min_bins_per_window: Minimal number of spectral bins
                                    per filtering window. Default is 10.
    :param str name: Polychromator name.

    .. code-block:: pycon

       >>> from raysect.optical import World
       >>> from raysect.optical.observer import FibreOptic
       >>> from cherab.tools.spectroscopy import Polychromator, TrapezoidalFilter
       >>>
       >>> world = World()
       >>> h_alpha_filter = TrapezoidalFilter(656.1, name='H-alpha filter')
       >>> ciii_465nm_filter = TrapezoidalFilter(464.8, name='CIII 465 nm filter')
       >>> polychromator = Polychromator([h_alpha_filter, ciii_465nm_filter], name='MyPolychromator')
       >>> fibreoptic = FibreOptic(name="MyFibreOptic", parent=world)
       >>> fibreoptic.min_wavelength = polychromator.min_wavelength
       >>> fibreoptic.max_wavelength = polychromator.max_wavelength
       >>> fibreoptic.spectral_bins = polychromator.spectral_bins
       >>> fibreoptic.pipelines = polychromator.create_pipelines()
    """

    def __init__(self, filters, min_bins_per_window=10, name=''):
        super().__init__(name)
        self.min_bins_per_window = min_bins_per_window
        self.filters = filters

    @property
    def min_bins_per_window(self):
        # Minimal number of spectral bins per filtering window.
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
        # List of the PolychromatorFilter instances.
        return self._filters

    @filters.setter
    def filters(self, value):
        for poly_filter in value:
            if not isinstance(poly_filter, PolychromatorFilter):
                raise TypeError('Property filters must contain only PolychromatorFilter instances.')

        self._filters = value
        self._clear_spectral_settings()
        self._pipeline_classes = None
        self._pipeline_kwargs = None

    def _update_pipeline_classes(self):
        self._pipeline_classes = [RadiancePipeline0D for poly_filter in self._filters]

    def _update_pipeline_kwargs(self):
        self._pipeline_kwargs = [{'name': self._name + ': ' + poly_filter.name, 'filter': poly_filter} for poly_filter in self._filters]

    def _update_spectral_settings(self):

        min_wavelength = np.inf
        max_wavelength = 0
        step = np.inf
        for poly_filter in self._filters:
            step = min(step, poly_filter.window / self._min_bins_per_window)
            min_wavelength = min(min_wavelength, poly_filter.min_wavelength)
            max_wavelength = max(max_wavelength, poly_filter.max_wavelength)

        self._min_wavelength = min_wavelength
        self._max_wavelength = max_wavelength
        self._spectral_bins = int(np.ceil((max_wavelength - min_wavelength) / step))
