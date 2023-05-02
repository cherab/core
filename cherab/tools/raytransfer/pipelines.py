# -*- coding: utf-8 -*-
#
# Copyright 2016-2018 Euratom
# Copyright 2016-2018 United Kingdom Atomic Energy Authority
# Copyright 2016-2018 Centro de Investigaciones Energéticas, Medioambientales y Tecnológicas
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
#
# The following code is created by Vladislav Neverov (NRC "Kurchatov Institute") for Cherab Spectroscopy Modelling Framework

"""
Very simple but fast pipelines for ray transfer matrix (geometry matrix) calculation.
When calculating the ray transfer matrix, the spectral array is used to store the radiance
from individual unit light sources and not the actual spectrum. In this case the spectral
array may contain ~ 10,000 spectral bins but the wavelengths for all of them are equal.
Spectral pipelines from Raysect still can be used, but they are slower compared to ray
transfer pipelines. Standard error is not calculated in these pipelines, only the mean value.
Dispersive rendering and adaptive sampling features are removed to improve the performance.
"""

import numpy as np
from raysect.optical.observer.base import Pipeline0D, Pipeline1D, Pipeline2D, PixelProcessor


class RayTransferPipelineBase():

    def __init__(self, name=None, units='power'):

        self.name = name
        self._matrix = None
        self._samples = 0
        self._bins = 0
        self.units = units

    @property
    def units(self):
        """
        The units in which the matrix is calculated. Can be 'power' or 'radiance'.
        The 'power' stands for [m^3 sr] and the 'radiance' stands for [m].
        """
        return self._units

    @units.setter
    def units(self, value):
        _units = value.lower()
        if _units in ('power', 'radiance'):
            self._units = _units
        else:
            raise ValueError("The units property must be 'power' or 'radiance'.")

    @property
    def matrix(self):
        return self._matrix


class RayTransferPipeline0D(Pipeline0D, RayTransferPipelineBase):
    """
    Simple 0D pipeline for ray transfer matrix (geometry matrix) calculation.

    :param str name: The name of the pipeline. Default is 'RayTransferPipeline0D'.
    :param str units: The units in which the matrix is calculated. Can
                      be 'power' (default) or 'radiance'.
                      The 'power' stands for [m^3 sr] and when the matrix is collapsed with
                      the emission profile [W m^-3 sr-1 nm-1] it gives the power [W nm-1].
                      The 'radiance' stands for [m] and when the matrix is collapsed with
                      the emission profile it gives the radiance [W m^-2 sr-1 nm-1].
                      If the 'power' is selected, the matrix is multiplied by the detector sensitivity.
                      Note that if the detector sensitivity is 1, the 'power' and 'radiance'
                      give the same results.

    :ivar np.ndarray matrix: Ray transfer matrix, a 1D array of size :math:`N_{bin}`.

    .. code-block:: pycon

       >>> from cherab.tools.raytransfer import RayTransferPipeline0D
       >>> pipeline = RayTransferPipeline0D(units='radiance')
    """

    def __init__(self, name='RayTransferPipeline0D', units='power'):

        RayTransferPipelineBase.__init__(self, name, units)

    def initialise(self, min_wavelength, max_wavelength, spectral_bins, spectral_slices, quiet):
        self._samples = 0
        self._bins = spectral_bins
        self._matrix = np.zeros(spectral_bins)

    def pixel_processor(self, slice_id):
        if self._units == 'power':
            return PowerRayTransferPixelProcessor(self._bins)
        else:
            return RadianceRayTransferPixelProcessor(self._bins)

    def update(self, slice_id, packed_result, pixel_samples):
        self._samples += pixel_samples
        self._matrix += packed_result[0]

    def finalise(self):
        self._matrix /= self._samples


class RayTransferPipeline1D(Pipeline1D, RayTransferPipelineBase):
    """
    Simple 1D pipeline for ray transfer matrix (geometry matrix) calculation.

    :param str name: The name of the pipeline. Default is 'RayTransferPipeline0D'.
    :param str units: The units in which the matrix is calculated. Can
                      be 'power' (default) or 'radiance'.
                      The 'power' stands for [m^3 sr] and when the matrix is collapsed with
                      the emission profile [W m^-3 sr-1 nm-1] it gives the power [W nm-1].
                      The 'radiance' stands for [m] and when the matrix is collapsed with
                      the emission profile it gives the radiance [W m^-2 sr-1 nm-1].
                      If the 'power' is selected, the matrix is multiplied by the detector sensitivity.
                      Note that if the detector sensitivity is 1, the 'power' and 'radiance'
                      give the same results.

    :ivar np.ndarray matrix: Ray transfer matrix, a 2D array of shape :math:`(N_{pixel}, N_{bin})`.

    .. code-block:: pycon

       >>> from cherab.tools.raytransfer import RayTransferPipeline1D
       >>> pipeline = RayTransferPipeline1D(units='radiance')
    """

    def __init__(self, name='RayTransferPipeline1D', units='power'):

        RayTransferPipelineBase.__init__(self, name, units)
        self._pixels = None

    def initialise(self, pixels, pixel_samples, min_wavelength, max_wavelength, spectral_bins, spectral_slices, quiet):
        self._pixels = pixels
        self._samples = pixel_samples
        self._bins = spectral_bins
        self._matrix = np.zeros((pixels, spectral_bins))

    def pixel_processor(self, pixel, slice_id):
        if self._units == 'power':
            return PowerRayTransferPixelProcessor(self._bins)
        else:
            return RadianceRayTransferPixelProcessor(self._bins)

    def update(self, pixel, slice_id, packed_result):
        self._matrix[pixel] = packed_result[0] / self._samples

    def finalise(self):
        pass


class RayTransferPipeline2D(Pipeline2D, RayTransferPipelineBase):
    """
    Simple 2D pipeline for ray transfer matrix (geometry matrix) calculation.

    :param str name: The name of the pipeline. Default is 'RayTransferPipeline0D'.
    :param str units: The units in which the matrix is calculated. Can
                      be 'power' (default) or 'radiance'.
                      The 'power' stands for [m^3 sr] and when the matrix is collapsed with
                      the emission profile [W m^-3 sr-1 nm-1] it gives the power [W nm-1].
                      The 'radiance' stands for [m] and when the matrix is collapsed with
                      the emission profile it gives the radiance [W m^-2 sr-1 nm-1].
                      If the 'power' is selected, the matrix is multiplied by the detector sensitivity.
                      Note that if the detector sensitivity is 1, the 'power' and 'radiance'
                      give the same results.

    :ivar np.ndarray matrix: Ray transfer matrix, a 3D array of shape :math:`(N_x, N_y, N_{bin})`.

    .. code-block:: pycon

       >>> from cherab.tools.raytransfer import RayTransferPipeline2D
       >>> pipeline = RayTransferPipeline2D(units='radiance')
    """

    def __init__(self, name='RayTransferPipeline2D', units='power'):

        RayTransferPipelineBase.__init__(self, name, units)
        self._pixels = None

    def initialise(self, pixels, pixel_samples, min_wavelength, max_wavelength, spectral_bins, spectral_slices, quiet):
        self._pixels = pixels
        self._samples = pixel_samples
        self._bins = spectral_bins
        self._matrix = np.zeros((pixels[0], pixels[1], spectral_bins))

    def pixel_processor(self, x, y, slice_id):
        if self._units == 'power':
            return PowerRayTransferPixelProcessor(self._bins)
        else:
            return RadianceRayTransferPixelProcessor(self._bins)

    def update(self, x, y, slice_id, packed_result):
        self._matrix[x, y] = packed_result[0] / self._samples

    def finalise(self):
        pass


class RayTransferPixelProcessorBase(PixelProcessor):
    """
    Base class for PixelProcessor that stores ray transfer matrix for each pixel.
    """

    def __init__(self, bins):
        self._matrix = np.zeros(bins)

    def pack_results(self):
        return (self._matrix, 0)


class RadianceRayTransferPixelProcessor(RayTransferPixelProcessorBase):
    """
    PixelProcessor that stores ray transfer matrix in the units of [m] for each pixel.
    """

    def add_sample(self, spectrum, sensitivity):
        self._matrix += spectrum.samples


class PowerRayTransferPixelProcessor(RayTransferPixelProcessorBase):
    """
    PixelProcessor that stores ray transfer matrix in the units of [m^3 sr] for each pixel.
    """

    def add_sample(self, spectrum, sensitivity):
        self._matrix += spectrum.samples * sensitivity
