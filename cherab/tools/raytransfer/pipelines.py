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

    def __init__(self, name=None, kind='power'):

        self.name = name
        self._matrix = None
        self._samples = 0
        self._bins = 0
        self.kind = kind

    @property
    def kind(self):
        """
        The kind of the pipeline. Can be 'power' or 'radiance'.
        In the case of 'power', the resulting matrix is multiplied by the sensitivity
        of the detector, and the units of the matrix are [m^3 sr], which gives the units
        of power [W] for the product of the ray transfer matrix and the emission profile.
        In case of 'radiance', the sensitivity is not taken into account and
        the matrix is calculated in [m], which gives the units of radiance [W m^-2 sr^-1]
        for the product of the ray transfer matrix and the emission profile.
        """
        return self._kind

    @kind.setter
    def kind(self, value):
        _kind = value.lower()
        if _kind in ('power', 'radiance'):
            self._kind = _kind
        else:
            raise ValueError("The kind property must be 'power' or 'radiance'.")

    @property
    def matrix(self):
        return self._matrix


class RayTransferPipeline0D(Pipeline0D, RayTransferPipelineBase):
    """
    Simple 0D pipeline for ray transfer matrix (geometry matrix) calculation.

    :param str name: The name of the pipeline. Default is 'RayTransferPipeline0D'.
    :param str kind: The kind of the pipeline. Can be 'power' (default) or 'radiance'.
        In the case of 'power', the resulting matrix is multiplied by the sensitivity
        of the detector, and the units of the matrix are [m^3 sr], which gives the units
        of power [W] for the product of the ray transfer matrix and the emission profile.
        In case of 'radiance', the sensitivity is not taken into account and
        the matrix is calculated in [m], which gives the units of radiance [W m^-2 sr^-1]
        for the product of the ray transfer matrix and the emission profile.
        Note that if the sensitivity of the detector is 1 (e.g. `PinholeCamera`, `VectorCamera`),
        the 'power' and 'radiance' give the same results.

    :ivar np.ndarray matrix: Ray transfer matrix, a 1D array of size :math:`N_{bin}`.

    .. code-block:: pycon

       >>> from cherab.tools.raytransfer import RayTransferPipeline0D
       >>> pipeline = RayTransferPipeline0D(kind='radiance')
    """

    def __init__(self, name='RayTransferPipeline0D', kind='power'):

        RayTransferPipelineBase.__init__(self, name, kind)

    def initialise(self, min_wavelength, max_wavelength, spectral_bins, spectral_slices, quiet):
        self._samples = 0
        self._bins = spectral_bins
        self._matrix = np.zeros(spectral_bins)

    def pixel_processor(self, slice_id):
        if self._kind == 'power':
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
    :param str kind: The kind of the pipeline. Can be 'power' (default) or 'radiance'.
        In the case of 'power', the resulting matrix is multiplied by the sensitivity
        of the detector, and the units of the matrix are [m^3 sr], which gives the units
        of power [W] for the product of the ray transfer matrix and the emission profile.
        In case of 'radiance', the sensitivity is not taken into account and
        the matrix is calculated in [m], which gives the units of radiance [W m^-2 sr^-1]
        for the product of the ray transfer matrix and the emission profile.
        Note that if the sensitivity of the detector is 1 (e.g. `PinholeCamera`, `VectorCamera`),
        the 'power' and 'radiance' give the same results.

    :ivar np.ndarray matrix: Ray transfer matrix, a 2D array of shape :math:`(N_{pixel}, N_{bin})`.

    .. code-block:: pycon

       >>> from cherab.tools.raytransfer import RayTransferPipeline1D
       >>> pipeline = RayTransferPipeline1D(kind='radiance')
    """

    def __init__(self, name='RayTransferPipeline1D', kind='power'):

        RayTransferPipelineBase.__init__(self, name, kind)
        self._pixels = None

    def initialise(self, pixels, pixel_samples, min_wavelength, max_wavelength, spectral_bins, spectral_slices, quiet):
        self._pixels = pixels
        self._samples = pixel_samples
        self._bins = spectral_bins
        self._matrix = np.zeros((pixels, spectral_bins))

    def pixel_processor(self, pixel, slice_id):
        if self._kind == 'power':
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
    :param str kind: The kind of the pipeline. Can be 'power' (default) or 'radiance'.
        In the case of 'power', the resulting matrix is multiplied by the sensitivity
        of the detector, and the units of the matrix are [m^3 sr], which gives the units
        of power [W] for the product of the ray transfer matrix and the emission profile.
        In case of 'radiance', the sensitivity is not taken into account and
        the matrix is calculated in [m], which gives the units of radiance [W m^-2 sr^-1]
        for the product of the ray transfer matrix and the emission profile.
        Note that if the sensitivity of the detector is 1 (e.g. `PinholeCamera`, `VectorCamera`),
        the 'power' and 'radiance' give the same results.

    :ivar np.ndarray matrix: Ray transfer matrix, a 3D array of shape :math:`(N_x, N_y, N_{bin})`.

    .. code-block:: pycon

       >>> from cherab.tools.raytransfer import RayTransferPipeline2D
       >>> pipeline = RayTransferPipeline2D(kind='radiance')
    """

    def __init__(self, name='RayTransferPipeline2D', kind='power'):

        RayTransferPipelineBase.__init__(self, name, kind)
        self._pixels = None

    def initialise(self, pixels, pixel_samples, min_wavelength, max_wavelength, spectral_bins, spectral_slices, quiet):
        self._pixels = pixels
        self._samples = pixel_samples
        self._bins = spectral_bins
        self._matrix = np.zeros((pixels[0], pixels[1], spectral_bins))

    def pixel_processor(self, x, y, slice_id):
        if self._kind == 'power':
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
