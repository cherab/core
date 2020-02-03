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


class RayTransferPipeline0D(Pipeline0D):
    """
    Simple 0D pipeline for ray transfer matrix (geometry matrix) calculation.

    :ivar np.ndarray matrix: Ray transfer matrix, a 1D array of size :math:`N_{bin}`.

    .. code-block:: pycon

       >>> from cherab.tools.raytransfer import RayTransferPipeline0D
       >>> pipeline = RayTransferPipeline0D()
    """

    def __init__(self, name=None):

        self.name = name or 'RayTransferPipeline0D'
        self._matrix = None
        self._samples = 0
        self._bins = 0

    def initialise(self, min_wavelength, max_wavelength, spectral_bins, spectral_slices, quiet):
        self._samples = 0
        self._bins = spectral_bins
        self._matrix = np.zeros(spectral_bins)

    def pixel_processor(self, slice_id):
        return RayTransferPixelProcessor(self._bins)

    def update(self, slice_id, packed_result, pixel_samples):
        self._samples += pixel_samples
        self._matrix += packed_result[0]

    def finalise(self):
        self._matrix /= self._samples

    @property
    def matrix(self):
        return self._matrix


class RayTransferPipeline1D(Pipeline1D):
    """
    Simple 1D pipeline for ray transfer matrix (geometry matrix) calculation.

    :ivar np.ndarray matrix: Ray transfer matrix, a 2D array of shape :math:`(N_{pixel}, N_{bin})`.

    .. code-block:: pycon

       >>> from cherab.tools.raytransfer import RayTransferPipeline1D
       >>> pipeline = RayTransferPipeline1D()
    """

    def __init__(self, name=None):

        self.name = name or 'RayTransferPipeline1D'
        self._matrix = None
        self._pixels = None
        self._samples = 0
        self._bins = 0

    def initialise(self, pixels, pixel_samples, min_wavelength, max_wavelength, spectral_bins, spectral_slices, quiet):
        self._pixels = pixels
        self._samples = pixel_samples
        self._bins = spectral_bins
        self._matrix = np.zeros((pixels, spectral_bins))

    def pixel_processor(self, pixel, slice_id):
        return RayTransferPixelProcessor(self._bins)

    def update(self, pixel, slice_id, packed_result):
        self._matrix[pixel] = packed_result[0] / self._samples

    def finalise(self):
        pass

    @property
    def matrix(self):
        return self._matrix


class RayTransferPipeline2D(Pipeline2D):
    """
    Simple 2D pipeline for ray transfer matrix (geometry matrix) calculation.

    :ivar np.ndarray matrix: Ray transfer matrix, a 3D array of shape :math:`(N_x, N_y, N_{bin})`.

    .. code-block:: pycon

       >>> from cherab.tools.raytransfer import RayTransferPipeline2D
       >>> pipeline = RayTransferPipeline2D()
    """

    def __init__(self, name=None):

        self.name = name or 'RayTransferPipeline2D'
        self._matrix = None
        self._pixels = None
        self._samples = 0
        self._bins = 0

    def initialise(self, pixels, pixel_samples, min_wavelength, max_wavelength, spectral_bins, spectral_slices, quiet):
        self._pixels = pixels
        self._samples = pixel_samples
        self._bins = spectral_bins
        self._matrix = np.zeros((pixels[0], pixels[1], spectral_bins))

    def pixel_processor(self, x, y, slice_id):
        return RayTransferPixelProcessor(self._bins)

    def update(self, x, y, slice_id, packed_result):
        self._matrix[x, y] = packed_result[0] / self._samples

    def finalise(self):
        pass

    @property
    def matrix(self):
        return self._matrix


class RayTransferPixelProcessor(PixelProcessor):
    """
    PixelProcessor that stores ray transfer matrix for each pixel.
    """

    def __init__(self, bins):
        self._matrix = np.zeros(bins)

    def add_sample(self, spectrum, sensitivity):
        self._matrix += spectrum.samples * sensitivity

    def pack_results(self):
        return (self._matrix, 0)
