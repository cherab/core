# Copyright 2016-2023 Euratom
# Copyright 2016-2023 United Kingdom Atomic Energy Authority
# Copyright 2016-2023 Centro de Investigaciones Energéticas, Medioambientales y Tecnológicas
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
cimport numpy as np
from libc.math cimport INFINITY
from raysect.core.math.function.float cimport Function1D, Interpolator1DArray

cimport cython

np.import_array()

DEF MULTIPLET_WAVELENGTH = 0
DEF MULTIPLET_RATIO = 1

DEF PI_POLARISATION = 0
DEF SIGMA_PLUS_POLARISATION = 1
DEF SIGMA_MINUS_POLARISATION = -1

cdef class ZeemanStructure(CoreZeemanStructure):
    r"""
    Provides wavelengths and ratios of
    :math:`\pi`-/:math:`\sigma`-polarised Zeeman components for any given value of
    magnetic field strength.

    :param dict data: Dictionary containing the central wavelengths and relative
           intensities of Zeeman components with the following keys:
    |      'b': A 1D array of shape (N,) with magnetic field strength.
    |      'polarisation': A 1D array of shape (M,) with component polarisation
    |                      0 for :math:`\pi`-polarisation,
    |                      -1 for :math:`\sigma-`-polarisation,
    |                      1 for :math:`\sigma+`-polarisation.
    |      'wavelength': A 2D array of shape (M, N) with component wavelength as functions of
    |                    magnetic field strength.
    |      'ratio': A 2D array of shape (M, N) with component relative intensities
    |               as functions of magnetic field strength.
    :param bint extrapolate: Enable extrapolation (default=False).

    :ivar tuple b_range: The interpolation range of magnetic field strength.
    :ivar dict raw_data: Dictionary containing the raw data.
    """

    def __init__(self, dict data, bint extrapolate=False):

        self.raw_data = data

        b = data['b']
        polarisation = data['polarisation']
        wvl = data['wavelength']
        ratio = data['ratio']

        self.b_range = b.min(), b.max()

        extrapolation_type = 'quadratic' if extrapolate else 'none'

        components = {PI_POLARISATION: [], SIGMA_PLUS_POLARISATION: [], SIGMA_MINUS_POLARISATION: []}

        for pol in (PI_POLARISATION, SIGMA_PLUS_POLARISATION, SIGMA_MINUS_POLARISATION):
            indx, = np.where(polarisation == pol)
            for i in indx:
                wvl_func = Interpolator1DArray(b, wvl[i], 'cubic', extrapolation_type, INFINITY)
                ratio_func = Interpolator1DArray(b, ratio[i], 'cubic', extrapolation_type, INFINITY)
                components[pol].append((wvl_func, ratio_func))

        self._pi_components = components[PI_POLARISATION]
        self._sigma_plus_components = components[SIGMA_PLUS_POLARISATION]
        self._sigma_minus_components = components[SIGMA_MINUS_POLARISATION]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @cython.cdivision(True)
    cdef double[:, :] evaluate(self, double b, int polarisation):

        cdef int i
        cdef np.npy_intp multiplet_shape[2]
        cdef double ratio_sum
        cdef np.ndarray multiplet
        cdef double[:, :] multiplet_mv
        cdef Function1D wavelength, ratio
        cdef list components

        if b < 0:  # safety check in case if used stand-alone
            raise ValueError('Argument "b" (magnetic field strength) must be non-negative.')

        if polarisation == PI_POLARISATION:
            components = self._pi_components
        elif polarisation == SIGMA_PLUS_POLARISATION:
            components = self._sigma_plus_components
        elif polarisation == SIGMA_MINUS_POLARISATION:
            components = self._sigma_minus_components
        else:
            raise ValueError('Argument "polarisation" must be 0, 1 or -1, {} given.'.format(polarisation))

        multiplet_shape[0] = 2
        multiplet_shape[1] = len(components)
        multiplet = np.PyArray_SimpleNew(2, multiplet_shape, np.NPY_FLOAT64)
        multiplet_mv = multiplet

        ratio_sum = 0
        for i in range(len(components)):
            wavelength = components[i][MULTIPLET_WAVELENGTH]
            ratio = components[i][MULTIPLET_RATIO]

            multiplet_mv[MULTIPLET_WAVELENGTH, i] = wavelength.evaluate(b)
            multiplet_mv[MULTIPLET_RATIO, i] = ratio.evaluate(b)

            ratio_sum += multiplet_mv[MULTIPLET_RATIO, i]

        # normalising ratios
        if ratio_sum > 0:
            for i in range(multiplet_mv.shape[1]):
                multiplet_mv[MULTIPLET_RATIO, i] /= ratio_sum

        return multiplet_mv
