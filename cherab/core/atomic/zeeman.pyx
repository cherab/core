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

import numpy as np
cimport numpy as np

from cherab.core.math.function cimport autowrap_function1d, Function1D

cimport cython

np.import_array()

DEF MULTIPLET_WAVELENGTH = 0
DEF MULTIPLET_RATIO = 1

DEF PI_POLARISATION = 0
DEF SIGMA_POLARISATION = 1

cdef class ZeemanStructure():
    r"""
    Provides wavelengths and ratios of
    :math:`\pi`-/:math:`\sigma`-polarised Zeeman components for any given value of
    magnetic field strength.

    :param list wavelengths_pi: A list of Function1D objects that provide the wavelengths
                                of individual :math:`\pi`-polarised Zeeman components for a given
                                magnetic field strength.
    :param list ratios_pi: A list of Function1D objects that provide the ratios
                           of individual :math:`\pi`-polarised Zeeman components for a given
                           magnetic field strength. The sum of all ratios must be equal to 1.
    :param list wavelengths_sigma: A list of Function1D objects that provide the wavelengths
                                   of individual :math:`\sigma`-polarised Zeeman components fo
                                   a given magnetic field strength.
    :param list ratios_sigma: A list of Function1D objects that provide the ratios
                              of individual :math:`\sigma`-polarised Zeeman components for a given
                              magnetic field strength. The sum of all ratios must be equal to 1.
    """

    def __init__(self, wavelengths_pi, ratios_pi, wavelengths_sigma, ratios_sigma):

        if len(wavelengths_pi) != len(ratios_pi):
            raise ValueError('The lengths of "wavelengths_pi" ({}) and "ratios_pi" ({}) do not match.'.format(len(wavelengths_pi),
                                                                                                              len(ratios_pi)))

        if len(wavelengths_sigma) != len(ratios_sigma):
            raise ValueError('The lengths of "wavelengths_sigma" ({}) and "ratios_sigma" ({}) do not match.'.format(len(wavelengths_sigma),
                                                                                                                    len(ratios_sigma)))

        self._number_of_pi_lines = len(wavelengths_pi)
        self._number_of_sigma_lines = len(wavelengths_sigma)

        self._wavelengths = wavelengths_pi + wavelengths_sigma
        self._ratios = ratios_pi + ratios_sigma

        for i in range(len(self._wavelengths)):
            self._wavelengths[i] = autowrap_function1d(self._wavelengths[i])
            self._ratios[i] = autowrap_function1d(self._ratios[i])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @cython.cdivision(True)
    cdef double[:, :] evaluate(self, double b, bint polarisation):

        cdef int i, start, number_of_lines
        cdef np.npy_intp multiplet_shape[2]
        cdef double ratio_sum
        cdef np.ndarray multiplet
        cdef double[:, :] multiplet_mv
        cdef Function1D wavelength, ratio

        b = max(0, b)

        if polarisation == PI_POLARISATION:
            start = 0
            number_of_lines = self._number_of_pi_lines
        else:
            start = self._number_of_pi_lines
            number_of_lines = self._number_of_sigma_lines

        multiplet_shape[0] = 2
        multiplet_shape[1] = number_of_lines
        multiplet = np.PyArray_SimpleNew(2, multiplet_shape, np.NPY_FLOAT64)
        multiplet_mv = multiplet

        ratio_sum = 0
        for i in range(number_of_lines):
            wavelength = self._wavelengths[start + i]
            ratio = self._ratios[start + i]
            multiplet_mv[MULTIPLET_WAVELENGTH, i] = wavelength.evaluate(b)
            multiplet_mv[MULTIPLET_RATIO, i] = ratio.evaluate(b)
            ratio_sum += multiplet_mv[1, i]

        # normalising ratios
        if ratio_sum > 0:
            for i in range(number_of_lines):
                multiplet_mv[1, i] /= ratio_sum

        return multiplet_mv

    def __call__(self, double b, str polarisation):

        if polarisation == 'pi':
            return np.asarray(self.evaluate(b, PI_POLARISATION))

        if polarisation == 'sigma':
            return np.asarray(self.evaluate(b, SIGMA_POLARISATION))

        raise ValueError('Argument "polarisation" must be "pi" or "sigma", {} given.'.fotmat(polarisation))
