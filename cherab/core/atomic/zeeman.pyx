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
DEF SIGMA_PLUS_POLARISATION = 1
DEF SIGMA_MINUS_POLARISATION = -1

cdef class ZeemanStructure():
    r"""
    Provides wavelengths and ratios of
    :math:`\pi`-/:math:`\sigma`-polarised Zeeman components for any given value of
    magnetic field strength.

    :param list pi_components: A list of 2-tuples of Function1D objects that provide the
                               wavelengths and ratios of individual :math:`\pi`-polarised
                               Zeeman components for a given magnetic field strength:
                               [(wvl_func1, ratio_func1), (wvl_func2, ratio_func2), ...]
    :param list sigma_plus_components: A list of 2-tuples of Function1D objects that provide the
                                       wavelengths and ratios of individual
                                       :math:`\sigma^{+}`-polarised Zeeman components for
                                       a given magnetic field strength:
                                       [(wvl_func1, ratio_func1), (wvl_func2, ratio_func2), ...]
    :param list sigma_minus_components: A list of 2-tuples of Function1D objects that provide the
                                        wavelengths and ratios of individual
                                        :math:`\sigma^{-}`-polarised
                                        Zeeman components for a given magnetic field strength:
                                        [(wvl_func1, ratio_func1), (wvl_func2, ratio_func2), ...]
    """

    def __init__(self, pi_components, sigma_plus_components, sigma_minus_components):

        cdef tuple component

        self._pi_components = []
        self._sigma_plus_components = []
        self._sigma_minus_components = []

        for component in pi_components:
            if len(component) != 2:
                raise ValueError('Argument "pi_components" must be a list of 2-tuples.')
            self._pi_components.append((autowrap_function1d(component[MULTIPLET_WAVELENGTH]),
                                        autowrap_function1d(component[MULTIPLET_RATIO])))

        for component in sigma_plus_components:
            if len(component) != 2:
                raise ValueError('Argument "sigma_plus_components" must be a list of 2-tuples.')
            self._sigma_plus_components.append((autowrap_function1d(component[MULTIPLET_WAVELENGTH]),
                                                autowrap_function1d(component[MULTIPLET_RATIO])))

        for component in sigma_minus_components:
            if len(component) != 2:
                raise ValueError('Argument "sigma_minus_components" must be a list of 2-tuples.')
            self._sigma_minus_components.append((autowrap_function1d(component[MULTIPLET_WAVELENGTH]),
                                                 autowrap_function1d(component[MULTIPLET_RATIO])))

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

    def __call__(self, double b, str polarisation):

        if polarisation.lower() == 'pi':
            return np.asarray(self.evaluate(b, PI_POLARISATION))

        if polarisation.lower() == 'sigma_plus':
            return np.asarray(self.evaluate(b, SIGMA_PLUS_POLARISATION))

        if polarisation.lower() == 'sigma_minus':
            return np.asarray(self.evaluate(b, SIGMA_MINUS_POLARISATION))

        raise ValueError('Argument "polarisation" must be "pi", "sigma_plus" or "sigma_minus", {} given.'.fotmat(polarisation))
