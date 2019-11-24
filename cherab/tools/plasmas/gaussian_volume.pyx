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

cimport cython
from libc.math cimport exp
from cherab.core.math.function cimport Function3D


cdef class GaussianVolume(Function3D):

    cdef double peak
    cdef double sigma
    cdef double _constant
    cdef bint _cache
    cdef double _cache_x, _cache_y, _cache_z, _cache_v

    def __init__(self, peak, sigma):
        self.peak = peak
        self.sigma = sigma
        self._constant = (2*self.sigma*self.sigma)

        # last value cache
        self._cache = False
        self._cache_x = 0.0
        self._cache_y = 0.0
        self._cache_z = 0.0
        self._cache_v = 0.0

    @cython.cdivision(True)
    cdef double evaluate(self, double x, double y, double z) except? -1e999:

        cdef double v

        if self._cache:
            if x == self._cache_x and y == self._cache_y and z == self._cache_z:
                return self._cache_v

        v = self.peak * exp(-(x*x + y*y + z*z) / self._constant)
        self._cache = True
        self._cache_x = x
        self._cache_y = y
        self._cache_z = z
        self._cache_v = v
        return v
