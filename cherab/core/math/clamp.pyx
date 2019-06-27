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

from libc.math cimport INFINITY

from cherab.core.math.function cimport autowrap_function1d, autowrap_function2d, autowrap_function3d
from raysect.core.math.cython cimport clamp


# todo: add docstrings

cdef class ClampOutput1D(Function1D):
    """

    """

    def __init__(self, object f, double min=-INFINITY, double max=INFINITY):

        if min >= max:
            raise ValueError('The minimum clamp value must be less than the maximum.')

        self._f = autowrap_function1d(f)
        self._min = min
        self._max = max

    cdef double evaluate(self, double x) except? -1e999:
        return clamp(self._f(x), self._min, self._max)


cdef class ClampOutput2D(Function2D):
    """

    """

    def __init__(self, object f, double min=-INFINITY, double max=INFINITY):

        if min >= max:
            raise ValueError('The minimum clamp value must be less than the maximum.')

        self._f = autowrap_function2d(f)
        self._min = min
        self._max = max

    cdef double evaluate(self, double x, double y) except? -1e999:
        return clamp(self._f(x, y), self._min, self._max)


cdef class ClampOutput3D(Function3D):
    """

    """

    def __init__(self, object f, double min=-INFINITY, double max=INFINITY):

        if min >= max:
            raise ValueError('The minimum clamp value must be less than the maximum.')

        self._f = autowrap_function3d(f)
        self._min = min
        self._max = max

    cdef double evaluate(self, double x, double y, double z) except? -1e999:
        return clamp(self._f(x, y, z), self._min, self._max)


cdef class ClampInput1D(Function1D):
    """

    """

    def __init__(self, object f, double xmin=-INFINITY, double xmax=INFINITY):

        if xmin >= xmax:
            raise ValueError('The minimum clamp value must be less than the maximum.')

        self._f = autowrap_function1d(f)
        self._xmin = xmin
        self._xmax = xmax

    cdef double evaluate(self, double x) except? -1e999:
        x = clamp(x, self._xmin, self._xmax)
        return self._f(x)


cdef class ClampInput2D(Function2D):
    """

    """

    def __init__(self, object f, double xmin=-INFINITY, double xmax=INFINITY, double ymin=-INFINITY, double ymax=INFINITY):

        if xmin >= xmax:
            raise ValueError('The minimum x clamp value must be less than the maximum.')

        if ymin >= ymax:
            raise ValueError('The minimum y clamp value must be less than the maximum.')

        self._f = autowrap_function2d(f)
        self._xmin = xmin
        self._xmax = xmax
        self._ymin = ymin
        self._ymax = ymax

    cdef double evaluate(self, double x, double y) except? -1e999:
        x = clamp(x, self._xmin, self._xmax)
        y = clamp(y, self._ymin, self._ymax)
        return self._f(x, y)


cdef class ClampInput3D(Function3D):
    """

    """

    def __init__(self, object f, double xmin=-INFINITY, double xmax=INFINITY, double ymin=-INFINITY, double ymax=INFINITY, double zmin=-INFINITY, double zmax=INFINITY):

        if xmin >= xmax:
            raise ValueError('The minimum x clamp value must be less than the maximum.')

        if ymin >= ymax:
            raise ValueError('The minimum y clamp value must be less than the maximum.')

        if zmin >= zmax:
            raise ValueError('The minimum z clamp value must be less than the maximum.')

        self._f = autowrap_function3d(f)
        self._xmin = xmin
        self._xmax = xmax
        self._ymin = ymin
        self._ymax = ymax
        self._zmin = zmin
        self._zmax = zmax

    cdef double evaluate(self, double x, double y, double z) except? -1e999:
        x = clamp(x, self._xmin, self._xmax)
        y = clamp(y, self._ymin, self._ymax)
        z = clamp(z, self._zmin, self._zmax)
        return self._f(x, y, z)

