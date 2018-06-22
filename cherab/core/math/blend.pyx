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

from cherab.core.math.function cimport autowrap_function1d, autowrap_function2d, autowrap_function3d
from raysect.core.math.cython cimport clamp


cdef class Blend1D(Function1D):
    """
    Performs a linear interpolation between two 1D functions controlled by a third.

    This function is a functional equivalent to the computer graphics Lerp
    function. The value returned when this function is evaluated is a linear
    interpolation between the sampled values of f1 and f2. The interpolation
    weighting is supplied by a third 'mask' function that must return a value
    in the range [0, 1].

    Mathematically the value returned by this function is as follows:

    .. math::
        r = (1 - f_m(x)) f_1(x) + f_m(x) f_2(x)

    The value of the mask function is clamped to the range [0, 1] if the
    sampled value exceeds the required range.
    """

    def __init__(self, object f1, object f2, object mask):
        self._f1 = autowrap_function1d(f1)
        self._f2 = autowrap_function1d(f2)
        self._mask = autowrap_function1d(mask)

    cdef double evaluate(self, double x) except? -1e999:
        cdef w = clamp(self._mask.evaluate(x), 0.0, 1.0)

        # only evaluate single function is at end of mask range
        if w == 0:
            return self._f1.evaluate(x)

        if w == 1:
            return self._f2.evaluate(x)

        # perform lerp
        cdef f1 = self._f1.evaluate(x)
        cdef f2 = self._f2.evaluate(x)
        return (1 - w) * f1 + w * f2


cdef class Blend2D(Function2D):
    """
    Performs a linear interpolation between two 2D functions controlled by a third.

    This function is a functional equivalent to the computer graphics Lerp
    function. The value returned when this function is evaluated is a linear
    interpolation between the sampled values of f1 and f2. The interpolation
    weighting is supplied by a third 'mask' function that must return a value
    in the range [0, 1].

    Mathematically the value returned by this function is as follows:

    .. math::
        r = (1 - f_m(x,y)) f_1(x,y) + f_m(x,y) f_2(x,y)

    The value of the mask function is clamped to the range [0, 1] if the
    sampled value exceeds the required range.
    """

    def __init__(self, object f1, object f2, object mask):
        self._f1 = autowrap_function2d(f1)
        self._f2 = autowrap_function2d(f2)
        self._mask = autowrap_function2d(mask)

    cdef double evaluate(self, double x, double y) except? -1e999:
        cdef w = clamp(self._mask.evaluate(x, y), 0.0, 1.0)

        # only evaluate single function is at end of mask range
        if w == 0:
            return self._f1.evaluate(x, y)

        if w == 1:
            return self._f2.evaluate(x, y)

        # perform lerp
        cdef f1 = self._f1.evaluate(x, y)
        cdef f2 = self._f2.evaluate(x, y)
        return (1 - w) * f1 + w * f2


cdef class Blend3D(Function3D):
    """
    Performs a linear interpolation between two 3D functions controlled by a third.

    This function is a functional equivalent to the computer graphics Lerp
    function. The value returned when this function is evaluated is a linear
    interpolation between the sampled values of f1 and f2. The interpolation
    weighting is supplied by a third 'mask' function that must return a value
    in the range [0, 1].

    Mathematically the value returned by this function is as follows:

    .. math::
        r = (1 - f_m(x,y,z)) f_1(x,y,z) + f_m(x,y,z) f_2(x,y,z)

    The value of the mask function is clamped to the range [0, 1] if the
    sampled value exceeds the required range.
    """

    def __init__(self, object f1, object f2, object mask):
        self._f1 = autowrap_function3d(f1)
        self._f2 = autowrap_function3d(f2)
        self._mask = autowrap_function3d(mask)

    cdef double evaluate(self, double x, double y, double z) except? -1e999:
        cdef w = clamp(self._mask.evaluate(x, y, z), 0.0, 1.0)

        # only evaluate single function is at end of mask range
        if w == 0:
            return self._f1.evaluate(x, y, z)

        if w == 1:
            return self._f2.evaluate(x, y, z)

        # perform lerp
        cdef f1 = self._f1.evaluate(x, y, z)
        cdef f2 = self._f2.evaluate(x, y, z)
        return (1 - w) * f1 + w * f2