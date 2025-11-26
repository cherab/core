# cython: language_level=3

# Copyright 2016-2025 Euratom
# Copyright 2016-2025 United Kingdom Atomic Energy Authority
# Copyright 2016-2025 Centro de Investigaciones Energéticas, Medioambientales y Tecnológicas
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

from cherab.core.math.function.float.function6d.autowrap cimport autowrap_function6d
from raysect.core.math.cython cimport clamp


cdef class Blend6D(Function6D):
    """
    Performs a linear interpolation between two scalar functions, modulated by a 3rd scalar function.

    The value of the scalar mask function is used to interpolated between the
    values returned by the two functions. Mathematically the value returned by
    this function is as follows:

    .. math::
        v = (1 - f_m(x, y, z, u, w, v)) f_1(x, y, z, u, w, v) + f_m(x, y, z, u, w, v) f_2(x, y, z, u, w, v)

    The value of the mask function is clamped to the range [0, 1] if the sampled
    value exceeds the required range.
    """

    def __init__(self, object f1, object f2, object mask):
        """
        :param float.Function6D f1: First scalar function.
        :param float.Function6D f2: Second scalar function.
        :param float.Function6D mask: Scalar function returning a value in the range [0, 1].
        """

        self._f1 = autowrap_function6d(f1)
        self._f2 = autowrap_function6d(f2)
        self._mask = autowrap_function6d(mask)

    cdef double evaluate(self, double x, double y, double z, double u, double w, double v) except? -1e999:

        cdef double t = clamp(self._mask.evaluate(x, y, z, u, w, v), 0.0, 1.0)

        # sample endpoints directly
        if t == 0:
            return self._f1.evaluate(x, y, z, u, w, v)

        if t == 1:
            return self._f2.evaluate(x, y, z, u, w, v)

        # lerp between function values
        cdef double f1 = self._f1.evaluate(x, y, z, u, w, v)
        cdef double f2 = self._f2.evaluate(x, y, z, u, w, v)
        return (1 - t) * f1 + t * f2
