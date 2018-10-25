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

from numpy cimport ndarray
from cherab.core.math.function cimport Function1D


cdef class _Interpolate1DBase(Function1D):

    cdef:
        double[::1] _x, _f
        bint _constant
        int _extrapolation_type
        double _extrapolation_range

    cdef object _build(self, ndarray x, ndarray f)

    cdef double evaluate(self, double px) except? -1e999

    cpdef double derivative(self, double px, int order) except? -1e999

    cdef double _evaluate(self, double px, int order, int index) except? -1e999

    cdef double _extrapolate(self, double px, int order, int index, double rx) except? -1e999

    cdef double _extrapol_linear(self, double px, int order, int index, double rx) except? -1e999

    cdef double _extrapol_quadratic(self, double px, int order, int index, double rx) except? -1e999


cdef class Interpolate1DLinear(_Interpolate1DBase):
    pass


cdef class Interpolate1DCubic(_Interpolate1DBase):

    cdef:
        int _continuity_order
        double _ox, _sx, _of, _sf
        double[:,::1] _k

    cdef double _calc_polynomial_derivative(self, int i, double p, int order)
