# Copyright 2014-2017 United Kingdom Atomic Energy Authority
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

    cdef readonly:
        ndarray x_np, data_np
        double[::1] x_domain_view
        bint extrapolate
        int extrapolation_type
        double extrapolation_range
        int top_index

    cdef double evaluate(self, double px) except? -1e999

    cdef double _evaluate(self, double px, int index)

    cdef inline double _extrapolate(self, double px, int index, double nearest_px)

    cdef double _extrapol_linear(self, double px, int index, double nearest_px)

    cdef double _extrapol_quadratic(self, double px, int index, double nearest_px)

    cdef void _set_constant(self)


cdef class Interpolate1DLinear(_Interpolate1DBase):

    cdef readonly double[::1] x_view, data_view


cdef class Interpolate1DCubic(_Interpolate1DBase):

    cdef readonly:
        double x_min, x_delta_inv, data_min, data_delta

    cdef readonly double[:,:] coeffs_view

    cdef inline double _evaluate_polynomial_derivative(self, int i_x, double px, int der_x)
