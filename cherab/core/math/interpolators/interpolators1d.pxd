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

    cdef readonly:
        ndarray x, data
        double[::1] x_mv, data_mv
        bint extrapolate
        bint constant
        int extrapolation_type
        double extrapolation_range
        int top_index

    cdef double evaluate(self, double px) except? -1e999

    cdef double _evaluate(self, double px, int index) except? -1e999

    cdef double _extrapolate(self, double px, int index, double nearest_px) except? -1e999

    cdef double _extrapol_linear(self, double px, int index, double nearest_px) except? -1e999

    cdef double _extrapol_quadratic(self, double px, int index, double nearest_px) except? -1e999


cdef class Interpolate1DLinear(_Interpolate1DBase):
    pass


cdef class Interpolate1DCubic(_Interpolate1DBase):

    cdef readonly:
        double x_min, x_delta_inv, data_min, data_delta

    cdef readonly double[:,:] coeffs_view

    cdef object _calc_coefficients(self, int continuity_order)

    cdef double _evaluate_polynomial_derivative(self, int i_x, double px, int der_x)
