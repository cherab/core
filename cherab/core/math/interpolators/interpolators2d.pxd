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

from cherab.core.math.function cimport Function2D
from numpy cimport ndarray, int8_t


cdef class _Interpolate2DBase(Function2D):

    cdef:
        double[::1] _x, _y
        int _extrapolation_type
        double _extrapolation_range

    cdef object _build(self, ndarray x, ndarray y, ndarray f)

    cdef double evaluate(self, double px, double py) except? -1e999

    cpdef double derivative(self, double px, double py, int order_x, int order_y) except? -1e999

    cdef double _dispatch(self, double px, double py, int order_x, int order_y) except? -1e999

    cdef double _evaluate(self, double px, double py, int order_x, int order_y, int ix, int iy) except? -1e999

    cdef double _extrapolate(self, double px, double py, int order_x, int order_y, int ix, int iy, double rx, double ry, bint inside_x, bint inside_y) except? -1e999

    cdef double _extrapol_nearest(self, double px, double py, int order_x, int order_y, int ix, int iy, double nearest_px, double nearest_py, bint inside_x, bint inside_y) except? -1e999

    cdef double _extrapol_linear(self, double px, double py, int order_x, int order_y, int ix, int iy, double rx, double ry, bint inside_x, bint inside_y) except? -1e999

    cdef double _extrapol_quadratic(self, double px, double py, int order_x, int order_y, int ix, int iy, double rx, double ry, bint inside_x, bint inside_y) except? -1e999


cdef class Interpolate2DLinear(_Interpolate2DBase):

    cdef:
        double[::1] _wx, _wy
        double[:,::1] _wf


cdef class Interpolate2DCubic(_Interpolate2DBase):

    cdef:
        double _sx, _sy, _sf
        double _ox, _oy, _of
        double[::1] _wx, _wx2, _wx3
        double[::1] _wy, _wy2, _wy3
        double[:,::1] _wf
        double[:,:,::1] _k
        int8_t[:,::1] _available

    cdef int _calc_polynomial(self, int ix, int iy) except -1

    cdef double _calc_polynomial_derivative(self, int ix, int iy, double px, double py, int order_x, int order_y)
