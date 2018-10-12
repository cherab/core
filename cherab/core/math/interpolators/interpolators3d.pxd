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

from cherab.core.math.function cimport Function3D
from numpy cimport ndarray, int8_t


cdef class _Interpolate3DBase(Function3D):

    cdef:
        double[::1] _x, _y, _z
        int _extrapolation_type
        double _extrapolation_range

    cdef object _build(self, ndarray x, ndarray y, ndarray z, ndarray f)

    cdef double evaluate(self, double px, double py, double pz) except? -1e999

    cdef double _evaluate(self, double px, double py, double pz, int ix, int iy, int iz) except? -1e999

    cdef double _extrapolate(self, double px, double py, double pz, int ix, int iy, int iz, double nearest_px, double nearest_py, double nearest_pz) except? -1e999

    cdef double _extrapol_linear(self, double px, double py, double pz, int ix, int iy, int iz, double nearest_px, double nearest_py, double nearest_pz) except? -1e999

    cdef double _extrapol_quadratic(self, double px, double py, double pz, int ix, int iy, int iz, double nearest_px, double nearest_py, double nearest_pz) except? -1e999


cdef class Interpolate3DLinear(_Interpolate3DBase):

    cdef:
        double[::1] _wx, _wy, _wz
        double[:,:,::1] _wf


cdef class Interpolate3DCubic(_Interpolate3DBase):

    cdef:
        double _sx, _sy, _sz, _sf
        double _ox, _oy, _oz, _of
        double[::1] _wx, _wx2, _wx3
        double[::1] _wy, _wy2, _wy3
        double[::1] _wz, _wz2, _wz3
        double[:,:,::1] _wf
        double[:,:,:,::1] _k
        int8_t[:,:,::1] _available

    cdef object _constraints3d(self, double[::1] c, int u, int v, int w, bint dx, bint dy, bint dz)

    cdef double _calc_polynomial_derivative(self, int ix, int iy, int iz, double px, double py, double pz, int der_x, int der_y, int der_z)




