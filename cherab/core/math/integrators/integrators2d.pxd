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

from raysect.core.math.function.float cimport Function1D, Function2D


cdef class Integrator2D:

    cdef:
        Function2D function

    cdef double evaluate(self,double x_lower, double x_upper, Function1D y_lower, Function1D y_upper) except? -1e999


cdef class GaussianQuadrature2D(Integrator2D):

    cdef:
        int _x_min_order, _x_max_order, _y_min_order, _y_max_order
        double _rtol
        object _x_roots, _x_weights, _y_roots, _y_weights
        double[:] _x_roots_mv, _x_weights_mv, _y_roots_mv, _y_weights_mv

    cdef _build_cache(self)
    
    cpdef double profile_evaluate(self, int n, double x_lower, double x_upper, Function1D y_lower, Function1D y_upper)

    cpdef double evaluate_overhead(self, int n)