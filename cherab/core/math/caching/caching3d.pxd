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

cdef class Caching3D(Function3D):

    cdef readonly:
        Function3D function
        int no_boundary_error
        ndarray x_np, y_np, z_np
        double[::1] x_domain_view, y_domain_view, z_domain_view
        int top_index_x, top_index_y, top_index_z
        double x_min, x_delta_inv, y_min, y_delta_inv, z_min, z_delta_inv
        double data_min, data_max, data_delta, data_delta_inv
        double[::1] x_view, x2_view, x3_view
        double[::1] y_view, y2_view, y3_view
        double[::1] z_view, z2_view, z3_view
        double[:,:,::1] data_view
        double[:,:,:,::1] coeffs_view
        int8_t[:,:,::1] calculated_view

    cdef double evaluate(self, double px, double py, double pz) except? -1e999

    cdef double _evaluate(self, double px, double py, double pz, int i_x, int i_y, int i_z)

    cdef double _evaluate_polynomial_derivative(self, int i_x, int i_y, int i_z, double px, double py, double pz, int der_x, int der_y, der_z)

    cdef double[::1] _constraints3d(self, int u, int v, int w, bint x_der, bint y_der, bint z_der)