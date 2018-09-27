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

from cherab.core.math.function cimport Function1D
from numpy cimport ndarray, int8_t

cdef class Caching1D(Function1D):

    cdef readonly:
        Function1D function
        int no_boundary_error
        ndarray x_np
        double[::1] x_domain_view
        int top_index_x
        double x_min, x_delta_inv
        double data_min, data_max, data_delta, data_delta_inv
        double[::1] x_view, x2_view, x3_view
        double[::1] data_view
        double[:,::1] coeffs_view
        int8_t[::1] calculated_view

    cdef double evaluate(self, double px) except? -1e999

    cdef double _evaluate(self, double px, int i_x)

    cdef double _evaluate_polynomial_derivative(self, int i_x, double px, int der_x)
