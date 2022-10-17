# Copyright 2016-2022 Euratom
# Copyright 2016-2022 United Kingdom Atomic Energy Authority
# Copyright 2016-2022 Centro de Investigaciones Energéticas, Medioambientales y Tecnológicas
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
from raysect.core.math.function.float cimport Function1D, Function2D


cdef class Integrator1D:

    cdef:
        Function1D function

    cdef double evaluate(self, double a, double b) except? -1e999


cdef class GaussianQuadrature(Integrator1D):

    cdef:
        int _min_order, _max_order
        double _rtol
        ndarray _roots, _weights
        double[:] _roots_mv, _weights_mv

    cdef _build_cache(self)
