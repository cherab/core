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

from cherab.core.math cimport Function2D


cdef class FreeFreeGauntFactor():

    cpdef double evaluate(self, double z, double temperature, double wavelength) except? -1e999


cdef class InterpolatedFreeFreeGauntFactor(FreeFreeGauntFactor):

    cdef:
        readonly tuple u_range, gamma2_range
        readonly dict raw_data
        double _u_min, _u_max, _gamma2_min, _gamma2_max
        Function2D _gaunt_factor


cdef class MaxwellianFreeFreeGauntFactor(InterpolatedFreeFreeGauntFactor):

    pass

