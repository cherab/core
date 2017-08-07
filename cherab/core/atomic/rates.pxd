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

cdef class _PECRate:
    cpdef double evaluate(self, double density, double temperature)


cdef class ImpactExcitationRate(_PECRate):
    pass


cdef class RecombinationRate(_PECRate):
    pass


cdef class ThermalCXRate(_PECRate):
    pass


cdef class BeamCXRate:
    cpdef double evaluate(self, double energy, double temperature, double density, double z_effective, double b_field) except? -1e999


cdef class _BeamRate:
    cpdef double evaluate(self, double energy, double density, double temperature) except? -1e999


cdef class BeamStoppingRate(_BeamRate):
    pass


cdef class BeamPopulationRate(_BeamRate):
    pass


cdef class BeamEmissionRate(_BeamRate):
    pass