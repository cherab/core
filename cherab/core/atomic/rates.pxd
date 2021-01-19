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

from cherab.core.atomic.elements cimport Element


cdef class IonisationRate:

    cpdef double evaluate(self, double density, double temperature) except? -1e999


cdef class RecombinationRate:

    cpdef double evaluate(self, double density, double temperature) except? -1e999


cdef class ThermalCXRate:

    cpdef double evaluate(self, double density, double temperature) except? -1e999


cdef class _PECRate:
    cpdef double evaluate(self, double density, double temperature) except? -1e999


cdef class ImpactExcitationPEC(_PECRate):
    pass


cdef class RecombinationPEC(_PECRate):
    pass


cdef class ThermalCXPEC(_PECRate):
    pass


cdef class BeamCXPEC:
    cpdef double evaluate(self, double energy, double temperature, double density, double z_effective, double b_field) except? -1e999


cdef class _BeamRate:
    cpdef double evaluate(self, double energy, double density, double temperature) except? -1e999


cdef class BeamStoppingRate(_BeamRate):
    pass


cdef class BeamPopulationRate(_BeamRate):
    pass


cdef class BeamEmissionPEC(_BeamRate):
    pass


cdef class TotalRadiatedPower:

    cdef:
        readonly Element element

    cdef double evaluate(self, double electron_density, double electron_temperature) except? -1e999


cdef class _RadiatedPower:

    cdef:
        readonly Element element
        readonly int charge

    cdef double evaluate(self, double electron_density, double electron_temperature) except? -1e999


cdef class LineRadiationPower(_RadiatedPower):
    pass


cdef class ContinuumPower(_RadiatedPower):
    pass


cdef class CXRadiationPower(_RadiatedPower):
    pass


cdef class FractionalAbundance:

    cdef:
        readonly Element element
        readonly int charge
        public str name

    cdef double evaluate(self, double electron_density, double electron_temperature) except? -1e999
