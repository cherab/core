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

from raysect.optical cimport Vector3D

from cherab.core.math cimport Function3D, VectorFunction3D


cdef class DistributionFunction:

    cdef object notifier

    cdef double evaluate(self, double x, double y, double z, double vx, double vy, double vz) except? -1e999

    cpdef Vector3D bulk_velocity(self, double x, double y, double z)

    cpdef double effective_temperature(self, double x, double y, double z) except? -1e999

    cpdef double density(self, double x, double y, double z) except? -1e999


cdef class ZeroDistribution(DistributionFunction):
    pass


cdef class Maxwellian(DistributionFunction):

    cdef readonly:
        Function3D _density, _temperature
        VectorFunction3D _velocity
        double _atomic_mass

