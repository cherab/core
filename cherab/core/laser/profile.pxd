# Copyright 2016-2021 Euratom
# Copyright 2016-2021 United Kingdom Atomic Energy Authority
# Copyright 2016-2021 Centro de Investigaciones Energéticas, Medioambientales y Tecnológicas
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


from raysect.core.math.function.float cimport Function3D
from raysect.core.math.function.vector3d cimport Function3D as VectorFunction3D

from raysect.optical cimport Spectrum, Point3D, Vector3D

from cherab.core.laser.node cimport Laser


cdef class LaserProfile:

    cdef:
        VectorFunction3D _polarization3d, _pointing3d
        Function3D _energy_density3d
        readonly object notifier

    cpdef Vector3D get_pointing(self, double x, double y, double z)

    cpdef Vector3D get_polarization(self, double x, double y, double z)

    cpdef double get_energy_density(self, double x, double y, double z)

    cpdef list generate_geometry(self)