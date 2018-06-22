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

from raysect.optical cimport Spectrum, Point3D, Vector3D

from cherab.core.plasma.node cimport Plasma
from cherab.core.beam.node cimport Beam
from cherab.core.atomic cimport AtomicData


cdef class BeamModel:

    cdef:
        Plasma _plasma
        Beam _beam
        AtomicData _atomic_data

    cdef object __weakref__

    cpdef Spectrum emission(self, Point3D beam_point, Point3D plasma_point, Vector3D beam_direction, Vector3D observation_direction, Spectrum spectrum)


cdef class BeamAttenuator:

    cdef:
        readonly object notifier
        Plasma _plasma
        Beam _beam
        AtomicData _atomic_data

    cdef object __weakref__

    cpdef double density(self, double x, double y, double z) except? -1e999