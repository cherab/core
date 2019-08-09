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

from raysect.optical cimport Point3D, Vector3D, Node, Spectrum, Primitive
from raysect.optical.material.emitter.inhomogeneous cimport VolumeIntegrator

from raysect.core cimport translate, rotate_x

from cherab.core.atomic cimport AtomicData, Element
from cherab.core.plasma cimport Plasma
from cherab.core.beam.model cimport BeamAttenuator
from cherab.core.beam.model cimport BeamModel


cdef class ModelManager:

    cdef:
        list _models
        readonly object notifier

    cpdef object set(self, object models)

    cpdef object add(self, BeamModel model)

    cpdef object clear(self)


cdef class Beam(Node):

    cdef:
        object notifier
        Vector3D BEAM_AXIS
        double _energy, _power, _temperature
        Element _element
        double _divergence_x, _divergence_y
        double _length, _sigma
        Plasma _plasma
        AtomicData _atomic_data
        ModelManager _models
        BeamAttenuator _attenuator
        Primitive _geometry
        VolumeIntegrator _integrator

    cdef object __weakref__

    cpdef double density(self, double x, double y, double z) except? -1e999

    cpdef Vector3D direction(self, double x, double y, double z)

    cdef double get_energy(self)

    cdef double get_power(self)

    cdef double get_temperature(self)

    cdef Element get_element(self)

    cdef double get_divergence_x(self)

    cdef double get_divergence_y(self)

    cdef double get_length(self)

    cdef double get_sigma(self)

    cdef Plasma get_plasma(self)

    cdef int _modified(self) except -1
