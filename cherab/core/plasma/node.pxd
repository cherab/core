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

from raysect.optical cimport Node, Primitive, AffineMatrix3D
from raysect.optical.material.emitter.inhomogeneous cimport VolumeIntegrator

from cherab.core.atomic cimport AtomicData, Element
from cherab.core.distribution cimport DistributionFunction
from cherab.core.species cimport Species
from cherab.core.math cimport VectorFunction3D
from cherab.core.plasma.model cimport PlasmaModel


cdef class Composition:

    cdef:
        dict _species
        readonly object notifier

    cpdef object set(self, object species)

    cpdef object add(self, Species species)

    cpdef Species get(self, Element element, int charge)

    cpdef object clear(self)


cdef class ModelManager:

    cdef:
        list _models
        readonly object notifier

    cpdef object set(self, object models)

    cpdef object add(self, PlasmaModel model)

    cpdef object clear(self)


cdef class Plasma(Node):

    cdef:

        readonly object notifier
        VectorFunction3D _b_field
        DistributionFunction _electron_distribution
        Composition _composition
        AtomicData _atomic_data
        Primitive _geometry
        AffineMatrix3D _geometry_transform
        ModelManager _models
        VolumeIntegrator _integrator

    cdef object __weakref__

    cdef VectorFunction3D get_b_field(self)

    cdef DistributionFunction get_electron_distribution(self)

    cdef Composition get_composition(self)

    cpdef double z_effective(self, double x, double y, double z) except -1

    cpdef double ion_density(self, double x, double y, double z)



