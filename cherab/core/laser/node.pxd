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


from raysect.optical cimport Point3D, Vector3D, Node, Spectrum, Primitive
from raysect.optical.material.emitter.inhomogeneous cimport VolumeIntegrator
from raysect.primitive cimport Cylinder

from cherab.core.plasma cimport Plasma
from cherab.core.laser.profile cimport LaserProfile
from cherab.core.laser.laserspectrum cimport LaserSpectrum
from cherab.core.laser.model cimport LaserModel


cdef class ModelManager:

    cdef:
        list _models
        readonly object notifier

    cpdef object set(self, object models)

    cpdef object add(self, LaserModel model)

    cpdef object clear(self)


cdef class Laser(Node):

    cdef:
        readonly object notifier
        double _importance
        Plasma _plasma
        ModelManager  _models
        LaserProfile _laser_profile
        LaserSpectrum _laser_spectrum
        list _geometry
        VolumeIntegrator _integrator
        
    cdef object __weakref__

    cdef Plasma get_plasma(self)