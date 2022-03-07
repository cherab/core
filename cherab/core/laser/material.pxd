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


from raysect.core.scenegraph._nodebase cimport _NodeBase
from raysect.core.math cimport AffineMatrix3D
from raysect.optical.material.emitter cimport InhomogeneousVolumeEmitter

from cherab.core.laser.node cimport Laser


cdef class LaserMaterial(InhomogeneousVolumeEmitter):

    cdef:
        AffineMatrix3D _laser_to_plasma, _laser_segment_to_laser_node
        list _models
