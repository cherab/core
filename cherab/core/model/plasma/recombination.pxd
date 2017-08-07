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

from cherab.core.atomic cimport Line, RecombinationRate
from cherab.core.plasma cimport PlasmaModel
from cherab.core.species cimport Species


cdef class RecombinationLine(PlasmaModel):

    cdef:
        Line _line
        double _wavelength
        Species _target_species
        RecombinationRate _rates

    # cpdef double radiance_at(self, Point3D point, Vector3D direction)

    cdef inline int _populate_cache(self) except -1

