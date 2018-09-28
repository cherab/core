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

cimport numpy as np
from raysect.optical cimport Vector3D
from cherab.core.math cimport Function1D
from cherab.core.beam cimport BeamAttenuator


cdef class SingleRayAttenuator(BeamAttenuator):

    cdef readonly:
        Function1D _density
        list _stopping_data
        double _step, _clamp_sigma_sqr, _tanxdiv, _tanydiv, _source_density
        bint clamp_to_zero

    cpdef calculate_attenuation(self)

    cdef void _calc_attenuation(self)

    cdef np.ndarray _beam_attenuation(self, np.ndarray axis, np.ndarray x, np.ndarray y, np.ndarray z,
                                          double energy, double power, double mass, Vector3D direction)

    cdef double _beam_stopping(self, double x, double y, double z, Vector3D beam_velocity)

    cdef int _populate_stopping_data_cache(self) except -1
