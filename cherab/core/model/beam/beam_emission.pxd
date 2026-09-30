# Copyright 2016-2023 Euratom
# Copyright 2016-2023 United Kingdom Atomic Energy Authority
# Copyright 2016-2023 Centro de Investigaciones Energéticas, Medioambientales y Tecnológicas
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


from raysect.core.math cimport Vector3D
from cherab.core cimport Line
from cherab.core.beam cimport BeamModel
from cherab.core.model.lineshape cimport BeamLineShapeModel


cdef class BeamEmissionLine(BeamModel):

    cdef:
        Line _line
        double _wavelength
        list _rates_list
        BeamLineShapeModel _lineshape
        object _lineshape_class, _lineshape_args, _lineshape_kwargs

    cdef double _beam_emission_rate(self, double x, double y, double z, Vector3D beam_velocity) except? -1e999

    cdef int _populate_cache(self) except -1
