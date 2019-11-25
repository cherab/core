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

from cherab.core cimport BeamStoppingRate as CoreBeamStoppingRate
from cherab.core cimport BeamPopulationRate as CoreBeamPopulationRate
from cherab.core cimport BeamEmissionPEC as CoreBeamEmissionPEC
from cherab.core.math cimport Function1D, Function2D


cdef class BeamStoppingRate(CoreBeamStoppingRate):

    cdef readonly:
        dict raw_data
        Function2D _npl_eb
        Function1D _tp
        tuple beam_energy_range
        tuple density_range
        tuple temperature_range


cdef class NullBeamStoppingRate(CoreBeamStoppingRate):
    pass


cdef class BeamPopulationRate(CoreBeamPopulationRate):

    cdef readonly:
        dict raw_data
        Function2D _npl_eb
        Function1D _tp
        tuple beam_energy_range
        tuple density_range
        tuple temperature_range


cdef class NullBeamPopulationRate(CoreBeamPopulationRate):
    pass


cdef class BeamEmissionPEC(CoreBeamEmissionPEC):

    cdef readonly:
        dict raw_data
        Function2D _npl_eb
        Function1D _tp
        tuple beam_energy_range
        tuple density_range
        tuple temperature_range


cdef class NullBeamEmissionPEC(CoreBeamEmissionPEC):
    pass
