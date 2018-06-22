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

from cherab.core.atomic cimport Element
from cherab.core.atomic.rates cimport *


cdef class AtomicData:

    cpdef double wavelength(self, Element ion, int ionisation, tuple transition)

    cpdef list beam_cx_rate(self, Element donor_ion, Element receiver_ion, int receiver_ionisation, tuple transition)

    cpdef BeamStoppingRate beam_stopping_rate(self, Element beam_ion, Element plasma_ion, int ionisation)

    cpdef BeamPopulationRate beam_population_rate(self, Element beam_ion, int metastable, Element plasma_ion, int ionisation)

    cpdef BeamEmissionRate beam_emission_rate(self, Element beam_ion, Element plasma_ion, int ionisation, tuple transition)

    cpdef ImpactExcitationRate impact_excitation_rate(self, Element ion, int ionisation, tuple transition)

    cpdef RecombinationRate recombination_rate(self, Element ion, int ionisation, tuple transition)

    cpdef RadiatedPower radiated_power_rate(self, Element element, str radiation_type)

    cpdef StageResolvedLineRadiation stage_resolved_line_radiation_rate(self, Element ion, int ionisation)

    cpdef FractionalAbundance fractional_abundance(self, Element ion, int ionisation)
