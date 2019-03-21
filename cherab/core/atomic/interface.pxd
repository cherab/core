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

    cpdef double wavelength(self, Element ion, int charge, tuple transition)

    cpdef IonisationRate ionisation_rate(self, Element ion, int charge)

    cpdef RecombinationRate recombination_rate(self, Element ion, int charge)

    cpdef ThermalCXRate thermal_cx_rate(self, Element donor_ion, int donor_charge, Element receiver_ion, int receiver_charge)

    cpdef list beam_cx_pec(self, Element donor_ion, Element receiver_ion, int receiver_charge, tuple transition)

    cpdef BeamStoppingRate beam_stopping_rate(self, Element beam_ion, Element plasma_ion, int charge)

    cpdef BeamPopulationRate beam_population_rate(self, Element beam_ion, int metastable, Element plasma_ion, int charge)

    cpdef BeamEmissionPEC beam_emission_pec(self, Element beam_ion, Element plasma_ion, int charge, tuple transition)

    cpdef ImpactExcitationPEC impact_excitation_pec(self, Element ion, int charge, tuple transition)

    cpdef RecombinationPEC recombination_pec(self, Element ion, int charge, tuple transition)

    cpdef TotalRadiatedPower total_radiated_power(self, Element element)

    cpdef LineRadiationPower line_radiated_power_rate(self, Element element, int charge)

    cpdef ContinuumPower continuum_radiated_power_rate(self, Element element, int charge)

    cpdef CXRadiationPower cx_radiated_power_rate(self, Element element, int charge)

    cpdef FractionalAbundance fractional_abundance(self, Element ion, int charge)
