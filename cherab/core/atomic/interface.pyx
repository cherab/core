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


cdef class AtomicData:
    """
    Atomic data source abstraction layer.

    This base class specifies a standardised set of methods for obtaining
    atomic data.
    """

    cpdef double wavelength(self, Element ion, int charge, tuple transition):
        """
        Returns the natural wavelength of the specified transition in nm.
        """

        raise NotImplementedError("The wavelength() virtual method is not implemented for this atomic data source.")

    cpdef IonisationRate ionisation_rate(self, Element ion, int charge):
        raise NotImplementedError("The ionisation_rate() virtual method is not implemented for this atomic data source.")

    cpdef RecombinationRate recombination_rate(self, Element ion, int charge):
        raise NotImplementedError("The recombination_rate() virtual method is not implemented for this atomic data source.")

    cpdef ThermalCXRate thermal_cx_rate(self, Element donor_ion, int donor_charge, Element receiver_ion, int receiver_charge):
        raise NotImplementedError("The thermal_cx_rate() virtual method is not implemented for this atomic data source.")

    cpdef list beam_cx_pec(self, Element donor_ion, Element receiver_ion, int receiver_charge, tuple transition):
        """
        Returns a list of applicable charge exchange emission rates in W.m^3.
        """

        raise NotImplementedError("The cxs_rates() virtual method is not implemented for this atomic data source.")

    cpdef BeamStoppingRate beam_stopping_rate(self, Element beam_ion, Element plasma_ion, int charge):
        """
        Returns a list of applicable beam stopping coefficients in m^3/s.
        """

        raise NotImplementedError("The beam_stopping() virtual method is not implemented for this atomic data source.")

    cpdef BeamPopulationRate beam_population_rate(self, Element beam_ion, int metastable, Element plasma_ion, int charge):
        """
        Returns a list of applicable dimensionless beam population coefficients.
        """

        raise NotImplementedError("The beam_population() virtual method is not implemented for this atomic data source.")

    cpdef BeamEmissionPEC beam_emission_pec(self, Element beam_ion, Element plasma_ion, int charge, tuple transition):
        """
        Returns a list of applicable beam emission coefficients in W.m^3.
        """

        raise NotImplementedError("The beam_emission() virtual method is not implemented for this atomic data source.")

    cpdef ImpactExcitationPEC impact_excitation_pec(self, Element ion, int charge, tuple transition):
        raise NotImplementedError("The impact_excitation() virtual method is not implemented for this atomic data source.")

    cpdef RecombinationPEC recombination_pec(self, Element ion, int charge, tuple transition):
        raise NotImplementedError("The recombination() virtual method is not implemented for this atomic data source.")

    cpdef TotalRadiatedPower total_radiated_power(self, Element element):
        raise NotImplementedError("The total_radiated_power() virtual method is not implemented for this atomic data source.")

    cpdef LineRadiationPower line_radiated_power_rate(self, Element element, int charge):
        raise NotImplementedError("The line_radiated_power_rate() virtual method is not implemented for this atomic data source.")

    cpdef ContinuumPower continuum_radiated_power_rate(self, Element element, int charge):
        raise NotImplementedError("The continuum_radiated_power_rate() virtual method is not implemented for this atomic data source.")

    cpdef CXRadiationPower cx_radiated_power_rate(self, Element element, int charge):
        raise NotImplementedError("The cx_radiated_power_rate() virtual method is not implemented for this atomic data source.")

    cpdef FractionalAbundance fractional_abundance(self, Element ion, int charge):
        raise NotImplementedError("The fractional_abundance() virtual method is not implemented for this atomic data source.")

