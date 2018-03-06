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


cdef class AtomicData:
    """
    Atomic data source abstraction layer.

    This base class specifies a standardised set of methods for obtaining
    atomic data.
    """

    cpdef double wavelength(self, Line line):
        """
        Returns the natural wavelength of the specified transition in nm.
        """

        if line.wavelength > 0:
            return line.wavelength
        else:
            return self.lookup_wavelength(line.element, line.ionisation, line.transition)

    cpdef double lookup_wavelength(self, Element ion, int ionisation, tuple transition):
        """
        Looks up the natural wavelength of the specified transition in this atomic data source (nm).
        """

        raise NotImplementedError("The lookup_wavelength() virtual method is not implemented for this atomic data source.")

    cpdef list beam_cx_rate(self, Line line, Element donor_ion):
        """
        Returns a list of applicable charge exchange emission rates in W.m^3.
        """

        raise NotImplementedError("The cxs_rates() virtual method is not implemented for this atomic data source.")

    cpdef BeamStoppingRate beam_stopping_rate(self, Element beam_ion, Element plasma_ion, int ionisation):
        """
        Returns a list of applicable beam stopping/emission coefficients in m^3/s.
        """

        raise NotImplementedError("The beam_stopping() virtual method is not implemented for this atomic data source.")

    cpdef BeamPopulationRate beam_population_rate(self, Element beam_ion, int metastable, Element plasma_ion, int ionisation):
        """
        Returns a list of applicable beam stopping/emission coefficients in m^3/s.
        """

        raise NotImplementedError("The beam_population() virtual method is not implemented for this atomic data source.")

    cpdef BeamEmissionRate beam_emission_rate(self, Element beam_ion, Element plasma_ion, int ionisation, tuple transition):
        """
        Returns a list of applicable beam stopping/emission coefficients in m^3/s.
        """

        raise NotImplementedError("The beam_emission() virtual method is not implemented for this atomic data source.")

    cpdef ImpactExcitationRate impact_excitation_rate(self, Element ion, int ionisation, tuple transition):
        raise NotImplementedError("The impact_excitation() virtual method is not implemented for this atomic data source.")

    cpdef RecombinationRate recombination_rate(self, Element ion, int ionisation, tuple transition):
        raise NotImplementedError("The recombination() virtual method is not implemented for this atomic data source.")

    cpdef RadiatedPower radiated_power_rate(self, Element element, str radiation_type):
        raise NotImplementedError("The radiated_power() virtual method is not implemented for this atomic data source.")

    cpdef StageResolvedLineRadiation stage_resolved_line_radiation_rate(self, Element ion, int ionisation):
        raise NotImplementedError("The stage_resolved_line_radiation() virtual method is not implemented for this atomic data source.")

    cpdef FractionalAbundance fractional_abundance(self, Element ion, int ionisation):
        raise NotImplementedError("The fractional_abundance() virtual method is not implemented for this atomic data source.")

