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

from .gaunt import MaxwellianFreeFreeGauntFactor
from .stark import InterpolatedStarkStructure
from .elements import Isotope
import json
from os import path


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

    cpdef ZeemanStructure zeeman_structure(self, Line line, object b_field=None):
        raise NotImplementedError("The zeeman_structure() virtual method is not implemented for this atomic data source.")

    cpdef tuple zeeman_triplet_parameters(self, Line line):
        """
        Returns Zeeman truplet parameters. See Table 1 in A. Blom and C. Jupén.
        "Parametrisation of the Zeeman effect for hydrogen-like spectra in
        high-temperature plasmas", Plasma Phys. Control. Fusion 44 (2002) `1229-1241
        <https://doi.org/10.1088/0741-3335/44/7/312>`_.
        """

        symbol = line.element.symbol.lower()
        upper, lower = line.transition
        encoded_transition = '{} -> {}'.format(str(upper).lower(), str(lower).lower())

        try:
            with open(path.join(path.dirname(__file__), "data/lineshape/zeeman/parametrised/{}.json".format(symbol))) as f:
                data = json.load(f)
            coefficients = data[str(line.charge)][encoded_transition]
        except (FileNotFoundError, KeyError):
            raise RuntimeError('Requested Zeeman triplet parameters (element={}, charge={}, transition={})'
                               ' are not available.'.format(line.element.symbol, line.charge, line.transition))

        return tuple(coefficients)

    cpdef tuple stark_model_coefficients(self, Line line):
        """
        Returns Stark model coefficients. See Table 1 in B. Lomanowski, et al.
        "Inferring divertor plasma properties from hydrogen Balmer
        and Paschen series spectroscopy in JET-ILW." Nuclear Fusion 55.12 (2015)
        `123028 <https://doi.org/10.1088/0029-5515/55/12/123028>`_.
        """

        symbol = line.element.symbol.lower()
        upper, lower = line.transition
        encoded_transition = '{} -> {}'.format(str(upper).lower(), str(lower).lower())

        try:
            with open(path.join(path.dirname(__file__), "data/lineshape/stark/{}.json".format(symbol))) as f:
                data = json.load(f)
            coefficients = data[str(line.charge)][encoded_transition]
        except (FileNotFoundError, KeyError):
            raise RuntimeError('Requested Stark model coefficients (element={}, charge={}, transition={})'
                               ' are not available.'.format(line.element.symbol, line.charge, line.transition))

        return tuple(coefficients)

    cpdef StarkStructure stark_structure(self, Line line):
        """
        Returns interpolated ratios of linear Stark components of the MSE spectrum when
        the observation direction is perependicular to the electric field.

        Use data from A.V. Demura, D.S. Leontiev, V.S. Lisitsa. Polarization characteristics
        of electrodynamic Stark effect. Accepted to J. Exp. Theor. Phys. (2024)
        provided by Dmitry Leontiev.
        """

        element = line.element
        if isinstance(element, Isotope):
            element = element.element

        symbol = element.symbol.lower()
        upper, lower = line.transition
        encoded_transition = '{} -> {}'.format(str(upper).lower(), str(lower).lower())

        try:
            with open(path.join(path.dirname(__file__), "data/lineshape/mse/{}/{}.json".format(symbol, line.charge))) as f:
                data = json.load(f)
            data = data[encoded_transition]
        except (FileNotFoundError, KeyError):
            raise RuntimeError('Requested MSE Stark component ratios (element={}, charge={}, transition={})'
                               ' are not available.'.format(symbol, line.charge, line.transition))

        return InterpolatedStarkStructure(data, extrapolate=True)

    cpdef FreeFreeGauntFactor free_free_gaunt_factor(self):
        """
        Returns the Maxwellian-averaged free-free Gaunt factor interpolated over the data
        from Table A.1 in M.A. de Avillez and D. Breitschwerdt, 2015, Astron. & Astrophys. 580,
        `A124 <https://www.aanda.org/articles/aa/full_html/2015/08/aa26104-15/aa26104-15.html>`_.

        The Born approximation is used outside the interpolation range.
        """

        return MaxwellianFreeFreeGauntFactor()
