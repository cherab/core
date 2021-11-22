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

from cherab.core import AtomicData
from cherab.core.atomic.elements import Isotope
from cherab.openadas.repository import DEFAULT_REPOSITORY_PATH

from .rates import *
from cherab.openadas import repository


class OpenADAS(AtomicData):
    """
    OpenADAS atomic data source.

    :param str data_path: OpenADAS local repository path.
    :param bool permit_extrapolation: If true, informs interpolation objects to allow extrapolation
                                      beyond the limits of the tabulated data. Default is False.
    :param bool missing_rates_return_null: If true, allows Null rate objects to be returned when
                                           the requested atomic data is missing. Default is False.
    :param bool wavelength_element_fallback: If true, allows to use the element's wavelength when
                                             the isotope's wavelength is not available.
                                             Default is False.
    """

    def __init__(self, data_path=None, permit_extrapolation=False, missing_rates_return_null=False,
                 wavelength_element_fallback=False):

        super().__init__()
        self._data_path = data_path or DEFAULT_REPOSITORY_PATH

        self._permit_extrapolation = permit_extrapolation

        self._missing_rates_return_null = missing_rates_return_null

        self._wavelength_element_fallback = wavelength_element_fallback

    @property
    def data_path(self):
        return self._data_path

    def wavelength(self, ion, charge, transition):
        """
        :param ion: Element object defining the ion type.
        :param charge: Charge state of the ion.
        :param transition: Tuple containing (initial level, final level)
        :return: Wavelength in nanometers.
        """

        if isinstance(ion, Isotope) and self._wavelength_element_fallback:
            try:
                return repository.get_wavelength(ion, charge, transition, repository_path=self._data_path)
            except RuntimeError:
                return repository.get_wavelength(ion.element, charge, transition, repository_path=self._data_path)

        return repository.get_wavelength(ion, charge, transition, repository_path=self._data_path)

    def ionisation_rate(self, ion, charge):

        # extract element from isotope because there are no isotope rates in ADAS
        if isinstance(ion, Isotope):
            ion = ion.element

        try:
            # read ionisation rate from json file in the repository
            data = repository.get_ionisation_rate(ion, charge, repository_path=self._data_path)

        except RuntimeError:
            if self._missing_rates_return_null:
                return NullIonisationRate()
            raise

        return IonisationRate(data, extrapolate=self._permit_extrapolation)

    def recombination_rate(self, ion, charge):

        # extract element from isotope because there are no isotope rates in ADAS
        if isinstance(ion, Isotope):
            ion = ion.element

        try:
            # read recombination rate from json file in the repository
            data = repository.get_recombination_rate(ion, charge, repository_path=self._data_path)

        except RuntimeError:
            if self._missing_rates_return_null:
                return NullRecombinationRate()
            raise

        return RecombinationRate(data, extrapolate=self._permit_extrapolation)

    def thermal_cx_rate(self, donor_element, donor_charge, receiver_element, receiver_charge):

        # extract elements from isotopes because there are no isotope rates in ADAS
        if isinstance(donor_element, Isotope):
            donor_element = donor_element.element

        if isinstance(receiver_element, Isotope):
            receiver_element = receiver_element.element

        try:
            # read thermal CX rate from json file in the repository
            data = repository.get_thermal_cx_rate(donor_element, donor_charge,
                                                  receiver_element, receiver_charge,
                                                  repository_path=self._data_path)

        except RuntimeError:
            if self._missing_rates_return_null:
                return NullThermalCXRate()
            raise

        return ThermalCXRate(data, extrapolate=self._permit_extrapolation)

    def beam_cx_pec(self, donor_ion, receiver_ion, receiver_charge, transition):
        """

        :param donor_ion:
        :param receiver_ion:
        :param receiver_charge:
        :param transition:
        :return:
        """

        # extract element from donor isotope because there are no isotope rates in ADAS
        if isinstance(donor_ion, Isotope):
            donor_ion = donor_ion.element

        # extract element from receiver isotope, but keep the receiver isotope for the wavelength
        receiver_ion_element = receiver_ion.element if isinstance(receiver_ion, Isotope) else receiver_ion

        try:
            # read element CX rate from json file in the repository
            data = repository.get_beam_cx_rates(donor_ion, receiver_ion_element, receiver_charge, transition,
                                                repository_path=self._data_path)

        except RuntimeError:
            if self._missing_rates_return_null:
                return [NullBeamCXPEC()]
            raise

        # obtain isotope's rest wavelength for a given transition
        # the wavelength is used ot convert the PEC from photons/s/m3 to W/m3
        wavelength = self.wavelength(receiver_ion, receiver_charge - 1, transition)

        # load and interpolate the relevant transition data from each file
        rates = []
        for donor_metastable, rate_data in data:
            rates.append(BeamCXPEC(donor_metastable, wavelength, rate_data, extrapolate=self._permit_extrapolation))
        return rates

    def beam_stopping_rate(self, beam_ion, plasma_ion, charge):
        """

        :param beam_ion:
        :param plasma_ion:
        :param charge:
        :return:
        """

        # extract elements from isotopes because there are no isotope rates in ADAS
        if isinstance(beam_ion, Isotope):
            beam_ion = beam_ion.element

        if isinstance(plasma_ion, Isotope):
            plasma_ion = plasma_ion.element

        try:
            # read beam stopping rate from json file in the repository
            data = repository.get_beam_stopping_rate(beam_ion, plasma_ion, charge, repository_path=self._data_path)

        except RuntimeError:
            if self._missing_rates_return_null:
                return NullBeamStoppingRate()
            raise

        # load and interpolate data
        return BeamStoppingRate(data, extrapolate=self._permit_extrapolation)

    def beam_population_rate(self, beam_ion, metastable, plasma_ion, charge):
        """

        :param beam_ion:
        :param metastable:
        :param plasma_ion:
        :param charge:
        :return:
        """

        # extract elements from isotopes because there are no isotope rates in ADAS
        if isinstance(beam_ion, Isotope):
            beam_ion = beam_ion.element

        if isinstance(plasma_ion, Isotope):
            plasma_ion = plasma_ion.element

        try:
            # read beam population rate from json file in the repository
            data = repository.get_beam_population_rate(beam_ion, metastable, plasma_ion, charge,
                                                       repository_path=self._data_path)

        except RuntimeError:
            if self._missing_rates_return_null:
                return NullBeamPopulationRate()
            raise

        # load and interpolate data
        return BeamPopulationRate(data, extrapolate=self._permit_extrapolation)

    def beam_emission_pec(self, beam_ion, plasma_ion, charge, transition):
        """

        :param beam_ion:
        :param plasma_ion:
        :param charge:
        :param transition:
        :return:
        """

        # extract element from beam isotope, but keep the beam isotope for the wavelength
        beam_ion_element = beam_ion.element if isinstance(beam_ion, Isotope) else beam_ion

        # extract element from plasma isotope because there are no isotope rates in ADAS
        if isinstance(plasma_ion, Isotope):
            plasma_ion = plasma_ion.element

        try:
            # read beam emission PEC from json file in the repository
            data = repository.get_beam_emission_rate(beam_ion_element, plasma_ion, charge, transition,
                                                     repository_path=self._data_path)

        except RuntimeError:
            if self._missing_rates_return_null:
                return NullBeamEmissionPEC()
            raise

        # obtain isotope's rest wavelength for a given transition
        # the wavelength is used ot convert the PEC from photons/s/m3 to W/m3
        wavelength = self.wavelength(beam_ion, 0, transition)

        # load and interpolate data
        return BeamEmissionPEC(data, wavelength, extrapolate=self._permit_extrapolation)

    def impact_excitation_pec(self, ion, charge, transition):
        """

        :param ion:
        :param charge:
        :param transition:
        :return:
        """

        # extract element from isotope because there are no isotope rates in ADAS
        # keep the isotope for the wavelength
        ion_element = ion.element if isinstance(ion, Isotope) else ion

        try:
            # read electron impact excitation PEC from json file in the repository
            data = repository.get_pec_excitation_rate(ion_element, charge, transition, repository_path=self._data_path)

        except RuntimeError:
            if self._missing_rates_return_null:
                return NullImpactExcitationPEC()
            raise

        # obtain isotope's rest wavelength for a given transition
        # the wavelength is used ot convert the PEC from photons/s/m3 to W/m3
        wavelength = self.wavelength(ion, charge, transition)

        return ImpactExcitationPEC(wavelength, data, extrapolate=self._permit_extrapolation)

    def recombination_pec(self, ion, charge, transition):
        """

        :param ion:
        :param charge:
        :param transition:
        :return:
        """

        # extract element from isotope because there are no isotope rates in ADAS
        # keep the isotope for the wavelength
        ion_element = ion.element if isinstance(ion, Isotope) else ion

        try:
            # read free electron recombination PEC from json file in the repository
            data = repository.get_pec_recombination_rate(ion_element, charge, transition, repository_path=self._data_path)

        except (FileNotFoundError, KeyError):
            if self._missing_rates_return_null:
                return NullRecombinationPEC()
            raise

        # obtain isotope's rest wavelength for a given transition
        # the wavelength is used ot convert the PEC from photons/s/m3 to W/m3
        wavelength = self.wavelength(ion, charge, transition)

        return RecombinationPEC(wavelength, data, extrapolate=self._permit_extrapolation)

    def line_radiated_power_rate(self, ion, charge):

        # extract element from isotope because there are no isotope rates in ADAS
        if isinstance(ion, Isotope):
            ion = ion.element

        try:
            # read total line radiated power rate from json file in the repository
            data = repository.get_line_radiated_power_rate(ion, charge, repository_path=self._data_path)

        except RuntimeError:
            if self._missing_rates_return_null:
                return NullLineRadiationPower(ion, charge)
            raise

        return LineRadiationPower(ion, charge, data, extrapolate=self._permit_extrapolation)

    def continuum_radiated_power_rate(self, ion, charge):

        # extract element from isotope because there are no isotope rates in ADAS
        if isinstance(ion, Isotope):
            ion = ion.element

        try:
            # read continuum radiated power rate from json file in the repository
            data = repository.get_continuum_radiated_power_rate(ion, charge, repository_path=self._data_path)

        except RuntimeError:
            if self._missing_rates_return_null:
                return NullContinuumPower(ion, charge)
            raise

        return ContinuumPower(ion, charge, data, extrapolate=self._permit_extrapolation)

    def cx_radiated_power_rate(self, ion, charge):

        # extract element from isotope because there are no isotope rates in ADAS
        if isinstance(ion, Isotope):
            ion = ion.element

        try:
            # read CX radiated power rate from json file in the repository
            data = repository.get_cx_radiated_power_rate(ion, charge, repository_path=self._data_path)

        except RuntimeError:
            if self._missing_rates_return_null:
                return NullCXRadiationPower(ion, charge)
            raise

        return CXRadiationPower(ion, charge, data, extrapolate=self._permit_extrapolation)
