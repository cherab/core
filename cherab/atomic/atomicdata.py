# Copyright 2016-2022 Euratom
# Copyright 2016-2022 United Kingdom Atomic Energy Authority
# Copyright 2016-2022 Centro de Investigaciones Energéticas, Medioambientales y Tecnológicas
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

from cherab.core import AtomicData as _BaseAtomicData
from cherab.core.atomic.elements import Isotope
from cherab.atomic.repository import DEFAULT_REPOSITORY_PATH

from .rates import *
from .gaunt import FreeFreeGauntFactor
from .zeeman import ZeemanStructure
from cherab.atomic import repository


class AtomicData(_BaseAtomicData):
    """
    Atomic data source.

    :param str data_path: Path to local atomic data repository.
    :param bool permit_extrapolation: If true, informs interpolation objects to allow extrapolation
                                      beyond the limits of the tabulated data. Default is False.
    :param bool missing_rates_return_null: If true, allows Null rate objects to be returned when
                                           the requested atomic data is missing. Default is False.
    :param bool wavelength_element_fallback: If true, allows to use the element's wavelength when
                                             the isotope's wavelength is not available.
                                             Default is False.
    :param bool rate_element_fallback: If true, allows to use the element's rate when
                                       the isotope's rate is not available.
                                       Default is True.
    """

    def __init__(self, data_path=None, permit_extrapolation=False, missing_rates_return_null=False,
                 wavelength_element_fallback=False, rate_element_fallback=True):

        super().__init__()
        self._data_path = data_path or DEFAULT_REPOSITORY_PATH

        self._permit_extrapolation = permit_extrapolation

        self._missing_rates_return_null = missing_rates_return_null

        self._wavelength_element_fallback = wavelength_element_fallback

        self._rate_element_fallback = rate_element_fallback

    @property
    def data_path(self):
        return self._data_path

    def wavelength(self, ion, charge, transition):
        """
        Spectral line wavelength for a given transition.

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
        """
        Electron impact ionisation rate for a given species.

        Data is interpolated with cubic spline in log-log space.
        Nearest neighbour extrapolation is used when permit_extrapolation is True.

        :param ion: Element object defining the ion type.
        :param charge: Charge state of the ion.
        :return: Ionisation rate in m^3/s as a function of electron density and temperature.
        """

        # try to read the rate for the isotope, fallback to the element if fails
        try:
            # read ionisation rate from json file in the repository
            data = repository.get_ionisation_rate(ion, charge, repository_path=self._data_path)
        except RuntimeError:
            if isinstance(ion, Isotope) and self._rate_element_fallback:
                try:
                    data = repository.get_ionisation_rate(ion.element, charge, repository_path=self._data_path)
                except RuntimeError:
                    if self._missing_rates_return_null:
                        return NullIonisationRate()
                    raise
            elif self._missing_rates_return_null:
                return NullIonisationRate()
            else:
                raise

        return IonisationRate(data, extrapolate=self._permit_extrapolation)

    def recombination_rate(self, ion, charge):
        """
        Recombination rate for a given species.

        Data is interpolated with cubic spline in log-log space.
        Nearest neighbour extrapolation is used when permit_extrapolation is True.

        :param ion: Element object defining the ion type.
        :param charge: Charge state of the ion.
        :return: Recombination rate in m^3/s as a function of electron density and temperature.
        """

        # try to read the rate for the isotope, fallback to the element if fails
        try:
            # read recombination rate from json file in the repository
            data = repository.get_recombination_rate(ion, charge, repository_path=self._data_path)
        except RuntimeError:
            if isinstance(ion, Isotope) and self._rate_element_fallback:
                try:
                    data = repository.get_recombination_rate(ion.element, charge, repository_path=self._data_path)
                except RuntimeError:
                    if self._missing_rates_return_null:
                        return NullRecombinationRate()
                    raise
            elif self._missing_rates_return_null:
                return NullRecombinationRate()
            else:
                raise

        return RecombinationRate(data, extrapolate=self._permit_extrapolation)

    def thermal_cx_rate(self, donor_element, donor_charge, receiver_element, receiver_charge):
        """
        Thermal charge exchange effective rate coefficient for a given donor and receiver species.

        Data is interpolated with cubic spline in log-log space.
        Linear extrapolation is used when permit_extrapolation is True.

        :param donor_element: Element object defining the donor ion type.
        :param donor_charge: Charge state of the donor ion.
        :param receiver_element: Element object defining the receiver ion type.
        :param receiver_charge: Charge state of the receiver ion.
        :return: Thermal charge exchange rate in m^3/s as a function of electron density and
                 temperature.
        """

        # try to read the rates for the isotopes
        try:
            # read thermal CX rate from json file in the repository
            data = repository.get_thermal_cx_rate(donor_element, donor_charge,
                                                  receiver_element, receiver_charge,
                                                  repository_path=self._data_path)
        except RuntimeError:
            # if fails, fallback to the element for the receiver
            if isinstance(receiver_element, Isotope) and self._rate_element_fallback:
                try:
                    data = repository.get_thermal_cx_rate(donor_element, donor_charge,
                                                          receiver_element.element, receiver_charge,
                                                          repository_path=self._data_path)
                except RuntimeError:
                    # if fails, fallback to the element for the donor
                    if isinstance(donor_element, Isotope):
                        try:
                            data = repository.get_thermal_cx_rate(donor_element.element, donor_charge,
                                                                  receiver_element, receiver_charge,
                                                                  repository_path=self._data_path)
                        except RuntimeError:
                            # if fails, fallback to the elements for both
                            try:
                                data = repository.get_thermal_cx_rate(donor_element.element, donor_charge,
                                                                      receiver_element.element, receiver_charge,
                                                                      repository_path=self._data_path)
                            except RuntimeError:
                                if self._missing_rates_return_null:
                                    return NullThermalCXRate()
                                raise
                    elif self._missing_rates_return_null:
                        return NullThermalCXRate()
                    else:
                        raise
            # if the receiver is not an isotope, fallback to the element for the donor
            elif isinstance(donor_element, Isotope) and self._rate_element_fallback:
                try:
                    data = repository.get_thermal_cx_rate(donor_element.element, donor_charge,
                                                          receiver_element, receiver_charge,
                                                          repository_path=self._data_path)
                except RuntimeError:
                    if self._missing_rates_return_null:
                        return NullThermalCXRate()
                    raise
            elif self._missing_rates_return_null:
                return NullThermalCXRate()
            else:
                raise

        return ThermalCXRate(data, extrapolate=self._permit_extrapolation)

    def beam_cx_pec(self, donor_ion, receiver_ion, receiver_charge, transition):
        """
        Effective charge exchange photon emission coefficient for a given donor (beam)
        and receiver (plasma) species and a given transition.

        The data for "qeb" is interpolated with a cubic spline in log-log space.
        The data for "qti", "qni", "qz" and "qb" are interpolated with a cubic spline
        in linear space.
        Quadratic extrapolation is used for "qeb" and nearest neighbour extrapolation is used for
        "qti", "qni", "qz" and "qb" when permit_extrapolation is True.


        :param donor_ion: Element object defining the donor ion type.
        :param receiver_ion: Element object defining the receiver ion type.
        :param receiver_charge: Charge state of the receiver ion.
        :param transition: Tuple containing (initial level, final level) of the receiver species.
        :return: Charge exchange photon emission coefficient in W.m^3 as a function of
                 interaction energy, receiver ion temperature, receiver ion density,
                 plasma Z-effective, magnetic field magnitude.
        """

        # try to read the rates for the isotopes
        try:
            # read beam CX PEC from json file in the repository
            data = repository.get_beam_cx_rates(donor_ion, receiver_ion,
                                                receiver_charge, transition,
                                                repository_path=self._data_path)
        except RuntimeError:
            # if fails, fallback to the element for the receiver
            if isinstance(receiver_ion, Isotope) and self._rate_element_fallback:
                try:
                    data = repository.get_beam_cx_rates(donor_ion, receiver_ion.element,
                                                        receiver_charge, transition,
                                                        repository_path=self._data_path)
                except RuntimeError:
                    # if fails, fallback to the element for the donor
                    if isinstance(donor_ion, Isotope):
                        try:
                            data = repository.get_beam_cx_rates(donor_ion.element, receiver_ion,
                                                                receiver_charge, transition,
                                                                repository_path=self._data_path)
                        except RuntimeError:
                            # if fails, fallback to the elements for both
                            try:
                                data = repository.get_beam_cx_rates(donor_ion.element, receiver_ion.element,
                                                                    receiver_charge, transition,
                                                                    repository_path=self._data_path)
                            except RuntimeError:
                                if self._missing_rates_return_null:
                                    return [NullBeamCXPEC()]
                                raise
                    elif self._missing_rates_return_null:
                        return [NullBeamCXPEC()]
                    else:
                        raise
            # if the receiver is not an isotope, fallback to the element for the donor
            elif isinstance(donor_ion, Isotope) and self._rate_element_fallback:
                try:
                    data = repository.get_beam_cx_rates(donor_ion.element, receiver_ion,
                                                        receiver_charge, transition,
                                                        repository_path=self._data_path)
                except RuntimeError:
                    if self._missing_rates_return_null:
                        return [NullBeamCXPEC()]
                    raise
            elif self._missing_rates_return_null:
                return [NullBeamCXPEC()]
            else:
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
        Beam stopping coefficient for a given beam and target species.

        The data is interpolated with cubic spline in log-log space.
        Linear and quadratic extrapolations are used for "sen" and "st" respectively
        when permit_extrapolation is True.

        :param beam_ion: Element object defining the beam ion type.
        :param plasma_ion: Element object defining the target ion type.
        :param charge: Charge state of the target ion.
        :return: The beam stopping coefficient in m^3.s^-1 as a function of interaction energy,
                 target equivalent electron density, target temperature.
        """

        # try to read the rates for the isotopes
        try:
            # read beam stopping rate from json file in the repository
            data = repository.get_beam_stopping_rate(beam_ion, plasma_ion, charge,
                                                     repository_path=self._data_path)
        except RuntimeError:
            # if fails, fallback to the element for the plasma ion
            if isinstance(plasma_ion, Isotope) and self._rate_element_fallback:
                try:
                    data = repository.get_beam_stopping_rate(beam_ion, plasma_ion.element, charge,
                                                             repository_path=self._data_path)
                except RuntimeError:
                    # if fails, fallback to the element for the beam ion
                    if isinstance(beam_ion, Isotope):
                        try:
                            data = repository.get_beam_stopping_rate(beam_ion.element, plasma_ion, charge,
                                                                     repository_path=self._data_path)
                        except RuntimeError:
                            # if fails, fallback to the elements for both
                            try:
                                data = repository.get_beam_stopping_rate(beam_ion.element, plasma_ion.element, charge,
                                                                         repository_path=self._data_path)
                            except RuntimeError:
                                if self._missing_rates_return_null:
                                    return NullBeamStoppingRate()
                                raise
                    elif self._missing_rates_return_null:
                        return NullBeamStoppingRate()
                    else:
                        raise
            # if the receiver is not an isotope, fallback to the element for the beam ion
            elif isinstance(beam_ion, Isotope) and self._rate_element_fallback:
                try:
                    data = repository.get_beam_stopping_rate(beam_ion.element, plasma_ion, charge,
                                                             repository_path=self._data_path)
                except RuntimeError:
                    if self._missing_rates_return_null:
                        return NullBeamStoppingRate()
                    raise
            elif self._missing_rates_return_null:
                return NullBeamStoppingRate()
            else:
                raise

        # load and interpolate data
        return BeamStoppingRate(data, extrapolate=self._permit_extrapolation)

    def beam_population_rate(self, beam_ion, metastable, plasma_ion, charge):
        """
        Beam population coefficient for a given beam and target species.

        The data is interpolated with cubic spline in log-log space.
        Linear and quadratic extrapolations are used for "sen" and "st" respectively
        when permit_extrapolation is True.

        :param beam_ion: Element object defining the beam ion type.
        :param metastable: The beam ion metastable number.
        :param plasma_ion: Element object defining the target ion type.
        :param charge: Charge state of the target ion.
        :return: The beam population coefficient in dimensionless units as a function of
                 interaction energy, target equivalent electron density, target temperature.
        """

        # try to read the rates for the isotopes
        try:
            # read beam population rate from json file in the repository
            data = repository.get_beam_population_rate(beam_ion, metastable, plasma_ion, charge,
                                                       repository_path=self._data_path)
        except RuntimeError:
            # if fails, fallback to the element for the plasma ion
            if isinstance(plasma_ion, Isotope) and self._rate_element_fallback:
                try:
                    data = repository.get_beam_population_rate(beam_ion, metastable, plasma_ion.element, charge,
                                                               repository_path=self._data_path)
                except RuntimeError:
                    # if fails, fallback to the element for the beam ion
                    if isinstance(beam_ion, Isotope):
                        try:
                            data = repository.get_beam_population_rate(beam_ion.element, metastable, plasma_ion, charge,
                                                                       repository_path=self._data_path)
                        except RuntimeError:
                            # if fails, fallback to the elements for both
                            try:
                                data = repository.get_beam_population_rate(beam_ion.element, metastable, plasma_ion.element, charge,
                                                                           repository_path=self._data_path)
                            except RuntimeError:
                                if self._missing_rates_return_null:
                                    return NullBeamPopulationRate()
                                raise
                    elif self._missing_rates_return_null:
                        return NullBeamPopulationRate()
                    else:
                        raise
            # if the receiver is not an isotope, fallback to the element for the beam ion
            elif isinstance(beam_ion, Isotope) and self._rate_element_fallback:
                try:
                    data = repository.get_beam_stopping_rate(beam_ion.element, plasma_ion, charge,
                                                             repository_path=self._data_path)
                except RuntimeError:
                    if self._missing_rates_return_null:
                        return NullBeamPopulationRate()
                    raise
            elif self._missing_rates_return_null:
                return NullBeamPopulationRate()
            else:
                raise

        # load and interpolate data
        return BeamPopulationRate(data, extrapolate=self._permit_extrapolation)

    def beam_emission_pec(self, beam_ion, plasma_ion, charge, transition):
        """
        The beam photon emission coefficient for a given beam and target species
        and a given transition.

        The data is interpolated with cubic spline in log-log space.
        Linear and quadratic extrapolations are used for "sen" and "st" respectively
        when permit_extrapolation is True.

        :param beam_ion: Element object defining the beam ion type.
        :param plasma_ion: Element object defining the target ion type.
        :param charge: Charge state of the target ion.
        :param transition: Tuple containing (initial level, final level) of the beam ion.
        :return: The beam photon emission coefficient in W.m^3 as a function of
                 interaction energy, target equivalent electron density, target temperature.
        """

        # try to read the rates for the isotopes
        try:
            # read beam PEC from json file in the repository
            data = repository.get_beam_emission_rate(beam_ion, plasma_ion, charge, transition,
                                                     repository_path=self._data_path)
        except RuntimeError:
            # if fails, fallback to the element for the plasma ion
            if isinstance(plasma_ion, Isotope) and self._rate_element_fallback:
                try:
                    data = repository.get_beam_emission_rate(beam_ion, plasma_ion.element, charge, transition,
                                                             repository_path=self._data_path)
                except RuntimeError:
                    # if fails, fallback to the element for the beam ion
                    if isinstance(beam_ion, Isotope):
                        try:
                            data = repository.get_beam_emission_rate(beam_ion.element, plasma_ion, charge, transition,
                                                                     repository_path=self._data_path)
                        except RuntimeError:
                            # if fails, fallback to the elements for both
                            try:
                                data = repository.get_beam_emission_rate(beam_ion.element, plasma_ion.element, charge, transition,
                                                                         repository_path=self._data_path)
                            except RuntimeError:
                                if self._missing_rates_return_null:
                                    return NullBeamEmissionPEC()
                                raise
                    elif self._missing_rates_return_null:
                        return NullBeamEmissionPEC()
                    else:
                        raise
            # if the receiver is not an isotope, fallback to the element for the beam ion
            elif isinstance(beam_ion, Isotope) and self._rate_element_fallback:
                try:
                    data = repository.get_beam_emission_rate(beam_ion.element, plasma_ion, charge, transition,
                                                             repository_path=self._data_path)
                except RuntimeError:
                    if self._missing_rates_return_null:
                        return NullBeamEmissionPEC()
                    raise
            elif self._missing_rates_return_null:
                return NullBeamEmissionPEC()
            else:
                raise

        # obtain isotope's rest wavelength for a given transition
        # the wavelength is used ot convert the PEC from photons/s/m3 to W/m3
        wavelength = self.wavelength(beam_ion, 0, transition)

        # load and interpolate data
        return BeamEmissionPEC(data, wavelength, extrapolate=self._permit_extrapolation)

    def impact_excitation_pec(self, ion, charge, transition):
        """
        Electron impact excitation photon emission coefficient for a given species.

        The data is interpolated with cubic spline in log-log space.
        Nearest neighbour extrapolation is used when permit_extrapolation is True.

        :param ion: Element object defining the ion type.
        :param charge: Charge state of the ion.
        :param transition: Tuple containing (initial level, final level).
        :return: Impact excitation photon emission coefficient in W.m^3 as a
                 function of electron density and temperature.
        """

        # try to read the rate for the isotope, fallback to the element if fails
        try:
            # read excitation PEC from json file in the repository
            data = repository.get_pec_excitation_rate(ion, charge, transition, repository_path=self._data_path)
        except RuntimeError:
            if isinstance(ion, Isotope) and self._rate_element_fallback:
                try:
                    data = repository.get_pec_excitation_rate(ion.element, charge, transition, repository_path=self._data_path)
                except RuntimeError:
                    if self._missing_rates_return_null:
                        return NullExcitationPEC()
                    raise
            elif self._missing_rates_return_null:
                return NullExcitationPEC()
            else:
                raise

        # obtain isotope's rest wavelength for a given transition
        # the wavelength is used ot convert the PEC from photons/s/m3 to W/m3
        wavelength = self.wavelength(ion, charge, transition)

        return ImpactExcitationPEC(wavelength, data, extrapolate=self._permit_extrapolation)

    def recombination_pec(self, ion, charge, transition):
        """
        Recombination photon emission coefficient for a given species.

        The data is interpolated with cubic spline in log-log space.
        Nearest neighbour extrapolation is used when permit_extrapolation is True.

        :param ion: Element object defining the ion type.
        :param charge: Charge state of the ion after recombination.
        :param transition: Tuple containing (initial level, final level).
        :return: Recombination photon emission coefficient in W.m^3 as a function of electron
                 density and temperature.
        """

        # try to read the rate for the isotope, fallback to the element if fails
        try:
            # read recombination PEC from json file in the repository
            data = repository.get_pec_recombination_rate(ion, charge, transition, repository_path=self._data_path)
        except RuntimeError:
            if isinstance(ion, Isotope) and self._rate_element_fallback:
                try:
                    data = repository.get_pec_recombination_rate(ion.element, charge, transition, repository_path=self._data_path)
                except RuntimeError:
                    if self._missing_rates_return_null:
                        return NullRecombinationPEC()
                    raise
            elif self._missing_rates_return_null:
                return NullRecombinationPEC()
            else:
                raise

        # obtain isotope's rest wavelength for a given transition
        # the wavelength is used ot convert the PEC from photons/s/m3 to W/m3
        wavelength = self.wavelength(ion, charge, transition)

        return RecombinationPEC(wavelength, data, extrapolate=self._permit_extrapolation)

    def line_radiated_power_rate(self, ion, charge):
        """
        Line radiated power coefficient for a given species.

        The data is interpolated with cubic spline in log-log space.
        Nearest neighbour extrapolation is used when permit_extrapolation is True.

        :param ion: Element object defining the ion type.
        :param charge: Charge state of the ion.
        :return: Line radiated power coefficient in W.m^3 as a function of electron
                 density and temperature.
        """

        # try to read the rate for the isotope, fallback to the element if fails
        try:
            # read total line radiated power rate from json file in the repository
            data = repository.get_line_radiated_power_rate(ion, charge, repository_path=self._data_path)
        except RuntimeError:
            if isinstance(ion, Isotope) and self._rate_element_fallback:
                try:
                    data = repository.get_line_radiated_power_rate(ion.element, charge, repository_path=self._data_path)
                except RuntimeError:
                    if self._missing_rates_return_null:
                        return NullLineRadiationPower(ion, charge)
                    raise
            elif self._missing_rates_return_null:
                return NullLineRadiationPower(ion, charge)
            else:
                raise

        return LineRadiationPower(ion, charge, data, extrapolate=self._permit_extrapolation)

    def continuum_radiated_power_rate(self, ion, charge):
        """
        Recombination continuum radiated power coefficient for a given species.

        The data is interpolated with cubic spline in log-log space.
        Nearest neighbour extrapolation is used when permit_extrapolation is True.

        :param ion: Element object defining the ion type.
        :param charge: Charge state of the ion.
        :return: Continuum radiated power coefficient in W.m^3 as a function
                 of electron density and temperature.
        """

        # try to read the rate for the isotope, fallback to the element if fails
        try:
            # read continuum radiated power rate from json file in the repository
            data = repository.get_continuum_radiated_power_rate(ion, charge, repository_path=self._data_path)
        except RuntimeError:
            if isinstance(ion, Isotope) and self._rate_element_fallback:
                try:
                    data = repository.get_continuum_radiated_power_rate(ion.element, charge, repository_path=self._data_path)
                except RuntimeError:
                    if self._missing_rates_return_null:
                        return NullContinuumPower(ion, charge)
                    raise
            elif self._missing_rates_return_null:
                return NullContinuumPower(ion, charge)
            else:
                raise

        return ContinuumPower(ion, charge, data, extrapolate=self._permit_extrapolation)

    def cx_radiated_power_rate(self, ion, charge):
        """
        Charge exchange radiated power coefficient for a given species.

        The data is interpolated with cubic spline in log-log space.
        Linear extrapolation is used when permit_extrapolation is True.

        :param ion: Element object defining the ion type.
        :param charge: Charge state of the ion.
        :return: Charge exchange radiated power coefficient in W.m^3 as a function
                 of electron density and temperature.
        """

        # try to read the rate for the isotope, fallback to the element if fails
        try:
            # read CX radiated power rate from json file in the repository
            data = repository.get_cx_radiated_power_rate(ion, charge, repository_path=self._data_path)
        except RuntimeError:
            if isinstance(ion, Isotope) and self._rate_element_fallback:
                try:
                    data = repository.get_cx_radiated_power_rate(ion.element, charge, repository_path=self._data_path)
                except RuntimeError:
                    if self._missing_rates_return_null:
                        return NullCXRadiationPower(ion, charge)
                    raise
            elif self._missing_rates_return_null:
                return NullCXRadiationPower(ion, charge)
            else:
                raise

        return CXRadiationPower(ion, charge, data, extrapolate=self._permit_extrapolation)

    def total_radiated_power(self, ion):
        """
        Total radiated power coefficient in equilibrium conditions for a given species.

        The data is interpolated with cubic spline in log-log space.
        Nearest neighbour extrapolation is used when permit_extrapolation is True.

        :param ion: Element object defining the ion type.
        :return: Total radiated power coefficient in W.m^3 as a function
                 of electron density and temperature.
        """

        # try to read the rate for the isotope, fallback to the element if fails
        try:
            # read total radiated power rate from json file in the repository
            data = repository.get_total_radiated_power_rate(ion, repository_path=self._data_path)
        except RuntimeError:
            if isinstance(ion, Isotope) and self._rate_element_fallback:
                try:
                    data = repository.get_total_radiated_power_rate(ion.element, repository_path=self._data_path)
                except RuntimeError:
                    if self._missing_rates_return_null:
                        return NullTotalRadiatedPower(ion, charge)
                    raise
            elif self._missing_rates_return_null:
                return NullTotalRadiatedPower(ion, charge)
            else:
                raise

        return TotalRadiatedPower(ion, data, extrapolate=self._permit_extrapolation)

    def fractional_abundance(self, ion, charge):
        """
        Fractional abundance of a given species in thermodynamic equilibrium.

        The data is interpolated with cubic spline.
        Linear extrapolation is used when permit_extrapolation is True.

        :param ion: Element object defining the ion type.
        :param charge: Charge state of the ion.
        :return: Fractional abundance as a function
                 of electron density and temperature.
        """

        # try to read the rate for the isotope, fallback to the element if fails
        try:
            # read total radiated power rate from json file in the repository
            data = repository.get_fractional_abundance(ion, charge, repository_path=self._data_path)
        except RuntimeError:
            if isinstance(ion, Isotope) and self._rate_element_fallback:
                data = repository.get_fractional_abundance(ion.element, charge, repository_path=self._data_path)
            else:
                raise

        return FractionalAbundance(ion, charge, data, extrapolate=self._permit_extrapolation)

    def zeeman_structure(self, ion, charge, transition):
        r"""
        Wavelengths and ratios of
        :math:`\pi`-/:math:`\sigma`-polarised Zeeman components for any given value of
        magnetic field strength.

        :param ion: Element object defining the ion type.
        :param charge: Charge state of the ion after recombination.
        :param transition: Tuple containing (initial level, final level).
        :return: ZeemanStructure object.
        """

        try:
            # read Zeeman structure from json file in the repository
            data = repository.get_zeeman_structure(ion, charge, transition, repository_path=self._data_path)
        except RuntimeError:
            if isinstance(ion, Isotope) and self._rate_element_fallback:
                data = repository.get_zeeman_structure(ion.element, charge, transition, repository_path=self._data_path)
            else:
                raise

        return ZeemanStructure(data, extrapolate=self._permit_extrapolation)

    def free_free_gaunt_factor(self):
        r"""
        Free-free Gaunt factor used in the bremsstrahlung emission model.

        The Gaunt factor is defined in the space of parameters:
        :math:`u = h{\nu}/kT` and :math:`{\gamma}^{2} = Z^{2}Ry/kT`.
        See T.R. Carson, 1988, Astron. & Astrophys., 189,
        `319 <https://ui.adsabs.harvard.edu/#abs/1988A&A...189..319C/abstract>`_ for details.

        The cubic interpolation in a semi-log space is used.

        The Born approximation and classical limits are used outside the interpolation range.

        :return: Free-free Gaunt factor as a function of species charge, electron temperature
                 and wavelength.
        """

        data = repository.get_free_free_gaunt_factor(repository_path=self._data_path)

        return FreeFreeGauntFactor(data)
