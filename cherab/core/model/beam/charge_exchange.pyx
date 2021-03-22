# cython: language_level=3

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

"""Calculate CX emission with ADAS beam coefficients with beam"""

from scipy import constants

from libc.math cimport exp, sqrt, M_PI as pi
from numpy cimport ndarray
cimport cython
from raysect.optical.material.emitter.inhomogeneous import NumericalIntegrator

from cherab.core cimport Species, Plasma, Beam, Element, BeamPopulationRate
from cherab.core.model.lineshape cimport doppler_shift, thermal_broadening, add_gaussian_line
from cherab.core.utility.constants cimport RECIP_4_PI, ELEMENTARY_CHARGE, ATOMIC_MASS

cdef double RECIP_ELEMENTARY_CHARGE = 1 / ELEMENTARY_CHARGE
cdef double RECIP_ATOMIC_MASS = 1 / ATOMIC_MASS


cdef double evamu_to_ms(double x):
    return sqrt(2 * x * ELEMENTARY_CHARGE * RECIP_ATOMIC_MASS)


cdef double ms_to_evamu(double x):
    return 0.5 * (x ** 2) * RECIP_ELEMENTARY_CHARGE * ATOMIC_MASS


cdef class BeamCXLine(BeamModel):
    """Calculates CX emission for a beam.

    :param line:
    :param step: integration step in meters
    :return:
    """

    def __init__(self, Line line not None, Beam beam=None, Plasma plasma=None, AtomicData atomic_data=None):

        super().__init__(beam, plasma, atomic_data)

        self._line = line

        # initialise cache to empty
        self._target_species = None
        self._wavelength = 0.0
        self._ground_beam_rate = None
        self._excited_beam_data = None

    @property
    def line(self):
        return self._line

    @line.setter
    def line(self, Line value not None):
        # the data cache depends on the line configuration
        self._line = value
        self._change()

    # todo: escape early if data is not suitable for a calculation
    # todo: carefully review changes to maths
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cpdef Spectrum emission(self, Point3D beam_point, Point3D plasma_point, Vector3D beam_direction,
                            Vector3D observation_direction, Spectrum spectrum):

        cdef:
            double x, y, z
            double donor_density
            double receiver_temperature, receiver_density, receiver_ion_mass, interaction_speed, interaction_energy, emission_rate
            Vector3D receiver_velocity, donor_velocity, interaction_velocity
            double natural_wavelength, central_wavelength, radiance, sigma

        # cache data on first run
        if self._target_species is None:
            self._populate_cache()

        # obtain donor density from beam
        donor_density = self._beam.density(beam_point.x, beam_point.y, beam_point.z)

        # abort calculation if donor density is zero
        if donor_density == 0.0:
            return spectrum

        # extract for more compact code
        x = plasma_point.x
        y = plasma_point.y
        z = plasma_point.z

        # abort calculation if receiver density is zero
        receiver_density = self._target_species.distribution.density(x, y, z)
        if receiver_density == 0:
            return spectrum

        # abort calculation if receiver temperature is zero
        receiver_temperature = self._target_species.distribution.effective_temperature(x, y, z)
        if receiver_temperature == 0:
            return spectrum

        receiver_velocity = self._target_species.distribution.bulk_velocity(x, y, z)
        receiver_ion_mass = self._target_species.element.atomic_weight

        donor_velocity = beam_direction.normalise().mul(evamu_to_ms(self._beam.get_energy()))

        interaction_velocity = donor_velocity.sub(receiver_velocity)
        interaction_speed = interaction_velocity.get_length()
        interaction_energy = ms_to_evamu(interaction_speed)

        # calculate the composite charge-exchange emission coefficient
        emission_rate = self._composite_cx_rate(x, y, z, interaction_energy, donor_velocity, receiver_temperature, receiver_density)

        # calculate emission line central wavelength, doppler shifted along observation direction
        natural_wavelength = self._wavelength
        central_wavelength = doppler_shift(natural_wavelength, observation_direction, receiver_velocity)

        # spectral line emission in W/m^3/str
        radiance = RECIP_4_PI * donor_density * receiver_density * emission_rate
        sigma = thermal_broadening(natural_wavelength, receiver_temperature, receiver_ion_mass)
        return add_gaussian_line(radiance, central_wavelength, sigma, spectrum)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef double _composite_cx_rate(self, double x, double y, double z, double interaction_energy,
                                          Vector3D donor_velocity, double receiver_temperature, double receiver_density) except? -1e999:
        """
        Performs a beam population weighted average of the effective cx rates.

        .. math::
            q_c = \\frac{q_1 + \\sum_{i=2}^n k_i q_i}{1 + \\sum_{i=2}^n k_i}

        where,
        :math:`k_i` is the beam population of the ith meta-stable state
        relative to the ground state.
        :math:`q_i` is the effective charge exchange rate vs the beam's
        ith meta-stable population.

        :param x: The plasma space x coordinate in meters.
        :param y: The plasma space y coordinate in meters.
        :param z: The plasma space z coordinate in meters.
        :param interaction_energy: The donor-receiver interaction energy in eV/amu.
        :param donor_velocity: A Vector defining the donor particle velocity in m/s.
        :param receiver_temperature: The receiver species temperature in eV.
        :param receiver_density: The receiver species density in m^-3
        :return: The composite charge exchange rate in W.m^3.
        """

        cdef:
            double z_effective, b_field, rate, total_population, population, effective_rate
            BeamCXPEC cx_rate
            list population_data

        # calculate z_effective and the B-field magnitude
        z_effective = self._plasma.z_effective(x, y, z)
        b_field = self._plasma.get_b_field().evaluate(x, y, z).get_length()

        # rate for the ground state (metastable = 1)
        rate = self._ground_beam_rate.evaluate(interaction_energy,
                                               receiver_temperature,
                                               receiver_density,
                                               z_effective,
                                               b_field)

        # accumulate the total fractional population or normalisation
        # starts at 1 as populations are measured relative to the ground state
        total_population = 1

        # rates for the excited states (metastable > 1)
        for cx_rate, population_data in self._excited_beam_data:

            population = self._beam_population(x, y, z, donor_velocity, population_data)

            effective_rate = cx_rate.evaluate(interaction_energy,
                                              receiver_temperature,
                                              receiver_density,
                                              z_effective,
                                              b_field)

            rate += population * effective_rate
            total_population += population

        # normalise to give population weighted average
        rate /= total_population

        return rate

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef double _beam_population(self, double x, double y, double z, Vector3D beam_velocity, list population_data) except? -1e999:
        """
        Calculates the relative beam population.

        See www.adas.ac.uk/man/chap3-04.pdf equation 4.4.7 and notes for details on this calculation.
        Note: As we have access to ni for each species, we have done away with the impurity fractions
        used in the above document

        :param x: The plasma space x coordinate in meters.
        :param y: The plasma space y coordinate in meters.
        :param z: The plasma space z coordinate in meters.
        :param beam_velocity: A Vector defining the beam particle velocity in m/s.
        :param population_data: A list of tuples containing the species and associated beam coefficients.
        :return: The relative beam population.
        """

        cdef:
            double density_sum, pop_coeff, total_ne, target_ne, target_ti
            double interaction_speed, interaction_energy, target_equiv_ne
            int target_z
            Vector3D target_velocity, interaction_velocity
            Species species
            BeamPopulationRate coeff

        # z-weighted density sum
        density_sum = 0
        for species, _ in population_data:
            density_sum += species.charge**2 * species.distribution.density(x, y, z)

        # combine population coefficients
        pop_coeff = 0
        total_ne = 0
        for species, coeff in population_data:

            target_z = species.charge
            target_ne = species.distribution.density(x, y, z) * target_z
            target_ti = species.distribution.effective_temperature(x, y, z)
            target_velocity = species.distribution.bulk_velocity(x, y, z)

            # calculate mean beam interaction energy
            interaction_velocity = beam_velocity.sub(target_velocity)
            interaction_speed = interaction_velocity.get_length()
            interaction_energy = ms_to_evamu(interaction_speed)

            # species equivalent electron density
            target_equiv_ne = density_sum / target_z

            pop_coeff += target_ne * coeff.evaluate(interaction_energy, target_equiv_ne, target_ti)
            total_ne += target_ne

        # normalise charge density weighted sum
        return pop_coeff / total_ne

    cdef int _populate_cache(self) except -1:

        cdef:
            Element receiver_element, donor_element
            int receiver_charge
            tuple transition
            Species species
            list rates, population_data
            BeamCXPEC rate
            BeamPopulationRate coeff

        # sanity checks
        if self._beam is None or self._plasma is None or self._atomic_data is None:
            raise RuntimeError("The emission model is not connected to a beam object.")

        if self._line is None:
            raise RuntimeError("The emission line has not been set.")

        receiver_element = self._line.element
        receiver_charge = self._line.charge + 1
        donor_element = self._beam.element
        transition = self._line.transition

        # locate target species
        try:
            self._target_species = self._plasma.composition.get(receiver_element, receiver_charge)
        except ValueError:
            raise RuntimeError("The plasma object does not contain the ion species for the specified cx line "
                               "(element={}, ionisation={}).".format(receiver_element.symbol, receiver_charge))

        # obtain wavelength for specified line
        self._wavelength = self._atomic_data.wavelength(receiver_element, receiver_charge - 1, transition)

        # obtain cx rates
        rates = self._atomic_data.beam_cx_pec(donor_element, receiver_element, receiver_charge, transition)

        # obtain beam population coefficients for each rate and assemble data
        # the data is assembled to make access efficient by linking the relevant rates and coefficients together:
        #
        #   ground_beam_rate = qeff(m=1)
        #   excited_beam_data = [
        #       (qeff(m=2), [(species, pop_coeff(m=2)),...]),
        #       (qeff(m=3), [(species, pop_coeff(m=3)),...]),
        #       etc...
        #   ]
        self._excited_beam_data = []
        for rate in rates:
            if rate.donor_metastable == 1:

                # other state populations are relative to the ground state
                # so the ground state does not require a relative population coefficient
                self._ground_beam_rate = rate

            else:

                # obtain population coefficients for all plasma species with which the beam interacts
                population_data = []
                for species in self._plasma.composition:

                    # bundle coefficient with its species
                    coeff = self._atomic_data.beam_population_rate(donor_element, rate.donor_metastable,
                                                                   species.element, species.charge)
                    population_data.append((species, coeff))

                # link each rate with its population data
                self._excited_beam_data.append((rate, population_data))

    def _change(self):

        # clear cache to force regeneration on first use
        self._target_species = None
        self._wavelength = 0.0
        self._ground_beam_rate = None
        self._excited_beam_data = None

