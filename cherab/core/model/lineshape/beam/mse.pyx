# cython: language_level=3

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

from libc.math cimport fabs, sqrt, M_PI
from raysect.optical cimport Spectrum, Point3D, Vector3D

from cherab.core.plasma cimport Plasma
from cherab.core.beam cimport Beam
from cherab.core.atomic cimport Line, AtomicData
from cherab.core.atomic.elements import Isotope, hydrogen
from cherab.core.math.function cimport autowrap_function2d
from cherab.core.utility.constants cimport ELEMENTARY_CHARGE, ELECTRON_REST_MASS, VACUUM_PERMITTIVITY, PLANCK_CONSTANT, HC_EV_NM
from cherab.core.model.lineshape.gaussian cimport add_gaussian_line
from cherab.core.model.lineshape.doppler cimport thermal_broadening, doppler_shift

cimport cython


# (3/2 * e0 * h^2)/(2pi * Me * e^2) in [eV*m/V]
cdef double STARK_ENERGY_SPLITTING_FACTOR = 3 * VACUUM_PERMITTIVITY * PLANCK_CONSTANT**2 / (2 * M_PI * ELECTRON_REST_MASS * ELEMENTARY_CHARGE**2)

DEF PI_POLARISATION = 0
DEF SIGMA_POLARISATION = 1
DEF NO_POLARISATION = 2

# example statistical weights supplied by E. Delabie for JET like plasmas
                            # [    Sigma group   ][        Pi group            ]
# STARK_STATISTICAL_WEIGHTS = [0.586167, 0.206917, 0.153771, 0.489716, 0.356513]
SIGMA_TO_PI = 0.56
SIGMA1_TO_SIGMA0 = 0.706  # s1*2/s0
PI2_TO_PI3 = 0.314  # pi2/pi3
PI4_TO_PI3 = 0.728  # pi4/pi3

# TODO - the sigma/pi line ratios should be moved to an atomic data source


cdef class BeamEmissionMultiplet(BeamLineShapeModel):
    """
    Produces Beam Emission Multiplet line shape, also known as the Motional Stark Effect spectrum.
    """

    def __init__(self, Line line, double wavelength, Beam beam, AtomicData atomic_data,
                 object sigma_to_pi=SIGMA_TO_PI, object sigma1_to_sigma0=SIGMA1_TO_SIGMA0,
                 object pi2_to_pi3=PI2_TO_PI3, object pi4_to_pi3=PI4_TO_PI3,
                 polarisation='no'):

        # extract element from isotope
        element = line.element.element if isinstance(line.element, Isotope) else line.element

        if element != hydrogen or line.charge != 0 or line.transition != (3, 2):
            raise ValueError('The BeamEmissionMultiplet model currently only support Balmer-Alpha as the line choice.')

        super().__init__(line, wavelength, beam, atomic_data)

        self.polarisation = polarisation

        self._sigma_to_pi = autowrap_function2d(sigma_to_pi)
        self._sigma1_to_sigma0 = autowrap_function2d(sigma1_to_sigma0)
        self._pi2_to_pi3 = autowrap_function2d(pi2_to_pi3)
        self._pi4_to_pi3 = autowrap_function2d(pi4_to_pi3)

    @property
    def polarisation(self):
        if self._polarisation == PI_POLARISATION:
            return 'pi'
        if self._polarisation == SIGMA_POLARISATION:
            return 'sigma'
        if self._polarisation == NO_POLARISATION:
            return 'no'

    @polarisation.setter
    def polarisation(self, value):
        if value.lower() == 'pi':
            self._polarisation = PI_POLARISATION
        elif value.lower() == 'sigma':
            self._polarisation = SIGMA_POLARISATION
        elif value.lower() == 'no':
            self._polarisation = NO_POLARISATION
        else:
            raise ValueError('Select between "pi", "sigma" or "no", {} is unsupported.'.format(value))

    @cython.cdivision(True)
    cpdef Spectrum add_line(self, double radiance, Point3D beam_point, Point3D plasma_point,
                            Vector3D beam_velocity, Vector3D observation_direction, Spectrum spectrum):

        cdef:
            double x, y, z
            Plasma plasma
            double te, ne, beam_energy, sigma, stark_split, beam_ion_mass, beam_temperature, e_magn
            double shifted_wavelength, photon_energy
            double cos_sqr, sin_sqr, sigma_to_pi, pi_to_total, intensity_sig, intensity_pi
            double s1_to_s0, intensity_s0, intensity_s1
            double pi2_to_pi3, pi4_to_pi3, intensity_pi2, intensity_pi3, intensity_pi4
            Vector3D b_field, e_field

        # extract for more compact code
        x = plasma_point.x
        y = plasma_point.y
        z = plasma_point.z

        plasma = self.beam.get_plasma()

        te = plasma.get_electron_distribution().effective_temperature(x, y, z)
        if te <= 0.0:
            return spectrum

        ne = plasma.get_electron_distribution().density(x, y, z)
        if ne <= 0.0:
            return spectrum

        beam_energy = self.beam.get_energy()

        # calculate Stark splitting
        b_field = plasma.get_b_field().evaluate(x, y, z)
        e_field = beam_velocity.cross(b_field)
        e_magn = e_field.get_length()
        stark_split = STARK_ENERGY_SPLITTING_FACTOR * e_magn

        # calculate emission line central wavelength, doppler shifted along observation direction
        shifted_wavelength = doppler_shift(self.wavelength, observation_direction, beam_velocity)
        photon_energy = HC_EV_NM / shifted_wavelength

        # calculate doppler broadening
        beam_ion_mass = self.beam.get_element().atomic_weight
        beam_temperature = self.beam.get_temperature()
        sigma = thermal_broadening(self.wavelength, beam_temperature, beam_ion_mass)

        if e_magn == 0:
            # no splitting if electric field strength is zero
            if self._polarisation == NO_POLARISATION:
                return add_gaussian_line(radiance, shifted_wavelength, sigma, spectrum)

            return add_gaussian_line(0.5 * radiance, shifted_wavelength, sigma, spectrum)

        # coefficients for intensities parallel and perpendicular to electric field
        cos_sqr = e_field.normalise().dot(observation_direction.normalise())**2
        sin_sqr = 1. - cos_sqr

        # calculate relative intensities of sigma and pi lines
        sigma_to_pi = self._sigma_to_pi.evaluate(ne, beam_energy)
        pi_to_total = 1 / (1 + sigma_to_pi)

        # add Sigma lines to output
        if self._polarisation != PI_POLARISATION:
            intensity_sig = (sin_sqr * sigma_to_pi * pi_to_total + cos_sqr) * radiance
            s1_to_s0 = self._sigma1_to_sigma0.evaluate(ne, beam_energy)
            intensity_s0 = 1 / (s1_to_s0 + 1)
            intensity_s1 = 0.5 * s1_to_s0 * intensity_s0  # 0.5 for each +/- component

            spectrum = add_gaussian_line(intensity_sig * intensity_s0, shifted_wavelength, sigma, spectrum)
            spectrum = add_gaussian_line(intensity_sig * intensity_s1, HC_EV_NM / (photon_energy - stark_split), sigma, spectrum)
            spectrum = add_gaussian_line(intensity_sig * intensity_s1, HC_EV_NM / (photon_energy + stark_split), sigma, spectrum)

        # add Pi lines to output
        if self._polarisation != SIGMA_POLARISATION:
            intensity_pi = sin_sqr * pi_to_total * radiance
            pi2_to_pi3 = self._pi2_to_pi3.evaluate(ne, beam_energy)
            pi4_to_pi3 = self._pi4_to_pi3.evaluate(ne, beam_energy)
            intensity_pi3 = 0.5 / (1 + pi2_to_pi3 + pi4_to_pi3)  # 0.5 for each +/- component
            intensity_pi2 = pi2_to_pi3 * intensity_pi3
            intensity_pi4 = pi4_to_pi3 * intensity_pi3

            spectrum = add_gaussian_line(intensity_pi * intensity_pi2, HC_EV_NM / (photon_energy - 2 * stark_split), sigma, spectrum)
            spectrum = add_gaussian_line(intensity_pi * intensity_pi2, HC_EV_NM / (photon_energy + 2 * stark_split), sigma, spectrum)
            spectrum = add_gaussian_line(intensity_pi * intensity_pi3, HC_EV_NM / (photon_energy - 3 * stark_split), sigma, spectrum)
            spectrum = add_gaussian_line(intensity_pi * intensity_pi3, HC_EV_NM / (photon_energy + 3 * stark_split), sigma, spectrum)
            spectrum = add_gaussian_line(intensity_pi * intensity_pi4, HC_EV_NM / (photon_energy - 4 * stark_split), sigma, spectrum)
            spectrum = add_gaussian_line(intensity_pi * intensity_pi4, HC_EV_NM / (photon_energy + 4 * stark_split), sigma, spectrum)

        return spectrum
