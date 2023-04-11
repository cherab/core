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

from libc.math cimport fabs, sqrt
from raysect.optical cimport Spectrum, Point3D, Vector3D

from cherab.core.plasma cimport Plasma
from cherab.core.beam cimport Beam
from cherab.core.atomic cimport AtomicData
from cherab.core.atomic cimport Line
from cherab.core.math.function cimport autowrap_function1d, autowrap_function2d
from cherab.core.utility.constants cimport ATOMIC_MASS, ELEMENTARY_CHARGE
from cherab.core.model.lineshape.gaussian cimport add_gaussian_line
from cherab.core.model.lineshape.doppler cimport thermal_broadening, doppler_shift

cimport cython


cdef double RECIP_ATOMIC_MASS = 1 / ATOMIC_MASS


cdef double evamu_to_ms(double x):
    return sqrt(2 * x * ELEMENTARY_CHARGE * RECIP_ATOMIC_MASS)


DEF STARK_SPLITTING_FACTOR = 2.77e-8


cdef class BeamEmissionMultiplet(BeamLineShapeModel):
    """
    Produces Beam Emission Multiplet line shape, also known as the Motional Stark Effect spectrum.
    """

    def __init__(self, Line line, double wavelength, Beam beam, AtomicData atomic_data,
                 object sigma_to_pi, object sigma1_to_sigma0, object pi2_to_pi3, object pi4_to_pi3):

        super().__init__(line, wavelength, beam, atomic_data)

        self._sigma_to_pi = autowrap_function2d(sigma_to_pi)
        self._sigma1_to_sigma0 = autowrap_function1d(sigma1_to_sigma0)
        self._pi2_to_pi3 = autowrap_function1d(pi2_to_pi3)
        self._pi4_to_pi3 = autowrap_function1d(pi4_to_pi3)

    @cython.cdivision(True)
    cpdef Spectrum add_line(self, double radiance, Point3D beam_point, Point3D plasma_point,
                            Vector3D beam_direction, Vector3D observation_direction, Spectrum spectrum):

        cdef double x, y, z
        cdef Plasma plasma
        cdef double te, ne, beam_energy, sigma, stark_split, beam_ion_mass, beam_temperature
        cdef double natural_wavelength, central_wavelength
        cdef double sigma_to_pi, d, intensity_sig, intensity_pi, e_field
        cdef double s1_to_s0, intensity_s0, intensity_s1
        cdef double pi2_to_pi3, pi4_to_pi3, intensity_pi2, intensity_pi3, intensity_pi4
        cdef Vector3D b_field, beam_velocity

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
        beam_velocity = beam_direction.normalise().mul(evamu_to_ms(beam_energy))
        e_field = beam_velocity.cross(b_field).get_length()
        stark_split = fabs(STARK_SPLITTING_FACTOR * e_field)  # TODO - calculate splitting factor? Reject other lines?

        # calculate emission line central wavelength, doppler shifted along observation direction
        natural_wavelength = self.wavelength
        central_wavelength = doppler_shift(natural_wavelength, observation_direction, beam_velocity)

        # calculate doppler broadening
        beam_ion_mass = self.beam.get_element().atomic_weight
        beam_temperature = self.beam.get_temperature()
        sigma = thermal_broadening(self.wavelength, beam_temperature, beam_ion_mass)

        # calculate relative intensities of sigma and pi lines
        sigma_to_pi = self._sigma_to_pi.evaluate(ne, beam_energy)
        d = 1 / (1 + sigma_to_pi)
        intensity_sig = sigma_to_pi * d * radiance
        intensity_pi = 0.5 * d * radiance

        # add Sigma lines to output
        s1_to_s0 = self._sigma1_to_sigma0.evaluate(ne)
        intensity_s0 = 1 / (s1_to_s0 + 1)
        intensity_s1 = 0.5 * s1_to_s0 * intensity_s0

        spectrum = add_gaussian_line(intensity_sig * intensity_s0, central_wavelength, sigma, spectrum)
        spectrum = add_gaussian_line(intensity_sig * intensity_s1, central_wavelength + stark_split, sigma, spectrum)
        spectrum = add_gaussian_line(intensity_sig * intensity_s1, central_wavelength - stark_split, sigma, spectrum)

        # add Pi lines to output
        pi2_to_pi3 = self._pi2_to_pi3.evaluate(ne)
        pi4_to_pi3 = self._pi4_to_pi3.evaluate(ne)
        intensity_pi3 = 1 / (1 + pi2_to_pi3 + pi4_to_pi3)
        intensity_pi2 = pi2_to_pi3 * intensity_pi3
        intensity_pi4 = pi4_to_pi3 * intensity_pi3

        spectrum = add_gaussian_line(intensity_pi * intensity_pi2, central_wavelength + 2 * stark_split, sigma, spectrum)
        spectrum = add_gaussian_line(intensity_pi * intensity_pi2, central_wavelength - 2 * stark_split, sigma, spectrum)
        spectrum = add_gaussian_line(intensity_pi * intensity_pi3, central_wavelength + 3 * stark_split, sigma, spectrum)
        spectrum = add_gaussian_line(intensity_pi * intensity_pi3, central_wavelength - 3 * stark_split, sigma, spectrum)
        spectrum = add_gaussian_line(intensity_pi * intensity_pi4, central_wavelength + 4 * stark_split, sigma, spectrum)
        spectrum = add_gaussian_line(intensity_pi * intensity_pi4, central_wavelength - 4 * stark_split, sigma, spectrum)

        return spectrum
