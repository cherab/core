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
from cherab.core.utility.constants cimport ELEMENTARY_CHARGE, ELECTRON_REST_MASS, VACUUM_PERMITTIVITY, PLANCK_CONSTANT, HC_EV_NM
from cherab.core.model.lineshape.gaussian cimport add_gaussian_line
from cherab.core.model.lineshape.doppler cimport thermal_broadening, doppler_shift

cimport cython


# (3/2 * e0 * h^2)/(2pi * Me * e^2) in [eV*m/V]
cdef double STARK_ENERGY_SPLITTING_FACTOR = 3 * VACUUM_PERMITTIVITY * PLANCK_CONSTANT**2 / (2 * M_PI * ELECTRON_REST_MASS * ELEMENTARY_CHARGE**2)

DEF PI_POLARISATION = 0
DEF SIGMA_POLARISATION = 1
DEF NO_POLARISATION = 2


cdef class BeamEmissionMultiplet(BeamLineShapeModel):
    """
    Produces Beam Emission Multiplet line shape, also known as the Motional Stark Effect spectrum.
    """

    def __init__(self, Line line, double wavelength, Beam beam, AtomicData atomic_data,
                 StarkStructure stark_structure=None, polarisation='no'):

        super().__init__(line, wavelength, beam, atomic_data)

        self._stark_structure = stark_structure or self.atomic_data.stark_structure(line)

        self.polarisation = polarisation

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
            int i, index
            double x, y, z
            Plasma plasma
            double te, ne, beam_energy, sigma, stark_split, beam_ion_mass, beam_temperature, e_magn, b_magn
            double shifted_wavelength, photon_energy
            double cos_sqr, sin_sqr, sigma_to_total, intensity
            double[:] ratios_mv
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

        # get relative ratios of Stark intensities perpendicular to electric field
        b_magn = b_field.get_length()
        ratios_mv = self._stark_structure.evaluate(beam_energy, ne, b_magn)

        # add Sigma lines to output
        if self._polarisation != PI_POLARISATION:
            sigma_to_total = 0
            for i in range(ratios_mv.shape[0]):
                if self._stark_structure.polarisation_mv[i] == SIGMA_POLARISATION:
                    sigma_to_total += 2 * ratios_mv[i] if self._stark_structure.index_mv[i] else ratios_mv[i]

            if sigma_to_total:
                intensity = (sin_sqr + cos_sqr / sigma_to_total) * radiance
                for i in range(ratios_mv.shape[0]):
                    if self._stark_structure.polarisation_mv[i] == SIGMA_POLARISATION:
                        index = self._stark_structure.index_mv[i]
                        if index == 0:
                            spectrum = add_gaussian_line(intensity * ratios_mv[i], shifted_wavelength, sigma, spectrum)
                        else:
                            spectrum = add_gaussian_line(intensity * ratios_mv[i], HC_EV_NM / (photon_energy - index * stark_split), sigma, spectrum)
                            spectrum = add_gaussian_line(intensity * ratios_mv[i], HC_EV_NM / (photon_energy + index * stark_split), sigma, spectrum)

        # add Pi lines to output
        if self._polarisation != SIGMA_POLARISATION:
            intensity = sin_sqr * radiance
            for i in range(ratios_mv.shape[0]):
                if self._stark_structure.polarisation_mv[i] == PI_POLARISATION:
                    index = self._stark_structure.index_mv[i]
                    spectrum = add_gaussian_line(intensity * ratios_mv[i], HC_EV_NM / (photon_energy - index * stark_split), sigma, spectrum)
                    spectrum = add_gaussian_line(intensity * ratios_mv[i], HC_EV_NM / (photon_energy + index * stark_split), sigma, spectrum)

        return spectrum
