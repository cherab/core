# Copyright 2016-2021 Euratom
# Copyright 2016-2021 United Kingdom Atomic Energy Authority
# Copyright 2016-2021 Centro de Investigaciones Energéticas, Medioambientales y Tecnológicas
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


from libc.math cimport exp, sqrt, cos, M_PI, sin
cimport cython

from raysect.optical cimport Vector3D, Point3D
from raysect.optical.spectrum cimport Spectrum

from cherab.core cimport Plasma
from cherab.core.laser cimport LaserModel, LaserProfile, LaserSpectrum
from cherab.core.utility.constants cimport DEGREES_TO_RADIANS
from cherab.core.utility.constants cimport SPEED_OF_LIGHT, ELECTRON_CLASSICAL_RADIUS, ELECTRON_REST_MASS, ELEMENTARY_CHARGE


cdef class SeldenMatobaThomsonSpectrum(LaserModel):
    r"""
    Thomson Scattering based on Selden-Matoba.

    The class calculates Thomson scattering of the laser to the spectrum. The model of the scattered spectrum used is based on
    the semi-empirical model by Selden and the Thomson scattering cross-section is taken from Matoba articles. The spectral contribution
    of the scattered laser light c is calculated as a sum of contributions of all laser wavelengths

    .. math::
         c(\lambda) =  c r_e^2 n_e cos^2\\theta \\sum_{\\lambda_L} \\frac{E_L(\\lambda_l) S(\\frac{\\lambda}{\\lambda_L} - 1, \\varphi, T_e)}{\\lambda_L},
    

    where :math:`\\lambda` is the spectrum's wavelength, :math:`r_e` is the classical electron radius, :math:`n_e` is the electron delsity,
    :math:`\\theta` is the angle between the laser polarisation and scattering vectors, :math:`c` is the vacuum speed of light
    :math:`\\lambda_L` is the laser wavelength, :math:`E_L` is the laser energy density, :math:`\\varphi` is the scattering angle and :math:`T_e` is the electron
    temperature. The scattering function S is taken from the Matoba article. The multiplication by the speed of light is added to transfer the Thomson scattering
    cross section into a reaction rate.

    .. seealso::
         The Prunty article provides a thorough introduction into the phyiscs of Thomson scattering. The articles by Selden and Matoba were used to build
         this model.
         
         :Selden: `Selden, A.C., 1980. Simple analytic form of the relativistic Thomson scattering spectrum. Physics Letters A, 79(5-6), pp.405-406.`
         :Matoba: `Matoba, T., et al., 1979. Analytical approximations in the theory of relativistic Thomson scattering for high temperature fusion plasma.
                  Japanese Journal of Applied Physics, 18(6), p.1127.`
         :Prunty: `Prunty, S.L., 2014. A primer on the theory of Thomson scattering for high-temperature fusion plasmas. Physica Scripta, 89(12), p.128001.`

    """

    def __init__(self, LaserProfile laser_profile=None, LaserSpectrum laser_spectrum=None, Plasma plasma=None):

        super().__init__(laser_profile, laser_spectrum, plasma)

        # Selden, A.C., 1980. Simple analytic form of the relativistic Thomson scattering spectrum. Physics Letters A, 79(5-6), pp.405-406.
        self._CONST_ALPHA = ELECTRON_REST_MASS * SPEED_OF_LIGHT ** 2 / (2 * ELEMENTARY_CHARGE)  #constant alpha, rewritten for Te in eV
        
        # from: Prunty, S. L. "A primer on the theory of Thomson scattering for high-temperature fusion plasmas."
        # TS cross section equiation ~ 3.28 or
        # Matoba, T., et al., 1979. Analytical approximations in the theory of relativistic Thomson scattering for high temperature fusion plasma.
        # Japanese Journal of Applied Physics, 18(6), p.1127., TS cross section equiation 18 
        # speed of light for correct normalisation of the scattered intensity calculation (from x-section to rate constant)
        self._RATE_TS = ELECTRON_CLASSICAL_RADIUS ** 2 * SPEED_OF_LIGHT

        self._RECIP_M_PI = 1 / M_PI

    @cython.cdivision(True)
    cdef double seldenmatoba_spectral_shape(self, double epsilon, double const_theta, double alpha):

        cdef:
            double c, a, b

        # const_theta is 2 * (1 - cos(theta))

        c = sqrt(alpha * self._RECIP_M_PI) * (1 - 15. / (16. * alpha) + 345. / (512. * alpha ** 2))
        a = (1 + epsilon) ** 3 * sqrt(const_theta * (1 + epsilon) + epsilon ** 2)
        b = sqrt(1 + epsilon ** 2 / (const_theta * (1 + epsilon))) - 1

        return c / a * exp(-2 * alpha * b)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef Spectrum emission(self, Point3D point_plasma, Vector3D observation_plasma, Point3D point_laser,
                            Vector3D observation_laser, Spectrum spectrum):
        cdef:
            double angle_scattering, angle_pointing, angle_polarization
            double te, ne, laser_energy_density, laser_energy
            double plasma_x, plasma_y, plasma_z, laser_x, laser_y, laser_z
            double[::1] laser_wavelength_mv, laser_spectrum_power_mv
            int bins
            Vector3D pointing_vector, polarisation_vector
            Py_ssize_t index

        plasma_x = point_plasma.x
        plasma_y = point_plasma.y
        plasma_z = point_plasma.z

        # get electron parameters for the plasma point
        te = self._plasma.get_electron_distribution().effective_temperature(plasma_x, plasma_y, plasma_z)
        
        #terminate early if electron temperature is 0
        if te <= 0:
            return spectrum
        
        ne = self._plasma.get_electron_distribution().density(plasma_x, plasma_y, plasma_z)
        
        #terminate early if electron density is 0
        if ne <= 0:
            return spectrum

        laser_x = point_laser.x
        laser_y = point_laser.y
        laser_z = point_laser.z

        #get laser volumetric power
        laser_energy_density = self._laser_profile.get_energy_density(laser_x, laser_y, laser_z)

        #terminate early if laser power is 0
        if laser_energy_density == 0:
            return spectrum

        pointing_vector = self._laser_profile.get_pointing(laser_x, laser_y, laser_z)

        #angle between observation and pointing vector
        angle_pointing = observation_laser.angle(pointing_vector)  # angle between observation and pointing vector of laser

        angle_scattering = (180. - angle_pointing)  # scattering direction is the opposite to obervation direction

        # angle between polarisation and observation
        polarisation_vector = self._laser_profile.get_polarization(laser_x, laser_y, laser_z)
        angle_polarization = observation_laser.angle(polarisation_vector) # scattering direction is the opposite to obervation direction

        laser_wavelength_mv = self._laser_spectrum.wavelengths_mv
        laser_spectrum_power_mv = self._laser_spectrum.power_mv  # power in spectral bins (PSD * delta wavelength)
        bins = self._laser_spectrum.get_spectral_bins()

        for index in range(bins):
            laser_energy = laser_spectrum_power_mv[index] * laser_energy_density 
            if laser_energy > 0:
                spectrum = self._add_spectral_contribution(ne, te, laser_energy, angle_scattering,
                                                           angle_polarization, laser_wavelength_mv[index], spectrum)

        return spectrum

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef Spectrum _add_spectral_contribution(self, double ne, double te, double laser_energy, double angle_scattering,
                                             double angle_polarization, double laser_wavelength, Spectrum spectrum):

        cdef:
            int index, nbins
            double alpha, epsilon, cos_anglescat, wavelength, min_wavelength, delta_wavelength
            double const_theta, recip_laser_wavelength, scattered_power, spectrum_norm
            double sin2_angle_pol

        alpha = self._CONST_ALPHA / te
        # scattering angle of the photon = pi - observation_angle
        cos_anglescat = cos(angle_scattering * DEGREES_TO_RADIANS)
        
        # pre-calculate constants for Selden-Matoba shape 
        const_theta = 2 * (1 - cos_anglescat)

        nbins = spectrum.bins
        min_wavelength = spectrum.min_wavelength
        delta_wavelength = spectrum.delta_wavelength
        recip_laser_wavelength = 1 / laser_wavelength

        # dipole radiation has a cos ** 2 characteristic, here angle shifted by 90 deg
        sin2_angle_pol = sin(angle_polarization * DEGREES_TO_RADIANS) ** 2

        #from d_lambda to d_epsilon:d_epsilon = d_lambda / laser_wavelength
        scattered_power = ne * self._RATE_TS * laser_energy * recip_laser_wavelength * sin2_angle_pol
        for index in range(nbins):
            wavelength = min_wavelength + (0.5 + index) * delta_wavelength
            epsilon = (wavelength * recip_laser_wavelength) - 1
            spectrum_norm = self.seldenmatoba_spectral_shape(epsilon, const_theta, alpha)
            spectrum.samples_mv[index] += spectrum_norm * scattered_power

        return spectrum

    cpdef Spectrum calculate_spectrum(self, double ne, double te, double laser_energy_density, double laser_wavelength,
                                      double observation_angle, double angle_polarization, Spectrum spectrum):
        """
        Calculates scattered spectrum for the given parameters.
        
        The method returns the Thomson scattered spectrum given the plasma parameters, without the need of specifying
        plasma or laser.

        :param float ne: Plasma electron density in m**-3
        :param float te: Plasma electron temperature in eV
        :param float laser_energy_density: Energy density of the laser light in J * m**-3
        :param float laser_wavelength: The laser light wavelength in nm
        :param float observation_angle: The angle of observation is the angle between the observation direction and the direction
                                        of the Poynting vector.
        :param float angle_polarization: The angle between the observation direction and the polarisation direction of the laser light.
        
        :return: Spectrum
        """
        # check for nonzero laser power, ne, te, wavelength
        if ne <= 0 or te <= 0 or not laser_energy_density > 0:
            return spectrum
        if laser_wavelength <= 0:
            raise ValueError("laser wavelength has to be larger than 0")

        angle_scattering = (180. - observation_angle)  # scattering direction is the opposite to obervation direction

        return self._add_spectral_contribution(ne, te, laser_energy_density, angle_scattering, angle_polarization, laser_wavelength, spectrum)

    