from raysect.optical cimport Vector3D, Point3D
from raysect.optical.spectrum cimport Spectrum

from cherab.core cimport Plasma
from cherab.core.laser.node cimport Laser
from cherab.core.laser.models.profile_base cimport LaserProfile
from cherab.core.utility.constants cimport DEGREES_TO_RADIANS, ATOMIC_MASS, RECIP_4_PI
from cherab.core.laser.models.laserspectrum_base cimport LaserSpectrum
from cherab.core.utility.constants cimport PLANCK_CONSTANT, SPEED_OF_LIGHT, ELECTRON_CLASSICAL_RADIUS, ELECTRON_REST_MASS, ELEMENTARY_CHARGE

from libc.math cimport exp, sqrt, cos, M_PI
cimport cython


cdef class LaserModel:

    def __init__(self, Laser laser):

        self.laser = laser

    cpdef Spectrum emission(self, Point3D point_plasma, Vector3D observation_plasma, Point3D point_laser, Vector3D observation_laser,
                            Spectrum spectrum):

        raise NotImplementedError('Virtual method must be implemented in a sub-class.')

    @property
    def laser_profile(self):
        return self._laser_profile

    @laser_profile.setter
    def laser_profile(self, LaserProfile value):
        self._laser_profile = value

    @property
    def plasma(self):
        return self._plasma

    @plasma.setter
    def plasma(self, Plasma value):
        self._plasma = value

    @property
    def laser_spectrum(self):
        return self._laser_spectrum

    @laser_spectrum.setter
    def laser_spectrum(self, LaserSpectrum value):

        self._laser_spectrum = value

cdef class SeldenMatobaThomsonSpectrum(LaserModel):

    def __init__(self):
        # Selden, A.C., 1980. Simple analytic form of the relativistic Thomson scattering spectrum. Physics Letters A, 79(5-6), pp.405-406.
        self._CONST_ALPHA = ELECTRON_REST_MASS * SPEED_OF_LIGHT ** 2 / (2 * ELEMENTARY_CHARGE)  #constant alpha, rewritten for Te in eV
        
        # from: Prunty, S. L. "A primer on the theory of Thomson scattering for high-temperature fusion plasmas."
        # Thomson scattering reaction rate from Prunty eq. (4.39)
        # speed of light for correct normalisation of the scattered intensity calculation (from x-section to rate constant)
        self._CONST_TS = ELECTRON_CLASSICAL_RADIUS ** 2 * SPEED_OF_LIGHT


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
            double[::1] laser_wavelength_mv, laser_spectrum_power_mv
            int bins
            Vector3D pointing_vector
            Py_ssize_t index

        # get electron parameters for the plasma point
        te = self._plasma.get_electron_distribution().effective_temperature(point_plasma.x, point_plasma.y, point_plasma.z)
        ne = self._plasma.get_electron_distribution().density(point_plasma.x, point_plasma.y, point_plasma.z)
        
        #terminate early if electron density is 0
        if ne == 0:
            return spectrum
        #get laser volumetric power
        laser_energy_density = self._laser_profile.get_energy_density(point_laser.x, point_laser.y, point_laser.z)

        #terminate early if laser power is 0
        if laser_energy_density == 0:
            return spectrum

        pointing_vector = self._laser_profile.get_pointing(point_laser.x, point_laser.y, point_laser.z)

        #angle between observation and pointing vector
        angle_pointing = observation_laser.angle(pointing_vector)  # angle between observation and pointing vector of laser

        angle_scattering = (180. - angle_pointing)  # scattering direction is the opposite to obervation direction

        angle_polarization = 90.

        laser_wavelength_mv = self._laser_spectrum._wavelengths_mv
        laser_spectrum_power_mv = self._laser_spectrum._power_mv  # power in spectral bins (PSD * delta wavelength)
        bins = self._laser_spectrum._bins

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

        alpha = self._CONST_ALPHA / te
        # scattering angle of the photon = pi - observation_angle
        cos_anglescat = cos(angle_scattering * DEGREES_TO_RADIANS)
        
        # pre-calculate constants for Selden-Matoba shape 
        const_theta = 2 * (1 - cos_anglescat)

        nbins = spectrum.bins
        min_wavelength = spectrum.min_wavelength
        delta_wavelength = spectrum.delta_wavelength
        recip_laser_wavelength = 1 / laser_wavelength

        #from d_lambda to d_epsilon:d_epsilon = d_lambda / laser_wavelength
        scattered_power = ne * self._CONST_TS * laser_energy * recip_laser_wavelength

        for index in range(nbins):
            wavelength = min_wavelength + (0.5 + index) * delta_wavelength
            epsilon = (wavelength * recip_laser_wavelength) - 1
            spectrum_norm = self.seldenmatoba_spectral_shape(epsilon, const_theta, alpha)
            spectrum.samples_mv[index] += spectrum_norm * scattered_power

        return spectrum

    cpdef Spectrum calculate_spectrum(self, double ne, double te, double laser_energy_density, double laser_wavelength,
                                      double observation_angle, Spectrum spectrum):

        # check for nonzero laser power, ne, te, wavelength
        if not ne > 0 or not te > 0 or not laser_energy_density > 0:
            return spectrum
        if not laser_wavelength >= 0:
            raise ValueError("laser wavelength has to be larger than 0")

        angle_scattering = (180. - observation_angle)  # scattering direction is the opposite to obervation direction
        angle_polarisation = 90.

        return self._add_spectral_contribution(ne, te, laser_energy_density, angle_scattering, angle_polarisation, laser_wavelength, spectrum)

    