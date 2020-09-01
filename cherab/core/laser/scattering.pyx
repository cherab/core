from raysect.optical cimport Vector3D, Point3D
from raysect.optical.spectrum cimport Spectrum

from cherab.core cimport Plasma
from cherab.core.laser.node cimport Laser
from cherab.core.laser.models.model_base cimport LaserModel
from cherab.core.utility.constants cimport DEGREES_TO_RADIANS, ATOMIC_MASS, RECIP_4_PI
from cherab.core.laser.models.laserspectrum_base cimport LaserSpectrum
from cherab.core.utility.constants cimport PLANCK_CONSTANT, SPEED_OF_LIGHT, ELECTRON_CLASSICAL_RADIUS, ELECTRON_REST_MASS, ELEMENTARY_CHARGE

from libc.math cimport exp, sqrt, cos, M_PI
cimport cython


cdef class LaserEmissionModel:

    def __init__(self, Laser laser):

        self.laser = laser

    cpdef Spectrum emission(self, Point3D point_plasma, Vector3D observation_plasma, Point3D point_laser, Vector3D observation_laser,
                            Spectrum spectrum):

        raise NotImplementedError('Virtual method must be implemented in a sub-class.')

    @property
    def laser_model(self):
        return self._laser_model

    @laser_model.setter
    def laser_model(self, LaserModel value):
        self._laser_model = value

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

cdef class SeldenMatobaThomsonSpectrum(LaserEmissionModel):

    def __init__(self):
        # from: Prunty, S. L. "A primer on the theory of Thomson scattering for high-temperature fusion plasmas."
        # Physica Scripta 89.12 (2014): 128001.
        self._CONST_ALPHA = ELECTRON_REST_MASS * SPEED_OF_LIGHT ** 2 / (2 * ELEMENTARY_CHARGE)  # rewritten for Te in eV
        self._CONST_TS = 2 / 3 * ELECTRON_CLASSICAL_RADIUS ** 2  # rewritten per solid angle
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
            double te, ne, laser_power_density
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
        laser_volumetric_power = self._laser_model.get_power_density(point_laser.x, point_laser.y, point_laser.z)

        #terminate early if laser power is 0
        if laser_volumetric_power == 0:
            return spectrum

        pointing_vector = self._laser_model.get_pointing(point_laser.x, point_laser.y, point_laser.z)

        #angle between observation and pointing vector
        angle_pointing = observation_laser.angle(pointing_vector)  # angle between observation and pointing vector of laser

        angle_scattering = (180. - angle_pointing)  # scattering direction is the opposite to obervation direction

        angle_polarization = 90.

        laser_wavelength_mv = self._laser_spectrum._wavelengths_mv
        laser_spectrum_power_mv = self._laser_spectrum._power_mv  # power in spectral bins (PSD * delta wavelength)
        bins = self._laser_spectrum._bins

        for index in range(bins):
            laser_power_density = laser_spectrum_power_mv[index] * laser_volumetric_power 
            if laser_power_density > 0:
                spectrum = self._add_spectral_contribution(ne, te, laser_power_density, angle_scattering,
                                                           angle_polarization, laser_wavelength_mv[index], spectrum)

        return spectrum

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef Spectrum _add_spectral_contribution(self, double ne, double te, double laser_power, double angle_scattering,
                                             double angle_polarization, double laser_wavelength, Spectrum spectrum):

        cdef:
            int index, nbins
            double alpha, epsilon, cos_anglescat, wavelength, min_wavelength, delta_wavelength, recip_delta_wavelength
            double const_theta_epsilon, recip_laser_wavelength

        alpha = self._CONST_ALPHA / te

        # scattering angle of the photon = pi - observation_angle
        cos_anglescat = cos(angle_scattering * DEGREES_TO_RADIANS)
        
        # pre-calculate constants for Selden-Matoba shape 
        const_theta = 2 * (1 - cos_anglescat)

        nbins = spectrum.bins
        min_wavelength = spectrum.min_wavelength
        delta_wavelength = spectrum.delta_wavelength
        recip_delta_wavelength = 1 / delta_wavelength
        recip_laser_wavelength = 1 / laser_wavelength

        for index in range(nbins):
            wavelength = min_wavelength + (0.5 + index) * delta_wavelength
            epsilon = (wavelength * recip_laser_wavelength) - 1
            spectrum_norm = self.seldenmatoba_spectral_shape(epsilon, const_theta, alpha)
            spectrum.samples_mv[index] += spectrum_norm * ne * self._CONST_TS * laser_power * recip_delta_wavelength

        return spectrum

    cpdef Spectrum calculate_spectrum(self, double ne, double te, double laser_power_density, double laser_wavelength,
                                      double observation_angle, Spectrum spectrum):

        # check for nonzero laser power, ne, te, wavelength
        if not ne > 0 or not te > 0 or not laser_power_density > 0:
            return spectrum
        if not laser_wavelength >= 0:
            raise ValueError("laser wavelength has to be larger than 0")

        angle_scattering = (180. - observation_angle)  # scattering direction is the opposite to obervation direction
        angle_polarisation = 90.

        return self._add_spectral_contribution(ne, te, laser_power_density, angle_scattering, angle_polarisation, laser_wavelength, spectrum)

    