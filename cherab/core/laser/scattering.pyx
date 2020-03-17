from raysect.optical cimport Vector3D, Point3D
from raysect.optical.spectrum cimport Spectrum

from cherab.core.utility.constants cimport DEGREES_TO_RADIANS, ATOMIC_MASS, RECIP_4_PI
from cherab.core.utility.constants cimport PLANCK_CONSTANT, SPEED_OF_LIGHT, ELECTRON_CLASSICAL_RADIUS, ELECTRON_REST_MASS, ELEMENTARY_CHARGE
from cherab.core.laser.node cimport Laser
from cherab.core.laser.models.model_base cimport LaserModel
from cherab.core.laser.models.laserspectrum_base cimport LaserSpectrum
from cherab.core cimport Plasma
from cherab.core.utility import Notifier

from libc.math cimport exp, sqrt, cos, M_PI
cimport cython


cdef class ScatteringModel:

    cpdef Spectrum emission(self, double ne, double te, double laser_power_density,
                            double laser_wavelength, Vector3D direction_observation,
                            Vector3D pointing_vector, Vector3D polarization_vector, Spectrum spectrum):

        raise NotImplementedError('Virtual method must be implemented in a sub-class.')


cdef class SeldenMatobaThomsonSpectrum(ScatteringModel):

    def __init__(self):
        # from: Prunty, S. L. "A primer on the theory of Thomson scattering for high-temperature fusion plasmas."
        # Physica Scripta 89.12 (2014): 128001.
        self._CONST_ALPHA = ELECTRON_REST_MASS * SPEED_OF_LIGHT ** 2 / (2 * ELEMENTARY_CHARGE)  # rewritten for Te in eV
        self._CONST_TS = 2 / 3 * ELECTRON_CLASSICAL_RADIUS ** 2  # rewritten per solid angle

    @cython.cdivision(True)
    cdef double seldenmatoba_spectral_shape(self, double epsilon, double cos_theta, double alpha):

        cdef:
            double c, a, b

        c = sqrt(alpha / M_PI) * (1 - 15. / (16. * alpha) + 345. / (512. * alpha ** 2))
        a = (1 + epsilon) ** 3 * sqrt(2 * (1 - cos_theta) * (1 + epsilon) + epsilon ** 2)
        b = sqrt(1 + epsilon ** 2 / (2 * (1 - cos_theta) * (1 + epsilon))) - 1

        return c / a * exp(-2 * alpha * b)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef Spectrum emission(self, double ne, double te, double laser_power_density,
                            double laser_wavelength, Vector3D direction_observation,
                            Vector3D pointing_vector, Vector3D polarization_vector, Spectrum spectrum):
        cdef:
            double angle_scattering, angle_pointing, angle_polarization

        #angle between observation and pointing vector
        angle_pointing = direction_observation.angle(pointing_vector)  # angle between observation and pointing vector of laser

        angle_scattering = (180 - angle_pointing)  # scattering direction is the opposite to obervation direction

        angle_polarization = 90.
        # todo: uncomment if influence of polarization angle is verified
        # Polarization vector (Vector of electric field of the laser) and angle between observation and polarisation
        # polarization_vector = self._laser_model.get_polarization(position_laser.x, position_laser.y, position_laser.z)
        # angle_polarization = direction_observation.angle(pointing_vector)

        # no scattering contribution cases
        spectrum = self._add_spectral_contribution(ne, te, laser_power_density, angle_scattering,
                                                   angle_polarization, laser_wavelength, spectrum)

        return spectrum

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef Spectrum _add_spectral_contribution(self, double ne, double te, double laser_power, double angle_scattering,
                                             double angle_polarization, double laser_wavelength, Spectrum spectrum):

        cdef:
            int index, nbins
            double alpha, epsilon, cos_anglescat, wavelength, min_wavelength, delta_wavelength
            # double sin2_polarisation

        alpha = self._CONST_ALPHA / te

        # scattering angle of the photon = pi - observation_angle
        cos_anglescat = cos(angle_scattering * DEGREES_TO_RADIANS)

        # todo: verify that angle between observation and polarization influences only cross section of
        # scattering by sin(angle)**2 and does not influence spectrum shape. If yes, calculate sin2_polarisation and
        # multiply scattered spectral power density with it
        # sin2_polarisation = sin(angle_polarization * DEGREES_TO_RADIANS) ** 2 #sin2 of observation to polarisation

        nbins = spectrum.bins
        min_wavelength = spectrum.min_wavelength
        delta_wavelength = spectrum.delta_wavelength
        # sin2 of angle_polarisation takes into account dipole nature (sin2) of thomson scattering radiation of the scattered wave

        for index in range(nbins):
            wavelength = min_wavelength + (0.5 + index) * delta_wavelength
            epsilon = (wavelength / laser_wavelength) - 1
            spectrum_norm = self.seldenmatoba_spectral_shape(epsilon, cos_anglescat, alpha)
            spectrum.samples_mv[index] += spectrum_norm * ne * self._CONST_TS * laser_power / delta_wavelength

        return spectrum

    cpdef Spectrum calculate_spectrum(self, double ne, double te, double laser_power_density, double laser_wavelength,
                                      Vector3D direction_observation, Vector3D pointing_vector, Vector3D polarization_vector, Spectrum spectrum):

        # check for nonzero laser power, ne, te, wavelength
        if not ne > 0 or not te > 0 or not laser_power_density > 0:
            return spectrum
        if not laser_wavelength >= 0:
            raise ValueError("laser wavelength has to be larger than 0")

        return self.emission(ne, te, laser_power_density, laser_wavelength, direction_observation,
                             pointing_vector, polarization_vector, spectrum)
