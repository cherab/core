from raysect.optical cimport Vector3D, Point3D
from raysect.optical.spectrum cimport Spectrum

from cherab.core.utility.constants cimport DEGREES_TO_RADIANS, ATOMIC_MASS, RECIP_4_PI
from cherab.core.utility.constants cimport PLANCK_CONSTANT, SPEED_OF_LIGHT, ELECTRON_CLASSICAL_RADIUS, ELECTRON_REST_MASS, ELEMENTARY_CHARGE
from cherab.core.laser.node cimport Laser
from cherab.core.laser.models.model_base cimport LaserModel
from cherab.core.laser.models.laserspectrum_base cimport LaserSpectrum
from cherab.core cimport Plasma

cimport cython
from libc.math cimport exp, sqrt, cos, M_PI

cdef double RE_SQUARED = ELECTRON_CLASSICAL_RADIUS ** 2  # cross section for thomson scattering for SeldenMatoba
cdef double E_TO_NPHOT = 10e-9 / (PLANCK_CONSTANT * SPEED_OF_LIGHT)  # N_photons(wlen, E_laser) = E_laser * wlen * E_TO_PHOT in [J, nm]
cdef double NPHOT_TO_E = 1 / E_TO_NPHOT  # E(wlen, N_photons) = n_photons/wlen * NPHOT_TO_E in [nm, J]


cdef class ScatteringModel:

    cpdef Spectrum emission(self, Point3D position_plasma, Point3D position_laser, Vector3D direction_observation, Spectrum spectrum):

        raise NotImplementedError('Virtual method must be implemented in a sub-class.')

    @property
    def laser(self):
        """
        link to Laser node the cattering model can be attached to.
        :return:
        """
        return self._laser

    @laser.setter
    def laser(self, Laser value):

        self._laser = value
        self._laser.notifier.add(self._laser_changed)

        self._laser_changed()

    @property
    def plasma(self):
        return self._plasma

    @plasma.setter
    def plasma(self, value):

        if not isinstance(value, Plasma):
            raise TypeError("Value has to be of type Plasma, but {0} passed".format(type(value)))

        self._plasma = value

    @property
    def laser_model(self):
        return self._laser_model

    @laser_model.setter
    def laser_model(self, LaserModel value):

        self._laser_model = value

    def set_laser_spectrum(self, LaserSpectrum value):
        # unregister callback
        if self._laser_spectrum is not None:
            self._laser_spectrum._notifier.remove(self._laser_spectrum_changed)

        self._laser_spectrum = value
        self._laser_spectrum._notifier.add(self._laser_spectrum_changed)
        self._laser_spectrum_changed()

    def _laser_spectrum_changed(self):
        self._laser_bins = self._laser_spectrum._bins
        self._laser_wavelength_mv = self._laser_spectrum._wavelengths_mv
        self._laser_power_mv = self._laser_spectrum._power_mv

cdef class SeldenMatobaThomsonSpectrum(ScatteringModel):

    def __init__(self, Laser laser=None, LaserModel laser_models=None, Plasma plasma=None):
        # from article: 2 * alpha = m_e * c **2 /(k * T_e), here rewritten for Te in eV
        self._CONST_ALPHA = ELECTRON_REST_MASS * SPEED_OF_LIGHT ** 2 / (2 * ELEMENTARY_CHARGE)
        if laser is not None:
            self.laser = laser
        else:
            if laser_models is not None:
                self.laser_model = laser_models
            if plasma is not None:
                self.plasma = plasma

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
    cpdef Spectrum emission(self, Point3D position_plasma, Point3D position_laser, Vector3D direction_observation,
                            Spectrum spectrum):

        cdef:
            double ne, te, laser_power_density, angle_pointing, angle_polarization, laser_wavelength
            Vector3D pointing_vector, polarization_vector
            Spectrum laser_spectrum
            Py_ssize_t index
        
        ne = self._plasma.get_electron_distribution().density(position_plasma.x, position_plasma.y, position_plasma.z)
        te = self._plasma.get_electron_distribution().effective_temperature(position_plasma.x, position_plasma.y, position_plasma.z)

        # Pointing vector and angle between observation and pointing vector
        pointing_vector = self._laser_model.get_pointing(position_laser.x, position_laser.y, position_laser.z)
        angle_pointing = direction_observation.angle(pointing_vector)

        angle_polarization = 90.
        # todo: uncomment if influence of polarization angle is verified
        # Polarization vector (Vector of electric field of the laser) and angle between observation and polarisation
        # polarization_vector = self._laser_model.get_polarization(position_laser.x, position_laser.y, position_laser.z)
        # angle_polarization = direction_observation.angle(pointing_vector)

        # no scattering contribution cases
        if ne <= 0 or te <= 0:
            # print("ne = {0} or te = {1}".format(ne, te))
            return spectrum
        for index in range(self._laser_bins):
            laser_power_density = self._laser_power_mv[index] * self._laser_model.get_power_density(position_laser.x, position_laser.y, position_laser.z)

            if laser_power_density > 0:
                spectrum = self.add_spectral_contribution(ne, te, laser_power_density,
                                                          angle_pointing, angle_polarization, self._laser_wavelength_mv[index], spectrum)

        return spectrum



    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef Spectrum add_spectral_contribution(self, double ne, double te, double laser_power, double angle_pointing,
                                            double angle_polarization, double laser_wavelength, Spectrum spectrum):

        cdef:
            int index, nbins
            double alpha, epsilon, cos_scatangle, wavelength, min_wavelength, delta_wavelength
            # double sin2_polarisation

        alpha = self._CONST_ALPHA / te
        nbins = spectrum.bins

        # scattering angle of the photon = pi - observation_angle
        cos_scatangle = cos((180 - angle_pointing) * DEGREES_TO_RADIANS)

        # todo: verify that angle between observation and polarization influences only cross section of
        # scattering by sin(angle)**2 and does not influence spectrum shape. If yes, calculate sin2_polarisation and
        # multiply scattered spectral power density with it
        # sin2_polarisation = sin(angle_polarization * DEGREES_TO_RADIANS) ** 2 #sin2 of observation to polarisation

        min_wavelength = spectrum.min_wavelength
        delta_wavelength = spectrum.delta_wavelength

        # sin2 of angle_polarisation takes into account dipole nature (sin2) of thomson scattering radiation of the scattered wave
        for index in range(nbins):
            wavelength = (min_wavelength + delta_wavelength * index)
            epsilon = (wavelength - laser_wavelength) / laser_wavelength
            spectrum_norm = self.seldenmatoba_spectral_shape(epsilon, cos_scatangle, alpha)
            spectrum.samples_mv[index] += spectrum_norm * ne * RE_SQUARED * laser_power / delta_wavelength

        return spectrum

    cpdef Spectrum calculate_spectrum(self, double ne, double te, double laser_power_density, double angle_pointing,
                                      double angle_polarization, Spectrum laser_spectrum, Spectrum spectrum):
        # todo: make this function return spectrum without the need of plasma and laser model.
        # check for nonzero laser power, ne, te, wavelength

        if not ne > 0 or not te > 0 or not laser_power_density > 0:
            return spectrum

        cdef double laser_wavelength

        for index in range(laser_spectrum.bins):
            laser_wavelength = self._laser_model.laser_spectrum.wavelengths[index]

            if laser_power_density > 0:
                spectrum = self.add_spectral_contribution(ne, te, laser_power_density,
                                                          angle_pointing, angle_polarization, laser_wavelength, spectrum)

        return spectrum

    def _laser_changed(self):

        self._laser_model = self._laser._laser_model
        self._plasma = self._laser._plasma
