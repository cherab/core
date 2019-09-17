from raysect.optical.spectrum cimport Spectrum
from cherab.core.model.lineshape import add_gaussian_line, thermal_broadening
from cherab.core.utility.constants cimport DEGREES_TO_RADIANS, ATOMIC_MASS, RECIP_4_PI
from cherab.core.model.lineshape cimport thermal_broadening, add_gaussian_line
from cherab.core.utility.constants cimport PLANCK_CONSTANT, SPEED_OF_LIGHT, ELECTRON_CLASSICAL_RADIUS, ELECTRON_REST_MASS, ELEMENTARY_CHARGE
from cherab.core.laser.node cimport Laser
from cherab.core.laser.model cimport LaserModel
from cherab.core cimport Plasma

cimport cython
from libc.math cimport exp, sqrt, sin, cos, M_PI as pi

cdef double RE_SQUARED = ELECTRON_CLASSICAL_RADIUS ** 2 #cross section for thomson scattering for SeldenMatoba
cdef double E_TO_NPHOT = 10e-9 / (PLANCK_CONSTANT * SPEED_OF_LIGHT) # N_photons(wlen, E_laser) = E_laser * wlen * E_TO_PHOT in [J, nm]
cdef double NPHOT_TO_E = 1/ E_TO_NPHOT # E(wlen, N_photons) = n_photons/wlen * NPHOT_TO_E in [nm, J]


cdef class ScatteringModel:

    cpdef Spectrum emission(self, Point3D position_plasma, Point3D position_laser, Vector3D direction_observation, Spectrum spectrum):

        raise NotImplementedError('Virtual method must be implemented in a sub-class.')

cdef class SeldenMatobaThomsonSpectrum(ScatteringModel):

    def __init__(self, Laser laser=None, LaserModel laser_models=None, Plasma plasma=None):
        # from article: 2 * alpha = m_e * c **2 /(k * T_e), here rewritten for Te in eV
        self._CONST_ALPHA = ELECTRON_REST_MASS * SPEED_OF_LIGHT ** 2 / ( 2 * ELEMENTARY_CHARGE)

        if laser is not None:
            self.laser = laser
        else:
            if laser_models is not None:
                self.laser_model = laser_models
            if plasma is not None:
                self.plasma = plasma


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

    @cython.cdivision(True)
    cdef double seldenmatoba_spectral_shape(self, epsilon, cos_theta, alpha):

        cdef:
            double c, a, b, bin

        c = sqrt(alpha / pi) * (1 - 15. / (16. * alpha) + 345. / (512. * alpha ** 2))
        a = (1 + epsilon) ** 3 * sqrt(2 * (1 - cos_theta) * (1 + epsilon) + epsilon ** 2)
        b = sqrt(1 + epsilon ** 2 / (2 * (1 - cos_theta) * (1 + epsilon))) - 1

        bin = c / a * exp(-2 * alpha * b)

        return bin

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef Spectrum emission(self, Point3D position_plasma, Point3D position_laser, Vector3D direction_observation,
                            Spectrum spectrum):

        cdef:
            double ne, te, laser_power_density, angle_pointing, angle_polarization, laser_wavelength
            Vector3D pointing_vector, polarization_vector
            Spectrum laser_spectrum


        ne = self._plasma.get_electron_distribution().density(position_plasma.x, position_plasma.y, position_plasma.z)
        te = self._plasma.get_electron_distribution().effective_temperature(position_plasma.x, position_plasma.y, position_plasma.z)

        laser_power_density = self._laser_model.get_power_density(position_laser.x, position_laser.y, position_laser.z)
        #Pointing vector and angle between observation and pointing vector
        pointing_vector = self._laser_model.get_pointing(position_laser.x, position_laser.y, position_laser.z)
        angle_pointing = direction_observation.angle(pointing_vector)

        angle_polarization = 90.
        #todo: uncomment if influence of polarization angle is verified
        #Polarization vector (Vector of electric field of the laser) and angle between observation and polarisation
        #polarization_vector = self._laser_model.get_polarization(position_laser.x, position_laser.y, position_laser.z)
        #angle_polarization = direction_observation.angle(pointing_vector)

        #connected model takes laser spectrum as infinitely thin spectral line with position equivalent to the wavelength of the maximum of the laser spectrum
        laser_spectrum = self._laser_model.laser_spectrum
        laser_wavelength = laser_spectrum.wavelengths[laser_spectrum.samples.argmax()]
        spectrum = self.add_spectral_contribution(ne, te, laser_power_density, angle_pointing, angle_polarization, laser_wavelength, spectrum)

        return spectrum



    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef Spectrum add_spectral_contribution(self, double ne, double te, double laser_power_density, double angle_pointing,
                                             double angle_polarization, double laser_wavelength, Spectrum spectrum):

        cdef:
            double alpha, epsilon, min_wavelength, cos_observation, sin2_polarisation, wavelength
            int index

        #no scattering contribution cases
        if ne <= 0 or te <=0 or laser_power_density <=0 or laser_wavelength <=0:
            return spectrum

        alpha = self._CONST_ALPHA / te
        nbins = spectrum.bins
        cos_observation = cos(angle_pointing * DEGREES_TO_RADIANS)
        #todo: verify that angle between observation and polarization influences only cross section of
        # scattering by sin(angle)**2 and does not influence spectrum shape. If yes, calculate sin2_polarisation and
        # multiply photons_persec with it
        #sin2_polarisation = sin(angle_polarization * DEGREES_TO_RADIANS) ** 2 #sin2 of observation to polarisation
        min_wavelength = spectrum.min_wavelength


        #photon density per second
        #sin2 of angle_polarisation takes into account dipole nature (sin2) of thomson scattering radiation of the scattered wave
        photons_persec = ne * RE_SQUARED * laser_power_density * laser_wavelength * E_TO_NPHOT
        for index in range(nbins):
            wavelength = (spectrum.min_wavelength + spectrum.delta_wavelength * index)
            epsilon =  (wavelength - laser_wavelength) / laser_wavelength
            spectrum_norm = self.seldenmatoba_spectral_shape(epsilon, cos_observation, alpha)

            spectrum.samples_mv[index] += spectrum_norm * photons_persec / wavelength * NPHOT_TO_E / spectrum.delta_wavelength

        return spectrum

    cpdef Spectrum calculate_spectrum(self, double ne, double te, double laser_power_density, double angle_pointing,
                                             double angle_polarization, double laser_wavelength, Spectrum spectrum):
        #todo: make this function return spectrum withou the need of plasma and laser model.
        pass

    def _laser_changed(self):

        self._laser_model = self._laser._laser_model
        self._plasma = self._laser._plasma
