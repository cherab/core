from raysect.optical.spectrum cimport Spectrum
from cherab.core.model.lineshape import add_gaussian_line, thermal_broadening
from cherab.core.utility.constants cimport DEGREES_TO_RADIANS, ATOMIC_MASS, RECIP_4_PI
from cherab.core.model.lineshape cimport thermal_broadening, add_gaussian_line
from cherab.core.utility.constants cimport PLANCK_CONSTANT, SPEED_OF_LIGHT, ELECTRON_CLASSICAL_RADIUS, ELECTRON_MASS, ELEMENTARY_CHARGE
cimport cython
from libc.math cimport exp, sqrt, cos, M_PI as pi

cdef double RE_SQUARED = ELECTRON_CLASSICAL_RADIUS ** 2 #cross section for thomson scattering for SeldenMatoba
cdef double E_TO_NPHOT = 10e-9 / (PLANCK_CONSTANT * SPEED_OF_LIGHT) # N_photons(wlen, E_laser) = E_laser * wlen * E_TO_PHOT in [J, nm]
cdef double NPHOT_TO_E = 1/ E_TO_NPHOT # E(wlen, N_photons) = n_photons/wlen * NPHOT_TO_E in [nm, J]


cdef class ScatteringModel:

    cpdef Spectrum emission(self, double ne, double te, double laser_power, Vector3D pointing_vector,
                            Vector3D observation_direction, Vector3D polarisation_direction,
                            Spectrum laser_spectrum, Spectrum spectrum):

        raise NotImplementedError('Virtual method must be implemented in a sub-class.')


cdef class ThomsonScattering_tester(ScatteringModel):

    def __init__(self):
        super().__init__()

        self.ELECTRON_AMU = 9.10938356e-31 / ATOMIC_MASS
        self._scattering_crosssection = 1e-16

    cpdef Spectrum emission(self, double ne, double te, double laser_power, Vector3D observation_direction,
                            Vector3D pointing_vector, Vector3D polarisation_direction,
                            Spectrum laser_spectrum, Spectrum spectrum):
        cdef:
            double obsangle, wavelength, broadening, radiance

        if ne <= 0 or te <=0:
            return spectrum

        obsangle = pointing_vector.angle(observation_direction) * DEGREES_TO_RADIANS
        wavelength = laser_spectrum.wavelengths[laser_spectrum.samples.argmax()]
        broadening = thermal_broadening(wavelength, te, self.ELECTRON_AMU)
        radiance = ne * laser_power * obsangle * self._scattering_crosssection * RECIP_4_PI
        spectrum = add_gaussian_line(radiance, wavelength, broadening, spectrum)

        return spectrum


cdef class SeldenMatobaThomsonSpectrum(ScatteringModel):

    def __init__(self):
        # from article: 2 * alpha = m_e * c **2 /(k * T_e), here rewritten for Te in eV
        self._CONST_ALPHA = ELECTRON_MASS * SPEED_OF_LIGHT ** 2 / ( 2 * ELEMENTARY_CHARGE)

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
    cpdef Spectrum emission(self, double ne, double te, double laser_power, Vector3D pointing_vector,
                            Vector3D observation_direction, Vector3D polarisation_direction,
                            Spectrum laser_spectrum, Spectrum spectrum):

        cdef:
            double alpha, laser_wavelength, epsilon, angle_scattering, min_wavelength, angle_polarisation, cos_observation, cos2_polarisation
            double wavelength
            int index

        if ne <= 0 or te <=0:
            return spectrum

        alpha = self._CONST_ALPHA / te
        nbins = spectrum.bins
        laser_wavelength = laser_spectrum.wavelengths[laser_spectrum.samples.argmax()]
        cos_observation = cos(observation_direction.angle(pointing_vector) * DEGREES_TO_RADIANS)
        cos2_polarisation = cos(observation_direction.angle(polarisation_direction) * DEGREES_TO_RADIANS) ** 2 #cos2 of observation to polarisation
        min_wavelength = spectrum.min_wavelength


        #photon density per second
        photons_persec = ne * RE_SQUARED * laser_power * laser_wavelength * E_TO_NPHOT

        for i in range(nbins):
            wavelength = (spectrum.min_wavelength + spectrum.delta_wavelength * i)
            epsilon =  (wavelength - laser_wavelength) / laser_wavelength
            spectrum_norm = self.seldenmatoba_spectral_shape(epsilon, cos_observation, alpha)

            spectrum.samples_mv[i] += spectrum_norm * photons_persec / wavelength * NPHOT_TO_E / spectrum.delta_wavelength

        return spectrum


