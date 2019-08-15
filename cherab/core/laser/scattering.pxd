from raysect.optical.spectrum cimport

Spectrum

from raysect.optical cimport

Vector3D

cdef double RE_SQUARED

cdef class ScatteringModel:
    cpdef Spectrum emission(self, double ne, double te, double laser_power, Vector3D pointing_vector,
                            Vector3D polarisation_direction,
                            Vector3D observation_direction, Spectrum laser_spectrum, Spectrum spectrum)

cdef class ThomsonScattering_tester(ScatteringModel):
    cdef:
        double ELECTRON_AMU, _scattering_crosssection
#    cdef spectrum(self, ne, te, laser_power, pointing_vector, observation_direction, wavelength)

cdef class SeldenMatobaThomsonSpectrum(ScatteringModel):
    cdef:
        double re, _CONST_ALPHA

    cdef double seldenmatoba_spectral_shape(self, epsilon, cos_theta, alpha)
