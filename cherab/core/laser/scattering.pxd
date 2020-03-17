from raysect.optical cimport Vector3D, Point3D
from raysect.optical.spectrum cimport Spectrum
from cherab.core.laser.node cimport Laser
from cherab.core.laser.models.model_base cimport LaserModel
from cherab.core.laser.models.laserspectrum_base cimport LaserSpectrum
cimport cython

cdef double RE_SQUARED


cdef class ScatteringModel:

    cpdef Spectrum emission(self, double ne, double te, double laser_power_density, double laser_wavelength, Vector3D direction_observation,
                            Vector3D pointing_vector, Vector3D polarization_vector, Spectrum spectrum)

    cdef object __weakref__

cdef class SeldenMatobaThomsonSpectrum(ScatteringModel):
    cdef:
        double re, _CONST_ALPHA, _CONST_TS

    cpdef Spectrum emission(self, double ne, double te, double laser_power_density, double laser_wavelength, Vector3D direction_observation,
                            Vector3D pointing_vector, Vector3D polarization_vector, Spectrum spectrum)

    cdef double seldenmatoba_spectral_shape(self, double epsilon, double cos_theta, double alpha)

    cdef Spectrum _add_spectral_contribution(self, double ne, double te, double laser_power, double angle_pointing,
                                             double angle_polarization, double laser_wavelength, Spectrum spectrum)

    cpdef Spectrum calculate_spectrum(self, double ne, double te, double laser_power_density, double laser_wavelength,
                                      Vector3D direction_observation, Vector3D pointing_vector, Vector3D polarization_vector, Spectrum spectrum)