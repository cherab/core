from raysect.optical cimport Vector3D, Point3D
from raysect.optical.spectrum cimport Spectrum
from cherab.core.laser.node cimport Laser
from cherab.core.laser.models.model_base cimport LaserModel
from cherab.core.laser.models.laserspectrum_base cimport LaserSpectrum
from cherab.core cimport Plasma
cimport cython

cdef double RE_SQUARED

cdef class ScatteringModel:

    cdef:
        Plasma _plasma
        LaserModel _laser_model
        Laser _laser
        LaserSpectrum _laser_spectrum
        double[::1] _laser_wavelength_mv, _laser_power_mv
        int _laser_bins

    cpdef Spectrum emission(self, Point3D position_plasma, Point3D position_laser, Vector3D direction_observation, Spectrum spectrum)

cdef class SeldenMatobaThomsonSpectrum(ScatteringModel):
    cdef:
        double re, _CONST_ALPHA

    cdef object __weakref__

    cpdef Spectrum emission(self, Point3D position_plasma, Point3D position_laser, Vector3D direction_observation, Spectrum spectrum)

    cdef double seldenmatoba_spectral_shape(self, double epsilon, double cos_theta, double alpha)

    cdef Spectrum add_spectral_contribution(self, double ne, double te, double laser_power, double angle_pointing,
                                             double angle_polarization, double laser_wavelength, Spectrum spectrum)

    cpdef Spectrum calculate_spectrum(self, double ne, double te, double laser_power_density, double angle_pointing,
                                             double angle_polarization, Spectrum laser_spectrum, Spectrum spectrum)