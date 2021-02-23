from raysect.optical cimport Vector3D, Point3D
from raysect.optical.spectrum cimport Spectrum

from cherab.core cimport Plasma
from cherab.core.distribution cimport DistributionFunction
from cherab.core.laser.node cimport Laser
from cherab.core.laser.models.profile_base cimport LaserProfile
from cherab.core.laser.models.laserspectrum_base cimport LaserSpectrum
cimport cython

#cdef double RE_SQUARED


cdef class LaserEmissionModel:
    cdef:
        Plasma _plasma
        LaserSpectrum _laser_spectrum
        LaserProfile _laser_profile


    cpdef Spectrum emission(self, Point3D point_plasma, Vector3D observation_plasma, Point3D point_laser,
                            Vector3D observation_laser, Spectrum spectrum)

    cdef object __weakref__

cdef class SeldenMatobaThomsonSpectrum(LaserEmissionModel):
    cdef:
        double re, _CONST_ALPHA, _CONST_TS, _RECIP_M_PI


    cpdef Spectrum emission(self, Point3D point_plasma, Vector3D observation_plasma, Point3D point_laser,
                            Vector3D observation_laser, Spectrum spectrum)

    cdef double seldenmatoba_spectral_shape(self, double epsilon, double cos_theta, double alpha)

    cdef Spectrum _add_spectral_contribution(self, double ne, double te, double laser_power, double angle_pointing,
                                             double angle_polarization, double laser_wavelength, Spectrum spectrum)

    cpdef Spectrum calculate_spectrum(self, double ne, double te, double laser_energy_density, double laser_wavelength,
                                      double observation_angle, Spectrum spectrum)