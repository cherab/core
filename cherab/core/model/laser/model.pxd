from raysect.optical cimport Vector3D, Point3D
from raysect.optical.spectrum cimport Spectrum

from cherab.core.laser.model cimport LaserModel
from cherab.core.laser.node cimport Laser
from cherab.core.laser.profile cimport LaserProfile
from cherab.core.laser.laserspectrum cimport LaserSpectrum


cdef class SeldenMatobaThomsonSpectrum(LaserModel):
    
    cdef:
        double re, _CONST_ALPHA, _RATE_TS, _RECIP_M_PI

    cpdef Spectrum emission(self, Point3D point_plasma, Vector3D observation_plasma, Point3D point_laser,
                            Vector3D observation_laser, Spectrum spectrum)

    cdef double seldenmatoba_spectral_shape(self, double epsilon, double cos_theta, double alpha)

    cdef Spectrum _add_spectral_contribution(self, double ne, double te, double laser_energy, double angle_pointing,
                                             double angle_polarization, double laser_wavelength, Spectrum spectrum)

    cpdef Spectrum calculate_spectrum(self, double ne, double te, double laser_energy_density, double laser_wavelength,
                                      double observation_angle, double angle_polarization, Spectrum spectrum)