from raysect.optical cimport Spectrum, Point3D, Vector3D

from cherab.core.laser.models.model_base cimport LaserModel

cdef class UniformPowerDensity(LaserModel):

    cdef:
        double _power

cdef class GaussianBeamAxisymmetric(LaserModel):

    cdef:
        double _laser_power, _const_width, _waist_radius, _waist_const, _waist2, _power_const, _focus_z
        double _beam_quality_factor

    cpdef double get_beam_width2(self, double z, double wavelength)

    cpdef double get_power_axis(self, double z, double wavelength)


cdef class AxisymmetricGaussian(LaserModel):

    cdef:
        double _laser_sigma, _half_recip_laser_sigma2, _const_width
