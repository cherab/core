from raysect.core.math.function.float cimport Function3D
from raysect.optical cimport Spectrum, Point3D, Vector3D

from cherab.core.laser.models.model_base cimport LaserModel

cdef class UniformPowerDensity(LaserModel):

    cdef:
        double _power_density


cdef class ConstantAxisymmetricGaussian(LaserModel):

    cdef:
        double _stddev, _pulse_energy, _pulse_length
        Function3D _distribution


cdef class ConstantBivariateGaussian(LaserModel):

    cdef:
        double _stddev_x, _stddev_y, _pulse_energy, _pulse_length
        Function3D _distribution


cdef class TrivariateGaussian(LaserModel):

    cdef:
        double _stddev_x, _stddev_y, _stddev_z, _mean_z, _pulse_energy, _pulse_length
        Function3D _distribution


cdef class GaussianBeamAxisymmetric(LaserModel):

    cdef:
        double _pulse_energy, _pulse_length, _stddev_waist, _waist_z, _laser_wavelength
        Function3D _distribution
