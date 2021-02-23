from raysect.core.math.function.float cimport Function3D
from raysect.optical cimport Spectrum, Point3D, Vector3D

from cherab.core.laser.models.profile_base cimport LaserProfile

cdef class UniformEnergyDensity(LaserProfile):

    cdef:
        double _energy_density


cdef class ConstantAxisymmetricGaussian(LaserProfile):

    cdef:
        double _stddev, _pulse_energy, _pulse_length
        Function3D _distribution


cdef class ConstantBivariateGaussian(LaserProfile):

    cdef:
        double _stddev_x, _stddev_y, _pulse_energy, _pulse_length
        Function3D _distribution


cdef class TrivariateGaussian(LaserProfile):

    cdef:
        double _stddev_x, _stddev_y, _stddev_z, _mean_z, _pulse_energy, _pulse_length
        Function3D _distribution


cdef class GaussianBeamAxisymmetric(LaserProfile):

    cdef:
        double _pulse_energy, _pulse_length, _stddev_waist, _waist_z, _laser_wavelength
        Function3D _distribution
