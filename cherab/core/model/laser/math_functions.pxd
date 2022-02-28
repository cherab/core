from raysect.core.math.function.float cimport Function3D

cdef class ConstantAxisymmetricGaussian3D(Function3D):
    cdef:
        double _stddev, _recip_negative_2stddev2, _recip_2pistddev2

    cdef double evaluate(self, double x, double y, double z) except? -1e999


cdef class ConstantBivariateGaussian3D(Function3D):

    cdef:
        double _stddev_x, _stddev_y, _kx, _ky,_normalisation


cdef class TrivariateGaussian3D(Function3D):

    cdef:
        double _mean_z, _stddev_x, _stddev_y, _stddev_z, _kx, _ky
        double _kz, _normalisation


cdef class GaussianBeamModel(Function3D):
    cdef:
        double _waist_z, _stddev_waist, _stddev_waist2, _wavelength, _rayleigh_range

    cdef double evaluate(self, double x, double y, double z) except? -1e999
