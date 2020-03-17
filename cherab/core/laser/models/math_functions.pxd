from raysect.core.math.function cimport Function3D

cdef class ConstantAxisymmetricGaussian3D(Function3D):
    cdef:
        double _stddev, _recip_negative_2stddev2, _recip_2pistddev2

    cdef double evaluate(self, double x, double y, double z) except? -1e999


cdef class ConstantBivariateGaussian3D(Function3D):

    cdef:
        double _stddev_x, _stddev_y, _negative_recip_2stddevx2, _negative_recip_2stddevy2,_recip_2pistddevxstddevy


cdef class TrivariateGaussian3D(Function3D):

    cdef:
        double _mean_z, _stddev_x, _stddev_y, _stddev_z, _negative_recip_2stddevx2, _negative_recip_2stddevy2
        double _negative_recip_2stddevz2, _recip_2pistddevxstddevystddevz


cdef class GaussianBeamModel(Function3D):
    cdef:
        double _waist_z, _stddev_waist, _stddev_waist2, _wavelength, _rayleigh_range

    cdef double evaluate(self, double x, double y, double z) except? -1e999
