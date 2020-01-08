from raysect.core.math.function cimport Function3D
from raysect.optical cimport Spectrum

cdef class AxisymmetricGaussian3D(Function3D):

    cdef:
        double _sigma, _recip_negative_2sigma2, _recip_sqrt_2pisigma2

    cdef double evaluate(self, double x, double y, double z) except? -1e999

cdef class GaussianBeamModel(Function3D):

    cdef:
        double _focus_z, _focus_width, _wavelength, _m2, _wavelengthm_div_pi, _focus_width2m

    cdef double evaluate(self, double x, double y, double z) except? -1e999