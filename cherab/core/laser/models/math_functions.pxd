from raysect.core.math.function cimport Function1D
from raysect.optical cimport Spectrum


cdef class MathFunction(Function1D):

    cdef double evaluate(self, double x) except? -1e999

    cpdef Spectrum spectrum(self, double min_wavelength, double max_wavelength, int nbins, bint zero_edges=*)


cdef class Normal(MathFunction):

    cdef:
        double _mean, _sigma, _recip_negative_2sigma2, _recip_sqrt_2pisigma2

    cdef double evaluate(self, double x) except? -1e999

cdef class BellShape(MathFunction):

    cdef:
        double _mean, _sigma, _recip_negative_2sigma2

cdef class Delta(MathFunction):

    cdef:
        double _position

