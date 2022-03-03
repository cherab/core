from cherab.core.laser.laserspectrum cimport LaserSpectrum


cdef class ConstantSpectrum(LaserSpectrum):

    cdef double evaluate(self, double x) except? -1e999


cdef class GaussianSpectrum(LaserSpectrum):

    cdef:
        double _stddev, _spectral_mean, _recip_stddev, _normalisation, _mean
        double _norm_cdf

    cdef double evaluate(self, double x) except? -1e999
