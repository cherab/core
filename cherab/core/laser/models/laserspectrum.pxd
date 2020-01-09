from cherab.core.laser.models.laserspectrum_base cimport LaserSpectrum_base

cdef class ConstantSpectrum(LaserSpectrum_base):

    cdef double evaluate(self, double x) except? -1e999

cdef class GaussianSpectrum(LaserSpectrum_base):
    
    cdef:
        double _sigma, _spectral_mu, _recip_sigma, _normalisation, _mu

    cdef double evaluate(self, double x) except? -1e999