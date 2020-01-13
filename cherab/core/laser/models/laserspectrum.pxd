from cherab.core.laser.models.laserspectrum_base cimport LaserSpectrum

cdef class ConstantSpectrum(LaserSpectrum):

    cdef double evaluate(self, double x) except? -1e999

cdef class GaussianSpectrum(LaserSpectrum):
    
    cdef:
        double _sigma, _spectral_mu, _recip_sigma, _normalisation, _mu

    cdef double evaluate(self, double x) except? -1e999