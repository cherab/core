from cherab.core.laser.models.laserspectrum_base cimport LaserSpectrum_base
from libc.math cimport sqrt, exp, M_PI

cdef class ConstantSpectrum(LaserSpectrum_base):

    def __init__(self, double min_wavelength, double max_wavelength, int bins):

        super().__init__(min_wavelength, max_wavelength, bins)

    cdef double evaluate(self, double x) except? -1e999:
        
        cdef:
            double spectrum_width
            int index

        if x < self._max_wavelength and x > self._min_wavelength:
            return 1.0/ (self._max_wavelength - self._min_wavelength)
        else:
            return 0

cdef class GaussianSpectrum(LaserSpectrum_base):

    def __init__(self, double min_wavelength, double max_wavelength, int bins, double mu, double sigma):

        self.sigma = sigma
        self.mu = mu
        super().__init__(min_wavelength, max_wavelength, bins)

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, value):
        if not value >0:
            raise ValueError("Value has to be larger than 0")

        self._sigma = value
        self._recip_sigma = 1 / value
        self._normalisation = 1 / (value * sqrt(2 * M_PI))

    @property
    def mu(self):
        return self._mu

    @mu.setter
    def mu(self, double value):
        if not value >0:
            raise ValueError("Value has to be larger than 0")
        
        self._mu = value


    cdef double evaluate(self, double x) except? -1e999:
        return self._normalisation * exp(-0.5 * ((x - self._mu) * self._recip_sigma) ** 2)

