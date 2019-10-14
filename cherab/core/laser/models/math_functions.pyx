from raysect.core.math.function cimport Function1D
from raysect.optical cimport Spectrum
from raysect.core.math.cython.utility cimport find_index

from libc.math cimport sqrt, exp, pi as PI


cdef class MathFunction(Function1D):

    def __init__(self):
        super().__init__()

    cpdef Spectrum spectrum(self, double min_wavelength, double max_wavelength, int nbins, bint zero_edges = 1):

        cdef:
            Spectrum spectrum
            double delta_wavelength, recip_delta_wavelength, wlen
            int index

        spectrum = Spectrum(min_wavelength, max_wavelength, nbins)

        recip_delta_wavelength = 1 / spectrum.delta_wavelength
        for index, wlen in enumerate(spectrum.wavelengths):
            spectrum.samples[index] = self.evaluate(wlen)

        if zero_edges:
            spectrum.samples[0] = 0
            if spectrum.bins > 1:
                    spectrum.samples[-1] = 0

        spectrum.samples_mv = spectrum.samples

        return spectrum

cdef class Normal(MathFunction):

    def __init__(self, mean = None, sigma = None):

        super().__init__()

        self.mean = mean
        self.sigma = sigma

    @property
    def mean(self):
        return self._mean

    @mean.setter
    def mean(self, value):

        self._mean = value

    @property
    def sigma(self):

        return self._sigma

    @sigma.setter
    def sigma(self, value):

        self._sigma = value
        self._recip_negative_2sigma2 = -1 / (2 * value ** 2)
        self._recip_sqrt_2pisigma2 = 1 / sqrt(2 * PI * value ** 2)

    cdef double evaluate(self, double x) except? -1e999:

        return self._recip_sqrt_2pisigma2 * exp(((x - self._mean) ** 2) * self._recip_negative_2sigma2)

cdef class BellShape(MathFunction):

    def __init__(self, mean = None, sigma = None):

        super().__init__()

        self.mean = mean
        self.sigma = sigma

    @property
    def mean(self):
        return self._mean

    @mean.setter
    def mean(self, value):

        self._mean = value

    @property
    def sigma(self):

        return self._sigma

    @sigma.setter
    def sigma(self, value):

        self._sigma = value
        self._recip_negative_2sigma2 = -1 / (2 * value ** 2)

    cdef double evaluate(self, double x) except? -1e999:

        return exp(((x - self._mean) ** 2) * self._recip_negative_2sigma2)

cdef class Delta(MathFunction):

    def __init__(self, position = None):

        super().__init__()

        self.position = position

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, value):

        self._position = value

    cdef double evaluate(self, double x) except? -1e999:

        if x == self._position:
            return 1
        else:
            return 0

    cpdef Spectrum spectrum(self, double min_wavelength, double max_wavelength, int nbins, bint zero_edges = 1):

        cdef:
            Spectrum spectrum
            double delta_wavelength, recip_delta_wavelength, wlen
            int index

        spectrum = Spectrum(min_wavelength, max_wavelength, nbins)

        index = find_index(spectrum.wavelengths, self._position)
        spectrum.samples[index] = 1
        if spectrum.bins > 1:
            spectrum.samples[index] /= spectrum.delta_wavelength
            #edges cant be zeros if number of bins is 1
            if zero_edges:
                    spectrum.samples[0] = 0
                    spectrum.samples[-1] = 0

        spectrum.samples_mv = spectrum.samples
        print(spectrum.samples)

        return spectrum
