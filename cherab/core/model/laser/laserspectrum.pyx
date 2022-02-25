from cherab.core.laser.models.laserspectrum_base cimport LaserSpectrum
from libc.math cimport sqrt, exp, M_PI

cdef class ConstantSpectrum(LaserSpectrum):
    """
    A laser spectrum with constant power.

    Has a constant, non-zero distribution of power spectral density
    between the min_wavelength and max_wavelength. The integral value
    of the power is 1 W.

    .. note::
        The ConstantSpectrum class is suitable for approximation
        of an infinitely thin laser spectrum, e.g.:
        ConstantSpectrum(1063.9, 1064.1, 1)
    """

    def __init__(self, double min_wavelength, double max_wavelength, int bins):

        super().__init__(min_wavelength, max_wavelength, bins)

    cdef double evaluate(self, double x) except? -1e999:
        """
        Returns the spectral power density for the given wavelength.

        :param float x: Wavelength in nm.

        :return: Power spectral density in W/nm. 
        """

        cdef:
            double spectrum_width
            int index

        if self._min_wavelength < x < self._max_wavelength:
            return 1.0 / (self._max_wavelength - self._min_wavelength)
        else:
            return 0

cdef class GaussianSpectrum(LaserSpectrum):
    """
    A laser spectrum with a normally distributed power spectral density.

    Has a Gaussian-like spectral shape. The inegral value of power is 1 W.

    :param float mean: The mean value of the Gaussian distribution
      of the laser spectrum in nm, can be thought of as the central
      wavelength of the laser.
    :param float stddev: Standard deviation of the Gaussian
      distribution of the laser spectrum.
    
    :ivar float stddev: Standard deviation of the Gaussian
      distribution of the laser spectrum.
    :ivar float mean: The mean value of the Gaussian distribution
      of the laser spectrum in nm, can be thought of as the central
      wavelength of the laser.
    """

    def __init__(self, double min_wavelength, double max_wavelength, int bins, double mean, double stddev):

        self.stddev = stddev
        self.mean = mean
        super().__init__(min_wavelength, max_wavelength, bins)

    @property
    def stddev(self):
        return self._stddev

    @stddev.setter
    def stddev(self, value):
        if not value > 0:
            raise ValueError("Value has to be larger than 0")

        self._stddev = value
        self._recip_stddev = 1 / value
        self._normalisation = 1 / (value * sqrt(2 * M_PI))

    @property
    def mean(self):
        return self._mean

    @mean.setter
    def mean(self, double value):
        if not value > 0:
            raise ValueError("Value has to be larger than 0")

        self._mean = value

    cdef double evaluate(self, double x) except? -1e999:
        """
        Returns the spectral power density for the given wavelength.

        :param float x: Wavelength in nm.

        :return: Power spectral density in W/nm. 
        """
        return self._normalisation * exp(-0.5 * ((x - self._mean) * self._recip_stddev) ** 2)
