# Copyright 2016-2021 Euratom
# Copyright 2016-2021 United Kingdom Atomic Energy Authority
# Copyright 2016-2021 Centro de Investigaciones Energéticas, Medioambientales y Tecnológicas
#
# Licensed under the EUPL, Version 1.1 or – as soon they will be approved by the
# European Commission - subsequent versions of the EUPL (the "Licence");
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at:
#
# https://joinup.ec.europa.eu/software/page/eupl5
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the Licence is distributed on an "AS IS" basis, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied.
#
# See the Licence for the specific language governing permissions and limitations
# under the Licence.


from cherab.core.laser cimport LaserSpectrum
from libc.math cimport sqrt, exp, M_PI, erf, M_SQRT2


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

        if self._min_wavelength <= x <= self._max_wavelength:
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
        if value <= 0:
            raise ValueError("Value has to be larger than 0")

        self._stddev = value
        self._recip_stddev = 1 / value
        self._normalisation = 1 / (value * sqrt(2 * M_PI))
        self._norm_cdf = 1 / (value * M_SQRT2)

    @property
    def mean(self):
        return self._mean

    @mean.setter
    def mean(self, double value):
        if value <= 0:
            raise ValueError("Value has to be larger than 0")

        self._mean = value

    cdef double evaluate(self, double x) except? -1e999:
        """
        Returns the spectral power density for the given wavelength.

        :param float x: Wavelength in nm.

        :return: Power spectral density in W/nm. 
        """
        return self._normalisation * exp(-0.5 * ((x - self._mean) * self._recip_stddev) ** 2)

    cpdef double _get_bin_power_spectral_density(self, double wavelength_lower, double wavelength_upper):
        """
        Returns the power spectral density in a bin.

        Overrides the parent method to deliver better precision.
        """

        cdef:
            double val_lower, val_upper

        val_lower = erf((wavelength_lower - self._mean) * self._norm_cdf)
        val_upper = erf((wavelength_upper - self._mean) * self._norm_cdf)
        return 0.5 * (val_upper - val_lower) / self._delta_wavelength