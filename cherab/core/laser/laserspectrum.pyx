
from raysect.core.math.function.float cimport Function1D
from raysect.optical cimport Point3D, Vector3D

from cherab.core.utility import Notifier
from cherab.core.laser.node import Laser
from cherab.core.utility.constants cimport SPEED_OF_LIGHT, PLANCK_CONSTANT

import numpy as np
cimport numpy as np


cdef class LaserSpectrum(Function1D):
    """
    Laser spectrum base class.

    This is an abstract class and cannot be used for observing.
    
    A 1D function holding information about the spectral properties
    of a laser.  The scattered spectrum is calculated as an iteration
    over the laser spectrum.


    .. warning::
        When adding a LaserSpectrum, a special care should be given
        to the integral power of the laser spectrum. During the
        scattering calculation, the spectral power can be multiplied
        by the power spatial distribution [W * m ** -3] of the laser
        power from the LaserProfile. If the integral power
        of the LaserSpectrum is not 1, unexpected values 
        might be obtained.

    .. note::
        It is expected that majority of the fusion applications can
        neglect the influence of the spectral shape of the
        laser and can use laser spectrum with a single 
        bin, which approximates an infinitely narrow laser spectrum.

    :param float min_wavelength: The minimum wavelength of the laser
      spectrum in nm.
    :param float max_wavelength: The maximum wavelength of the laser
      spectrum in nm.
    :param int bins: The number of spectral bins of the laser spectrum.
    :ivar float min_wavelength: The minimum wavelength of the laser
      spectrum in nm.
    :ivar float max_wavelength: The maximum wavelength of the laser
      spectrum in nm.
    :ivar int bins: The number of specral bins of the laser spectrum
    :ivar ndarray wavelengths: The wavelengt coordinate vector in nm.
    :ivar ndarray power_spectral_density: The values of the power
      spectral density in W / nm.
    :ivar ndarray photon_spectral_density: The values corresponding to the number
      of photons per nm.
    :ivar float delta_wavelength: Spectral width of the bins in nm.
    """

    def __init__(self, double min_wavelength, double max_wavelength, int bins):

        super().__init__()

        self._check_wavelength_validity(min_wavelength, max_wavelength)

        self._min_wavelength = min_wavelength
        self._max_wavelength = max_wavelength

        self.bins = bins


    @property
    def min_wavelength(self):
        return self._min_wavelength

    @min_wavelength.setter
    def min_wavelength(self, double value):

        self._check_wavelength_validity(value, self.max_wavelength)
        self._min_wavelength = value
        self._update_cache()

    @property
    def max_wavelength(self):
        return self._max_wavelength

    @max_wavelength.setter
    def max_wavelength(self, double value):

        self._check_wavelength_validity(self.min_wavelength, value)
        self._max_wavelength = value
        self._update_cache()

    @property
    def bins(self):
        return self._bins

    @bins.setter
    def bins(self, int value):
        if not value > 0:
            raise ValueError("Value has to be larger than 0")

        self._bins = value
        self._update_cache()

    @property
    def wavelengths(self):
        return self._wavelengths

    @property
    def power_spectral_density(self):
        return self._power_spectral_density

    @property
    def photon_spectral_density(self):
        return self._photon_spectral_density

    @property
    def delta_wavelength(self):
        return self._delta_wavelength

    def _check_wavelength_validity(self, min_wavelength, max_wavelength):

        if not min_wavelength > 0:
            raise ValueError("min_wavelength has to be larger than 0, but {} passed.".format(min_wavelength))
        if not max_wavelength > 0:
            raise ValueError("min_wavelength has to be larger than 0, but {} passed.".format(max_wavelength))

        if not min_wavelength < max_wavelength:
            raise ValueError("min_wavelength has to be smaller than max_wavelength: min_wavelength={} > max_wavelength={}".format(min_wavelength, max_wavelength))

    cpdef void _update_cache(self):

        cdef:
            Py_ssize_t index
            double delta_wvl_half, wvl_lower, wvl_upper, wvl

        self._delta_wavelength = (self._max_wavelength - self._min_wavelength) / self._bins
        self._wavelengths = np.zeros(self.bins, dtype=np.double)
        self._wavelengths_mv = self._wavelengths

        for index in range(self._bins):
            self._wavelengths[index] = self._min_wavelength + (0.5 + index) * self._delta_wavelength

        self._power_spectral_density = np.zeros(self._bins, dtype=np.double)  # power spectral density (PSD)
        self._power_spectral_density_mv = self._power_spectral_density
        
        self._power = np.zeros(self._bins, dtype=np.double)  # power in a spectral bin (PSD * delta wavelength)
        self._power_mv = self._power

        self._photon_spectral_density = np.zeros(self._bins, dtype=np.double)
        self._photon_spectral_density_mv = self._photon_spectral_density

        delta_wvl_half = self._delta_wavelength * 0.5
        wvl_lower = self._wavelengths_mv[0] - delta_wvl_half

        for index in range(self._bins):
            wvl = wvl_lower + delta_wvl_half
            wvl_upper = wvl_lower + self._delta_wavelength

            self._power_spectral_density_mv[index] = self._get_bin_power_spectral_density(wvl_lower, wvl_upper)
            self._power_mv[index] = self._power_spectral_density_mv[index] * self._delta_wavelength # power in the spectral bin for scattering calculations
            self._photon_spectral_density_mv[index] = self._power_spectral_density_mv[index] / self._photon_energy(wvl)

            wvl_lower = wvl_upper

    cdef double _photon_energy(self, double wavelength):
        return SPEED_OF_LIGHT * PLANCK_CONSTANT / (wavelength * 1e-9)

    cpdef double evaluate_integral(self, double lower_limit, double upper_limit):
        raise NotImplementedError('Virtual method must be implemented in a sub-class.')

    cpdef double _get_bin_power_spectral_density(self, double wavelength_lower, double wavelength_upper):
        """
        Returns the power spectral density in a bin.

        This method can be overidden if a better precision is needed.
        For example for distributions with known cumulative distribution function.
        """
        return 0.5 * (self.evaluate(wavelength_lower) + self.evaluate(wavelength_upper))
