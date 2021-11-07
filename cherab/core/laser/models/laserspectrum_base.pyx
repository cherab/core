
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
    :ivar ndarray photos: The values corresponding to the number
      of photons.
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
    def photons(self):
        return self._photons

    @property
    def delta_wavelength(self):
        return self._delta_wavelength

    def _check_wavelength_validity(self, min_wavelength, max_wavelength):

        if not min_wavelength > 0:
            raise ValueError("min_wavelength has to be larger than 0, but {} passed.".format(min_wavelength))
        if not max_wavelength > 0:
            raise ValueError("min_wavelength has to be larger than 0, but {} passed.".format(max_wavelength))

        if min_wavelength > max_wavelength:
            raise ValueError("min_wavelength has to be smaller than max_wavelength: min_wavelength={} > max_wavelength={}".format(min_wavelength, max_wavelength))

    cpdef void _update_cache(self):

        cdef:
            Py_ssize_t index

        self._delta_wavelength = (self._max_wavelength - self._min_wavelength) / self._bins
        self._wavelengths = np.zeros(self.bins, dtype=np.double)
        self._wavelengths_mv = self._wavelengths

        for index in range(self._bins):
            self._wavelengths[index] = self._min_wavelength + (0.5 + index) * self._delta_wavelength

        self._power_spectral_density = np.zeros(self._bins, dtype=np.double)  # power spectral density (PSD)
        self._power_spectral_density_mv = self._power_spectral_density
        
        self._power = np.zeros(self._bins, dtype=np.double)  # power in a spectral bin (PSD * delta wavelength)
        self._power_mv = self._power

        self._photons = np.zeros(self._bins, dtype=np.double)
        self._photons_mv = self._photons

        for index in range(self._bins):
            self._power_spectral_density_mv[index] = self.evaluate(self._wavelengths_mv[index])
            self._power_mv[index] = self._power_spectral_density_mv[index] * self._delta_wavelength # power in the spectral bin for scattering calculations
            self._photons_mv[index] = self._power_spectral_density_mv[index] / self._photon_energy(self._wavelengths_mv[index])

    cdef double _photon_energy(self, double wavelength):
        return SPEED_OF_LIGHT * PLANCK_CONSTANT / (wavelength * 1e-9)

    cpdef double evaluate_integral(self, double lower_limit, double upper_limit):
        raise NotImplementedError('Virtual method must be implemented in a sub-class.')
