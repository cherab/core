
from raysect.core.math.function cimport Function1D
from raysect.optical cimport Point3D, Vector3D

from cherab.core.utility import Notifier
from cherab.core.laser.node import Laser
import numpy as np
cimport numpy as np

from cherab.core.utility.constants cimport SPEED_OF_LIGHT, PLANCK_CONSTANT
cdef double CONSTANT_HC = SPEED_OF_LIGHT * PLANCK_CONSTANT

cdef class LaserSpectrum_base(Function1D):

    def __init__(self, double min_wavelength, double max_wavelength, int bins):

        super().__init__()

        self._notifier = Notifier()

        self._check_wavelength_validity(min_wavelength, max_wavelength)

        self._min_wavelength = min_wavelength
        self._max_wavelength = max_wavelength

        self.bins = bins


    @property
    def min_wavelength(self):
        return self._min_wavelength

    @min_wavelength.setter
    def min_wavelength(self, double value):

        self._min_wavelength = value
        self._update_cache()

    @property
    def max_wavelength(self):
        return self._max_wavelength

    @max_wavelength.setter
    def max_wavelength(self, double value):

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
    def psd(self):
        return self._psd

    @property
    def photons(self):
        return self._photons

    @property
    def delta_wavelength(self):
        return self._delta_wavelength

    def _check_wavelength_validity(self, min_wavelength = None, max_wavelength = None):

        if min_wavelength is None:
            min_wavelength = self._min_wavelength
        if max_wavelength is None:
            max_wavelength = self._max_wavelength

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
        self._wavelengths = np.zeros((self.bins), dtype=np.double)
        self._wavelengths_mv = self._wavelengths
        
        for index in range(self._bins):
            self._wavelengths[index] = self._min_wavelength + (0.5 + index) * self._delta_wavelength

        self._psd = np.zeros((self._bins), dtype=np.double)
        self._power_mv = np.zeros((self._bins), dtype=np.double)
        
        self._psd_mv = self._psd

        self._photons = np.zeros((self._bins), dtype=np.double)
        self._photons_mv = self._photons  

        for index in range(self._bins):
            self._power_mv[index] = self.evaluate(self._wavelengths_mv[index])
            self._psd_mv[index] = self._power_mv[index] / self._delta_wavelength
            self._photons_mv[index] = self._psd_mv[index] / self._photon_energy(self._wavelengths_mv[index])

        #notify about changes
        self._notifier.notify()

    cdef double _photon_energy(self, double wavelength):
        return CONSTANT_HC / (wavelength * 1e-9)

    cpdef double evaluate_integral(self, double lower_limit, double upper_limit):
        raise NotImplementedError('Virtual method must be implemented in a sub-class.')
        