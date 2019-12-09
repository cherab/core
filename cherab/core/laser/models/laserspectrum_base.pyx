
from raysect.core.math.function cimport Function1D
from raysect.optical cimport Point3D, Vector3D

from cherab.core.utility import Notifier
from cherab.core.laser.scattering cimport ScatteringModel

import numpy as np
cimport numpy as np

cdef class LaserSpectrum_base(Function1D):

    def __init__(self, min_wavelength, max_wavelength, bins, central_wavelength):

        super().__init__()
        self._check_wavelength_validity(min_wavelength, max_wavelength, central_wavelength)

        self.min_wavelength = min_wavelength
        self.max_wavelength = max_wavelength
        self.central_wavelength = central_wavelength

    @property
    def central_wavelength(self):
        return self._central_wavelength

    @central_wavelength.setter
    def central_wavelength(self, double value):

        self._central_wavelength = value
        self._check_wavelength_validity()

    @property
    def min_wavelength(self):
        return self._min_wavelength

    @min_wavelength.setter
    def min_wavelength(self, double value):

        self._min_wavelength = value
        self._check_wavelength_validity()

    @property
    def max_wavelength(self):
        return self._max_wavelength

    @max_wavelength.setter
    def max_wavelength(self, double value):

        self._max_wavelength = value
        self._check_wavelength_validity()

    def _check_wavelength_validity(self, min_wavelength = None, max_wavelength = None, central_wavelength = None):

        if min_wavelength is None:
            min_wavelength = self._min_wavelength
        if max_wavelength is None:
            max_wavelength = self._max_wavelength
        if central_wavelength is None:
            central_wavelength = self._central_wavelength
        
        if not min_wavelength > 0:
            raise ValueError("min_wavelength has to be larger than 0, but {} passed.".format(min_wavelength))
        if not max_wavelength > 0:
            raise ValueError("min_wavelength has to be larger than 0, but {} passed.".format(max_wavelength))
        if not central_wavelength > 0:
            raise ValueError("min_wavelength has to be larger than 0, but {} passed.".format(central_wavelength))

        if min_wavelength > central_wavelength:
            raise ValueError("Central wavelength has to be larger than Min_wavelength: min_wavelength={} > central_wavelength={}".format(min_wavelength, central_wavelength))
        if max_wavelength < central_wavelength:
            raise ValueError("Central wavelength has to be smaller than max_wavelength: max_wavelength={} < central_wavelength={}".format(min_wavelength, central_wavelength))
    
    cpdef void _create_cache_spectrum_mv(self):

        raise NotImplementedError('Virtual method must be implemented in a sub-class.')

    cpdef double evaluate_integral(self, double lower_limit, double upper_limit):
        raise NotImplementedError('Virtual method must be implemented in a sub-class.')
        