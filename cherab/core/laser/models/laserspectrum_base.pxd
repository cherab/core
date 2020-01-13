
from raysect.core.math.function cimport Function1D

from cherab.core.laser.node cimport Laser
from cherab.core.utility.constants cimport SPEED_OF_LIGHT, PLANCK_CONSTANT
from numpy cimport ndarray


cdef double CONSTANT_HC = SPEED_OF_LIGHT * PLANCK_CONSTANT

cdef class LaserSpectrum(Function1D):

    cdef:
        double _min_wavelength, _max_wavelength, _delta_wavelength
        int _bins
        ndarray _power, _psd, _photons, _wavelengths # psd: power spectral density [w/nm]
        double[::1] _power_mv, _psd_mv, _photons_mv, _wavelengths_mv 
        object _notifier
        
    cpdef double evaluate_integral(self, double lower_limit, double upper_limit)

    cpdef void _update_cache(self)

    cdef double _photon_energy(self, double wavelength)