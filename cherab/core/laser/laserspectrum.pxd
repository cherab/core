
from raysect.core.math.function.float cimport Function1D

from cherab.core.utility.constants cimport SPEED_OF_LIGHT, PLANCK_CONSTANT

from numpy cimport ndarray


cdef class LaserSpectrum(Function1D):

    cdef:
        double _min_wavelength, _max_wavelength, _delta_wavelength
        readonly int _bins
        ndarray _power, _power_spectral_density, _photon_spectral_density, _wavelengths  # power_spectral_density [w/nm]
        double[::1] _power_mv, _power_spectral_density_mv, _photon_spectral_density_mv, _wavelengths_mv

    cpdef double evaluate_integral(self, double lower_limit, double upper_limit)

    cpdef void _update_cache(self)

    cdef double _photon_energy(self, double wavelength)

    cpdef double _get_bin_power_spectral_density(self, double wavelength_lower, double wavelength_upper)