
from raysect.core.math.function cimport Function1D

import numpy as np
cimport numpy as np

cdef class LaserSpectrum_base(Function1D):

    cdef:
        double _min_wavelength, _max_wavelength, _central_wavelength
        int bins
        np.ndarray _radiance, _photons, _wavelengths
        cdef np.float_t [:] _radiance_mv, _photons_mv, _wavelngths_mv
        
    cpdef void _create_cache_spectrum_mv(sefl)

    cpdef double evaluate_integral(self, double lower_limit, double upper_limit)

    cpdef void _create_cache_spectrum_mv(sefl)