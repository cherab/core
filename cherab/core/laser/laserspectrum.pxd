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


from raysect.core.math.function.float cimport Function1D

from cherab.core.utility.constants cimport SPEED_OF_LIGHT, PLANCK_CONSTANT

from numpy cimport ndarray


cdef class LaserSpectrum(Function1D):

    cdef:
        double _min_wavelength, _max_wavelength, _delta_wavelength
        int _bins
        ndarray _power, _power_spectral_density, _wavelengths  # power_spectral_density [w/nm]
        double[::1] power_mv, power_spectral_density_mv, wavelengths_mv

    cpdef double evaluate_integral(self, double lower_limit, double upper_limit)

    cpdef void _update_cache(self)

    cpdef double get_min_wavelenth(self)

    cpdef double get_max_wavelenth(self)

    cpdef int get_spectral_bins(self)

    cpdef double get_delta_wavelength(self)

    cpdef double _get_bin_power_spectral_density(self, double wavelength_lower, double wavelength_upper)