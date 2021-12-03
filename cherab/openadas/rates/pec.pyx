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

import numpy as np

from libc.math cimport INFINITY, log10

from raysect.core.math.function.float cimport Interpolator2DArray
from cherab.core.utility.conversion import PhotonToJ


cdef class ImpactExcitationPEC(CoreImpactExcitationPEC):

    def __init__(self, double wavelength, dict data, extrapolate=False):
        """
        :param wavelength: Resting wavelength of corresponding emission line in nm.
        :param data: Dictionary containing rate data.
        :param extrapolate: Enable extrapolation (default=False).
        """

        self.wavelength = wavelength
        self.raw_data = data

        # unpack
        ne = data['ne']
        te = data['te']
        rate = data['rate']

        # pre-convert data to W m^3 from Photons s^-1 cm^3 prior to interpolation
        rate = np.log10(PhotonToJ.to(rate, wavelength))

        # store limits of data
        self.density_range = ne.min(), ne.max()
        self.temperature_range = te.min(), te.max()

        # interpolate rate
        # using nearest extrapolation to avoid infinite values at 0 for some rates
        extrapolation_type = 'nearest' if extrapolate else 'none'
        self._rate = Interpolator2DArray(np.log10(ne), np.log10(te), rate, 'cubic', extrapolation_type, INFINITY, INFINITY)

    cpdef double evaluate(self, double density, double temperature) except? -1e999:

        # need to handle zeros, also density and temperature can become negative due to cubic interpolation
        if density < 1.e-300:
            density = 1.e-300

        if temperature < 1.e-300:
            temperature = 1.e-300

        # calculate rate and convert from log10 space to linear space
        return 10 ** self._rate.evaluate(log10(density), log10(temperature))


cdef class NullImpactExcitationPEC(CoreImpactExcitationPEC):
    """
    A PEC rate that always returns zero.
    Needed for use cases where the required atomic data is missing.
    """

    cpdef double evaluate(self, double density, double temperature) except? -1e999:
        return 0.0


cdef class RecombinationPEC(CoreRecombinationPEC):

    def __init__(self, double wavelength, dict data, extrapolate=False):
        """
        :param wavelength: Resting wavelength of corresponding emission line in nm.
        :param data: Dictionary containing rate data.
        :param extrapolate: Enable extrapolation (default=False).
        """

        self.wavelength = wavelength
        self.raw_data = data

        # unpack
        ne = data['ne']
        te = data['te']
        rate = data['rate']

        # pre-convert data to W m^3 from Photons s^-1 cm^3 prior to interpolation
        rate = np.log10(PhotonToJ.to(rate, wavelength))

        # store limits of data
        self.density_range = ne.min(), ne.max()
        self.temperature_range = te.min(), te.max()

        # interpolate rate
        # using nearest extrapolation to avoid infinite values at 0 for some rates
        extrapolation_type = 'nearest' if extrapolate else 'none'
        self._rate = Interpolator2DArray(np.log10(ne), np.log10(te), rate, 'cubic', extrapolation_type, INFINITY, INFINITY)

    cpdef double evaluate(self, double density, double temperature) except? -1e999:

        # need to handle zeros, also density and temperature can become negative due to cubic interpolation
        if density < 1.e-300:
            density = 1.e-300

        if temperature < 1.e-300:
            temperature = 1.e-300

        # calculate rate and convert from log10 space to linear space
        return 10 ** self._rate.evaluate(log10(density), log10(temperature))


cdef class NullRecombinationPEC(CoreRecombinationPEC):
    """
    A PEC rate that always returns zero.
    Needed for use cases where the required atomic data is missing.
    """

    cpdef double evaluate(self, double density, double temperature) except? -1e999:
        return 0.0
