
# Copyright 2016-2024 Euratom
# Copyright 2016-2024 United Kingdom Atomic Energy Authority
# Copyright 2016-2024 Centro de Investigaciones Energéticas, Medioambientales y Tecnológicas
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


cdef class LineRadiationPower(CoreLineRadiationPower):
    """
    Line radiated power coefficient.

    The data is interpolated with cubic spline in log-log space.
    Nearest neighbour extrapolation is used if extrapolate is True.

    :param Element species: Element object defining the ion type.
    :param int ionisation: Charge state of the ion.
    :param dict data: Line radiated power rate dictionary containing the following entries:

    |      'ne': 1D array of size (N) with electron density in m^-3,
    |      'te': 1D array of size (M) with electron temperature in eV,
    |      'rate': 2D array of size (N, M) with radiated power rate in W.m^3.

    :param bint extrapolate: Enable extrapolation (default=False).

    :ivar tuple density_range: Electron density interpolation range.
    :ivar tuple temperature_range: Electron temperature interpolation range.
    :ivar dict raw_data: Dictionary containing the raw data.
    """

    def __init__(self, species, ionisation, dict data, extrapolate=False):

        super().__init__(species, ionisation)

        self.raw_data = data

        # unpack
        ne = data['ne']
        te = data['te']
        rate = np.log10(data['rate'])

        # store limits of data
        self.density_range = ne.min(), ne.max()
        self.temperature_range = te.min(), te.max()

        # interpolate rate
        # using nearest extrapolation to avoid infinite values at 0 for some rates
        extrapolation_type = 'nearest' if extrapolate else 'none'
        self._rate = Interpolator2DArray(np.log10(ne), np.log10(te), rate, 'cubic', extrapolation_type, INFINITY, INFINITY)

    cpdef double evaluate(self, double electron_density, double electron_temperature) except? -1e999:

        # need to handle zeros, also density and temperature can become negative due to cubic interpolation
        if electron_density <= 0 or electron_temperature <= 0:
            return 0

        # calculate rate and convert from log10 space to linear space
        return 10 ** self._rate.evaluate(log10(electron_density), log10(electron_temperature))


cdef class NullLineRadiationPower(CoreLineRadiationPower):
    """
    A line radiation power rate that always returns zero.
    Needed for use cases where the required atomic data is missing.
    """

    cpdef double evaluate(self, double electron_density, double electron_temperature) except? -1e999:
        return 0.0


cdef class ContinuumPower(CoreContinuumPower):
    """
    Recombination continuum radiated power coefficient.

    The data is interpolated with cubic spline in log-log space.
    Nearest neighbour extrapolation is used if extrapolate is True.

    :param Element species: Element object defining the ion type.
    :param int ionisation: Charge state of the ion.
    :param dict data: Recombination continuum radiated power rate dictionary containing the following entries:

    |      'ne': 1D array of size (N) with electron density in m^-3,
    |      'te': 1D array of size (M) with electron temperature in eV,
    |      'rate': 2D array of size (N, M) with radiated power rate in W.m^3.

    :param bint extrapolate: Enable extrapolation (default=False).

    :ivar tuple density_range: Electron density interpolation range.
    :ivar tuple temperature_range: Electron temperature interpolation range.
    :ivar dict raw_data: Dictionary containing the raw data.
    """

    def __init__(self, species, ionisation, dict data, extrapolate=False):

        super().__init__(species, ionisation)

        self.raw_data = data

        # unpack
        ne = data['ne']
        te = data['te']
        rate = np.log10(data['rate'])

        # store limits of data
        self.density_range = ne.min(), ne.max()
        self.temperature_range = te.min(), te.max()

        # interpolate rate
        # using nearest extrapolation to avoid infinite values at 0 for some rates
        extrapolation_type = 'nearest' if extrapolate else 'none'
        self._rate = Interpolator2DArray(np.log10(ne), np.log10(te), rate, 'cubic', extrapolation_type, INFINITY, INFINITY)

    cpdef double evaluate(self, double electron_density, double electron_temperature) except? -1e999:

        # need to handle zeros, also density and temperature can become negative due to cubic interpolation
        if electron_density <= 0 or electron_temperature <= 0:
            return 0

        # calculate rate and convert from log10 space to linear space
        return 10 ** self._rate.evaluate(log10(electron_density), log10(electron_temperature))


cdef class NullContinuumPower(CoreContinuumPower):
    """
    A continuum radiation power rate that always returns zero.
    Needed for use cases where the required atomic data is missing.
    """

    cpdef double evaluate(self, double electron_density, double electron_temperature) except? -1e999:
        return 0.0


cdef class CXRadiationPower(CoreCXRadiationPower):
    """
    Charge exchange radiated power coefficient.

    The data is interpolated with cubic spline in log-log space.
    Linear extrapolation is used if extrapolate is True.

    :param Element species: Element object defining the ion type.
    :param int ionisation: Charge state of the ion.
    :param dict data: CX radiated power rate dictionary containing the following entries:

    |      'ne': 1D array of size (N) with electron density in m^-3,
    |      'te': 1D array of size (M) with electron temperature in eV,
    |      'rate': 2D array of size (N, M) with radiated power rate in W.m^3.

    :param bint extrapolate: Enable extrapolation (default=False).

    :ivar tuple density_range: Electron density interpolation range.
    :ivar tuple temperature_range: Electron temperature interpolation range.
    :ivar dict raw_data: Dictionary containing the raw data.
    """

    def __init__(self, species, ionisation, dict data, extrapolate=False):

        super().__init__(species, ionisation)

        self.raw_data = data

        # unpack
        ne = data['ne']
        te = data['te']
        rate = np.log10(data['rate'])

        # store limits of data
        self.density_range = ne.min(), ne.max()
        self.temperature_range = te.min(), te.max()

        # interpolate rate
        extrapolation_type = 'linear' if extrapolate else 'none'
        self._rate = Interpolator2DArray(np.log10(ne), np.log10(te), rate, 'cubic', extrapolation_type, INFINITY, INFINITY)

    cpdef double evaluate(self, double electron_density, double electron_temperature) except? -1e999:

        # need to handle zeros, also density and temperature can become negative due to cubic interpolation
        if electron_density <= 0 or electron_temperature <= 0:
            return 0

        # calculate rate and convert from log10 space to linear space
        return 10 ** self._rate.evaluate(log10(electron_density), log10(electron_temperature))


cdef class NullCXRadiationPower(CoreCXRadiationPower):
    """
    A CX radiation power rate that always returns zero.
    Needed for use cases where the required atomic data is missing.
    """

    cpdef double evaluate(self, double electron_density, double electron_temperature) except? -1e999:
        return 0.0
