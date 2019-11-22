
# Copyright 2016-2018 Euratom
# Copyright 2016-2018 United Kingdom Atomic Energy Authority
# Copyright 2016-2018 Centro de Investigaciones Energéticas, Medioambientales y Tecnológicas
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


cdef class IonisationRate(CoreIonisationRate):

    def __init__(self, dict data, extrapolate=False):
        """
        :param data: Dictionary containing rate data.
        :param extrapolate: Enable extrapolation (default=False).
        """

        self.raw_data = data

        # unpack
        ne = data['ne']
        te = data['te']
        rate =  np.log10(data['rate'])

        # store limits of data
        self.density_range = ne.min(), ne.max()
        self.temperature_range = te.min(), te.max()

        # interpolate rate
        self._rate = Interpolate2DCubic(
            ne, te, rate, extrapolate=extrapolate, extrapolation_type="quadratic"
        )

    cpdef double evaluate(self, double density, double temperature) except? -1e999:

        # calculate rate and convert from log10 space to linear space
        return 10 ** self._rate.evaluate(density, temperature)


cdef class NullIonisationRate(CoreIonisationRate):
    """
    A PEC rate that always returns zero.
    Needed for use cases where the required atomic data is missing.
    """

    cpdef double evaluate(self, double density, double temperature) except? -1e999:
        return 0.0


cdef class RecombinationRate(CoreRecombinationRate):

    def __init__(self, dict data, extrapolate=False):
        """
        :param data: Dictionary containing rate data.
        :param extrapolate: Enable extrapolation (default=False).
        """

        self.raw_data = data

        # unpack
        ne = data['ne']
        te = data['te']
        rate =  np.log10(data['rate'])

        # store limits of data
        self.density_range = ne.min(), ne.max()
        self.temperature_range = te.min(), te.max()

        # interpolate rate
        self._rate = Interpolate2DCubic(
            ne, te, rate, extrapolate=extrapolate, extrapolation_type="quadratic"
        )


    cpdef double evaluate(self, double density, double temperature) except? -1e999:

        # calculate rate and convert from log10 space to linear space
        return 10 ** self._rate.evaluate(density, temperature)


cdef class NullRecombinationRate(CoreRecombinationRate):
    """
    A PEC rate that always returns zero.
    Needed for use cases where the required atomic data is missing.
    """

    cpdef double evaluate(self, double density, double temperature) except? -1e999:
        return 0.0


cdef class ThermalCXRate(CoreThermalCXRate):

    def __init__(self, dict data, extrapolate=False):
        """
        :param data: Dictionary containing rate data.
        :param extrapolate: Enable extrapolation (default=False).
        """

        self.raw_data = data

        # unpack
        ne = data['ne']
        te = data['te']
        rate =  np.log10(data['rate'])

        # store limits of data
        self.density_range = ne.min(), ne.max()
        self.temperature_range = te.min(), te.max()

        # interpolate rate
        self._rate = Interpolate2DCubic(
            ne, te, rate, extrapolate=extrapolate, extrapolation_type="quadratic"
        )

    cpdef double evaluate(self, double density, double temperature) except? -1e999:

        # calculate rate and convert from log10 space to linear space
        return 10 ** self._rate.evaluate(density, temperature)


cdef class NullThermalCXRate(CoreThermalCXRate):
    """
    A PEC rate that always returns zero.
    Needed for use cases where the required atomic data is missing.
    """

    cpdef double evaluate(self, double density, double temperature) except? -1e999:
        return 0.0