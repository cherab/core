
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


cdef class LineRadiationPower(CoreLineRadiationPower):
    """Base class for radiated powers."""

    def __init__(self, species, ionisation, dict data, extrapolate=False):

        super().__init__(species, ionisation)

        self.raw_data = data

        # unpack
        ne = data['ne']
        te = data['te']
        rate =  data['rate']

        # store limits of data
        self.density_range = ne.min(), ne.max()
        self.temperature_range = te.min(), te.max()

        # interpolate rate
        self._rate = Interpolate2DCubic(
            ne, te, rate, extrapolate=extrapolate, extrapolation_type="quadratic"
        )

    cdef double evaluate(self, double electron_density, double electron_temperature) except? -1e999:

        # prevent -ve values (possible if extrapolation enabled)
        return max(0, self._rate.evaluate(electron_density, electron_temperature))


cdef class NullLineRadiationPower(CoreLineRadiationPower):
    """
    A line radiation power rate that always returns zero.
    Needed for use cases where the required atomic data is missing.
    """

    cdef double evaluate(self, double electron_density, double electron_temperature) except? -1e999:
        return 0.0


cdef class ContinuumPower(CoreContinuumPower):
    """Base class for radiated powers."""

    def __init__(self, species, ionisation, dict data, extrapolate=False):

        super().__init__(species, ionisation)

        self.raw_data = data

        # unpack
        ne = data['ne']
        te = data['te']
        rate =  data['rate']

        # store limits of data
        self.density_range = ne.min(), ne.max()
        self.temperature_range = te.min(), te.max()

        # interpolate rate
        self._rate = Interpolate2DCubic(
            ne, te, rate, extrapolate=extrapolate, extrapolation_type="quadratic"
        )

    cdef double evaluate(self, double electron_density, double electron_temperature) except? -1e999:

        # prevent -ve values (possible if extrapolation enabled)
        return max(0, self._rate.evaluate(electron_density, electron_temperature))


cdef class NullContinuumPower(CoreContinuumPower):
    """
    A continuum radiation power rate that always returns zero.
    Needed for use cases where the required atomic data is missing.
    """

    cdef double evaluate(self, double electron_density, double electron_temperature) except? -1e999:
        return 0.0


cdef class CXRadiationPower(CoreCXRadiationPower):
    """Base class for radiated powers."""

    def __init__(self, species, ionisation, dict data, extrapolate=False):

        super().__init__(species, ionisation)

        self.raw_data = data

        # unpack
        ne = data['ne']
        te = data['te']
        rate =  data['rate']

        # store limits of data
        self.density_range = ne.min(), ne.max()
        self.temperature_range = te.min(), te.max()

        # interpolate rate
        self._rate = Interpolate2DCubic(
            ne, te, rate, extrapolate=extrapolate, extrapolation_type="quadratic"
        )

    cdef double evaluate(self, double electron_density, double electron_temperature) except? -1e999:

        # prevent -ve values (possible if extrapolation enabled)
        return max(0, self._rate.evaluate(electron_density, electron_temperature))


cdef class NullCXRadiationPower(CoreCXRadiationPower):
    """
    A CX radiation power rate that always returns zero.
    Needed for use cases where the required atomic data is missing.
    """

    cdef double evaluate(self, double electron_density, double electron_temperature) except? -1e999:
        return 0.0
