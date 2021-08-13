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

from cherab.core.utility.conversion import PhotonToJ

cimport cython
from libc.math cimport INFINITY
from raysect.core.math.function.float cimport Interpolator1DArray, Interpolator2DArray, Constant1D, Constant2D

# todo: clarify variables

cdef class FunctionY2D(Function2D):
    cdef Function1D _function

    def __init__(self, Function1D function):
        self._function = function

    cdef double evaluate(self, double x, double y) except? -1e999:
        return self._function(y)


cdef class FunctionX2D(Function2D):
    cdef Function1D _function

    def __init__(self, Function1D function):
        self._function = function

    cdef double evaluate(self, double x, double y) except? -1e999:
        return self._function(x)


cdef class BeamStoppingRate(CoreBeamStoppingRate):
    """
    The beam stopping coefficient interpolation class.

    :param data: A dictionary holding the beam coefficient data.
    :param extrapolate: Set to True to enable extrapolation, False to disable (default).
    """

    @cython.cdivision(True)
    def __init__(self, dict data, bint extrapolate=False):

        self.raw_data = data

        # unpack
        e = data["e"]                          # eV/amu
        n = data["n"]                          # m^-3
        t = data["t"]                          # eV
        sen = data["sen"]                      # m^3/s
        st = data["st"] / data["sref"]         # dimensionless

        # store limits of data
        self.beam_energy_range = e.min(), e.max()
        self.density_range = n.min(), n.max()
        self.temperature_range = t.min(), t.max()

        # interpolate
        extrapolation_type_2d = 'nearest' if extrapolate else 'none'
        extrapolation_type_1d = 'quadratic' if extrapolate else 'none'
        if len(e) == 1 and len(n) == 1:
            self._npl_eb = Constant2D(sen[0, 0])
        elif len(e) == 1:
            self._npl_eb = FunctionY2D(Interpolator1DArray(n, sen[0], 'cubic', extrapolation_type_1d, INFINITY))
        elif len(n) == 1:
            self._npl_eb = FunctionX2D(Interpolator1DArray(e, sen[:, 0], 'cubic', extrapolation_type_1d, INFINITY))
        else:
            self._npl_eb = Interpolator2DArray(e, n, sen, 'cubic', extrapolation_type_2d, INFINITY, INFINITY)
        self._tp = Interpolator1DArray(t, st, 'cubic', extrapolation_type_1d, INFINITY)

    cpdef double evaluate(self, double energy, double density, double temperature) except? -1e999:
        """
        Interpolates and returns the beam coefficient for the supplied parameters.

        If the requested data is out-of-range then the call with throw a ValueError exception.

        :param energy: Interaction energy in eV/amu.
        :param density: Target electron density in m^-3
        :param temperature: Target temperature in eV.
        :return: The beam stopping coefficient in m^3.s^-1
        """

        cdef double val

        val = self._npl_eb.evaluate(energy, density)
        if val <= 0:
            return 0.0

        val *= self._tp.evaluate(temperature)
        if val <= 0:
            return 0.0

        return val


cdef class NullBeamStoppingRate(CoreBeamStoppingRate):
    """
    A beam rate that always returns zero.
    Needed for use cases where the required atomic data is missing.
    """

    cpdef double evaluate(self, double energy, double density, double temperature) except? -1e999:
        return 0.0


cdef class BeamPopulationRate(CoreBeamPopulationRate):
    """
    The beam population coefficient interpolation class.

    :param data: A dictionary holding the beam coefficient data.
    :param extrapolate: Set to True to enable extrapolation, False to disable (default).
    """

    @cython.cdivision(True)
    def __init__(self, dict data, bint extrapolate=False):

        self.raw_data = data

        # unpack
        e = data["e"]                          # eV/amu
        n = data["n"]                          # m^-3
        t = data["t"]                          # eV
        sen = data["sen"]                      # dimensionless
        st = data["st"] / data["sref"]         # dimensionless

        # store limits of data
        self.beam_energy_range = e.min(), e.max()
        self.density_range = n.min(), n.max()
        self.temperature_range = t.min(), t.max()

        # interpolate
        extrapolation_type_2d = 'nearest' if extrapolate else 'none'
        extrapolation_type_1d = 'quadratic' if extrapolate else 'none'
        if len(e) == 1 and len(n) == 1:
            self._npl_eb = Constant2D(sen[0, 0])
        elif len(e) == 1:
            self._npl_eb = FunctionY2D(Interpolator1DArray(n, sen[0], 'cubic', extrapolation_type_1d, INFINITY))
        elif len(n) == 1:
            self._npl_eb = FunctionX2D(Interpolator1DArray(e, sen[:, 0], 'cubic', extrapolation_type_1d, INFINITY))
        else:
            self._npl_eb = Interpolator2DArray(e, n, sen, 'cubic', extrapolation_type_2d, INFINITY, INFINITY)
        self._tp = Interpolator1DArray(t, st, 'cubic', extrapolation_type_1d, INFINITY)

    cpdef double evaluate(self, double energy, double density, double temperature) except? -1e999:
        """
        Interpolates and returns the beam coefficient for the supplied parameters.

        If the requested data is out-of-range then the call with throw a ValueError exception.

        :param energy: Interaction energy in eV/amu.
        :param density: Target electron density in m^-3
        :param temperature: Target temperature in eV.
        :return: The beam population coefficient in dimensionless units.
        """

        cdef double val

        val = self._npl_eb.evaluate(energy, density)
        if val <= 0:
            return 0.0

        val *= self._tp.evaluate(temperature)
        if val <= 0:
            return 0.0

        return val


cdef class NullBeamPopulationRate(CoreBeamPopulationRate):
    """
    A beam rate that always returns zero.
    Needed for use cases where the required atomic data is missing.
    """

    cpdef double evaluate(self, double energy, double density, double temperature) except? -1e999:
        return 0.0


cdef class BeamEmissionPEC(CoreBeamEmissionPEC):
    """
    The beam emission coefficient interpolation class.

    :param data: A dictionary holding the beam coefficient data.
    :param wavelength: The natural wavelength of the emission line associated with the rate data in nm.
    :param extrapolate: Set to True to enable extrapolation, False to disable (default).
    """

    @cython.cdivision(True)
    def __init__(self, dict data, double wavelength, bint extrapolate=False):

        self.raw_data = data

        # unpack
        e = data["e"]                                   # eV/amu
        n = data["n"]                                   # m^-3
        t = data["t"]                                   # eV
        sen = PhotonToJ.to(data["sen"], wavelength)     # W.m^3/s
        st = data["st"] / data["sref"]                  # dimensionless

        # store limits of data
        self.beam_energy_range = e.min(), e.max()
        self.density_range = n.min(), n.max()
        self.temperature_range = t.min(), t.max()

        # interpolate
        extrapolation_type_2d = 'nearest' if extrapolate else 'none'
        extrapolation_type_1d = 'quadratic' if extrapolate else 'none'
        if len(e) == 1 and len(n) == 1:
            self._npl_eb = Constant2D(sen[0, 0])
        elif len(e) == 1:
            self._npl_eb = FunctionY2D(Interpolator1DArray(n, sen[0], 'cubic', extrapolation_type_1d, INFINITY))
        elif len(n) == 1:
            self._npl_eb = FunctionX2D(Interpolator1DArray(e, sen[:, 0], 'cubic', extrapolation_type_1d, INFINITY))
        else:
            self._npl_eb = Interpolator2DArray(e, n, sen, 'cubic', extrapolation_type_2d, INFINITY, INFINITY)
        self._tp = Interpolator1DArray(t, st, 'cubic', extrapolation_type_1d, INFINITY)

    cpdef double evaluate(self, double energy, double density, double temperature) except? -1e999:
        """
        Interpolates and returns the beam coefficient for the supplied parameters.

        If the requested data is out-of-range then the call with throw a ValueError exception.

        :param energy: Interaction energy in eV/amu.
        :param density: Target electron density in m^-3
        :param temperature: Target temperature in eV.
        :return: The beam emission coefficient in m^3.s^-1
        """

        cdef double val

        val = self._npl_eb.evaluate(energy, density)
        if val <= 0:
            return 0.0

        val *= self._tp.evaluate(temperature)
        if val <= 0:
            return 0.0

        return val


cdef class NullBeamEmissionPEC(CoreBeamEmissionPEC):
    """
    A beam rate that always returns zero.
    Needed for use cases where the required atomic data is missing.
    """

    cpdef double evaluate(self, double energy, double density, double temperature) except? -1e999:
        return 0.0
