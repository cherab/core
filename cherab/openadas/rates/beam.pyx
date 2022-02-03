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

from cherab.core.utility.conversion import PhotonToJ

cimport cython
from libc.math cimport INFINITY, log10
from raysect.core.math.function.float cimport Interpolator1DArray, Interpolator2DArray, Constant2D, Arg2D
from cherab.core.math cimport IsoMapper2D

# todo: clarify variables

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
        sen = np.log10(data["sen"])                      # m^3/s
        st = np.log10(data["st"] / data["sref"])         # dimensionless

        # store limits of data
        self.beam_energy_range = e.min(), e.max()
        self.density_range = n.min(), n.max()
        self.temperature_range = t.min(), t.max()

        # interpolate
        extrapolation_type_2d = 'linear' if extrapolate else 'none'
        extrapolation_type_1d = 'quadratic' if extrapolate else 'none'
        if len(e) == 1 and len(n) == 1:
            self._npl_eb = Constant2D(sen[0, 0])
        elif len(e) == 1:
            self._npl_eb = IsoMapper2D(Arg2D('y'), Interpolator1DArray(np.log10(n), sen[0], 'cubic', extrapolation_type_1d, INFINITY))
        elif len(n) == 1:
            self._npl_eb = IsoMapper2D(Arg2D('x'), Interpolator1DArray(np.log10(e), sen[:, 0], 'cubic', extrapolation_type_1d, INFINITY))
        else:
            self._npl_eb = Interpolator2DArray(np.log10(e), np.log10(n), sen, 'cubic', extrapolation_type_2d, INFINITY, INFINITY)
        self._tp = Interpolator1DArray(np.log10(t), st, 'cubic', extrapolation_type_1d, INFINITY)

    cpdef double evaluate(self, double energy, double density, double temperature) except? -1e999:
        """
        Interpolates and returns the beam coefficient for the supplied parameters.

        If the requested data is out-of-range then the call with throw a ValueError exception.

        :param energy: Interaction energy in eV/amu.
        :param density: Target electron density in m^-3
        :param temperature: Target temperature in eV.
        :return: The beam stopping coefficient in m^3.s^-1
        """

        # need to handle zeros, also density and temperature can become negative due to cubic interpolation
        if energy < 1.e-300:
            energy = 1.e-300

        if density < 1.e-300:
            density = 1.e-300

        if temperature < 1.e-300:
            temperature = 1.e-300

        # calculate rate and convert from log10 space to linear space
        return 10 ** (self._npl_eb.evaluate(log10(energy), log10(density)) + self._tp.evaluate(log10(temperature)))


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
        sen = np.log10(data["sen"])                      # dimensionless
        st = np.log10(data["st"] / data["sref"])         # dimensionless

        # store limits of data
        self.beam_energy_range = e.min(), e.max()
        self.density_range = n.min(), n.max()
        self.temperature_range = t.min(), t.max()

        # interpolate
        extrapolation_type_2d = 'linear' if extrapolate else 'none'
        extrapolation_type_1d = 'quadratic' if extrapolate else 'none'
        if len(e) == 1 and len(n) == 1:
            self._npl_eb = Constant2D(sen[0, 0])
        elif len(e) == 1:
            self._npl_eb = IsoMapper2D(Arg2D('y'), Interpolator1DArray(np.log10(n), sen[0], 'cubic', extrapolation_type_1d, INFINITY))
        elif len(n) == 1:
            self._npl_eb = IsoMapper2D(Arg2D('x'), Interpolator1DArray(np.log10(e), sen[:, 0], 'cubic', extrapolation_type_1d, INFINITY))
        else:
            self._npl_eb = Interpolator2DArray(np.log10(e), np.log10(n), sen, 'cubic', extrapolation_type_2d, INFINITY, INFINITY)
        self._tp = Interpolator1DArray(np.log10(t), st, 'cubic', extrapolation_type_1d, INFINITY)

    cpdef double evaluate(self, double energy, double density, double temperature) except? -1e999:
        """
        Interpolates and returns the beam coefficient for the supplied parameters.

        If the requested data is out-of-range then the call with throw a ValueError exception.

        :param energy: Interaction energy in eV/amu.
        :param density: Target electron density in m^-3
        :param temperature: Target temperature in eV.
        :return: The beam population coefficient in dimensionless units.
        """

        # need to handle zeros, also density and temperature can become negative due to cubic interpolation
        if energy < 1.e-300:
            energy = 1.e-300

        if density < 1.e-300:
            density = 1.e-300

        if temperature < 1.e-300:
            temperature = 1.e-300

        # calculate rate and convert from log10 space to linear space
        return 10 ** (self._npl_eb.evaluate(log10(energy), log10(density)) + self._tp.evaluate(log10(temperature)))


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

        self.wavelength = wavelength
        self.raw_data = data

        # unpack
        e = data["e"]                                   # eV/amu
        n = data["n"]                                   # m^-3
        t = data["t"]                                   # eV
        sen = np.log10(PhotonToJ.to(data["sen"], wavelength))     # W.m^3/s
        st = np.log10(data["st"] / data["sref"])                  # dimensionless

        # store limits of data
        self.beam_energy_range = e.min(), e.max()
        self.density_range = n.min(), n.max()
        self.temperature_range = t.min(), t.max()

        # interpolate
        extrapolation_type_2d = 'linear' if extrapolate else 'none'
        extrapolation_type_1d = 'quadratic' if extrapolate else 'none'
        if len(e) == 1 and len(n) == 1:
            self._npl_eb = Constant2D(sen[0, 0])
        elif len(e) == 1:
            self._npl_eb = IsoMapper2D(Arg2D('y'), Interpolator1DArray(np.log10(n), sen[0], 'cubic', extrapolation_type_1d, INFINITY))
        elif len(n) == 1:
            self._npl_eb = IsoMapper2D(Arg2D('x'), Interpolator1DArray(np.log10(e), sen[:, 0], 'cubic', extrapolation_type_1d, INFINITY))
        else:
            self._npl_eb = Interpolator2DArray(np.log10(e), np.log10(n), sen, 'cubic', extrapolation_type_2d, INFINITY, INFINITY)
        self._tp = Interpolator1DArray(np.log10(t), st, 'cubic', extrapolation_type_1d, INFINITY)

    cpdef double evaluate(self, double energy, double density, double temperature) except? -1e999:
        """
        Interpolates and returns the beam coefficient for the supplied parameters.

        If the requested data is out-of-range then the call with throw a ValueError exception.

        :param energy: Interaction energy in eV/amu.
        :param density: Target electron density in m^-3
        :param temperature: Target temperature in eV.
        :return: The beam emission coefficient in m^3.s^-1
        """

        # need to handle zeros, also density and temperature can become negative due to cubic interpolation
        if energy < 1.e-300:
            energy = 1.e-300

        if density < 1.e-300:
            density = 1.e-300

        if temperature < 1.e-300:
            temperature = 1.e-300

        # calculate rate and convert from log10 space to linear space
        return 10 ** (self._npl_eb.evaluate(log10(energy), log10(density)) + self._tp.evaluate(log10(temperature)))


cdef class NullBeamEmissionPEC(CoreBeamEmissionPEC):
    """
    A beam rate that always returns zero.
    Needed for use cases where the required atomic data is missing.
    """

    cpdef double evaluate(self, double energy, double density, double temperature) except? -1e999:
        return 0.0
