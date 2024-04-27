# Copyright 2016-2023 Euratom
# Copyright 2016-2023 United Kingdom Atomic Energy Authority
# Copyright 2016-2023 Centro de Investigaciones Energéticas, Medioambientales y Tecnológicas
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
from libc.math cimport log10, log, M_PI, sqrt
from raysect.core.math.function.float cimport Interpolator2DArray
from cherab.core.utility.constants cimport RYDBERG_CONSTANT_EV, SPEED_OF_LIGHT, ELEMENTARY_CHARGE, PLANCK_CONSTANT

cimport cython


DEF EULER_GAMMA = 0.5772156649015329

cdef double PH_TO_EV_FACTOR = PLANCK_CONSTANT * SPEED_OF_LIGHT * 1e9 / ELEMENTARY_CHARGE


cdef class FreeFreeGauntFactor(CoreFreeFreeGauntFactor):
    r"""
    The temperature-averaged free-free Gaunt factors interpolated in the space of parameters:
    :math:`u = h{\nu}/kT` and :math:`{\gamma}^{2} = Z^{2}Ry/kT`.
    See T.R. Carson, 1988, Astron. & Astrophys., 189,
    `319 <https://ui.adsabs.harvard.edu/#abs/1988A&A...189..319C/abstract>`_ for details.

    The cubic interpolation in a semi-log space is used.

    The Born approximation and classical limits are used outside the interpolation range.

    :param dict data: Dictionary containing the Gaunt factor data with the following keys:

    |      'u': A 1D array of real values.
    |      'gamma2': A 1D array of real values.
    |      'gaunt_factor': 2D array of real values storing the Gaunt factor values at u, gamma2.

    :ivar tuple u_range: The interpolation range of `u` parameter.
    :ivar tuple gamma2_range: The interpolation range of :math:`\\gamma^2` parameter.
    :ivar dict raw_data: Dictionary containing the raw data.
    """

    def __init__(self, dict data):

        self.raw_data = data

        u = data['u']
        gamma2 = data['gamma2']
        gaunt_factor = data['gaunt_factor']

        self._u_min = u.min()
        self._u_max = u.max()
        self._gamma2_min = gamma2.min()
        self._gamma2_max = gamma2.max()

        self.u_range = (self._u_min, self._u_max)
        self.gamma2_range = (self._gamma2_min, self._gamma2_max)

        self._gaunt_factor = Interpolator2DArray(np.log10(u), np.log10(gamma2), gaunt_factor, 'cubic', 'none', 0, 0)

    @cython.cdivision(True)
    cpdef double evaluate(self, double z, double temperature, double wavelength) except? -1e999:
        """
        Returns the temperature-averaged free-free Gaunt factor for the supplied parameters.

        :param double z: Species charge or effective plasma charge.
        :param double temperature: Electron temperature in eV.
        :param double wavelength: Spectral wavelength.

        :return: free-free Gaunt factor
        """

        cdef:
            double u, gamma2

        if z == 0:

            return 0

        gamma2 = z * z * RYDBERG_CONSTANT_EV / temperature
        u = PH_TO_EV_FACTOR / (temperature * wavelength)

        # classical limit
        if u >= self._u_max or gamma2 >= self._gamma2_max:

            return 1

        # Born approximation limit
        if u < self._u_min or gamma2 < self._gamma2_min:

            return sqrt(3) / M_PI * (log(4 / u) - EULER_GAMMA)

        return self._gaunt_factor.evaluate(log10(u), log10(gamma2))
