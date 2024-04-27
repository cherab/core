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

from libc.math cimport INFINITY

from raysect.core.math.function.float cimport Interpolator2DArray
from cherab.core.atomic cimport Element


cdef class FractionalAbundance(CoreFractionalAbundance):
    """
    Fractional abundance in thermodynamic equilibrium.

    The data is interpolated with cubic spline.
    Linear extrapolation is used when permit_extrapolation is True.

    :param Element species: the radiating element
    :param int ionisation: Charge state of the ion.
    :param dict data: Fractional abundance  dictionary containing the following fields:

    |      'ne': 1D array of size (N) with electron density in m^-3,
    |      'te': 1D array of size (M) with electron temperature in eV,
    |      'fractional_abundance': 2D array of size (N, M) with fractional abundance.

    :param bint extrapolate: Enable extrapolation (default=False).

    :ivar tuple density_range: Electron density interpolation range.
    :ivar tuple temperature_range: Electron temperature interpolation range.
    :ivar dict raw_data: Dictionary containing the raw data.
    """

    def __init__(self, Element species, int ionisation, dict data, bint extrapolate=False):

        super().__init__(species, ionisation)

        self.raw_data = data

        # unpack
        ne = data['ne']
        te = data['te']
        fractional_abundance = data['fractional_abundance']

        # store limits of data
        self.density_range = ne.min(), ne.max()
        self.temperature_range = te.min(), te.max()

        extrapolation_type = 'linear' if extrapolate else 'none'
        self._abundance = Interpolator2DArray(ne, te, fractional_abundance, 'cubic', extrapolation_type, INFINITY, INFINITY)

    cdef double evaluate(self, double electron_density, double electron_temperature) except? -1e999:

        if electron_density <= 0 or electron_temperature <= 0:
            return 0.0

        return self._abundance.evaluate(electron_density, electron_temperature)
