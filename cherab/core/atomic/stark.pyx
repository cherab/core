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

from raysect.core.math.function.float cimport Interpolator3DArray, Function3D

from libc.math cimport INFINITY, log10

cimport cython


DEF ZERO_THRESHOLD = 1.e-300


cdef class StarkStructure():
    """
    Provides ratios of linear Stark components of the MSE spectrum for any given values of
    beam energy, electron density and magnetic field strength when the observation direction
    is perependicular to the electric field.

    :ivar ndarray index: Stark component indices: :math:`\\Delta E = \\frac{3}{2} e a_0 (\\pm i) E`.
        Contains only non-negative values. It is assumed that components with negative and positive
        indices have the same intensity (electric field vector is perpendicular to collisional axis).
    :ivar ndarray polarisation: Polarisation of the Stark components.
        0 stands for :math:`\\pi` polarisation and 1 stands for :math:`\\sigma` polarisation.
    """

    cdef double[::1] evaluate(self, double energy, double density, double b_field):

        raise NotImplementedError("The evaluate() virtual method must be implemented.")

    def __call__(self, double energy, double density, double b_field):
        """
        Returns an array with Stark component relative intensities for the given values of
        beam energy, electron density and magnetic field strength.

        The intensities are normalised as follows: :math:`I_0 + 2\\sum_{i\\geq 1}I_i = 1`.

        :param double energy: Beam energy in eV/amu.
        :param double density: Electron density in m-3.
        :param double b_field: Magnetic field in T.
        """

        return np.asarray(self.evaluate(energy, density, b_field))


cdef class InterpolatedStarkStructure(StarkStructure):
    """
    Provides interpolated ratios of linear Stark components of the MSE spectrum when
    the observation direction is perependicular to the electric field.

    :param dict data: Dictionary containing the relative
           intensities of Stark components with the following keys:
    |      'energy': A 1D array of shape (N,) with beam energy in eV/amu,
    |      'ne: A 1D array of shape (M,) with electron density in m-3,
    |      'b': A 1D array of shape (K,) with magnetic field strength in T.
    |      'index': A 1D integer array of shape (L,) with Stark component indices:
    |                      :math:`\\Delta E = \\frac{3}{2} e a_0 (\\pm i) E`.
    |                      Contains only non-negative values.
    |      'polarisation': A 1D integer array of shape (L,) with component polarisation
    |                      0 for :math:`\\pi`-polarisation,
    |                      1 for :math:`\\sigma`-polarisation.
    |      'ratio': A 4D array of shape (N, M, K, L) with Stark component relative intensities.
    |                      The intensities are normalised as follows:
    |                      :math:`I_0 + 2\\sum_{i\\geq 1}I_i = 1`.

    :param bint extrapolate: Enable extrapolation (default=False).

    :ivar tuple beam_energy_range: The interpolation range of beam energy.
    :ivar tuple density_range: The interpolation range of electron density.
    :ivar tuple b_field_range: The interpolation range of magnetic field strength.
    :ivar dict raw_data: Dictionary containing the raw data.
    :ivar ndarray index: Stark component indices: :math:`\\Delta E = \\frac{3}{2} e a_0 (\\pm i) E`.
        Contains only non-negative values. It is assumed that components with negative and positive
        indices have the same intensity (electric field vector is perpendicular to collisional axis).
    :ivar ndarray polarisation: Polarisation of the Stark components.
        0 stands for :math:`\\pi` polarisation and 1 stands for :math:`\\sigma` polarisation.
    """

    def __init__(self, dict data, bint extrapolate=False):

        self.raw_data = data

        b = np.asarray(data['b'])
        ne = np.asarray(data['ne'])
        energy = np.asarray(data['energy'])
        ratio = np.asarray(data['ratio'])
        polarisation = np.array(data['polarisation'], dtype=np.int32)
        index = np.array(data['index'], dtype=np.int32)

        if ratio.shape[3] != index.shape[0] != polarisation.shape[0]:
            raise ValueError("Fields: 'index', 'polarisation' and 'ratio' must provide data for the same number of Stark components.")

        if np.any((polarisation < 0) + (polarisation > 1)):
            raise ValueError("Field 'polarisation' must contain only 0 (pi) or 1 (sigma).")

        if np.any(index < 0):
            raise ValueError("Field 'index' must contain non-negative values.")

        # protect polarisation and index arrays
        polarisation.flags.writeable = False
        index.flags.writeable = False
        self.polarisation = polarisation
        self.index = index

        # argument ranges
        self.beam_energy_range = energy.min(), energy.max()
        self.density_range = ne.min(), ne.max()
        self.b_field_range = b.min(), b.max()

        extrapolation_type = 'nearest' if extrapolate else 'none'

        # normalise input ratios
        ratio_sum = 0
        for i in range(ratio.shape[3]):
            ratio_sum += 2 * ratio[:, :, :, i] if index[i] else ratio[:, :, :, i]
        ratio /= ratio_sum[:, :, :, None]

        ratio_functions = []
        for i in range(ratio.shape[3]):
            ratio_func = Interpolator3DArray(energy, np.log10(ne), b, ratio[:, :, :, i], 'cubic', extrapolation_type, INFINITY, INFINITY, INFINITY)
            ratio_functions.append(ratio_func)
        self._ratio_functions = ratio_functions

        ratios = np.zeros(self.index.size, dtype=np.float64)

        self._ratios_mv = ratios
        self.index_mv = index
        self.polarisation_mv = polarisation

        self._cached_energy = 0
        self._cached_density = 0
        self._cached_b_field = 0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @cython.cdivision(True)
    cdef double[::1] evaluate(self, double energy, double density, double b_field):
        """
        Returns an array with Stark component relative intensities for the given values of
        beam energy, electron density and magnetic field strength.

        The intensities are normalised as follows: :math:`I_0 + 2\\sum_{i\\geq 1}I_i = 1`

        :param double energy: Beam energy in eV/amu.
        :param double density: Electron density in m-3.
        :param double b_field: Magnetic field in T.
        """

        cdef int i
        cdef double ratio_sum
        cdef Function3D ratio

        if self._cached_energy == energy and self._cached_density == density and self._cached_b_field == b_field:

            return self._ratios_mv

        # safety check in case if used stand-alone
        if energy < 0:
            raise ValueError('Argument "energy" (beam energy) must be non-negative.')
        if density < 0:
            raise ValueError('Argument "density" (electron density) must be non-negative.')
        if b_field < 0:
            raise ValueError('Argument "b_field" (magnetic field strength) must be non-negative.')

        if density < ZERO_THRESHOLD:
            density = ZERO_THRESHOLD

        ratio_sum = 0
        for i in range(self._ratios_mv.shape[0]):
            ratio = self._ratio_functions[i]
            self._ratios_mv[i] = ratio.evaluate(energy, log10(density), b_field)

            ratio_sum += 2 * self._ratios_mv[i] if self.index_mv[i] else self._ratios_mv[i]

        # normalise output ratios
        if ratio_sum > 0:
            for i in range(self._ratios_mv.shape[0]):
                self._ratios_mv[i] /= ratio_sum

        self._cached_energy = energy
        self._cached_density = density
        self._cached_b_field = b_field

        return self._ratios_mv
