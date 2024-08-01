# cython: language_level=3

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

from raysect.optical cimport Spectrum, Point3D, Vector3D

from cherab.core.atomic cimport Line, AtomicData
from cherab.core.species cimport Species
from cherab.core.plasma cimport Plasma
from cherab.core.model.lineshape.doppler cimport doppler_shift, thermal_broadening
from cherab.core.model.lineshape.gaussian cimport add_gaussian_line

cimport cython

# required by numpy c-api
np.import_array()


DEF MULTIPLET_WAVELENGTH = 0
DEF MULTIPLET_RATIO = 1


cdef class MultipletLineShape(LineShapeModel):
    """
    Produces Multiplet line shapes.

    The lineshape radiance is calculated from a base PEC rate that is unresolved. This
    radiance is then divided over a number of components as specified in the multiplet
    argument. The multiplet components are specified with an Nx2 array where N is the
    number of components in the multiplet. The first axis of the array contains the
    wavelengths of each component, the second contains the line ratio for each component.
    The component line ratios must sum to one. For example:

    :param Line line: The emission line object for the base rate radiance calculation.
    :param float wavelength: The rest wavelength of the base emission line.
    :param Species target_species: The target plasma species that is emitting.
    :param Plasma plasma: The emitting plasma object.
    :param AtomicData atomic_data: The atomic data provider.
    :param multiplet: An Nx2 array that specifies the multiplet wavelengths and line ratios.

    .. code-block:: pycon

       >>> from cherab.core.atomic import Line, nitrogen
       >>> from cherab.core.model import ExcitationLine, MultipletLineShape
       >>>
       >>> # multiplet specification in Nx2 array
       >>> multiplet = [[403.509, 404.132, 404.354, 404.479, 405.692], [0.205, 0.562, 0.175, 0.029, 0.029]]
       >>>
       >>> # Adding the multiplet to the plasma model.
       >>> nitrogen_II_404 = Line(nitrogen, 1, ("2s2 2p1 4f1 3G13.0", "2s2 2p1 3d1 3F10.0"))
       >>> excit = ExcitationLine(nitrogen_II_404, lineshape=MultipletLineShape, lineshape_args=[multiplet])
       >>> plasma.models.add(excit)
    """

    def __init__(self, Line line, double wavelength, Species target_species, Plasma plasma, AtomicData atomic_data,
                 object multiplet):

        super().__init__(line, wavelength, target_species, plasma, atomic_data)

        multiplet = np.array(multiplet, dtype=np.float64)

        if not (len(multiplet.shape) == 2 and multiplet.shape[0] == 2):
            raise ValueError("The multiplet specification must be an array of shape (Nx2).")

        if not multiplet[1, :].sum() == 1.0:
            raise ValueError("The multiplet line ratios should sum to one.")

        self._number_of_lines = multiplet.shape[1]
        self._multiplet = multiplet
        self._multiplet_mv = self._multiplet

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef Spectrum add_line(self, double radiance, Point3D point, Vector3D direction, Spectrum spectrum):

        cdef double ts, sigma, shifted_wavelength, component_wavelength, component_radiance
        cdef Vector3D ion_velocity

        ts = self.target_species.distribution.effective_temperature(point.x, point.y, point.z)
        if ts <= 0.0:
            return spectrum

        ion_velocity = self.target_species.distribution.bulk_velocity(point.x, point.y, point.z)

        # calculate the line width
        sigma = thermal_broadening(self.wavelength, ts, self.line.element.atomic_weight)

        for i in range(self._number_of_lines):

            component_wavelength = self._multiplet_mv[MULTIPLET_WAVELENGTH, i]
            component_radiance = radiance * self._multiplet_mv[MULTIPLET_RATIO, i]

            # calculate emission line central wavelength, doppler shifted along observation direction
            shifted_wavelength = doppler_shift(component_wavelength, direction, ion_velocity)

            spectrum = add_gaussian_line(component_radiance, shifted_wavelength, sigma, spectrum)

        return spectrum
