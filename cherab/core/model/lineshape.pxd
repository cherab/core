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
cimport numpy as np

from raysect.optical cimport Spectrum, Point3D, Vector3D
from cherab.core.atomic.line cimport Line
from cherab.core.species cimport Species
from cherab.core.plasma.node cimport Plasma

cpdef double doppler_shift(double wavelength, Vector3D observation_direction, Vector3D velocity)

cpdef double thermal_broadening(double wavelength, double temperature, double atomic_weight)

cpdef Spectrum add_gaussian_line(double radiance, double wavelength, double sigma, Spectrum spectrum)


cdef class LineShapeModel:

    cdef:
        Line line
        double wavelength
        Species target_species
        Plasma plasma

    cpdef Spectrum add_line(self, double radiance, Point3D point, Vector3D direction, Spectrum spectrum)


cdef class GaussianLine(LineShapeModel):
    pass


cdef class MultipletLineShape(LineShapeModel):

    cdef:
        int number_of_lines
        np.ndarray multiplet


cdef class StarkBroadenedLine(LineShapeModel):

    cdef double _aij, _bij, _cij

    pass
