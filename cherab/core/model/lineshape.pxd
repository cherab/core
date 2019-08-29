# cython: language_level=3

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
from cherab.core cimport Line, Species, Plasma, Beam
from cherab.core.math cimport Function1D, Function2D


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
        int _number_of_lines
        np.ndarray _multiplet
        double[:,::1] _multiplet_mv


cdef class StarkBroadenedLine(LineShapeModel):

    cdef double _aij, _bij, _cij

    pass


cdef class BeamLineShapeModel:

    cdef:

        Line line
        double wavelength
        Beam beam

    cpdef Spectrum add_line(self, double radiance, Point3D beam_point, Point3D plasma_point,
                            Vector3D beam_direction, Vector3D observation_direction, Spectrum spectrum)


cdef class BeamEmissionMultiplet(BeamLineShapeModel):

    cdef:

        Function2D _sigma_to_pi
        Function1D _sigma1_to_sigma0, _pi2_to_pi3, _pi4_to_pi3
