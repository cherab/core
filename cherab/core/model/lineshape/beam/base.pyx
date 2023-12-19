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


cdef class BeamLineShapeModel:
    """
    A base class for building beam emission line shapes.

    :param Line line: The emission line object for this line shape.
    :param float wavelength: The rest wavelength for this emission line.
    :param Beam beam: The beam class that is emitting.
    :param AtomicData atomic_data: The atomic data provider.
    """

    def __init__(self, Line line, double wavelength, Beam beam, AtomicData atomic_data):

        self.line = line
        self.wavelength = wavelength
        self.beam = beam
        self.atomic_data = atomic_data

    cpdef Spectrum add_line(self, double radiance, Point3D beam_point, Point3D plasma_point,
                            Vector3D beam_direction, Vector3D observation_direction, Spectrum spectrum):
        raise NotImplementedError('Child lineshape class must implement this method.')
