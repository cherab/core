# Copyright 2014-2017 United Kingdom Atomic Energy Authority
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

from numpy cimport ndarray
from raysect.optical cimport Node, World, Primitive, Ray, Spectrum, SpectralFunction, Point3D, Vector3D, AffineMatrix3D

from cherab.core cimport Species, Plasma, Beam, Line, AtomicData, BeamCXRate
from cherab.core.beam cimport BeamModel


cdef class BeamCXLine(BeamModel):

    cdef:
        Line _line
        Species _target_species
        double _wavelength
        BeamCXRate _ground_beam_rate
        list _excited_beam_data

    cdef inline double _composite_cx_rate(self, double x, double y, double z, double interaction_energy,
                                          Vector3D donor_velocity, double receiver_temperature, double receiver_density)

    cdef inline double _beam_population(self, double x, double y, double z, Vector3D beam_velocity, list population_data)

    cdef inline int _populate_cache(self) except -1
