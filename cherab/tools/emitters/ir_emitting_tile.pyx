
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

from raysect.optical cimport World, Primitive, Ray, Spectrum, Point3D, Vector3D, AffineMatrix3D, Normal3D
from raysect.optical.material.material cimport NullVolume
from raysect.optical.material.material cimport Material
cimport cython


cdef class IREmittingTile(NullVolume):
    """
    Custom material for IR tile emission with reflections.

    Combined behaviour of two materials. One represents the tiles thermal emission spectrum.
    The other represents the tiles reflection properties.

    :param Material surface_emission_material: The tiles thermal emission material.
    :param Material reflecting_tile_material: The tiles reflection properties.
    """

    cdef:
        Material surface_emission_material, reflecting_tile_material

    def __init__(self, Material surface_emission_material, Material reflecting_tile_material):
        super().__init__()
        self.surface_emission_material = surface_emission_material
        self.reflecting_tile_material = reflecting_tile_material

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef Spectrum evaluate_surface(self, World world, Ray ray, Primitive primitive, Point3D hit_point,
                                    bint exiting, Point3D inside_point, Point3D outside_point,
                                    Normal3D normal, AffineMatrix3D world_to_primitive, AffineMatrix3D primitive_to_world):

        cdef:
            Spectrum reflected_spectrum, emitted_spectrum
            int index

        reflected_spectrum = self.reflecting_tile_material.evaluate_surface(
            world, ray, primitive, hit_point, exiting, inside_point, outside_point,
            normal, world_to_primitive, primitive_to_world
        )

        emitted_spectrum = self.surface_emission_material.evaluate_surface(
            world, ray, primitive, hit_point, exiting, inside_point, outside_point,
            normal, world_to_primitive, primitive_to_world
        )

        reflected_spectrum.add_array(emitted_spectrum.samples_mv)

        return reflected_spectrum

