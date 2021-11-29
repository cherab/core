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

from scipy.constants import atomic_mass, electron_mass

from raysect.core import translate, Vector3D
from raysect.primitive import Cylinder, Subtract
from raysect.core.math.function.float import Discrete2DMesh

from cherab.core import Plasma, Species, Maxwellian
from cherab.core.math.function import ConstantVector3D
from cherab.core.math.mappers import AxisymmetricMapper


# TODO: add Discrete2DMesh to raysect.core.math.funciton.vector3d
#       and then add support for b_field, electron_velocity and
#       species_velocity.
def plasma_profiles_to_mesh2d(vertex_coords, triangles, electron_density, electron_temperature, species_density, species_temperature):
    """
    Converts plasma profiles defined on a triangular 2d mesh to `Discrete2DMesh` functions.
    Returns the dictionary with the following keys:
    'electron_density', 'electron_temperature', 'species_density', 'species_temperature'.

    :param ndarray vertex_coords: An array of vertex coordinates (R, Z) with shape Nx2.
    :param ndarray triangles: An array of vertex indices defining the mesh triangles,
                              with shape Mx3.
    :param ndarray electron_density: Electron density profile of shape Mx1.
    :param ndarray electron_temperature: Electron temperature profile of shape Mx1.
    :param ndarray species_density: A dict with (element/isotope, charge) keys
                                    containing density of each plasma species.
    :param ndarray species_temperature: A dict with (element/isotope, charge) keys
                                        containing temperature of each plasma species.

    :return: plasma_mesh2d
    """

    if species_density.keys() != species_temperature.keys():
        raise ValueError('Inconsistent data: "species_density" and "species_temperature" contain different plasma species.')

    plasma_mesh2d = {}
    plasma_mesh2d['electron_density'] = Discrete2DMesh(vertex_coords, triangles, electron_density, limit=False, default_value=0)
    plasma_mesh2d['electron_temperature'] = Discrete2DMesh.instance(plasma_mesh2d['electron_density'], electron_temperature)
    plasma_mesh2d['species_density'] = {key: Discrete2DMesh.instance(plasma_mesh2d['electron_density'], value)
                                        for key, value in species_density.items()}
    plasma_mesh2d['species_temperature'] = {key: Discrete2DMesh.instance(plasma_mesh2d['electron_density'], value)
                                            for key, value in species_temperature.items()}

    return plasma_mesh2d


def plasma_from_2d_profiles(vertex_coords, triangles, electron_density, electron_temperature, species_density, species_temperature,
                            parent=None, transform=None, name='Mesh2D Plasma'):
    """
    Creates Plasma object from the plasma profiles defined on a triangular 2d mesh.

    :param ndarray vertex_coords: An array of vertex coordinates (R, Z) with shape Nx2.
    :param ndarray triangles: An array of vertex indices defining the mesh triangles,
                              with shape Mx3.
    :param ndarray electron_density: Electron density profile of shape Mx1.
    :param ndarray electron_temperature: Electron temperature profile of shape Mx1.
    :param ndarray species_density: A dict with (element/isotope, charge) keys
                                    containing density of each plasma species.
    :param ndarray species_temperature: A dict with (element/isotope, charge) keys
                                        containing temperature of each plasma species.
    :param Node parent: The plasma's parent node in the scenegraph, e.g. a World object.
    :param AffineMatrix3D transform: Affine matrix describing the location and orientation
                                     of the plasma in the world.
    :param str name: User friendly name for this plasma (default = "Mesh2D Plasma").

    :rtype: Plasma
    """

    plasma_mesh2d = plasma_profiles_to_mesh2d(vertex_coords, triangles,
                                              electron_density, electron_temperature,
                                              species_density, species_temperature)

    plasma = Plasma(parent=parent, transform=transform, name=name)
    radius = vertex_coords[:, 0].max()
    inner_radius = vertex_coords[:, 0].min()
    minz = vertex_coords[:, 1].min()
    height = vertex_coords[:, 1].max() - minz
    plasma.geometry = Subtract(Cylinder(radius, height), Cylinder(inner_radius, height))
    plasma.geometry_transform = translate(0, 0, minz)

    electron_density_3d = AxisymmetricMapper(plasma_mesh2d['electron_density'])
    electron_temperature_3d = AxisymmetricMapper(plasma_mesh2d['electron_temperature'])
    electron_velocity_3d = ConstantVector3D(Vector3D(0, 0, 0))
    plasma.electron_distribution = Maxwellian(electron_density_3d, electron_temperature_3d, electron_velocity_3d, electron_mass)

    for species in plasma_mesh2d['species_density'].keys():
        density_3d = AxisymmetricMapper(plasma_mesh2d['species_density'][species])
        temperature_3d = AxisymmetricMapper(plasma_mesh2d['species_temperature'][species])
        velocity_3d = ConstantVector3D(Vector3D(0, 0, 0))
        species_type = species[0]
        charge = species[1]
        distribution = Maxwellian(density_3d, temperature_3d, velocity_3d, species_type.atomic_weight * atomic_mass)

        plasma.composition.add(Species(species_type, charge, distribution))

    return plasma
