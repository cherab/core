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


from raysect.core import Node, Point2D, Point3D, rotate_z
from raysect.primitive import Mesh
from raysect.optical import UnityVolumeEmitter, UnitySurfaceEmitter


def generate_annulus_mesh_segments(lower_corner, upper_corner, number_segments, world, material=None):
    """
    Generates an annulus from many smaller mesh segments.

    Used for calculating sensitivity matrices for poloidal inversion grids.

    :param Point2D lower_corner: the lower corner of the poloidal 2D cell.
    :param Point2D upper_corner: the upper corner of the poloidal 2D cell.
    :param int number_segments: The number of angular mesh segments used to build the annulus.
    :param World world: The scene-graph to which the annulus will be attached.
    :return: Node holding all the annulus segment primitives.
    :rtype: Node
    """

    material = material or UnityVolumeEmitter()

    annulus_node = Node(parent=world)

    theta_adjusted = 360 / number_segments

    # Set of points in x-z plane
    p1a = Point3D(lower_corner.x, 0, lower_corner.y)  # corresponds to lower corner is x-z plane
    p2a = Point3D(lower_corner.x, 0, upper_corner.y)
    p3a = Point3D(upper_corner.x, 0, upper_corner.y)  # corresponds to upper corner in x-z plane
    p4a = Point3D(upper_corner.x, 0, lower_corner.y)

    # Set of points rotated away from x-z plane
    p1b = p1a.transform(rotate_z(theta_adjusted))
    p2b = p2a.transform(rotate_z(theta_adjusted))
    p3b = p3a.transform(rotate_z(theta_adjusted))
    p4b = p4a.transform(rotate_z(theta_adjusted))

    vertices = [[p1a.x, p1a.y, p1a.z], [p2a.x, p2a.y, p2a.z],
                [p3a.x, p3a.y, p3a.z], [p4a.x, p4a.y, p4a.z],
                [p1b.x, p1b.y, p1b.z], [p2b.x, p2b.y, p2b.z],
                [p3b.x, p3b.y, p3b.z], [p4b.x, p4b.y, p4b.z]]

    triangles = [[1, 0, 3], [1, 3, 2],  # front face (x-z)
                 [7, 4, 5], [7, 5, 6],  # rear face (rotated out of x-z plane)
                 [5, 1, 2], [5, 2, 6],  # top face (x-y plane)
                 [3, 0, 4], [3, 4, 7],  # bottom face (x-y plane)
                 [4, 0, 5], [1, 5, 0],  # inner face (y-z plane)
                 [2, 3, 7], [2, 7, 6]]  # outer face (y-z plane)

    base_segment = Mesh(vertices=vertices, triangles=triangles, smoothing=False)

    # Construct annulus by duplicating and rotating base segment.
    for i in range(number_segments):
        theta_rotation = theta_adjusted * i
        segment = base_segment.instance(transform=rotate_z(theta_rotation), material=material, parent=annulus_node)

    return annulus_node
