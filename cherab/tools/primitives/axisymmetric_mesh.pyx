# Copyright 2016-2022 Euratom
# Copyright 2016-2022 United Kingdom Atomic Energy Authority
# Copyright 2016-2022 Centro de Investigaciones Energéticas, Medioambientales y Tecnológicas
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

from raysect.primitive.mesh.mesh cimport Mesh

from .toroidal_mesh import toroidal_mesh_from_polygon


cpdef Mesh axisymmetric_mesh_from_polygon(object polygon, int num_toroidal_segments=500):
    """
    Generates an Raysect Mesh primitive from the specified 2D polygon.

    :param object polygon: An object which can be converted to a numpy array with shape [N,2] 
                           specifying the wall outline polygon in the R-Z plane. The polygon 
                           should not be closed, i.e. vertex i = 0 and i = N should not be the
                           same vertex, but neighbours.
    :param int num_toroidal_segments: The number of repeating toroidal segments that will be used
                                      to construct the mesh.
    :return: A Raysect Mesh primitive constructed from the R-Z polygon using symmetry.

    .. code-block:: pycon

        >>> from cherab.tools.primitives import axisymmetric_mesh_from_polygon
        >>>
        >>> # wall_polygon is your (N, 2) ndarray describing the polygon
        >>> mesh = axisymmetric_mesh_from_polygon(wall_polygon)
    """

    return toroidal_mesh_from_polygon(polygon, toroidal_extent=360, num_toroidal_segments=num_toroidal_segments)
