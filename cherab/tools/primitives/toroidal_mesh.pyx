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

import numpy as np
cimport numpy as np
from libc.math cimport cos, sin
from raysect.primitive.mesh.mesh cimport Mesh
from raysect.core.math.polygon import triangulate2d
from raysect.core.math.cython cimport winding2d


cdef double DEG2RAD = 2 * np.pi / 360


cpdef Mesh toroidal_mesh_from_polygon(object polygon, double toroidal_extent, object polygon_triangles=None, int num_toroidal_segments=500):
    """
    Generates a watertight Raysect Mesh primitive from the specified 2D polygon in R-Z plane
    by extending it in toroidal direction by a given angle and closing the
    poloidal faces with triangulated polygons.

    :param object polygon: An object which can be converted to a numpy array with shape 
                           [N,2] specifying the wall outline polygon in the R-Z plane. 
                           The polygon should not be closed, i.e. vertex i = 0 and i = N 
                           should not be the same vertex, but neighbours.
    :param object polygon_triangles: An object which can be converted to a numpy array
                                     with shape [M,3] specifying the triangulation
                                     of a polygon (polygon_triangles = [[v1, v2, v3],...),
                                     where v1, v2, v3 are the vertex array indices specifying
                                     the triangle’s vertices. Should be with clockwise winding.
                                     Defaults to None.
                                     If not provided, the triangulation will be performed using
                                     `triangulate2d(polygon)` from raysect.core.math.polygon.
    :param float toroidal_extent: Angular extention of an element in toroidal direction (in degrees).
                                  Note that in the case of toroidal_extent=360  produces 
                                  an axisymmetric mesh which has no end faces
    :param int num_toroidal_segments: The number of repeating toroidal segments
                                      per given `toroidal_extent` that will be used to construct
                                      the mesh. Defaults to 500.

    :return: A watertight Raysect Mesh primitive constructed from the R-Z polygon.

    .. code-block:: pycon

        >>> from cherab.tools.primitives import toroidal_mesh_from_polygon
        >>>
        >>> # wall_polygon is your (N, 2) ndarray describing the polygon
        >>> mesh = toroidal_mesh_from_polygon(wall_polygon, toroidal_extent = 50)
    """

    cdef:
        int num_poloidal_vertices, num_toroidal_segments_loop
        int i, j, vid, v1_id, v2_id, v3_id, v4_id
        double theta, r, x, y, z
        list triangles
        double[:, :] polygon_mv, vertices_mv

    polygon = np.array(polygon, dtype=np.float64)

    if polygon.ndim != 2:
        raise ValueError("The 'polygon' must be a two-dimensional array.")

    if polygon.shape[0] < 2:
        raise ValueError("The 'polygon' must contain at least two vertices.")

    if polygon.shape[1] != 2:
        raise ValueError("The 'polygon' must have [N, 2] shape.")
    
    if (toroidal_extent<=0) or (toroidal_extent>360):
        raise ValueError("The 'toroidal_extent' should be in the range 0 < 'toroidal_extent' <= 360.")

    num_poloidal_vertices = polygon.shape[0]
    theta = toroidal_extent / num_toroidal_segments  # toroidal step

    vertices = np.zeros((num_poloidal_vertices * num_toroidal_segments, 3))
    vertices_mv = vertices
    polygon_mv = polygon

    for i in range(num_toroidal_segments):
        for j in range(num_poloidal_vertices):

            r = polygon_mv[j, 0]
            z = polygon_mv[j, 1]
            x = r * cos(i * theta * DEG2RAD)
            y = r * sin(i * theta * DEG2RAD)

            vid = i * num_poloidal_vertices + j
            vertices_mv[vid, 0] = x
            vertices_mv[vid, 1] = y
            vertices_mv[vid, 2] = z

    # assemble mesh triangles
    triangles = []

    if toroidal_extent != 360.:
        if polygon_triangles is None:
            # triangulating initial polygon in case of not full axisymmetric mesh
            polygon_triangles = triangulate2d(polygon)

        else:
            # user-defined triangulation
            polygon_triangles = np.array(polygon_triangles, dtype=np.int32)
            # check data sanity
            _check_polygon_triangulation(polygon, polygon_triangles)

        triangles = triangles + polygon_triangles[:, [1, 0, 2]].tolist()

    num_toroidal_segments_loop = num_toroidal_segments if toroidal_extent == 360. else num_toroidal_segments - 1

    for i in range(num_toroidal_segments_loop):
        for j in range(num_poloidal_vertices):

            if i == num_toroidal_segments - 1 and j == num_poloidal_vertices - 1:
                v1_id = i * num_poloidal_vertices + j
                v2_id = i * num_poloidal_vertices + 0
                v3_id = j
                v4_id = 0

            elif i == num_toroidal_segments - 1:
                v1_id = i * num_poloidal_vertices + j
                v2_id = i * num_poloidal_vertices + j + 1
                v3_id = j
                v4_id = j + 1

            elif j == num_poloidal_vertices - 1:
                v1_id = i * num_poloidal_vertices + j
                v2_id = i * num_poloidal_vertices
                v3_id = i * num_poloidal_vertices + num_poloidal_vertices + j
                v4_id = i * num_poloidal_vertices + num_poloidal_vertices

            else:
                v1_id = i * num_poloidal_vertices + j
                v2_id = i * num_poloidal_vertices + j + 1
                v3_id = i * num_poloidal_vertices + num_poloidal_vertices + j
                v4_id = i * num_poloidal_vertices + num_poloidal_vertices + j + 1

            triangles.append([v1_id, v2_id, v4_id])
            triangles.append([v4_id, v3_id, v1_id])

    if toroidal_extent != 360.:
        triangles = triangles + ((num_toroidal_segments - 1) * num_poloidal_vertices + polygon_triangles).tolist()

    return Mesh(vertices=vertices, triangles=triangles, smoothing=False)


cdef _check_polygon_triangulation(np.ndarray polygon, np.ndarray polygon_triangles):


    # check consistency
    if not np.array_equal(np.unique(polygon_triangles), np.arange(polygon.shape[0], dtype=np.int32)):
        raise ValueError("The data in 'polygon_triangles' does not match a given 'polygon'.")
    # check winding
    triangles = np.ascontiguousarray(polygon[polygon_triangles])
    if not all(winding2d(triangle) for triangle in triangles):
        raise ValueError("All triangles in 'polygon_triangles' must be wound clockwise.")
