
import numpy as np
cimport numpy as np
from libc.math cimport cos, sin
from raysect.primitive.mesh.mesh cimport Mesh


cdef double DEG2RAD = 2 * np.pi / 360


cpdef Mesh axisymmetric_mesh_from_polygon(np.ndarray polygon, int num_toroidal_segments=500):
    """
    Generates an Raysect Mesh primitive from the specified 2D polygon.

    :param np.ndarray polygon: A numpy array with shape [N,2] specifying the wall outline polygon
      in the R-Z plane. The polygon should not be closed, i.e. vertex i = 0 and i = N should not
      be the same vertex, but neighbours.  
    :param int num_toroidal_segments: The number of repeating toroidal segments that will be used
      to construct the mesh.
    :return: A Raysect Mesh primitive constructed from the R-Z polygon using symmetry.
    
    .. code-block:: pycon

        >>> from cherab.tools.primitives import axisymmetric_mesh_from_polygon
        >>>
        >>> # wall_polygon is your (N, 2) ndarray describing the polygon
        >>> mesh = axisymmetric_mesh_from_polygon(wall_polygon)
    """

    cdef:
        int num_poloidal_vertices
        int i, j, vid, v1_id, v2_id, v3_id, v4_id
        float theta, r, x, y, z
        np.ndarray vertices
        list triangles
        double[:,:] polygon_mv, vertices_mv

    num_poloidal_vertices = len(polygon)
    theta = 360 / num_toroidal_segments
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
    for i in range(num_toroidal_segments):
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

    return Mesh(vertices=vertices, triangles=triangles, smoothing=False)
