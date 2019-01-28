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

import itertools
import unittest

from matplotlib.patches import Polygon
from matplotlib.path import Path
import numpy as np

from raysect.core import Point2D, Point3D
from cherab.tools.inversions import AxisymmetricVoxel

TRIANGLE_VOXEL_COORDS = [
    # Triangles with all vertices having different R and z
    # First leaf
    [(4, 3), (3, 2), (2, -1)],
    [(4, 3), (3, -1), (2, -1)],
    [(4, 3), (3, 3), (2, -1)],
    # Second leaf
    [(4, 3), (3, -1), (2, 2)],
    [(4, 3), (3, 2), (2, 2)],
    [(4, 3), (3, -1), (2, 3)],
    # Third leaf
    [(4, 2), (3, -1), (2, 3)],
    [(4, -1), (3, -1), (2, 3)],
    [(4, 3), (3, -1), (2, 3)],
    # Fourth leaf
    [(4, 2), (3, 3), (2, -1)],
    [(4, -1), (3, 3), (2, -1)],
    [(4, 3), (3, 3), (2, -1)],
    # Fifth leaf
    [(4, -1), (3, 3), (2, 2)],
    [(4, -1), (3, 3), (2, -1)],
    [(4, -1), (3, 3), (2, 3)],
    # Sixth leaf
    [(4, -1), (3, 2), (2, 3)],
    [(4, -1), (3, -1), (2, 3)],
    [(4, -1), (3, 3), (2, 3)],
    # Triangles with horizontal or vertical edges (e.g. from rectilinear cells)
    # R1 = R2 > R3
    # First leaf
    [(4, 3), (4, 2), (2, -1)],
    [(4, 3), (4, -1), (2, -1)],
    [(4, 3), (4, 3), (2, -1)], # Should produce zero volume voxel
    # Second leaf
    [(4, 3), (4, -1), (2, 2)],
    [(4, 3), (4, -1), (2, -1)],
    [(4, 3), (4, -1), (2, 3)],
    # Third leaf
    [(4, 2), (4, -1), (2, 3)],
    [(4, -1), (4, -1), (2, 3)], # Should produce zero volume voxel
    [(4, 3), (4, -1), (2, 3)],
    # Fourth leaf
    [(4, 2), (4, 3), (2, -1)],
    [(4, -1), (4, 3), (2, -1)],
    [(4, 3), (4, 3), (2, -1)], # Should produce zero volume voxel
    # Fifth leaf
    [(4, -1), (4, 3), (2, 2)],
    [(4, -1), (4, 3), (2, -1)],
    [(4, -1), (4, 3), (2, 3)],
    # Sixth leaf
    [(4, -1), (4, 2), (2, 3)],
    [(4, -1), (4, 3), (2, 3)],
    [(4, -1), (4, -1), (2, 3)], # Should produce zero volume voxel
    # R1 > R2 = R3
    # First leaf
    [(4, 3), (2, 2), (2, -1)],
    [(4, 3), (2, -1), (2, -1)], # Should produce zero volume voxel
    [(4, 3), (2, 3), (2, -1)],
    # Second leaf
    [(4, 3), (2, -1), (2, 2)],
    [(4, 3), (2, -1), (2, -1)], # Should produce zero volume voxel
    [(4, 3), (2, -1), (2, 3)],
    # Third leaf
    [(4, 2), (2, -1), (2, 3)],
    [(4, -1), (2, -1), (2, 3)],
    [(4, 3), (2, -1), (2, 3)],
    # Fourth leaf
    [(4, 2), (2, 3), (2, -1)],
    [(4, -1), (2, 3), (2, -1)],
    [(4, 3), (2, 3), (2, -1)],
    # Fifth leaf
    [(4, -1), (2, 3), (2, 2)],
    [(4, -1), (2, 3), (2, -1)],
    [(4, -1), (2, 3), (2, 3)], # Should produce zero volume voxel
    # Sixth leaf
    [(4, -1), (2, 2), (2, 3)],
    [(4, -1), (2, -1), (2, 3)],
    [(4, -1), (2, 3), (2, 3)], # Should produce zero volume voxel
]

RECTANGULAR_VOXEL_COORDS = [
    [(2, -1), (2, 3), (4, 3), (4, -1)],
]

ARBITRARY_VOXEL_COORDS = [
    [(2, -1), (2, 2), (3, 4), (4, 3), (4, -1)],
    [(2, -1), (2, 2), (3, 4), (4, 3), (4, 0)],
    [(2, -1), (2, 2), (3, 4), (4, 3), (5, 0)],
]

class TestCSGVoxels(unittest.TestCase):
    """Test cases for CSG voxels.

    The basic test is as follows: for a list of voxel vertex coordinates,
    generate the CSG representation of that Axisymmetric voxel. Then
    generate a 2D polygon of the same cross section independently, using
    Matplotlib. If the voxel has the correct cross section shape, then
    the set of points (r, 0, z) which lie inside the voxel should be the
    same as the set of points (r, z) which lie inside the 2D polygon.

    So far, there are tests for triangular, rectangular and pentagonal voxels.
    In principle, any arbitrary shape could be tested, but if the triangular
    voxels are working correctly then arbitrary polygons will also work
    correctly, as arbitrary shapes are implemented as a union of triangles.
    """
    def voxel_matches_polygon(self, coordinate_list):
        for voxel_coords in coordinate_list:
            voxel_coords = np.asarray(voxel_coords)
            rmax = voxel_coords[:, 0].max()
            rmin = voxel_coords[:, 0].min()
            zmax = voxel_coords[:, 1].max()
            zmin = voxel_coords[:, 1].min()
            router = 1.5 * rmax
            rinner = 0.5 * rmin
            zupper = 1.5 * zmax if zmax > 0 else 0.5 * zmax
            zlower = 0.5 * zmin if zmin > 0 else 1.5 * zmin
            test_rs = np.linspace(rinner, router, int(100*(router-rinner)))
            test_zs = np.linspace(zlower, zupper, int(100*(zupper-zlower)))
            voxel_vertex_points = [Point2D(*v) for v in voxel_coords]
            voxel = AxisymmetricVoxel(voxel_vertex_points, parent=None,
                                      primitive_type='csg')
            polygon = Polygon(voxel_coords, closed=True)
            test_verts = list(itertools.product(test_rs, test_zs))
            try:
                inside_poly = polygon.contains_points(test_verts)
            except AttributeError:
                # Polygon.contains_points was only introduced in Matplotlib 2.2.
                # Before that we need to convert to Path
                inside_poly = Path(polygon.get_verts()).contains_points(test_verts)
            inside_csg = [any(child.contains(Point3D(r, 0, z)) for child in voxel.children)
                          for (r, z) in test_verts]
            self.assertSequenceEqual(inside_csg, inside_poly.tolist())

    def test_triangular_voxels(self):
        self.voxel_matches_polygon(TRIANGLE_VOXEL_COORDS)

    def test_rectangular_voxels(self):
        self.voxel_matches_polygon(RECTANGULAR_VOXEL_COORDS)

    def test_arbitrary_voxels(self):
        self.voxel_matches_polygon(ARBITRARY_VOXEL_COORDS)

