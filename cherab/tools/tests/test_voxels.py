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
from raysect.core.math import triangulate2d
from raysect.core.math.random import seed
from cherab.tools.inversions import AxisymmetricVoxel

try:
    import quadpy
except ImportError:
    HAVE_QUADPY = False
else:
    HAVE_QUADPY = True


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


class TestVoxelCalculations(unittest.TestCase):
    """Test cases for voxel calculations

    This tests properties such as the area and centroid calculations,
    and also calculations such as the emissivities
    """
    def test_triangle_area(self):
        for triangle in TRIANGLE_VOXEL_COORDS:
            coords = np.asarray(triangle)
            voxel_vertex_points = [Point2D(*v) for v in coords]
            voxel = AxisymmetricVoxel(voxel_vertex_points, parent=None)
            # Calculate the area with the shoelace formula
            x1, y1 = coords[0]
            x2, y2 = coords[1]
            x3, y3 = coords[2]
            expected_area = 0.5 * abs(x1 * y2 + x2 * y3 + x3 * y1
                                      - x2 * y1 - x3 * y2 - x1 * y3)
            # Different algorithms have some floating point rounding error
            self.assertAlmostEqual(voxel.cross_sectional_area, expected_area)

    def test_triangle_centroid(self):
        for triangle in TRIANGLE_VOXEL_COORDS:
            coords = np.asarray(triangle)
            voxel_vertex_points = [Point2D(*v) for v in coords]
            voxel = AxisymmetricVoxel(voxel_vertex_points, parent=None)
            # The centroid for a triangle is simply the mean position
            # of the vertices
            expected_centroid = Point2D(*coords.mean(axis=0))
            if voxel.cross_sectional_area == 0:
                self.assertRaises(ZeroDivisionError, getattr, voxel, "cross_section_centroid")
            else:
                self.assertEqual(voxel.cross_section_centroid, expected_centroid)

    def test_rectangle_area(self):
        for rectangle in RECTANGULAR_VOXEL_COORDS:
            coords = np.asarray(rectangle)
            voxel_vertex_points = [Point2D(*v) for v in coords]
            voxel = AxisymmetricVoxel(voxel_vertex_points, parent=None)
            dx = coords[:, 0].ptp()
            dy = coords[:, 1].ptp()
            expected_area = dx * dy
            self.assertEqual(voxel.cross_sectional_area, expected_area)

    def test_rectangle_centroid(self):
        for rectangle in RECTANGULAR_VOXEL_COORDS:
            coords = np.asarray(rectangle)
            voxel_vertex_points = [Point2D(*v) for v in coords]
            voxel = AxisymmetricVoxel(voxel_vertex_points, parent=None)
            x0, y0 = coords.mean(axis=0)
            expected_centroid = Point2D(x0, y0)
            self.assertEqual(voxel.cross_section_centroid, expected_centroid)

    def test_polygon_area(self):
        for polygon in ARBITRARY_VOXEL_COORDS:
            coords = np.asarray(polygon)
            voxel_vertex_points = [Point2D(*v) for v in coords]
            voxel = AxisymmetricVoxel(voxel_vertex_points, parent=None)
            # Shoelace formula
            x = coords[:, 0]
            y = coords[:, 1]
            expected_area = abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) / 2
            self.assertEqual(voxel.cross_sectional_area, expected_area)

    def test_polygon_centroid(self):
        for polygon in ARBITRARY_VOXEL_COORDS:
            coords = np.asarray(polygon)
            voxel_vertex_points = [Point2D(*v) for v in coords]
            voxel = AxisymmetricVoxel(voxel_vertex_points, parent=None)
            x = coords[:, 0]
            y = coords[:, 1]
            signed_area = (np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) / 2
            xroll = np.roll(x, 1)
            yroll = np.roll(y, 1)
            cx = np.sum((x + xroll) * (x * yroll - xroll * y)) / (6 * signed_area)
            cy = np.sum((y + yroll) * (x * yroll - xroll * y)) / (6 * signed_area)
            expected_centroid = Point2D(cx, cy)
            self.assertEqual(voxel.cross_section_centroid, expected_centroid)

    def test_constant_emissivity(self):
        def emiss_function(r, phi, z):
            return 5
        # Emissivity in the voxel should be the constant value of emiss_function
        for polygon in TRIANGLE_VOXEL_COORDS + RECTANGULAR_VOXEL_COORDS + ARBITRARY_VOXEL_COORDS:
            coords = np.asarray(polygon)
            voxel_vertex_points = [Point2D(*v) for v in coords]
            voxel = AxisymmetricVoxel(voxel_vertex_points, parent=None)
            emiss = voxel.emissivity_from_function(emiss_function, 1000)
            expected_emiss = emiss_function(0, 0, 0)
            self.assertEqual(emiss, expected_emiss)

    @unittest.skipUnless(HAVE_QUADPY, "Need quadpy package for integration")
    def test_variable_emissivity_triangular(self):
        # Use the same seed as Raysect's random tests
        seed(1234567890)

        def emiss_function(r, phi, z):
            return r * z
        # This should be equal to the integrated emissivity divided by the area
        for polygon in TRIANGLE_VOXEL_COORDS:
            coords = np.asarray(polygon)
            voxel_vertex_points = [Point2D(*v) for v in coords]
            voxel = AxisymmetricVoxel(voxel_vertex_points, parent=None)
            nsamples = 10000
            emiss = voxel.emissivity_from_function(emiss_function, nsamples)
            # TODO: include pre-calculated values of expected emissivity
            # for when quadpy is not installed
            # Calculate the expected emissivity analytically
            x1, y1 = coords[0]
            x2, y2 = coords[1]
            x3, y3 = coords[2]
            triangle_area = 0.5 * abs(x1 * y2 + x2 * y3 + x3 * y1
                                      - x2 * y1 - x3 * y2 - x1 * y3)
            # Doesn't make any sense to sample from a <2D cross section area
            if triangle_area == 0:
                return
            expected_emiss = quadpy.triangle.integrate(
                lambda x: emiss_function(x[0], 0, x[1]), coords, quadpy.triangle.Strang(6)
            ) / triangle_area
            max_relative_error = 0.0723  # Measured with seed(1234567890)
            self.assertAlmostEqual(emiss, expected_emiss, delta=emiss * max_relative_error)

    def test_variable_emissivity_rectangular(self):
        # Use the same seed as Raysect's random tests
        seed(1234567890)

        def emiss_function(r, phi, z):
            return r * z
        # This should be equal to the integrated emissivity divided by the area
        # We can calculate the integrated emissivity analytically
        for polygon in RECTANGULAR_VOXEL_COORDS:
            coords = np.asarray(polygon)
            voxel_vertex_points = [Point2D(*v) for v in coords]
            voxel = AxisymmetricVoxel(voxel_vertex_points, parent=None)
            nsamples = 10000
            emiss = voxel.emissivity_from_function(emiss_function, nsamples)
            rmax = coords[:, 0].max()
            rmin = coords[:, 0].min()
            zmax = coords[:, 1].max()
            zmin = coords[:, 1].min()
            area = (rmax - rmin) * (zmax - zmin)
            expected_emiss = (rmax**2 - rmin**2) * (zmax**2 - zmin**2) / 4 / area
            max_relative_error = 0.0221  # Measured with seed(1234567890)
            self.assertAlmostEqual(emiss, expected_emiss, delta=emiss * max_relative_error)

    @unittest.skipUnless(HAVE_QUADPY, "Need quadpy package for integration")
    def test_variable_emissivity_arbitrary(self):
        # Use the same seed as Raysect's random tests
        seed(1234567890)

        def emiss_function(r, phi, z):
            return r * z
        # This should be equal to the integrated emissivity divided by the area
        for polygon in ARBITRARY_VOXEL_COORDS:
            coords = np.asarray(polygon)
            voxel_vertex_points = [Point2D(*v) for v in coords]
            voxel = AxisymmetricVoxel(voxel_vertex_points, parent=None)
            nsamples = 10000
            emiss = voxel.emissivity_from_function(emiss_function, nsamples)
            # TODO: include pre-calculated values of expected emissivity
            # for when quadpy is not installed
            # Calculate the expected emissivity
            triangle_indices = triangulate2d(coords)
            triangles = coords[triangle_indices]
            expected_emiss = 0
            polygon_area = 0
            for triangle in triangles:
                # Calculate the area with the shoelace formula
                x1, y1 = triangle[0]
                x2, y2 = triangle[1]
                x3, y3 = triangle[2]
                triangle_area = 0.5 * abs(x1 * y2 + x2 * y3 + x3 * y1
                                          - x2 * y1 - x3 * y2 - x1 * y3)
                # Total emissivity is the area-weighted emissivity for each triangle
                expected_emiss += quadpy.triangle.integrate(
                    lambda x: emiss_function(x[0], 0, x[1]), triangle, quadpy.triangle.Strang(6)
                )
                polygon_area += triangle_area
            expected_emiss /= polygon_area
            max_relative_error = 0.0225  # Measured with seed(1234567890)
            self.assertAlmostEqual(emiss, expected_emiss, delta=emiss * max_relative_error)
