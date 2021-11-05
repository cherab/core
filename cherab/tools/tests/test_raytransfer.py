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

import unittest
import numpy as np
from raysect.optical import World, Ray, Point3D, Point2D, Vector3D, NumericalIntegrator
from raysect.primitive import Box, Cylinder, Subtract
from cherab.tools.raytransfer import RayTransferBox, RayTransferCylinder, CartesianRayTransferEmitter, CylindricalRayTransferEmitter
from cherab.tools.inversions import ToroidalVoxelGrid


class TestRayTransferCylinder(unittest.TestCase):
    """
    Test cases for RayTransferCylinder class.
    """

    def test_mask_2d(self):
        rtc = RayTransferCylinder(radius_outer=8., height=10., n_radius=4, n_height=10, radius_inner=4.)
        mask = np.zeros((4, 10), dtype=bool)
        mask[:, 3:6] = True
        rtc.mask = mask[:, None, :]
        voxel_map_ref = -1 * np.ones((4, 10), dtype=np.int32)
        voxel_map_ref[:, 3:6] = np.arange(12, dtype=int).reshape((4, 3))
        self.assertTrue(np.all(voxel_map_ref == rtc.voxel_map[:, 0, :]) and rtc.bins == 12)

    def test_voxel_map_2d(self):
        rtc = RayTransferCylinder(radius_outer=8., height=10., n_radius=4, n_height=10, radius_inner=4.)
        voxel_map = -1 * np.ones((4, 10), dtype=np.int32)
        voxel_map[1, 3:5] = 0
        voxel_map[2, 3:5] = 1
        voxel_map[1, 5:7] = 2
        voxel_map[2, 5:7] = 3
        rtc.voxel_map = voxel_map[:, None, :]
        mask_ref = np.zeros((4, 10), dtype=bool)
        mask_ref[1:3, 3:7] = True
        inv_vmap_ref = [(np.array([1, 1]), np.array([3, 4])), (np.array([2, 2]), np.array([3, 4])),
                        (np.array([1, 1]), np.array([5, 6])), (np.array([2, 2]), np.array([5, 6]))]
        self.assertTrue(np.all(mask_ref == rtc.mask[:, 0, :]) and
                        np.all(np.array(inv_vmap_ref) == np.array(rtc.invert_voxel_map())[:, ::2, :]) and
                        rtc.bins == 4)

    def test_mask_3d(self):
        rtc = RayTransferCylinder(radius_outer=8., height=10., n_radius=4, n_height=10, radius_inner=4., n_polar=10, period=10.)
        mask = np.zeros((4, 10, 10), dtype=bool)
        mask[1:3, 3:8, 4:6] = True
        rtc.mask = mask
        voxel_map_ref = -1 * np.ones((4, 10, 10), dtype=np.int32)
        voxel_map_ref[1:3, 3:8, 4:6] = np.arange(20, dtype=int).reshape((2, 5, 2))
        self.assertTrue(np.all(voxel_map_ref == rtc.voxel_map) and rtc.bins == 20)

    def test_voxel_map_3d(self):
        rtc = RayTransferCylinder(radius_outer=8., height=10., n_radius=4, n_height=10, radius_inner=4., n_polar=10, period=10.)
        voxel_map = -1 * np.ones((4, 10, 10), dtype=np.int32)
        voxel_map[1, 3:5, 3:5] = 0
        voxel_map[2, 3:5, 3:5] = 1
        voxel_map[1, 5:7, 3:5] = 2
        voxel_map[2, 5:7, 3:5] = 3
        voxel_map[1, 3:5, 5:7] = 4
        voxel_map[2, 3:5, 5:7] = 5
        voxel_map[1, 5:7, 5:7] = 6
        voxel_map[2, 5:7, 5:7] = 7
        rtc.voxel_map = voxel_map
        mask_ref = np.zeros((4, 10, 10), dtype=bool)
        mask_ref[1:3, 3:7, 3:7] = True
        inv_vmap_ref = [(np.array([1, 1, 1, 1]), np.array([3, 3, 4, 4]), np.array([3, 4, 3, 4])),
                        (np.array([2, 2, 2, 2]), np.array([3, 3, 4, 4]), np.array([3, 4, 3, 4])),
                        (np.array([1, 1, 1, 1]), np.array([5, 5, 6, 6]), np.array([3, 4, 3, 4])),
                        (np.array([2, 2, 2, 2]), np.array([5, 5, 6, 6]), np.array([3, 4, 3, 4])),
                        (np.array([1, 1, 1, 1]), np.array([3, 3, 4, 4]), np.array([5, 6, 5, 6])),
                        (np.array([2, 2, 2, 2]), np.array([3, 3, 4, 4]), np.array([5, 6, 5, 6])),
                        (np.array([1, 1, 1, 1]), np.array([5, 5, 6, 6]), np.array([5, 6, 5, 6])),
                        (np.array([2, 2, 2, 2]), np.array([5, 5, 6, 6]), np.array([5, 6, 5, 6]))]
        self.assertTrue(np.all(mask_ref == rtc.mask) and np.all(np.array(inv_vmap_ref) == np.array(rtc.invert_voxel_map())) and rtc.bins == 8)

    def test_integration_2d(self):
        """ Testing against ToroidalVoxelGrid"""
        world = World()
        rtc = RayTransferCylinder(radius_outer=4., height=2., n_radius=2, n_height=2, radius_inner=2., parent=world)
        rtc.step = 0.001 * rtc.step
        ray = Ray(origin=Point3D(4., 1., 2.), direction=Vector3D(-4., -1., -2.) / np.sqrt(21.),
                  min_wavelength=500., max_wavelength=501., bins=rtc.bins)
        spectrum = ray.trace(world)
        world = World()
        vertices = []
        for rv in [2., 3.]:
            for zv in [0., 1.]:
                vertices.append([Point2D(rv, zv + 1.), Point2D(rv + 1., zv + 1.), Point2D(rv + 1., zv), Point2D(rv, zv)])
        tvg = ToroidalVoxelGrid(vertices, parent=world, primitive_type='csg', active='all')
        tvg.set_active('all')
        spectrum_test = ray.trace(world)
        self.assertTrue(np.allclose(spectrum_test.samples, spectrum.samples, atol=0.001))

    def test_integration_3d(self):
        world = World()
        rtc = RayTransferCylinder(radius_outer=2., height=2., n_radius=2, n_height=2, n_polar=3, period=90., parent=world)
        rtc.step = 0.001 * rtc.step
        ray = Ray(origin=Point3D(np.sqrt(2.), np.sqrt(2.), 2.), direction=Vector3D(-1., -1., -np.sqrt(2.)) / 2.,
                  min_wavelength=500., max_wavelength=501., bins=rtc.bins)
        spectrum = ray.trace(world)
        spectrum_test = np.zeros(rtc.bins)
        spectrum_test[2] = spectrum_test[9] = np.sqrt(2.)
        self.assertTrue(np.allclose(spectrum_test, spectrum.samples, atol=0.001))


class TestRayTransferBox(unittest.TestCase):
    """
    Test cases for RayTransferCylinder class.
    """

    def test_mask(self):
        rtb = RayTransferBox(xmax=10., ymax=10., zmax=10., nx=10, ny=10, nz=10)
        mask = np.zeros((10, 10, 10), dtype=bool)
        mask[5:7, 5:7, 5:7] = True
        rtb.mask = mask
        voxel_map_ref = -1 * np.ones((10, 10, 10), dtype=np.int32)
        voxel_map_ref[5:7, 5:7, 5:7] = np.arange(8, dtype=int).reshape((2, 2, 2))
        self.assertTrue(np.all(voxel_map_ref == rtb.voxel_map) and rtb.bins == 8)

    def test_voxel_map(self):
        rtb = RayTransferBox(xmax=10., ymax=10., zmax=10., nx=10, ny=10, nz=10)
        voxel_map = -1 * np.ones((10, 10, 10), dtype=np.int32)
        voxel_map[:2, :2, :2] = 0
        voxel_map[:2, :2, 8:] = 1
        voxel_map[:2, 8:, :2] = 2
        voxel_map[:2, 8:, 8:] = 3
        voxel_map[8:, :2, :2] = 4
        voxel_map[8:, :2, 8:] = 5
        voxel_map[8:, 8:, :2] = 6
        voxel_map[8:, 8:, 8:] = 7
        rtb.voxel_map = voxel_map
        mask_ref = np.zeros((10, 10, 10), dtype=bool)
        mask_ref[:2, :2, :2] = True
        mask_ref[:2, :2, 8:] = True
        mask_ref[:2, 8:, :2] = True
        mask_ref[:2, 8:, 8:] = True
        mask_ref[8:, :2, :2] = True
        mask_ref[8:, :2, 8:] = True
        mask_ref[8:, 8:, :2] = True
        mask_ref[8:, 8:, 8:] = True
        inv_vmap_ref = [(np.array([0, 0, 0, 0, 1, 1, 1, 1]), np.array([0, 0, 1, 1, 0, 0, 1, 1]), np.array([0, 1, 0, 1, 0, 1, 0, 1])),
                        (np.array([0, 0, 0, 0, 1, 1, 1, 1]), np.array([0, 0, 1, 1, 0, 0, 1, 1]), np.array([8, 9, 8, 9, 8, 9, 8, 9])),
                        (np.array([0, 0, 0, 0, 1, 1, 1, 1]), np.array([8, 8, 9, 9, 8, 8, 9, 9]), np.array([0, 1, 0, 1, 0, 1, 0, 1])),
                        (np.array([0, 0, 0, 0, 1, 1, 1, 1]), np.array([8, 8, 9, 9, 8, 8, 9, 9]), np.array([8, 9, 8, 9, 8, 9, 8, 9])),
                        (np.array([8, 8, 8, 8, 9, 9, 9, 9]), np.array([0, 0, 1, 1, 0, 0, 1, 1]), np.array([0, 1, 0, 1, 0, 1, 0, 1])),
                        (np.array([8, 8, 8, 8, 9, 9, 9, 9]), np.array([0, 0, 1, 1, 0, 0, 1, 1]), np.array([8, 9, 8, 9, 8, 9, 8, 9])),
                        (np.array([8, 8, 8, 8, 9, 9, 9, 9]), np.array([8, 8, 9, 9, 8, 8, 9, 9]), np.array([0, 1, 0, 1, 0, 1, 0, 1])),
                        (np.array([8, 8, 8, 8, 9, 9, 9, 9]), np.array([8, 8, 9, 9, 8, 8, 9, 9]), np.array([8, 9, 8, 9, 8, 9, 8, 9]))]
        self.assertTrue(np.all(mask_ref == rtb.mask) and np.all(np.array(inv_vmap_ref) == np.array(rtb.invert_voxel_map())) and rtb.bins == 8)

    def test_integration(self):
        world = World()
        rtb = RayTransferBox(xmax=3., ymax=3., zmax=3., nx=3, ny=3, nz=3, parent=world)
        rtb.step = 0.01 * rtb.step
        ray = Ray(origin=Point3D(4., 4., 4.), direction=Vector3D(-1., -1., -1.) / np.sqrt(3),
                  min_wavelength=500., max_wavelength=501., bins=rtb.bins)
        spectrum = ray.trace(world)
        spectrum_test = np.zeros(rtb.bins)
        spectrum_test[0] = spectrum_test[13] = spectrum_test[26] = np.sqrt(3.)
        self.assertTrue(np.allclose(spectrum_test, spectrum.samples, atol=0.001))


class TestCartesianRayTransferEmitter(unittest.TestCase):
    """
    Test cases for CartesianRayTransferEmitter class.
    """

    def test_evaluate_function(self):
        """
        Unlike test_integration() in TestRayTransferBox here we test how
        CartesianRayTransferEmitter works with NumericalIntegrator.
        """
        world = World()
        material = CartesianRayTransferEmitter((3, 3, 3), (1., 1., 1.), integrator=NumericalIntegrator(0.0001))
        box = Box(lower=Point3D(0, 0, 0), upper=Point3D(2.99999, 2.99999, 2.99999),
                  material=material, parent=world)
        ray = Ray(origin=Point3D(4., 4., 4.), direction=Vector3D(-1., -1., -1.) / np.sqrt(3),
                  min_wavelength=500., max_wavelength=501., bins=material.bins)
        spectrum = ray.trace(world)
        spectrum_test = np.zeros(material.bins)
        spectrum_test[0] = spectrum_test[13] = spectrum_test[26] = np.sqrt(3.)
        self.assertTrue(np.allclose(spectrum_test, spectrum.samples, atol=0.001))


class TestCylindricalRayTransferEmitter(unittest.TestCase):
    """
    Test cases for CylindricalRayTransferEmitter class.
    """

    def test_evaluate_function_2d(self):
        """
        Unlike test_integration_2d() in TestRayTransferCylinder here we test how
        CylindricalRayTransferEmitter works with NumericalIntegrator in axysimmetric case.
        Testing against ToroidalVoxelGrid.
        """
        world = World()
        material = CylindricalRayTransferEmitter((2, 1, 2), (1., 360., 1.), rmin=2., integrator=NumericalIntegrator(0.0001))
        primitive = Subtract(Cylinder(3.999999, 1.999999), Cylinder(2.0, 1.999999),
                             material=material, parent=world)
        ray = Ray(origin=Point3D(4., 1., 2.), direction=Vector3D(-4., -1., -2.) / np.sqrt(21.),
                  min_wavelength=500., max_wavelength=501., bins=4)
        spectrum = ray.trace(world)
        world = World()
        vertices = []
        for rv in [2., 3.]:
            for zv in [0., 1.]:
                vertices.append([Point2D(rv, zv + 1.), Point2D(rv + 1., zv + 1.), Point2D(rv + 1., zv), Point2D(rv, zv)])
        tvg = ToroidalVoxelGrid(vertices, parent=world, primitive_type='csg', active='all')
        tvg.set_active('all')
        spectrum_test = ray.trace(world)
        self.assertTrue(np.allclose(spectrum_test.samples, spectrum.samples, atol=0.001))

    def test_evaluate_function_3d(self):
        """
        Unlike test_integration_3d() in TestRayTransferCylinder here we test how
        CylindricalRayTransferEmitter works with NumericalIntegrator in 3D case.
        """
        world = World()
        material = CylindricalRayTransferEmitter((2, 3, 2), (1., 30., 1.), integrator=NumericalIntegrator(0.0001))
        primitive = Subtract(Cylinder(1.999999, 1.999999), Cylinder(0.000001, 1.999999),
                             material=material, parent=world)
        ray = Ray(origin=Point3D(np.sqrt(2.), np.sqrt(2.), 2.), direction=Vector3D(-1., -1., -np.sqrt(2.)) / 2.,
                  min_wavelength=500., max_wavelength=501., bins=12)
        spectrum = ray.trace(world)
        spectrum_test = np.zeros(12)
        spectrum_test[2] = spectrum_test[9] = np.sqrt(2.)
        self.assertTrue(np.allclose(spectrum_test, spectrum.samples, atol=0.001))
