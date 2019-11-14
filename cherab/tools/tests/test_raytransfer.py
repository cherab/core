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
from cherab.tools.raytransfer import RayTransferBox, RayTransferCylinder


class TestRayTransferCylinder(unittest.TestCase):
    """
    Test cases for RayTransferCylinder class.
    """

    def test_mask_2d(self):
        rtc = RayTransferCylinder(radius_outer=8., height=10., n_radius=4, n_height=10, radius_inner=4.)
        mask = np.zeros((4, 10), dtype=np.bool)
        mask[:, 3:6] = True
        rtc.mask = mask
        voxel_map_ref = -1 * np.ones((4, 10), dtype=np.int)
        voxel_map_ref[:, 3:6] = np.arange(12, dtype=int).reshape((4, 3))
        self.assertTrue(np.all(voxel_map_ref == rtc.voxel_map) and rtc.bins == 12)

    def test_voxel_map_2d(self):
        rtc = RayTransferCylinder(radius_outer=8., height=10., n_radius=4, n_height=10, radius_inner=4.)
        voxel_map = -1 * np.ones((4, 10), dtype=np.int)
        voxel_map[1, 3:5] = 0
        voxel_map[2, 3:5] = 1
        voxel_map[1, 5:7] = 2
        voxel_map[2, 5:7] = 3
        rtc.voxel_map = voxel_map
        mask_ref = np.zeros((4, 10), dtype=np.bool)
        mask_ref[1:3, 3:7] = True
        inv_vmap_ref = [(np.array([1, 1]), np.array([3, 4])), (np.array([2, 2]), np.array([3, 4])),
                        (np.array([1, 1]), np.array([5, 6])), (np.array([2, 2]), np.array([5, 6]))]
        self.assertTrue(np.all(mask_ref == rtc.mask) and np.all(np.array(inv_vmap_ref) == np.array(rtc.invert_voxel_map())) and rtc.bins == 4)

    def test_mask_3d(self):
        rtc = RayTransferCylinder(radius_outer=8., height=10., n_radius=4, n_height=10, radius_inner=4., n_polar=10, period=10.)
        mask = np.zeros((4, 10, 10), dtype=np.bool)
        mask[1:3, 3:8, 4:6] = True
        rtc.mask = mask
        voxel_map_ref = -1 * np.ones((4, 10, 10), dtype=np.int)
        voxel_map_ref[1:3, 3:8, 4:6] = np.arange(20, dtype=int).reshape((2, 5, 2))
        self.assertTrue(np.all(voxel_map_ref == rtc.voxel_map) and rtc.bins == 20)

    def test_voxel_map_3d(self):
        rtc = RayTransferCylinder(radius_outer=8., height=10., n_radius=4, n_height=10, radius_inner=4., n_polar=10, period=10.)
        voxel_map = -1 * np.ones((4, 10, 10), dtype=np.int)
        voxel_map[1, 3:5, 3:5] = 0
        voxel_map[2, 3:5, 3:5] = 1
        voxel_map[1, 5:7, 3:5] = 2
        voxel_map[2, 5:7, 3:5] = 3
        voxel_map[1, 3:5, 5:7] = 4
        voxel_map[2, 3:5, 5:7] = 5
        voxel_map[1, 5:7, 5:7] = 6
        voxel_map[2, 5:7, 5:7] = 7
        rtc.voxel_map = voxel_map
        mask_ref = np.zeros((4, 10, 10), dtype=np.bool)
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


class TestRayTransferBox(unittest.TestCase):
    """
    Test cases for RayTransferCylinder class.
    """

    def test_mask(self):
        rtb = RayTransferBox(xmax=10., ymax=10., zmax=10., nx=10, ny=10, nz=10)
        mask = np.zeros((10, 10, 10), dtype=np.bool)
        mask[5:7, 5:7, 5:7] = True
        rtb.mask = mask
        voxel_map_ref = -1 * np.ones((10, 10, 10), dtype=np.int)
        voxel_map_ref[5:7, 5:7, 5:7] = np.arange(8, dtype=int).reshape((2, 2, 2))
        self.assertTrue(np.all(voxel_map_ref == rtb.voxel_map) and rtb.bins == 8)

    def test_voxel_map(self):
        rtb = RayTransferBox(xmax=10., ymax=10., zmax=10., nx=10, ny=10, nz=10)
        voxel_map = -1 * np.ones((10, 10, 10), dtype=np.int)
        voxel_map[:2, :2, :2] = 0
        voxel_map[:2, :2, 8:] = 1
        voxel_map[:2, 8:, :2] = 2
        voxel_map[:2, 8:, 8:] = 3
        voxel_map[8:, :2, :2] = 4
        voxel_map[8:, :2, 8:] = 5
        voxel_map[8:, 8:, :2] = 6
        voxel_map[8:, 8:, 8:] = 7
        rtb.voxel_map = voxel_map
        mask_ref = np.zeros((10, 10, 10), dtype=np.bool)
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
