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

from cherab.tools.inversions import generate_derivative_operators

class TestADMT(unittest.TestCase):
    """Tests for ADMT utilities

    The derivative operators are tested to ensure that the results match
    equations 37-41 in JET-R(99)08.
    """

    NROW = 3
    NCOL = 3
    DX = 2
    DY = 1

    # Voxels are ordered in successive descending columns
    VOXEL_COORDS = [
        [2, 4],
        [2, 3],
        [2, 2],
        [4, 4],
        [4, 3],
        [4, 2],
        [6, 4],
        [6, 3],
        [6, 2],
    ]

    VOXEL_VERTICES = []
    for (h, k) in VOXEL_COORDS:
        VOXEL_VERTICES.append(
            [(h + DX / 2, k + DY / 2), (h + DX / 2, k - DY / 2),
             (h - DX / 2, k - DY / 2), (h - DX / 2, k + DY / 2)]
        )

    # Note that in 2D the array is indexed (x, y), with y varying quickest
    GRID_1D_TO_2D_MAP = {
        0: (0, 0),
        1: (0, 1),
        2: (0, 2),
        3: (1, 0),
        4: (1, 1),
        5: (1, 2),
        6: (2, 0),
        7: (2, 1),
        8: (2, 2),
    }

    GRID_2D_TO_1D_MAP = {
        (0, 0): 0,
        (0, 1): 1,
        (0, 2): 2,
        (1, 0): 3,
        (1, 1): 4,
        (1, 2): 5,
        (2, 0): 6,
        (2, 1): 7,
        (2, 2): 8,
    }

    VOXEL_TEST_DATA = np.arange(10, 19)**3  # Varying second derivatives

    # Voxels are ordered in descending columns, as per Ingesson
    VOXELS_2D = np.reshape(VOXEL_COORDS, (NROW, NCOL, 2))

    TEST_DATA_2D = np.reshape(VOXEL_TEST_DATA, (NROW, NCOL))

    DERIVATIVE_OPERATORS = generate_derivative_operators(
        VOXEL_VERTICES, GRID_1D_TO_2D_MAP, GRID_2D_TO_1D_MAP
    )

    def test_dx(self):
        """D/Dx (Equations 37)"""
        DtestDx = self.DERIVATIVE_OPERATORS["Dx"] @ self.VOXEL_TEST_DATA
        data = self.TEST_DATA_2D
        for xi in range(self.NCOL):
            for yj in range(self.NROW):
                deriv = DtestDx[self.GRID_2D_TO_1D_MAP[(xi, yj)]]
                if xi == 0:  # Eq. 37b
                    eq_deriv = (data[1, yj] - data[0, yj]) / self.DX
                elif xi == self.NCOL - 1:  # Eq. 37c
                    eq_deriv = (data[xi, yj] - data[xi - 1, yj]) / self.DX
                else:  # Eq. 37a
                    eq_deriv = (data[xi + 1, yj] - data[xi - 1, yj]) / (2 * self.DX)
                self.assertEqual(eq_deriv, deriv, msg="Failed for ({}, {})".format(xi, yj))

    def test_dy(self):
        """D/Dy (Equations 38)"""
        DtestDy = self.DERIVATIVE_OPERATORS["Dy"] @ self.VOXEL_TEST_DATA
        data = self.TEST_DATA_2D
        for xi in range(self.NCOL):
            for yj in range(self.NROW):
                deriv = DtestDy[self.GRID_2D_TO_1D_MAP[(xi, yj)]]
                if yj == 0:  # Eq. 38b
                    eq_deriv = (data[xi, 0] - data[xi, 1]) / self.DY
                elif yj == self.NCOL - 1:  # Eq. 38c
                    eq_deriv = (data[xi, yj - 1] - data[xi, yj]) / self.DY
                else:  # Eq. 38a
                    eq_deriv = (data[xi, yj - 1] - data[xi, yj + 1]) / (2 * self.DY)
                self.assertEqual(eq_deriv, deriv, msg="Failed for ({}, {})".format(xi, yj))

    def test_dxx(self):
        """D/Dx2 (Equations 39)"""
        DtestDxx = self.DERIVATIVE_OPERATORS["Dxx"] @ self.VOXEL_TEST_DATA
        data = self.TEST_DATA_2D
        for xi in range(self.NCOL):
            for yj in range(self.NROW):
                deriv = DtestDxx[self.GRID_2D_TO_1D_MAP[(xi, yj)]]
                if xi == 0:  # Eq. 39b
                    eq_deriv = (data[1, yj] - data[0, yj]) / self.DX**2
                elif xi == self.NCOL - 1:  # Eq. 39c
                    eq_deriv = (data[xi, yj] - data[xi - 1, yj]) / self.DX**2
                else:  # Eq. 39a
                    eq_deriv = (data[xi + 1, yj] + data[xi - 1, yj] - 2 * data[xi, yj]) / self.DX**2
                self.assertEqual(eq_deriv, deriv, msg="Failed for ({}, {})".format(xi, yj))

    def test_dyy(self):
        """D/Dy2 (Equations 40)"""
        DtestDyy = self.DERIVATIVE_OPERATORS["Dyy"] @ self.VOXEL_TEST_DATA
        data = self.TEST_DATA_2D
        for xi in range(self.NCOL):
            for yj in range(self.NROW):
                deriv = DtestDyy[self.GRID_2D_TO_1D_MAP[(xi, yj)]]
                if yj == 0:  # Eq. 40b
                    eq_deriv = (data[xi, 0] - data[xi, 1]) / self.DY**2
                elif yj == self.NROW - 1:  # Eq. 40c
                    eq_deriv = (data[xi, yj - 1] - data[xi, yj]) / self.DY**2
                else:  # Eq. 40a
                    eq_deriv = (data[xi, yj - 1] + data[xi, yj + 1] - 2 * data[xi, yj]) / self.DY**2
                self.assertEqual(eq_deriv, deriv, msg="Failed for ({}, {})".format(xi, yj))

    def test_dyx(self):
        """D/DxDy (Equations 41)"""
        DtestDxy = self.DERIVATIVE_OPERATORS["Dxy"] @ self.VOXEL_TEST_DATA
        data = self.TEST_DATA_2D
        dxdy = self.DX * self.DY
        for xi in range(self.NCOL):
            for yj in range(self.NROW):
                deriv = DtestDxy[self.GRID_2D_TO_1D_MAP[(xi, yj)]]
                if xi == 0:
                    if yj == 0:  # Eq. 41f
                        eq_deriv = ((data[1, 0] + data[0, 1]
                                     - data[0, 0] - data[1, 1])
                                    / dxdy)
                    elif yj == self.NROW - 1:  # Eq. 41h
                        eq_deriv = ((data[1, yj - 1] + data[0, yj]
                                     - data[0, yj - 1] - data[1, yj])
                                    / dxdy)
                    else:  # Eq. 41d
                        eq_deriv = ((data[1, yj - 1] + data[0, yj + 1]
                                     - data[0, yj - 1] - data[1, yj + 1])
                                    / (2 * dxdy))
                elif xi == self.NCOL - 1:
                    if yj == 0:  # Eq. 41g
                        eq_deriv = ((data[xi, 0] + data[xi - 1, 1]
                                     - data[xi - 1, 0] - data[xi, 1])
                                    / dxdy)
                    elif yj == self.NROW - 1:  # Eq. 41i
                        eq_deriv = ((data[xi, yj - 1] + data[xi - 1, yj]
                                     - data[xi, yj] - data[xi - 1, yj - 1])
                                    / dxdy)
                    else:  # Eq. 41e
                        eq_deriv = ((data[xi, yj - 1] + data[xi - 1, yj + 1]
                                     - data[xi - 1, yj - 1] - data[xi, yj + 1])
                                    / (2 * dxdy))
                else:
                    if yj == 0: # Eq. 41b
                        eq_deriv = ((data[xi + 1, 0] + data[xi - 1, 1]
                                     - data[xi - 1, 0] - data[xi + 1, 1])
                                    / (2 * dxdy))
                    elif yj == self.NROW - 1:  # Eq. 41c
                        eq_deriv = ((data[xi + 1, yj - 1] + data[xi - 1, yj]
                                     - data[xi - 1, yj - 1] - data[xi + 1, yj])
                                    / (2 * dxdy))
                    else:  # Eq. 41a
                        eq_deriv = ((data[xi + 1, yj - 1] + data[xi - 1, yj + 1]
                                     - data[xi - 1, yj - 1] - data[xi + 1, yj + 1])
                                    / (4 * dxdy))
                self.assertEqual(eq_deriv, deriv, msg="Failed for ({}, {})".format(xi, yj))

    def test_objective(self):
        pass

if __name__ == "__main__":
    unittest.main()
