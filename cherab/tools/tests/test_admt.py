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
import numpy.testing as npt

from cherab.tools.inversions import generate_derivative_operators

class TestADMT(unittest.TestCase):
    """Tests for ADMT utilities"""

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

    # TODO: generalise tests to more than 3x3 grid (use -1 for X, Y indices)

    # Check that equations 38-41 in JET-R(99)08 are satisfied.
    def test_dx(self):
        """D/Dx (Equations 37)"""
        DtestDx = self.DERIVATIVE_OPERATORS["Dx"] @ self.VOXEL_TEST_DATA
        data = self.TEST_DATA_2D
        for i in range(self.NROW):  # Test each row of the grid
            i0 = self.GRID_2D_TO_1D_MAP[(0, i)]
            i1 = self.GRID_2D_TO_1D_MAP[(1, i)]
            i2 = self.GRID_2D_TO_1D_MAP[(2, i)]
            yX = (data[2, i] - data[1, i]) / self.DX
            y1 = (data[1, i] - data[0, i]) / self.DX
            yx = (data[2, i] - data[0, i]) / (2 * self.DX)
            npt.assert_equal(DtestDx[[i0, i1, i2]], (y1, yx, yX))

    def test_dy(self):
        """D/Dy (Equations 38)"""
        DtestDy = self.DERIVATIVE_OPERATORS["Dy"] @ self.VOXEL_TEST_DATA
        data = self.TEST_DATA_2D
        for i in range(self.NCOL):  # Test each column of the grid
            i0 = self.GRID_2D_TO_1D_MAP[(i, 0)]
            i1 = self.GRID_2D_TO_1D_MAP[(i, 1)]
            i2 = self.GRID_2D_TO_1D_MAP[(i, 2)]
            yx = (data[i, 0] - data[i, 2]) / (2 * self.DY)
            onex = (data[i, 0] - data[i, 1]) / self.DY
            Yx = (data[i, 1] - data[i, 2]) / self.DY
            npt.assert_equal(DtestDy[[i0, i1, i2]], (onex, yx, Yx))

    def test_dxx(self):
        """D/Dx2 (Equations 39)"""
        DtestDxx = self.DERIVATIVE_OPERATORS["Dxx"] @ self.VOXEL_TEST_DATA
        data = self.TEST_DATA_2D
        for i in range(self.NROW):  # Test each row of the grid
            i0 = self.GRID_2D_TO_1D_MAP[(0, i)]
            i1 = self.GRID_2D_TO_1D_MAP[(1, i)]
            i2 = self.GRID_2D_TO_1D_MAP[(2, i)]
            yx = (data[2, i] + data[0, i] - 2 * data[1, i]) / self.DX**2
            y1 = (data[1, i] - data[0, i]) / self.DX**2
            yX = (data[2, i] - data[1, i]) / self.DX**2
            npt.assert_equal(DtestDxx[[i0, i1, i2]], (y1, yx, yX))

    def test_dyy(self):
        """D/Dy2 (Equations 40)"""
        DtestDyy = self.DERIVATIVE_OPERATORS["Dyy"] @ self.VOXEL_TEST_DATA
        data = self.TEST_DATA_2D
        for i in range(self.NCOL):  # Test each column of the grid
            i0 = self.GRID_2D_TO_1D_MAP[(i, 0)]
            i1 = self.GRID_2D_TO_1D_MAP[(i, 1)]
            i2 = self.GRID_2D_TO_1D_MAP[(i, 2)]
            yx = (data[i, 0] + data[i, 2] - 2 * data[i, 1]) / self.DY**2
            onex = (data[i, 0] - data[i, 1]) / self.DY**2
            Yx = (data[i, 1] - data[i, 2]) / self.DY**2
            npt.assert_equal(DtestDyy[[i0, i1, i2]], (onex, yx, Yx))

    def test_dxy(self):  # pylint: disable=too-many-locals
        """D/DxDy (Equations 41)"""
        DtestDxy = self.DERIVATIVE_OPERATORS["Dxy"] @ self.VOXEL_TEST_DATA
        data = self.TEST_DATA_2D
        dxdy = self.DX * self.DY
        # Test corners of the grid (41f to 41i)
        i00 = self.GRID_2D_TO_1D_MAP[(0, 0)]
        iX0 = self.GRID_2D_TO_1D_MAP[(2, 0)]
        i0Y = self.GRID_2D_TO_1D_MAP[(0, 2)]
        iXY = self.GRID_2D_TO_1D_MAP[(2, 2)]
        oneone = (data[1, 0] + data[0, 1] - data[0, 0] - data[1, 1]) / dxdy
        oneX = (data[2, 0] + data[1, 1] - data[1, 0] - data[2, 1]) / dxdy
        Y1 = (data[1, 1] + data[0, 2] - data[0, 1] - data[1, 2]) / dxdy
        YX = (data[2, 1] + data[1, 2] - data[2, 2] - data[1, 1]) / dxdy
        npt.assert_equal(DtestDxy[[i00, iX0, i0Y, iXY]],
                         (oneone, oneX, Y1, YX))
        # Test edges (41b-41e)
        for i in range(3):
            i1x = self.GRID_2D_TO_1D_MAP[(i, 0)]
            iYx = self.GRID_2D_TO_1D_MAP[(i, 2)]
            iy1 = self.GRID_2D_TO_1D_MAP[(0, i)]
            iyX = self.GRID_2D_TO_1D_MAP[(2, i)]
            if 0 < i < 2:
                onex = ((data[i + 1, 0] + data[i - 1, 1] - data[i - 1, 0] - data[i + 1, 1])
                        / (2 * dxdy))
                Yx = ((data[i + 1, -2] + data[i - 1, -1] - data[i - 1, -2] - data[i + 1, -1])
                      / (2 * dxdy))
                y1 = ((data[1, i - 1] + data[0, i + 1] - data[0, i - 1] - data[1, i + 1])
                      / (2 * dxdy))
                yX = ((data[-1, i - 1] + data[-2, i + 1] - data[-2, i - 1] - data[-1, i + 1])
                      / (2 * dxdy))
                np.testing.assert_equal(DtestDxy[[i1x, iYx, iy1, iyX]],
                                        (onex, Yx, y1, yX))
        # Test inner (41a)
        for i in range(3):
            for j in range(3):
                if 0 < i < 2:
                    if 0 < j < 2:
                        ixy = self.GRID_2D_TO_1D_MAP[(i, j)]
                        yx = ((data[i + 1, j - 1] + data[i - 1, j + 1]
                               - data[i - 1, j - 1] - data[i + 1, j + 1])
                              / (4 * dxdy))
                        self.assertEqual(DtestDxy[ixy], yx)


    def test_objective(self):
        pass
