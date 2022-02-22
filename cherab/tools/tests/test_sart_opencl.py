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

import gc
import unittest
import os
import numpy as np
from cherab.tools.inversions import SartOpencl
from cherab.tools.inversions.opencl import device_select
try:
    import pyopencl as cl
except ImportError:
    _has_pyopencl = False
else:
    _has_pyopencl = True


@unittest.skipUnless(_has_pyopencl, "the pyopencl module is required to use the SartOpencl solver")
class TestSartOpencl(unittest.TestCase):
    """
    Test cases for SartOpencl solver.
    Tests unconstrained inversion with and without atomic operation and constrained inversion on GPU.
    Note that these tests will fail if pyopencl module is not installed
    or the system has no OpenCL-compatible GPU (or the OpenCL-compatible dirver).
    """

    def setUp(self):
        # geometry matrix in float32, shape: (npixel_x, npixel_y, nsource)
        gm = np.load(os.path.join(os.path.dirname(__file__), 'data/geometry_matrix.npy'))
        self.gm = gm.reshape((gm.shape[0] * gm.shape[1], gm.shape[2]))
        # receiver in float32, shape: (npixel_x, npixel_y)
        receiver = np.load(os.path.join(os.path.dirname(__file__), 'data/receiver.npy'))
        self.receiver = receiver.flatten()
        # true emissivity in float32, shape: (11, 8)
        true_emissivity = np.load(os.path.join(os.path.dirname(__file__), 'data/true_emissivity.npy'))
        self.true_emissivity = true_emissivity.flatten()
        # Any OpenCL device, including a CPU, will do for the tests. This enables
        # POCL to be installed as an OpenCL driver for testing.
        self.device = device_select(device_type=cl.device_type.ALL)

    def tearDown(self):
        # Ensure the OpenCL device is properly released between tests.
        self.device = None
        gc.collect()

    def test_inversion(self):
        with SartOpencl(self.gm, block_size=256, copy_column_major=True, use_atomic=False,
                        steps_per_thread=64, block_size_row_maj=64, device=self.device) as inv_sart:
            solution, residual = inv_sart(self.receiver)
        self.assertTrue(np.allclose(solution, self.true_emissivity, atol=1.e-2))

    def test_inversion_atomic(self):
        with SartOpencl(self.gm, block_size=256, copy_column_major=True, use_atomic=True,
                        steps_per_thread=64, block_size_row_maj=64, device=self.device) as inv_sart:
            solution, residual = inv_sart(self.receiver)
        self.assertTrue(np.allclose(solution, self.true_emissivity, atol=1.e-2))

    def test_inversion_constrained(self):
        # The emission profile is a sharp function here, so in this test the regularisation leads to inaccurate results.
        # The beta_laplace parameter is set to just 0.001 to reduce the impact of regularisation. This is a technical test only.
        laplacian_matrix = np.identity(self.gm.shape[1], dtype=np.float32)
        with SartOpencl(self.gm, laplacian_matrix=laplacian_matrix, block_size=256, copy_column_major=True, use_atomic=False,
                        steps_per_thread=64, block_size_row_maj=64, device=self.device) as inv_sart:
            solution, residual = inv_sart(self.receiver, beta_laplace=0.001)
        self.assertTrue(np.allclose(solution / solution.max(), self.true_emissivity, atol=1.e-2))
