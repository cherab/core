# -*- coding: utf-8 -*-
#
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
#
# The following code is created by Vladislav Neverov (NRC "Kurchatov Institute") for Cherab Spectroscopy Modelling Framework

from __future__ import print_function
import os
import numpy as np
from timeit import default_timer as timer
from .opencl_utils import device_select
try:
    import pyopencl as cl
except ImportError:
    _has_pyopencl = False
else:
    _has_pyopencl = True


class SartOpencl:
    """
    A GPU-accelerated version of SART inversion.
    The geometry matrix and Laplacian matrix are provided on initialisation because they must
    be copied to GPU memory, which takes time. Inversions may be performed multiple times
    for different measurement vectors without copying the matrices each time. If required,
    the Laplacian matrix can be updated by calling
    `update_laplacian_matrix(new_laplacian_matrix)` method. Note that computations are
    performed with single precision.

    :param np.ndarray geometry_matrix: The sensitivity matrix describing the coupling between
        the detectors and the voxels. Must be an array with shape :math:`(N_d, N_s)`.
    :param np.ndarray laplacian_matrix:  The laplacian regularisation matrix of
        shape :math:`(N_s, N_s)`. Default value: `laplacian_matrix=None`.
    :param pyopencl.Device device: OpenCL device which will be used for computations.
        Default value: `device=None` (autoselect).
    :param int block_size: Number of GPU threads per block. Must be the power of 2.
        For the best performance try from 256 to 1024 for Nvidia (use 1024 on high-end GPUs),
        from 64 to 256 for AMD and from 16 to 64 for Intel GPUs. Default value: `block_size=256`.
    :param bool copy_column_major: If True, the two copies of geometry matrix will be stored in
        GPU memory. One in row-major order and the other one in column-major order. This
        provides much better performance of the inversions but requires twice as much GPU memory.
        Default value: `copy_column_major=True`.
    :param int block_size_row_maj: If `copy_column_major` is set to False, this parameter defines
        the number of GPU threads per block in mat_vec_mult_row_maj() kernel used to calculate
        y_hat. Must be lower than `block_size`. Default value: `block_size_row_maj=64` (optimal
        value for Nvidia GPUs).
    :param bool use_atomic: If True, increases the number of thread blocks that can run in
        parallel with the help of atomic operations (custom atomic add on floats). Set this
        to False, if the atomic operations are running slow on your device (Nvidia GPUs before
        Kepler, some AMD APUs, some Intel GPUs). Default value: `use_atomic=True`.
    :param int steps_per_thread: If `use_atomic` is set to True, this parameters defines the
        maximum number of loop steps performed by the parallel threads in a single thread block.
        Default value: `steps_per_thread=64` (optimal for Nvidia GPUs).

    .. code-block:: pycon

        >>> with SartOpencl(geometry_matrix, block_size=1024) as invert_sart:
        >>>     solution, residual = invert_sart(measurement_vector)
        >>> ### or ###
        >>> inv_sart = SartOpencl(geometry_matrix, block_size=1024)
        >>> solution, residual = inv_sart(measurement_vector)
        >>> inv_sart.clean()
    """

    def __init__(self, geometry_matrix, laplacian_matrix=None, device=None, block_size=256, copy_column_major=True, block_size_row_maj=64,
                 use_atomic=True, steps_per_thread=64):
        if not _has_pyopencl:
            raise RuntimeError("The pyopencl module is required to use the SartOpencl() inversion class.")
        if geometry_matrix.dtype != np.float32:  # converting geometry_matrix to float32 if needed
            geometry_matrix = geometry_matrix.astype(np.float32)
        self.m_detectors, self.n_sources = geometry_matrix.shape
        cell_ray_densities = geometry_matrix.sum(0)
        ray_lengths = geometry_matrix.sum(1)
        device = device or device_select()
        self.use_atomic = use_atomic
        steps_per_thread = min(block_size, steps_per_thread)
        steps_per_thread_row_maj = min(block_size_row_maj, steps_per_thread)

        # creating OpenCL context
        self.cl_context = cl.Context([device])

        # reading and compiling OpenCL kernels
        kernels_filename = 'sart_kernels_atomic.cl' if use_atomic else 'sart_kernels.cl'
        kernel_source_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), kernels_filename)
        with open(kernel_source_file) as f_kernel:
            kernel_source = f_kernel.read()
        compile_options = ['-DBLOCK_SIZE=%d' % block_size, '-DSTEPS_PER_THREAD=%d' % steps_per_thread,
                           '-DSTEPS_PER_THREAD_ROW_MAJ=%d' % steps_per_thread_row_maj,
                           '-DBLOCK_SIZE_ROW_MAJ=%d' % block_size_row_maj, '-cl-fast-relaxed-math']
        self.cl_prog = cl.Program(self.cl_context, kernel_source).build(options=compile_options)

        # creating buffers in device memory
        mf = cl.mem_flags
        self.geometry_matrix_device = cl.Buffer(self.cl_context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=geometry_matrix)
        if copy_column_major:
            geometry_matric_col_maj = geometry_matrix.flatten(order='F')
            self.geometry_matric_col_maj_device = cl.Buffer(self.cl_context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=geometry_matric_col_maj)
        else:
            self.geometry_matric_col_maj_device = None
        if laplacian_matrix is not None:
            laplacian_matrix = laplacian_matrix.flatten(order='F').astype(np.float32)
            self.laplacian_matrix_device = cl.Buffer(self.cl_context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=laplacian_matrix)
        else:
            self.laplacian_matrix_device = None
        self.cell_ray_densities_device = cl.Buffer(self.cl_context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=cell_ray_densities)
        self.ray_lengths_device = cl.Buffer(self.cl_context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ray_lengths)
        grad_penalty = np.zeros(self.n_sources, dtype=np.float32)
        self.grad_penalty_device = cl.Buffer(self.cl_context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=grad_penalty)
        self.solution_device = cl.Buffer(self.cl_context, mf.READ_WRITE, cell_ray_densities.nbytes)
        self.detectors_device = cl.Buffer(self.cl_context, mf.READ_ONLY, ray_lengths.nbytes)
        self.y_hat_device = cl.Buffer(self.cl_context, mf.READ_WRITE, ray_lengths.nbytes)

        # calculating global and local work sizes
        nrem = self.n_sources % block_size
        gws_sources_x = self.n_sources + bool(nrem) * (block_size - nrem)
        mrem = self.m_detectors % block_size
        gws_detectors_x = self.m_detectors + bool(mrem) * (block_size - mrem)
        mrem_rm = self.m_detectors % block_size_row_maj
        gws_detectors_row_maj_x = self.m_detectors + bool(mrem_rm) * (block_size - mrem_rm)
        if use_atomic:
            gws_sources_row_maj_y = self.n_sources // steps_per_thread_row_maj + bool(self.n_sources % steps_per_thread_row_maj)
            gws_sources_y = self.n_sources // steps_per_thread + bool(self.n_sources % steps_per_thread)
            gws_detectors_y = self.m_detectors // steps_per_thread + bool(self.m_detectors % steps_per_thread)
        else:
            gws_sources_row_maj_y = gws_sources_y = gws_detectors_y = 1
        self.global_work_size = {}
        self.local_work_size = {'default': (block_size, 1)}
        self.global_work_size['trivial_sources'] = (gws_sources_x, 1)
        self.global_work_size['trivial_detectors'] = (gws_detectors_x, 1)
        self.global_work_size['iter'] = (gws_sources_x, gws_detectors_y)
        if copy_column_major:
            self.local_work_size['mult'] = self.local_work_size['default']
            self.global_work_size['mult'] = (gws_detectors_x, gws_sources_y)
        else:
            self.local_work_size['mult'] = (block_size_row_maj, 1)
            self.global_work_size['mult'] = (gws_detectors_row_maj_x, gws_sources_row_maj_y)
        self.global_work_size['grad'] = (gws_sources_x, gws_sources_y)

    def clean(self):
        """ Releases GPU buffers"""
        self.geometry_matrix_device.release()
        if self.geometry_matric_col_maj_device is not None:
            self.geometry_matric_col_maj_device.release()
        if self.laplacian_matrix_device is not None:
            self.laplacian_matrix_device.release()
        self.cell_ray_densities_device.release()
        self.ray_lengths_device.release()
        self.solution_device.release()
        self.grad_penalty_device.release()
        self.detectors_device.release()
        self.y_hat_device.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.clean()

    def update_laplacian_matrix(self, laplacian_matrix):
        """
        Updates the Laplacian matrix in GPU memory

        :param np.ndarray laplacian_matrix:  The laplacian regularisation matrix of
            shape :math:`(N_s, N_s)`.
        """
        if self.laplacian_matrix_device is not None:
            laplacian_matrix = laplacian_matrix.flatten(order='F').astype(np.float32)
            queue = cl.CommandQueue(self.cl_context)
            cl.enqueue_copy(queue, self.laplacian_matrix_device, laplacian_matrix)

    def __call__(self, measurement_vector, initial_guess=None, max_iterations=250, relaxation=1.0,
                 beta_laplace=0.01, conv_tol=1.e-4, time_limit=None):
        """
        Performs the inversion for a given measurement vector.

        :param np.ndarray measurement_vector: The measured power/radiance vector with
            shape :math:`(N_d)`.
        :param initial_guess: An optional initial guess, can be an array of shape :math:`(N_s)`
            or a constant value that will be used to seed the algorithm.
        :param int max_iterations: The maximum number of iterations to run the SART algorithm
            before returning a result, defaults to `max_iterations=250`.
        :param float relaxation: The relaxation hyperparameter, defaults to `relaxation=1`.
            Consult the reference papers for more information on this hyperparameter.
        :param float beta_laplace: The regularisation hyperparameter in the range [0, 1].
            Defaults to `beta_laplace=0.01`.
        :param float conv_tol: The convergence limit at which the algorithm will be terminated,
            unless the maximum number of iterations has been reached. The convergence is
            calculated as the normalised squared difference between the measurement and solution
            vectors. Note that reaching convergence lower than 1.e-6 is hardly possible
            on GPUs due to single precision calculations and relaxed math.
        :param float time_limit: If set, the iterations will stop after this time limit (in
            seconds) is reached. Default value: `time_limit=None`.

        :return: A tuple with the inverted solution vector :math:`\mathbf{x}` as an ndarray with
            shape :math:`(N_s)`, and the list of convergence values achieved after each iteration
            step.
        """
        time_start = timer()
        time_limit = time_limit or 1.e7
        if initial_guess is None:
            solution = np.zeros(self.n_sources, dtype=np.float32) + 1 / np.e
        elif isinstance(initial_guess, (float, int)):
            solution = np.zeros(self.n_sources, dtype=np.float32) + initial_guess
        else:
            solution = initial_guess.astype(np.float32)  # making a copy even if initial_guess is in float32 already
        measurement_max = measurement_vector.max()
        # normalising and converting to float32
        measurement_vector = (measurement_vector / measurement_max).astype(np.float32)
        measurement_squared = np.dot(measurement_vector, measurement_vector)
        y_hat_vector = np.empty_like(measurement_vector)  # host y_hat
        queue = cl.CommandQueue(self.cl_context)

        # copying initial guess and measurement_vector to device
        cl.enqueue_copy(queue, self.solution_device, solution)
        cl.enqueue_copy(queue, self.detectors_device, measurement_vector)

        # calculating y_hat on device
        self._calc_y_hat(queue)

        # starting iterations
        convergence = []
        conv_tol = np.float32(conv_tol)
        success = False
        for k in range(max_iterations):
            # print('Iteration: %d' % k)
            # making one iteration on device
            self._make_iteration(queue, relaxation, beta_laplace)

            # calculating y_hat on device
            self._calc_y_hat(queue)

            # copying y_hat to host
            cl.enqueue_copy(queue, y_hat_vector, self.y_hat_device)

            # calculating convergence
            y_hat_squared = np.dot(y_hat_vector, y_hat_vector)
            convergence.append((measurement_squared - y_hat_squared) / measurement_squared)
            time_passed = timer() - time_start

            # checking conditions
            if k > 0 and np.abs(convergence[k] - convergence[k - 1]) < conv_tol:
                success = True
                print('Convergence limit is reached in %.4f s with %d iterations' % (time_passed, k + 1))
                break
            if time_passed > time_limit:
                print('Time limit is exceeded')
                break

        if (not success) and k == max_iterations - 1:
            print('Maximum number of iterations is reached. Time passed: %.4f s' % time_passed)

        # copying solution to host
        cl.enqueue_copy(queue, solution, self.solution_device)

        return solution * measurement_max, convergence

    def _calc_y_hat(self, queue):
        if self.use_atomic:
            self.cl_prog.zero_all(queue, self.global_work_size['trivial_detectors'], self.local_work_size['default'],
                                  self.y_hat_device, np.uint32(self.m_detectors))
        if self.geometry_matric_col_maj_device is None:
            self.cl_prog.mat_vec_mult_row_major(queue, self.global_work_size['mult'], self.local_work_size['mult'],
                                                self.geometry_matrix_device, self.solution_device, self.y_hat_device,
                                                np.uint32(self.m_detectors), np.uint32(self.n_sources))
        else:
            self.cl_prog.mat_vec_mult_col_major(queue, self.global_work_size['mult'], self.local_work_size['mult'],
                                                self.geometry_matric_col_maj_device, self.solution_device, self.y_hat_device,
                                                np.uint32(self.m_detectors), np.uint32(self.n_sources))

    def _make_iteration(self, queue, relaxation, beta_laplace):
        if self.laplacian_matrix_device is not None:
            if self.use_atomic:
                self.cl_prog.zero_all(queue, self.global_work_size['trivial_sources'], self.local_work_size['default'],
                                      self.grad_penalty_device, np.uint32(self.n_sources))
            self.cl_prog.mat_vec_mult_col_major(queue, self.global_work_size['grad'], self.local_work_size['default'],
                                                self.laplacian_matrix_device, self.solution_device, self.grad_penalty_device,
                                                np.uint32(self.n_sources), np.uint32(self.n_sources))
            self.cl_prog.vec_scalar_mult(queue, self.global_work_size['trivial_sources'], self.local_work_size['default'],
                                         self.grad_penalty_device, np.float32(beta_laplace), np.uint32(self.n_sources))
        # grad_penalty is just an all-zero array if laplacian_matrix is not provided on initialisation
        self.cl_prog.sart_iteration(queue, self.global_work_size['iter'], self.local_work_size['default'],
                                    self.geometry_matrix_device, self.cell_ray_densities_device, self.ray_lengths_device,
                                    self.y_hat_device, self.detectors_device, self.solution_device, self.grad_penalty_device,
                                    np.float32(relaxation), np.uint32(self.n_sources), np.uint32(self.m_detectors))
        if self.use_atomic:
            self.cl_prog.zero_negative(queue, self.global_work_size['trivial_sources'], self.local_work_size['default'],
                                       self.solution_device, np.uint32(self.n_sources))
