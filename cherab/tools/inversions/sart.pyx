
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

import numpy as np
cimport numpy as np
cimport cython


@cython.boundscheck(False)
cpdef invert_sart(geometry_matrix, measurement_vector, object initial_guess=None, int max_iterations=250,
                  double relaxation=1.0, double conv_tol=1.0E-4):
    """
    Performs a SART inversion on the specified measurement vector.
    
    This function implements the Simultaneous Algebraic Reconstruction Technique (SART), as published in
    A. Andersen, and A. Kak, Ultrasonic imaging 6, 81 (1984). The SART method is an iterative inversion 
    scheme where the source cells are updated with the formula

    .. math::

       x_l^{(i+1)} = f_{sart}(x_l^{(i)}) = x_l^{(i)} + \\frac{\omega}{W_{\oplus,l}} \sum_{k=1}^{N_d} \\frac{W_{k,l}}{W_{k,\oplus}} (\Phi_k - \hat{\Phi}_k),

    where
    
    .. math::
       W_{k,\oplus} = \sum_{l=1}^{N_s} W_{k,l}, \quad W_{\oplus, l} = \sum_{k=1}^{N_d} W_{k,l}.

    Here :math:`x_l^{(i)}` is the previous estimate for the emission at voxel :math:`l` in iteration :math:`i`.
    The SART method effectively updates each cell by the weighted average error between the forward modelled
    :math:`\hat{\Phi}_k` and observed :math:`\Phi_k` measurements. The observed errors are weighted by both
    their proportion of the total ray length (:math:`W_{k,\oplus}`) and the sum of the effective ray paths 
    crossing that cell (:math:`W_{\oplus, l}`).
    
    :param np.ndarray geometry_matrix: The sensitivity matrix describing the coupling between the detectors
      and the voxels. Must be an array with shape :math:`(N_d, N_s)`.
    :param np.ndarray measurement_vector: The measured power/radiance vector with shape :math:`(N_d)`. 
    :param initial_guess: An optional initial guess, can be an array of shape :math:`(N_s)` or a constant
      value that will be used to seed the algorithm.
    :param int max_iterations: The maximum number of iterations to run the SART algorithm before returning
      a result, defaults to `max_iterations=250`.
    :param float relaxation: The relaxation hyperparameter, defaults to `relaxation=1`. Consult the reference
      papers for more information on this hyperparameter.
    :param float conv_tol: The convergence limit at which the algorithm will be terminated, unless the maximum
      number of iterations has been reached. The convergence is calculated as the normalised squared difference
      between the measurement and solution vectors.
    :return: A tuple with the inverted solution vector :math:`\mathbf{x}` as an ndarray with shape :math:`(N_s)`,
      and the convergence achieved as a float.
    
    .. code-block:: pycon
   
       >>> from cherab.tools.inversions import invert_sart
       >>> inverted_solution, conv = invert_sart(weight_matrix, observations, max_iterations=100)    
    """

    cdef:
        int m_observations, n_sources, ith_obs, jth_cell, k
        list convergence
        double x_j, x_j_new, relax_over_density, obs_diff, measurement_squared, y_hat_squared, prop_ray_length
        np.ndarray solution, solution_new, y_hat_vector, cell_ray_densities, ray_lengths
        double[:] obs_vector_mv, solution_mv, solution_new_mv, y_hat_vector_mv, cell_ray_densities_mv, ray_lengths_mv, inv_ray_lengths_mv
        double[:,:] geometry_matrix_mv

    m_observations, n_sources = geometry_matrix.shape  # (M, N) matrix

    if initial_guess is None:
        solution = np.zeros(n_sources) + np.exp(-1)
    elif isinstance(initial_guess, (float, int)):
        solution = np.zeros(n_sources) + initial_guess
    else:
        solution = initial_guess
    solution_mv = solution

    solution_new = np.zeros(n_sources)
    solution_new_mv = solution_new

    obs_vector_mv = measurement_vector
    geometry_matrix_mv = geometry_matrix

    # Create an array to monitor the convergence
    convergence = []

    # A_(+,j)  - the total length of all rays passing through jth cell, equivalent to ray density
    cell_ray_densities = np.sum(geometry_matrix, axis=0)
    cell_ray_densities_mv = cell_ray_densities

    # A_(i,+)  - the total length of each ray
    ray_lengths = np.sum(geometry_matrix, axis=1)
    ray_lengths_mv = ray_lengths
    inv_ray_lengths_mv = 1 / ray_lengths

    y_hat_vector = np.dot(geometry_matrix, solution)
    y_hat_vector_mv = y_hat_vector

    for k in range(max_iterations):

        for jth_cell in range(n_sources):

            x_j = solution_mv[jth_cell]  # previous solution value for this cell

            if cell_ray_densities_mv[jth_cell] > 0.0:

                with cython.cdivision(True):
                    relax_over_density = relaxation / cell_ray_densities_mv[jth_cell]
                obs_diff = 0
                for ith_obs in range(m_observations):
                    # Ray path length can be zero
                    if ray_lengths_mv[ith_obs] == 0:
                        continue
                    prop_ray_length = geometry_matrix_mv[ith_obs, jth_cell] * inv_ray_lengths_mv[ith_obs]  # fraction of ray length/volume
                    obs_diff += prop_ray_length * (obs_vector_mv[ith_obs] - y_hat_vector_mv[ith_obs])

                x_j_new = x_j + relax_over_density * obs_diff

            # It is possible that some cells will have no rays passing through them.
            else:
                x_j_new = x_j

            # Don't allow negativity
            if x_j_new < 0:
                x_j_new = 0.0

            solution_new_mv[jth_cell] = x_j_new

        # Calculate how quickly the code is converging

        y_hat_vector = np.dot(geometry_matrix, solution_new)
        y_hat_vector_mv = y_hat_vector

        measurement_squared = np.dot(measurement_vector, measurement_vector)
        y_hat_squared = np.dot(y_hat_vector, y_hat_vector)
        convergence.append((measurement_squared - y_hat_squared) / measurement_squared)

        # Set the new solution to be the old solution and get ready to repeat
        solution_mv[:] = solution_new_mv[:]

        # Check for convergence
        if k > 0:
            if np.abs(convergence[k]-convergence[k-1]) < conv_tol:
                break

    return solution, convergence


@cython.boundscheck(False)
cpdef invert_constrained_sart(geometry_matrix, laplacian_matrix, measurement_vector,
                              object initial_guess=None, int max_iterations=250, double relaxation=1.0,
                              double beta_laplace=0.01, double conv_tol=1.0E-4):
    """

    Performs a constrained SART inversion on the specified measurement vector.
    
    The core of the constrained SART algorithm is identical to the basic SART algorithm implemented in 
    `invert_sart()`. The only difference is that now the iterative update formula includes a 
    regularisation operator.

    .. math::

       x_l^{(i+1)} = f_{sart}(x_l^{(i)}) - \hat{\mathcal{L}}_{iso}(x_l^{(i)}).

    In this particular function we have implemented a isotropic Laplacian smoothness operator, 
    
    .. math::

       \hat{\mathcal{L}}_{iso}(x_l^{(i)}) = \\beta_L (Cx_l^{(i)} - \sum_{c=1}^C x_c^{(i)}).

    Here, :math:`c` is the index for the sum over the neighbouring voxels. The regularisation 
    hyperparameter :math:`\\beta_L` determines the amount of local smoothness imposed on the
    solution vector. When :math:`\\beta_L = 0`, the solution is fully determined by the 
    measurements, and as :math:`\\beta_L \\rightarrow 1`, the solution is dominated by the 
    smoothness operator.
    
    :param np.ndarray geometry_matrix: The sensitivity matrix describing the coupling between the detectors
      and the voxels. Must be an array with shape :math:`(N_d, N_s)`.
    :param np.ndarray laplacian_matrix: The laplacian regularisation matrix of shape :math:`(N_s, N_s)`.
    :param np.ndarray measurement_vector: The measured power/radiance vector with shape :math:`(N_d)`. 
    :param initial_guess: An optional initial guess, can be an array of shape :math:`(N_s)` or a constant
      value that will be used to seed the algorithm.
    :param int max_iterations: The maximum number of iterations to run the SART algorithm before returning
      a result, defaults to `max_iterations=250`.
    :param float relaxation: The relaxation hyperparameter, defaults to `relaxation=1`. Consult the reference
      papers for more information on this hyperparameter.
    :param float beta_laplace: The regularisation hyperparameter in the range [0, 1]. Defaults
      to `beta_laplace=0.01`.
    :param float conv_tol: The convergence limit at which the algorithm will be terminated, unless the maximum
      number of iterations has been reached. The convergence is calculated as the normalised squared difference
      between the measurement and solution vectors.
    :return: A tuple with the inverted solution vector :math:`\mathbf{x}` as an ndarray with shape :math:`(N_s)`,
      and the convergence achieved as a float.
    
    .. code-block:: pycon
   
       >>> from cherab.tools.inversions import invert_constrained_sart
       >>> inverted_solution, conv = invert_constrained_sart(weight_matrix, laplacian, observations)
    """

    cdef:
        int m_observations, n_sources, ith_obs, jth_cell, k
        list convergence
        double x_j, x_j_new, relax_over_density, obs_diff, measurement_squared, y_hat_squared, prop_ray_length
        np.ndarray solution, solution_new, y_hat_vector, cell_ray_densities, ray_lengths
        double[:] obs_vector_mv, solution_mv, solution_new_mv, y_hat_vector_mv, cell_ray_densities_mv, ray_lengths_mv, inv_ray_lengths_mv, grad_penalty_mv
        double[:,:] geometry_matrix_mv

    m_observations, n_sources = geometry_matrix.shape  # (M, N) matrix

    if initial_guess is None:
        solution = np.zeros(n_sources) + np.exp(-1)
    elif isinstance(initial_guess, (float, int)):
        solution = np.zeros(n_sources) + initial_guess
    else:
        solution = initial_guess
    solution_mv = solution

    solution_new = np.zeros(n_sources)
    solution_new_mv = solution_new

    obs_vector_mv = measurement_vector
    geometry_matrix_mv = geometry_matrix

    # Create an array to monitor the convergence
    convergence = []

    # A_(+,j)  - the total length of all rays passing through jth cell, equivalent to ray density
    cell_ray_densities = np.sum(geometry_matrix, axis=0)
    cell_ray_densities_mv = cell_ray_densities

    # A_(i,+)  - the total length of each ray
    ray_lengths = np.sum(geometry_matrix, axis=1)
    ray_lengths_mv = ray_lengths
    inv_ray_lengths_mv = 1 / ray_lengths

    y_hat_vector = np.dot(geometry_matrix, solution)
    y_hat_vector_mv = y_hat_vector

    for k in range(max_iterations):

        grad_penalty = np.dot(laplacian_matrix, solution) * beta_laplace
        grad_penalty_mv = grad_penalty

        for jth_cell in range(n_sources):

            x_j = solution_mv[jth_cell]  # previous solution value for this cell

            if cell_ray_densities_mv[jth_cell] > 0.0:

                with cython.cdivision(True):
                    relax_over_density = relaxation / cell_ray_densities_mv[jth_cell]

                obs_diff = 0
                for ith_obs in range(m_observations):
                    # Ray path length can be zero
                    if ray_lengths_mv[ith_obs] == 0:
                        continue
                    prop_ray_length = geometry_matrix_mv[ith_obs, jth_cell] * inv_ray_lengths_mv[ith_obs] # fraction of ray length/volume
                    obs_diff += prop_ray_length * (obs_vector_mv[ith_obs] - y_hat_vector_mv[ith_obs])

                x_j_new = x_j + relax_over_density * obs_diff - grad_penalty_mv[jth_cell]

            # It is possible that some cells will have no rays passing through them.
            else:
                x_j_new = x_j - grad_penalty_mv[jth_cell]

            # Don't allow negativity
            if x_j_new < 0:
                x_j_new = 0.0

            solution_new_mv[jth_cell] = x_j_new

        # Calculate how quickly the code is converging

        y_hat_vector = np.dot(geometry_matrix, solution_new)
        y_hat_vector_mv = y_hat_vector

        measurement_squared = np.dot(measurement_vector, measurement_vector)
        y_hat_squared = np.dot(y_hat_vector, y_hat_vector)
        convergence.append((measurement_squared - y_hat_squared) / measurement_squared)

        # Set the new solution to be the old solution and get ready to repeat
        solution_mv[:] = solution_new_mv[:]

        # Check for convergence
        if k > 0:
            if np.abs(convergence[k]-convergence[k-1]) < conv_tol:
                break

    return solution, convergence
