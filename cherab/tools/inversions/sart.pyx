
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

    cdef:
        int m_observations, n_sources, ith_obs, jth_cell, k
        list convergence
        double x_j, x_j_new, relax_over_density, obs_diff, measurement_squared, y_hat_squared, prop_ray_length
        np.ndarray solution, solution_new, y_hat_vector, cell_ray_densities, ray_lengths
        double[:] obs_vector_mv, solution_mv, solution_new_mv, y_hat_vector_mv, cell_ray_densities_mv, inv_ray_lengths_mv
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
    geometry_matrix_mv = geometry_matrix.T # Make second index ith_obs

    # Create an array to monitor the convergence
    convergence = []

    # A_(+,j)  - the total length of all rays passing through jth cell, equivalent to ray density
    cell_ray_densities = np.sum(geometry_matrix, axis=0)
    cell_ray_densities_mv = cell_ray_densities

    # A_(i,+)  - the total length of each ray
    ray_lengths = np.sum(geometry_matrix, axis=1)
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
                    prop_ray_length = geometry_matrix_mv[jth_cell, ith_obs] * inv_ray_lengths_mv[ith_obs]  # fraction of ray length/volume
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

    cdef:
        int m_observations, n_sources, ith_obs, jth_cell, k
        list convergence
        double x_j, x_j_new, relax_over_density, obs_diff, measurement_squared, y_hat_squared, prop_ray_length
        np.ndarray solution, solution_new, y_hat_vector, cell_ray_densities, ray_lengths
        double[:] obs_vector_mv, solution_mv, solution_new_mv, y_hat_vector_mv, cell_ray_densities_mv, inv_ray_lengths_mv, grad_penalty_mv
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
    geometry_matrix_mv = geometry_matrix.T # Make second index ith_obs

    # Create an array to monitor the convergence
    convergence = []

    # A_(+,j)  - the total length of all rays passing through jth cell, equivalent to ray density
    cell_ray_densities = np.sum(geometry_matrix, axis=0)
    cell_ray_densities_mv = cell_ray_densities

    # A_(i,+)  - the total length of each ray
    ray_lengths = np.sum(geometry_matrix, axis=1)
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
                    prop_ray_length = geometry_matrix_mv[jth_cell, ith_obs] * inv_ray_lengths_mv[ith_obs] # fraction of ray length/volume
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
