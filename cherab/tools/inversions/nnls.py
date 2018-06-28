
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
import scipy


def invert_regularised_nnls(w_matrix, b_vector, alpha=0.01, tikhonov_matrix=None):
    """
    Solve w_matrix · x = b_vector for the vector x, using Tikhonov
    regulariastion.

    This is a thin wrapper around scipy.optimize.nnls, which modifies
    the arguments to include the supplied Tikhonov regularisation matrix.

    If tikhonov_matrix is None, the matrix used is alpha times the
    identity matrix.

    Returns (x, norm), the solution vector and the residual norm.
    """

    # print('w_matrix shape', w_matrix.shape)

    m, n = w_matrix.shape

    if tikhonov_matrix is None:
        tikhonov_matrix = np.identity(n) * alpha

    # Extend W to have form ...
    c_matrix = np.zeros((m+n, n))
    c_matrix[0:m, :] = w_matrix[:, :]
    c_matrix[m:, :] = tikhonov_matrix[:, :]

    # Extend b to have form ...
    d_vector = np.zeros(m+n)
    d_vector[0:m] = b_vector[:]

    x_vector, rnorm = scipy.optimize.nnls(c_matrix, d_vector)

    # print('x_vector shape', x_vector.shape)

    return x_vector, rnorm
