
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


def invert_regularised_lstsq(w_matrix, b_vector, alpha=0.01, tikhonov_matrix=None):
    """
    Solves :math:`\mathbf{b} = \mathbf{W} \mathbf{x}` for the vector :math:`\mathbf{x}`,
    using Tikhonov regulariastion.

    This is a thin wrapper around numpy.linalg.lstsq, which modifies
    the arguments to include the supplied Tikhonov regularisation matrix.

    :param np.ndarray w_matrix: The sensitivity matrix describing the coupling between the
      detectors and the voxels. Must be an array with shape :math:`(N_d, N_s)`.
    :param np.ndarray b_vector: The measured power/radiance vector with shape :math:`(N_d)`.
    :param float alpha: The regularisation hyperparameter :math:`\alpha` which determines
      the regularisation strength of the tikhonov matrix.
    :param np.ndarray tikhonov_matrix: The tikhonov regularisation matrix operator, an array
      with shape :math:`(N_s, N_s)`. If None, the identity matrix is used.
    :return: (x, residuals), the solution and residual vectors.

    .. code-block:: pycon

       >>> from cherab.tools.inversions import invert_regularised_lstsq
       >>> x, rvec = invert_regularised_lstsq(w_matrix, b_vector, tikhonov_matrix=tikhonov_matrix)
    """

    m, n = w_matrix.shape

    if tikhonov_matrix is None:
        tikhonov_matrix = np.identity(n)

    tikhonov_matrix = alpha * tikhonov_matrix

    # Extend W to have form ...
    c_matrix = np.zeros((m+n, n))
    c_matrix[0:m, :] = w_matrix[:, :]
    c_matrix[m:, :] = tikhonov_matrix[:, :]

    # Extend b to have form ...
    d_vector = np.zeros(m+n)
    d_vector[0:m] = b_vector[:]

    x_vector, residuals, _, _ = np.linalg.lstsq(c_matrix, d_vector, rcond=None)

    return x_vector, residuals
