
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
from scipy import linalg


def invert_svd(w_matrix, b_vector):
    """
    Performs a Singular Value Decomposition (SVD) operation inversion.

    :param np.ndarray w_matrix: The sensitivity matrix describing the coupling between the
      detectors and the voxels. Must be an array with shape :math:`(N_d, N_s)`.
    :param np.ndarray b_vector: The measured power/radiance vector with shape :math:`(N_d)`.
    :return: The solution vector x as an ndarray.
    """

    # Compute the Moore-Penrose pseudo-inverse of a matrix from SVD
    inverse_w_matrix = np.matrix(linalg.pinv(w_matrix))

    # reshape b_vector into a column vector
    b_vector = b_vector.reshape((len(b_vector), 1))

    inverted_x_vector = (inverse_w_matrix * b_vector).flatten()

    return np.asarray(inverted_x_vector).flatten()
