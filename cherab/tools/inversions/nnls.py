
import numpy as np
import scipy


def invert_regularised_nnls(w_matrix, b_vector, alpha=0.01):

    # print('w_matrix shape', w_matrix.shape)

    m, n = w_matrix.shape

    alpha_identity = np.identity(n) * alpha

    # Extend W to have form ...
    c_matrix = np.zeros((m+n, n))
    c_matrix[0:m, :] = w_matrix[:, :]
    c_matrix[m:, :] = alpha_identity[:, :]

    # Extend b to have form ...
    d_vector = np.zeros(m+n)
    d_vector[0:m] = b_vector[:]

    x_vector, rnorm = scipy.optimize.nnls(c_matrix, d_vector)

    # print('x_vector shape', x_vector.shape)

    return x_vector, rnorm

