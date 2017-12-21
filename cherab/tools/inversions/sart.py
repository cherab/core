
# Copyright 2014-2017 United Kingdom Atomic Energy Authority
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
import scipy.sparse as sps


# Example SART algorithm provided by Dr J. Harrison, UKAEA.
def invert_sart(geometry_matrix, measurement_matrix, max_iterations=50, lam_start=1.0, conv_tol=1.0E-4):
    print()
    print("Sart test")

    csc_A = sps.csc_matrix(geometry_matrix)
    csc_b = sps.csc_matrix(measurement_matrix)

    print(csc_A.shape)
    print(csc_b.shape)

    shap = csc_A.shape
    lam = lam_start
    colsum = (csc_A.transpose()).dot(sps.csc_matrix(np.ones(shap[0])).transpose())
    lamda = colsum
    #lamda = lamda.multiply(colsum != 0)
    np.reciprocal(lamda.data, out=lamda.data)
    np.multiply(lamda.data, lam, out=lamda.data)

    # Set a convergergence threshold

    # Initialise output
    sol = sps.csc_matrix(np.zeros((shap[1], 1)) + np.exp(-1))
    # Create an array to monitor the convergence
    conv = np.zeros(max_iterations)

    for i in range(max_iterations):
        # Calculate sol_new = sol+lambda*(A'*(b-Ax))
        # print(i)
        tmp = csc_b.transpose()-csc_A.dot(sol)
        tmp2 = csc_A.transpose().dot(tmp)
        #newsol = sol+tmp2*lamda
        newsol = sol+tmp2.multiply(lamda)
        # Eliminate negative values
        newsol = newsol.multiply(newsol > 0.0)
        newsol.eliminate_zeros()
        # Calculate how quickly the code is converging
        conv[i] = (sol.multiply(sol).sum()-newsol.multiply(newsol).sum())/sol.multiply(sol).sum()
        # Set the new solution to be the old solution and repeat
        sol = newsol
        # Clear up memory leaks
        tmp = None
        tmp2 = None
        # Check for convergence
        if i > 0:
            if np.abs(conv[i]-conv[i-1]) < conv_tol:
                break

    print("i iterations", i)

    sol = None
    return newsol.todense(), conv
