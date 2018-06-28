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

"""
This script has been used to calculate the reference data for the 1D cubic interpolator C2 tests.
The reference interpolated data is stored in 'interp_data'.
The reference extrapolated data with linear method is stored in 'extr_data_lin'.
The reference extrapolated data with quadratic method is stored in 'extr_data_qua'.
"""

import numpy as np
from scipy.linalg import inv

np.set_printoptions(12, 30000, linewidth=90)

x = np.linspace(0, 1, 10)
x2 = x*x
x3 = x2*x
xsamples = np.linspace(0, 1, 30)

def f(a):
    return np.cos(10*a)

data = f(x)


const_mat = np.array(
[[x3[0], x2[0], x[0], 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [x3[1], x2[1], x[1], 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, x3[1], x2[1], x[1], 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, x3[2], x2[2], x[2], 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, x3[2], x2[2], x[2], 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, x3[3], x2[3], x[3], 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x3[3], x2[3], x[3], 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x3[4], x2[4], x[4], 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x3[4], x2[4], x[4], 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x3[5], x2[5], x[5], 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x3[5], x2[5], x[5], 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x3[6], x2[6], x[6], 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x3[6], x2[6], x[6], 1, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x3[7], x2[7], x[7], 1, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x3[7], x2[7], x[7], 1, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x3[8], x2[8], x[8], 1, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x3[8], x2[8], x[8], 1],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x3[9], x2[9], x[9], 1],
 [3*x2[1], 2*x[1], 1, 0, -3*x2[1], -2*x[1], -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 3*x2[2], 2*x[2], 1, 0, -3*x2[2], -2*x[2], -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 3*x2[3], 2*x[3], 1, 0, -3*x2[3], -2*x[3], -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3*x2[4], 2*x[4], 1, 0, -3*x2[4], -2*x[4], -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3*x2[5], 2*x[5], 1, 0, -3*x2[5], -2*x[5], -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3*x2[6], 2*x[6], 1, 0, -3*x2[6], -2*x[6], -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3*x2[7], 2*x[7], 1, 0, -3*x2[7], -2*x[7], -1, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3*x2[8], 2*x[8], 1, 0, -3*x2[8], -2*x[8], -1, 0],
 [6*x[1], 2, 0, 0, -6*x[1], -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 6*x[2], 2, 0, 0, -6*x[2], -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 6*x[3], 2, 0, 0, -6*x[3], -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6*x[4], 2, 0, 0, -6*x[4], -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6*x[5], 2, 0, 0, -6*x[5], -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6*x[6], 2, 0, 0, -6*x[6], -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6*x[7], 2, 0, 0, -6*x[7], -2, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6*x[8], 2, 0, 0, -6*x[8], -2, 0, 0],
 [6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0]], dtype=np.float64)

const_vec = np.array(
[data[0],
 data[1],
 data[1],
 data[2],
 data[2],
 data[3],
 data[3],
 data[4],
 data[4],
 data[5],
 data[5],
 data[6],
 data[6],
 data[7],
 data[7],
 data[8],
 data[8],
 data[9],
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0],
dtype=np.float64)

inv_const_mat = inv(const_mat)

coeffs = inv_const_mat.dot(const_vec)

def P(i):
    if 0 <= i <= 8:
        return lambda a: coeffs[4*i]*a*a*a + coeffs[4*i+1]*a*a + coeffs[4*i+2]*a + coeffs[4*i+3]
    else:
        raise ValueError("i must be between 0 and 8 included")

interp_data = np.zeros((30,), dtype=np.float64)
ind = 0
for i in range(30):
    while xsamples[i] > x[ind+1]:
        ind += 1
    interp_data[i] = P(ind)(xsamples[i])

print(interp_data)

def dP(i):
    if 0 <= i <= 8:
        return lambda a: 3*coeffs[4*i]*a*a + 2*coeffs[4*i+1]*a + coeffs[4*i+2]
    else:
        raise ValueError("i must be between 0 and 8 included")

def ddP(i):
    if 0 <= i <= 8:
        return lambda a: 6*coeffs[4*i]*a + 2*coeffs[4*i+1]
    else:
        raise ValueError("i must be between 0 and 8 included")

def e1_inf(a):
    if a <= x[0]:
        return P(0)(x[0]) + (a-x[0])*dP(0)(x[0])
    else:
        raise ValueError("not in good extrapolation range")

def e1_sup(a):
    if a >= x[9]:
        return P(8)(x[9]) + (a-x[9])*dP(8)(x[9])
    else:
        raise ValueError("not in good extrapolation range")

extr_data_lin = np.array([e1_inf(-0.08), e1_inf(-0.04), e1_sup(1.04), e1_sup(1.08)], dtype=np.float64)
print(extr_data_lin)

def e2_inf(a):
    if a <= x[0]:
        return e1_inf(a) + 0.5*(a-x[0])**2*ddP(0)(x[0])
    else:
        raise ValueError("not in good extrapolation range")

def e2_sup(a):
    if a >= x[9]:
        return e1_sup(a) + 0.5*(a-x[9])**2*ddP(8)(x[9])
    else:
        raise ValueError("not in good extrapolation range")

extr_data_qua = np.array([e2_inf(-0.08), e2_inf(-0.04), e2_sup(1.04), e2_sup(1.08)], dtype=np.float64)
print(extr_data_qua)
