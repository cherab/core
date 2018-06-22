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
This script has been used to calculate the reference data for the 2D interpolators tests.
To get the reference data for the 2D linear interpolators, uncomment the linear
part (l.106) and comment the cubic part (l.152). To get the reference data for
the 2D cubic interpolator, do the contrary.
Reference interpolated data is stored in 'interp_data'.
Reference extrapolated data with nearest method is stored in 'extrap_data_nea'
Reference extrapolated data with linear method is stored in 'extrap_data_lin'
Reference extrapolated data with quadratic method is stored in 'extrap_data_qua'
"""

import numpy as np
from scipy.interpolate import interp2d
from scipy.linalg import inv
from numpy.linalg import tensorsolve

# import sys
# sys.path.append('/home/cbert/code/Raysect')
# from cherab.math.mapping.interpolators import interpolators2d

np.set_printoptions(12, 30000, linewidth=230, formatter={'float': lambda x: format(x, ' 0.12e')})  # 10 values in a line

x = np.linspace(0, 1, 10)
y = np.linspace(0, 2, 10)
x2 = x*x
x3 = x2*x
y2 = y*y
y3 = y2*y
xsamples = np.linspace(0, 1, 30)
ysamples = np.linspace(0, 2, 30)
xsamples_ex = np.array([0 - 0.08, 0 - 0.05, 0 - 0.02] + list(xsamples) + [1 + 0.02, 1 + 0.05, 1 + 0.08], dtype=np.float64)
ysamples_ex = np.array([0 - 0.16, 0 - 0.10, 0 - 0.04] + list(ysamples) + [2 + 0.04, 2 + 0.10, 2 + 0.16], dtype=np.float64)
extrapol_xdomains = [(0, 3), (3, 30 + 3), (30 + 3, 30 + 6)]
extrapol_ydomains = [(0, 3), (3, 30 + 3), (30 + 3, 30 + 6)]

deltax = x[2] - x[0]
deltay = y[2] - y[0]

def point_to_indices(a, b):
    ia = 0
    ib = 0
    if a < x[0] or a > x[-1] or b < y[0] or b > y[-1]:
        raise ValueError("point outside interpolation area")
    while not (x[ia] <= a <= x[ia+1]):
        ia += 1
    while not (y[ib] <= b <= y[ib+1]):
        ib += 1

    return ia, ib

interp_data = np.zeros((30, 30), dtype=np.float64)

extrap_data_nea = [[
                    np.zeros((3, 3), dtype=np.float64),
                    np.zeros((3, 30), dtype=np.float64),
                    np.zeros((3, 3), dtype=np.float64)
                   ],[
                    np.zeros((30, 3), dtype=np.float64),
                    None,
                    np.zeros((30, 3), dtype=np.float64)
                   ],[
                    np.zeros((3, 3), dtype=np.float64),
                    np.zeros((3, 30), dtype=np.float64),
                    np.zeros((3, 3), dtype=np.float64)
                   ]]

extrap_data_lin = [[
                    np.zeros((3, 3), dtype=np.float64),
                    np.zeros((3, 30), dtype=np.float64),
                    np.zeros((3, 3), dtype=np.float64)
                   ],[
                    np.zeros((30, 3), dtype=np.float64),
                    None,
                    np.zeros((30, 3), dtype=np.float64)
                   ],[
                    np.zeros((3, 3), dtype=np.float64),
                    np.zeros((3, 30), dtype=np.float64),
                    np.zeros((3, 3), dtype=np.float64)
                   ]]

extrap_data_qua = [[
                    np.zeros((3, 3), dtype=np.float64),
                    np.zeros((3, 30), dtype=np.float64),
                    np.zeros((3, 3), dtype=np.float64)
                   ],[
                    np.zeros((30, 3), dtype=np.float64),
                    None,
                    np.zeros((30, 3), dtype=np.float64)
                   ],[
                    np.zeros((3, 3), dtype=np.float64),
                    np.zeros((3, 30), dtype=np.float64),
                    np.zeros((3, 3), dtype=np.float64)
                   ]]

def f(a, b):
    return (a-b)*b*np.cos(10*a)

data = np.zeros((10, 10), dtype=np.float64)
for i in range(10):
    for j in range(10):
        data[i, j] = f(x[i], y[j])


# ##### linear ##### #

# interp_func = interp2d(x, y, data.T, kind='linear')
#
# for i in range(30):
#     for j in range(30):
#         interp_data[i, j] = interp_func(xsamples[i], ysamples[j])
#
# def e0(a, b):
#     a_n = max(min(a, x.max()), x.min())
#     b_n = max(min(b, y.max()), y.min())
#     return interp_func(a_n, b_n)
#
# def e1(a, b):
#     a_n = max(min(a, x.max()), x.min())
#     b_n = max(min(b, y.max()), y.min())
#     k = max(abs(a - a_n), abs(b - b_n)) * 1000
#     a_ref = a_n - (a - a_n) / k
#     b_ref = b_n - (b - b_n) / k
#
#     da = (interp_func(a_n, b_n) - interp_func(a_ref, b_n)) / (a_n - a_ref) if a_n != a_ref else 0.
#     db = (interp_func(a_n, b_n) - interp_func(a_n, b_ref)) / (b_n - b_ref) if b_n != b_ref else 0.
#     dadb = (interp_func(a_n, b_n) - interp_func(a_n, b_ref) - interp_func(a_ref, b_n) + interp_func(a_ref, b_ref)) / (a_n - a_ref) / (b_n - b_ref) if a_n != a_ref and b_n != b_ref else 0.
#
#     return interp_func(a_n, b_n) + \
#            (a - a_n) * da + \
#            (b - b_n) * db + \
#            (a - a_n) * (b - b_n) * dadb
#
# whole_calc_data_nea = np.zeros((36, 36), dtype=np.float64)
# whole_calc_data_nea[3:33, 3:33] = interp_data
# whole_calc_data_lin = np.zeros((36, 36), dtype=np.float64)
# whole_calc_data_lin[3:33, 3:33] = interp_data
#
# for nx in range(3):
#     for ny in range(3):
#         if nx != 1 or ny != 1:
#             mini, maxi = extrapol_xdomains[nx]
#             minj, maxj = extrapol_ydomains[ny]
#             for iex in range(mini, maxi):
#                 for jex in range(minj, maxj):
#                     extrap_data_nea[nx][ny][iex - mini, jex - minj] = e0(xsamples_ex[iex], ysamples_ex[jex])
#                     extrap_data_lin[nx][ny][iex - mini, jex - minj] = e1(xsamples_ex[iex], ysamples_ex[jex])
#             whole_calc_data_nea[mini:maxi, minj:maxj] = extrap_data_nea[nx][ny]
#             whole_calc_data_lin[mini:maxi, minj:maxj] = extrap_data_lin[nx][ny]

# ##### cubic ##### #

const_mat = np.zeros((9*9*16, 9*9*16), dtype=np.float64)  # ~100Mb

def elt_mat(i, j):
    return np.array([
[  x3[i  ]*y3[j  ],   x3[i  ]*y2[j  ],   x3[i  ]* y[j  ],   x3[i  ],   x2[i  ]*y3[j  ],   x2[i  ]*y2[j  ],   x2[i  ]* y[j  ],   x2[i  ],    x[i  ]*y3[j  ],    x[i  ]*y2[j  ], x[i  ]*y[j  ], x[i  ], y3[j  ], y2[j  ], y[j  ], 1.],
[  x3[i+1]*y3[j  ],   x3[i+1]*y2[j  ],   x3[i+1]* y[j  ],   x3[i+1],   x2[i+1]*y3[j  ],   x2[i+1]*y2[j  ],   x2[i+1]* y[j  ],   x2[i+1],    x[i+1]*y3[j  ],    x[i+1]*y2[j  ], x[i+1]*y[j  ], x[i+1], y3[j  ], y2[j  ], y[j  ], 1.],
[  x3[i  ]*y3[j+1],   x3[i  ]*y2[j+1],   x3[i  ]* y[j+1],   x3[i  ],   x2[i  ]*y3[j+1],   x2[i  ]*y2[j+1],   x2[i  ]* y[j+1],   x2[i  ],    x[i  ]*y3[j+1],    x[i  ]*y2[j+1], x[i  ]*y[j+1], x[i  ], y3[j+1], y2[j+1], y[j+1], 1.],
[  x3[i+1]*y3[j+1],   x3[i+1]*y2[j+1],   x3[i+1]* y[j+1],   x3[i+1],   x2[i+1]*y3[j+1],   x2[i+1]*y2[j+1],   x2[i+1]* y[j+1],   x2[i+1],    x[i+1]*y3[j+1],    x[i+1]*y2[j+1], x[i+1]*y[j+1], x[i+1], y3[j+1], y2[j+1], y[j+1], 1.],

[3*x2[i  ]*y3[j  ], 3*x2[i  ]*y2[j  ], 3*x2[i  ]* y[j  ], 3*x2[i  ], 2* x[i  ]*y3[j  ], 2* x[i  ]*y2[j  ], 2* x[i  ]* y[j  ], 2* x[i  ],   y3[j  ]        ,   y2[j  ]        , y[j  ]       , 1., 0., 0., 0., 0.],
[3*x2[i+1]*y3[j  ], 3*x2[i+1]*y2[j  ], 3*x2[i+1]* y[j  ], 3*x2[i+1], 2* x[i+1]*y3[j  ], 2* x[i+1]*y2[j  ], 2* x[i+1]* y[j  ], 2* x[i+1],   y3[j  ]        ,   y2[j  ]        , y[j  ]       , 1., 0., 0., 0., 0.],
[3*x2[i  ]*y3[j+1], 3*x2[i  ]*y2[j+1], 3*x2[i  ]* y[j+1], 3*x2[i  ], 2* x[i  ]*y3[j+1], 2* x[i  ]*y2[j+1], 2* x[i  ]* y[j+1], 2* x[i  ],   y3[j+1]        ,   y2[j+1]        , y[j+1]       , 1., 0., 0., 0., 0.],
[3*x2[i+1]*y3[j+1], 3*x2[i+1]*y2[j+1], 3*x2[i+1]* y[j+1], 3*x2[i+1], 2* x[i+1]*y3[j+1], 2* x[i+1]*y2[j+1], 2* x[i+1]* y[j+1], 2* x[i+1],   y3[j+1]        ,   y2[j+1]        , y[j+1]       , 1., 0., 0., 0., 0.],

[3*x3[i  ]*y2[j  ], 2*x3[i  ]* y[j  ],   x3[i  ]        , 0.       , 3*x2[i  ]*y2[j  ], 2*x2[i  ]* y[j  ],   x2[i  ]        , 0.       , 3* x[i  ]*y2[j  ], 2* x[i  ]* y[j  ], x[i  ]       , 0., 3*y2[j  ], 2*y[j  ], 1., 0.],
[3*x3[i+1]*y2[j  ], 2*x3[i+1]* y[j  ],   x3[i+1]        , 0.       , 3*x2[i+1]*y2[j  ], 2*x2[i+1]* y[j  ],   x2[i+1]        , 0.       , 3* x[i+1]*y2[j  ], 2* x[i+1]* y[j  ], x[i+1]       , 0., 3*y2[j  ], 2*y[j  ], 1., 0.],
[3*x3[i  ]*y2[j+1], 2*x3[i  ]* y[j+1],   x3[i  ]        , 0.       , 3*x2[i  ]*y2[j+1], 2*x2[i  ]* y[j+1],   x2[i  ]        , 0.       , 3* x[i  ]*y2[j+1], 2* x[i  ]* y[j+1], x[i  ]       , 0., 3*y2[j+1], 2*y[j+1], 1., 0.],
[3*x3[i+1]*y2[j+1], 2*x3[i+1]* y[j+1],   x3[i+1]        , 0.       , 3*x2[i+1]*y2[j+1], 2*x2[i+1]* y[j+1],   x2[i+1]        , 0.       , 3* x[i+1]*y2[j+1], 2* x[i+1]* y[j+1], x[i+1]       , 0., 3*y2[j+1], 2*y[j+1], 1., 0.],

[9*x2[i  ]*y2[j  ], 6*x2[i  ]* y[j  ], 3*x2[i  ]        , 0.       , 6* x[i  ]*y2[j  ], 4* x[i  ]* y[j  ], 2* x[i  ]        , 0.       , 3*y2[j  ]        , 2* y[j  ]        , 1.           , 0., 0., 0., 0., 0.],
[9*x2[i+1]*y2[j  ], 6*x2[i+1]* y[j  ], 3*x2[i+1]        , 0.       , 6* x[i+1]*y2[j  ], 4* x[i+1]* y[j  ], 2* x[i+1]        , 0.       , 3*y2[j  ]        , 2* y[j  ]        , 1.           , 0., 0., 0., 0., 0.],
[9*x2[i  ]*y2[j+1], 6*x2[i  ]* y[j+1], 3*x2[i  ]        , 0.       , 6* x[i  ]*y2[j+1], 4* x[i  ]* y[j+1], 2* x[i  ]        , 0.       , 3*y2[j+1]        , 2* y[j+1]        , 1.           , 0., 0., 0., 0., 0.],
[9*x2[i+1]*y2[j+1], 6*x2[i+1]* y[j+1], 3*x2[i+1]        , 0.       , 6* x[i+1]*y2[j+1], 4* x[i+1]* y[j+1], 2* x[i+1]        , 0.       , 3*y2[j+1]        , 2* y[j+1]        , 1.           , 0., 0., 0., 0., 0.]],
                    dtype=np.float64)

for i in range(9):
    for j in range(9):
        l = i + 9 * j
        const_mat[16*l:16*(l+1), 16*l:16*(l+1)] = elt_mat(i, j)

const_vec = np.zeros((9*9*16,), dtype=np.float64)

def d(i, j):
    ieff = max(min(i, 9), 0)
    jeff = max(min(j, 9), 0)
    return data[ieff, jeff]

def xf(i):
    ieff = max(min(i, 9), 0)
    return x[ieff]

def yf(j):
    jeff = max(min(j, 9), 0)
    return y[jeff]

def elt_vec(i, j):
    return np.array([
d(i+0, j+0),
d(i+1, j+0),
d(i+0, j+1),
d(i+1, j+1),

(d(i+1, j+0) - d(i-1, j+0))/(xf(i+1) - xf(i-1)),
(d(i+2, j+0) - d(i+0, j+0))/(xf(i+2) - xf(i  )),
(d(i+1, j+1) - d(i-1, j+1))/(xf(i+1) - xf(i-1)),
(d(i+2, j+1) - d(i+0, j+1))/(xf(i+2) - xf(i  )),

(d(i+0, j+1) - d(i+0, j-1))/(yf(j+1) - yf(j-1)),
(d(i+1, j+1) - d(i+1, j-1))/(yf(j+1) - yf(j-1)),
(d(i+0, j+2) - d(i+0, j+0))/(yf(j+2) - yf(j  )),
(d(i+1, j+2) - d(i+1, j+0))/(yf(j+2) - yf(j  )),

(d(i+1, j+1) - d(i+1, j-1) - d(i-1, j+1) + d(i-1, j-1))/(xf(i+1) - xf(i-1))/(yf(j+1) - yf(j-1)),
(d(i+2, j+1) - d(i+2, j-1) - d(i+0, j+1) + d(i+0, j-1))/(xf(i+2) - xf(i  ))/(yf(j+1) - yf(j-1)),
(d(i+1, j+2) - d(i+1, j+0) - d(i-1, j+2) + d(i-1, j+0))/(xf(i+1) - xf(i-1))/(yf(j+2) - yf(j  )),
(d(i+2, j+2) - d(i+2, j+0) - d(i+0, j+2) + d(i+0, j+0))/(xf(i+2) - xf(i  ))/(yf(j+2) - yf(j  )),
],
                    dtype=np.float64)

for i in range(9):
    for j in range(9):
        l = i + 9 * j
        const_vec[16*l:16*(l+1)] = elt_vec(i, j)

# inv_const_mat = inv(const_mat)
# coeffs = inv_const_mat.dot(const_vec)

coeffs = tensorsolve(const_mat, const_vec)


def P(i, j):
    if 0 <= i <= 8 and 0 <= j <= 8:
        l = 16*(i+9*j)
        return lambda a, b: coeffs[l+0 ]*a*a*a*b*b*b + \
                            coeffs[l+1 ]*a*a*a*b*b + \
                            coeffs[l+2 ]*a*a*a*b + \
                            coeffs[l+3 ]*a*a*a + \
                            coeffs[l+4 ]*a*a*b*b*b + \
                            coeffs[l+5 ]*a*a*b*b + \
                            coeffs[l+6 ]*a*a*b + \
                            coeffs[l+7 ]*a*a + \
                            coeffs[l+8 ]*a*b*b*b + \
                            coeffs[l+9 ]*a*b*b + \
                            coeffs[l+10]*a*b + \
                            coeffs[l+11]*a + \
                            coeffs[l+12]*b*b*b + \
                            coeffs[l+13]*b*b + \
                            coeffs[l+14]*b + \
                            coeffs[l+15]
    else:
        raise ValueError("i and j must be between 0 and 8 included")

def dP_dx(i, j):
    if 0 <= i <= 8 and 0 <= j <= 8:
        l = 16*(i+9*j)
        return lambda a, b: 3*coeffs[l+0 ]*a*a*b*b*b + \
                            3*coeffs[l+1 ]*a*a*b*b + \
                            3*coeffs[l+2 ]*a*a*b + \
                            3*coeffs[l+3 ]*a*a + \
                            2*coeffs[l+4 ]*a*b*b*b + \
                            2*coeffs[l+5 ]*a*b*b + \
                            2*coeffs[l+6 ]*a*b + \
                            2*coeffs[l+7 ]*a + \
                            coeffs[l+8 ]*b*b*b + \
                            coeffs[l+9 ]*b*b + \
                            coeffs[l+10]*b + \
                            coeffs[l+11]
    else:
        raise ValueError("i and j must be between 0 and 8 included")

def d2P_dx2(i, j):
    if 0 <= i <= 8 and 0 <= j <= 8:
        l = 16*(i+9*j)
        return lambda a, b: 6*coeffs[l+0 ]*a*b*b*b + \
                            6*coeffs[l+1 ]*a*b*b + \
                            6*coeffs[l+2 ]*a*b + \
                            6*coeffs[l+3 ]*a + \
                            2*coeffs[l+4 ]*b*b*b + \
                            2*coeffs[l+5 ]*b*b + \
                            2*coeffs[l+6 ]*b + \
                            2*coeffs[l+7 ]
    else:
        raise ValueError("i and j must be between 0 and 8 included")

def dP_dy(i, j):
    if 0 <= i <= 8 and 0 <= j <= 8:
        l = 16*(i+9*j)
        return lambda a, b: 3*coeffs[l+0 ]*a*a*a*b*b + \
                            2*coeffs[l+1 ]*a*a*a*b + \
                            coeffs[l+2 ]*a*a*a + \
                            3*coeffs[l+4 ]*a*a*b*b + \
                            2*coeffs[l+5 ]*a*a*b + \
                            coeffs[l+6 ]*a*a + \
                            3*coeffs[l+8 ]*a*b*b + \
                            2*coeffs[l+9 ]*a*b + \
                            coeffs[l+10]*a + \
                            3*coeffs[l+12]*b*b + \
                            2*coeffs[l+13]*b + \
                            coeffs[l+14]
    else:
        raise ValueError("i and j must be between 0 and 8 included")

def d2P_dy2(i, j):
    if 0 <= i <= 8 and 0 <= j <= 8:
        l = 16*(i+9*j)
        return lambda a, b: 6*coeffs[l+0 ]*a*a*a*b + \
                            2*coeffs[l+1 ]*a*a*a + \
                            6*coeffs[l+4 ]*a*a*b + \
                            2*coeffs[l+5 ]*a*a + \
                            6*coeffs[l+8 ]*a*b + \
                            2*coeffs[l+9 ]*a + \
                            6*coeffs[l+12]*b + \
                            2*coeffs[l+13]
    else:
        raise ValueError("i and j must be between 0 and 8 included")

def d2P_dxdy(i, j):
    if 0 <= i <= 8 and 0 <= j <= 8:
        l = 16*(i+9*j)
        return lambda a, b: 9*coeffs[l+0 ]*a*a*b*b + \
                            6*coeffs[l+1 ]*a*a*b + \
                            3*coeffs[l+2 ]*a*a + \
                            6*coeffs[l+4 ]*a*b*b + \
                            4*coeffs[l+5 ]*a*b + \
                            2*coeffs[l+6 ]*a + \
                            3*coeffs[l+8 ]*b*b + \
                            2*coeffs[l+9 ]*b + \
                            coeffs[l+10]
    else:
        raise ValueError("i and j must be between 0 and 8 included")

def interp_func(a, b):
    return P(*point_to_indices(a, b))(a, b)

for i in range(30):
    for j in range(30):
        interp_data[i, j] = interp_func(xsamples[i], ysamples[j])

# compare_func = interpolators2d.Interpolate2DCubic(x, y, data)
# compare_data = np.zeros((30, 30), dtype=np.float64)
# for i in range(30):
#     for j in range(30):
#         compare_data[i, j] = compare_func(xsamples[i], ysamples[j])


def e0(a, b):
    a_n = max(min(a, x.max()), x.min())
    b_n = max(min(b, y.max()), y.min())
    return interp_func(a_n, b_n)

def e1(a, b):
    a_n = max(min(a, x.max()), x.min())
    b_n = max(min(b, y.max()), y.min())
    i, j = point_to_indices(a_n, b_n)

    return P(i, j)(a_n, b_n) + \
           (a - a_n) * dP_dx(i, j)(a_n, b_n) + \
           (b - b_n) * dP_dy(i, j)(a_n, b_n)

def e2(a, b):
    a_n = max(min(a, x.max()), x.min())
    b_n = max(min(b, y.max()), y.min())
    i, j = point_to_indices(a_n, b_n)

    return e1(a, b) + \
           (a - a_n) * (b - b_n) * d2P_dxdy(i, j)(a_n, b_n) + \
           0.5 * (a - a_n)**2 * d2P_dx2(i, j)(a_n, b_n) + \
           0.5 * (b - b_n)**2 * d2P_dy2(i, j)(a_n, b_n)



whole_calc_data_nea = np.zeros((36, 36), dtype=np.float64)
whole_calc_data_nea[3:33, 3:33] = interp_data
whole_calc_data_lin = np.zeros((36, 36), dtype=np.float64)
whole_calc_data_lin[3:33, 3:33] = interp_data
whole_calc_data_qua = np.zeros((36, 36), dtype=np.float64)
whole_calc_data_qua[3:33, 3:33] = interp_data

for nx in range(3):
    for ny in range(3):
        if nx != 1 or ny != 1:
            mini, maxi = extrapol_xdomains[nx]
            minj, maxj = extrapol_ydomains[ny]
            for iex in range(mini, maxi):
                for jex in range(minj, maxj):
                    extrap_data_nea[nx][ny][iex - mini, jex - minj] = e0(xsamples_ex[iex], ysamples_ex[jex])
                    extrap_data_lin[nx][ny][iex - mini, jex - minj] = e1(xsamples_ex[iex], ysamples_ex[jex])
                    extrap_data_qua[nx][ny][iex - mini, jex - minj] = e2(xsamples_ex[iex], ysamples_ex[jex])
            whole_calc_data_nea[mini:maxi, minj:maxj] = extrap_data_nea[nx][ny]
            whole_calc_data_lin[mini:maxi, minj:maxj] = extrap_data_lin[nx][ny]
            whole_calc_data_qua[mini:maxi, minj:maxj] = extrap_data_qua[nx][ny]
