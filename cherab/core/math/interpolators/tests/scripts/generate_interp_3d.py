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
This script has been used to calculate the reference data for the 3D interpolators tests.
To get the reference data for the 3D linear interpolators, uncomment the linear
part (l.210) and comment the cubic part (l.287). To get the reference data for
the 3D cubic interpolator, do the contrary.
Reference interpolated data is stored in 'interp_data'.
Reference extrapolated data with nearest method is stored in 'extrap_data_nea'
Reference extrapolated data with linear method is stored in 'extrap_data_lin'
Reference extrapolated data with quadratic method is stored in 'extrap_data_qua'
The function 'write_in_file' has been made to write extrapolation data in a nice way.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import inv
from numpy.linalg import tensorsolve

import sys
# sys.path.append('/home/cbert/code/Raysect')
# from cherab.math.mapping.interpolators import interpolators3d

np.set_printoptions(12, 30000, linewidth=340, formatter={'float': lambda x: format(x, ' 0.12e')})  # 15 values in a line

NSX, NSY, NSZ = 15, 10, 15

x = np.linspace(0, 1, 10)
y = np.linspace(0, 2, 5)
z = np.linspace(-1, 1, 5)
x2 = x*x
x3 = x2*x
y2 = y*y
y3 = y2*y
z2 = z*z
z3 = z2*z
xsamples = np.linspace(0, 1, NSX)
ysamples = np.linspace(0, 2, NSY)
zsamples = np.linspace(-1, 1, NSZ)
xsamples_ex = np.array([0 - 0.08, 0 - 0.05, 0 - 0.02] + list(xsamples) + [1 + 0.02, 1 + 0.05, 1 + 0.08], dtype=np.float64)
ysamples_ex = np.array([0 - 0.16, 0 - 0.10, 0 - 0.04] + list(ysamples) + [2 + 0.04, 2 + 0.10, 2 + 0.16], dtype=np.float64)
zsamples_ex = np.array([-1 - 0.16, -1 - 0.10, -1 - 0.04] + list(zsamples) + [1 + 0.04, 1 + 0.10, 1 + 0.16], dtype=np.float64)
extrapol_xdomains = [(0, 3), (3, NSX + 3), (NSX + 3, NSX + 6)]
extrapol_ydomains = [(0, 3), (3, NSY + 3), (NSY + 3, NSY + 6)]
extrapol_zdomains = [(0, 3), (3, NSZ + 3), (NSZ + 3, NSZ + 6)]

deltax = x[2] - x[0]
deltay = y[2] - y[0]
deltaz = z[2] - z[0]


def point_to_indices(a, b, c):
    ia = 0
    ib = 0
    ic = 0
    if a < x[0] or a > x[-1] or b < y[0] or b > y[-1] or c < z[0] or c > z[-1]:
        raise ValueError("point outside interpolation area")
    while not (x[ia] <= a <= x[ia+1]):
        ia += 1
    while not (y[ib] <= b <= y[ib+1]):
        ib += 1
    while not (z[ic] <= c <= z[ic+1]):
        ic += 1

    return ia, ib, ic

interp_data = np.zeros((NSX, NSY, NSZ), dtype=np.float64)

extrap_data_nea = [[[np.zeros((3, 3, 3), dtype=np.float64),
                     np.zeros((3, 3, NSZ), dtype=np.float64),
                     np.zeros((3, 3, 3), dtype=np.float64)],
                    [np.zeros((3, NSY, 3), dtype=np.float64),
                     np.zeros((3, NSY, NSZ), dtype=np.float64),
                     np.zeros((3, NSY, 3), dtype=np.float64)],
                    [np.zeros((3, 3, 3), dtype=np.float64),
                     np.zeros((3, 3, NSZ), dtype=np.float64),
                     np.zeros((3, 3, 3), dtype=np.float64)]],
                   [[np.zeros((NSX, 3, 3), dtype=np.float64),
                     np.zeros((NSX, 3, NSZ), dtype=np.float64),
                     np.zeros((NSX, 3, 3), dtype=np.float64)],
                    [np.zeros((NSX, NSY, 3), dtype=np.float64),
                     None,
                     np.zeros((NSX, NSY, 3), dtype=np.float64)],
                    [np.zeros((NSX, 3, 3), dtype=np.float64),
                     np.zeros((NSX, 3, NSZ), dtype=np.float64),
                     np.zeros((NSX, 3, 3), dtype=np.float64)]],
                   [[np.zeros((3, 3, 3), dtype=np.float64),
                     np.zeros((3, 3, NSZ), dtype=np.float64),
                     np.zeros((3, 3, 3), dtype=np.float64)],
                    [np.zeros((3, NSY, 3), dtype=np.float64),
                     np.zeros((3, NSY, NSZ), dtype=np.float64),
                     np.zeros((3, NSY, 3), dtype=np.float64)],
                    [np.zeros((3, 3, 3), dtype=np.float64),
                     np.zeros((3, 3, NSZ), dtype=np.float64),
                     np.zeros((3, 3, 3), dtype=np.float64)]]]

extrap_data_lin = [[[np.zeros((3, 3, 3), dtype=np.float64),
                     np.zeros((3, 3, NSZ), dtype=np.float64),
                     np.zeros((3, 3, 3), dtype=np.float64)],
                    [np.zeros((3, NSY, 3), dtype=np.float64),
                     np.zeros((3, NSY, NSZ), dtype=np.float64),
                     np.zeros((3, NSY, 3), dtype=np.float64)],
                    [np.zeros((3, 3, 3), dtype=np.float64),
                     np.zeros((3, 3, NSZ), dtype=np.float64),
                     np.zeros((3, 3, 3), dtype=np.float64)]],
                   [[np.zeros((NSX, 3, 3), dtype=np.float64),
                     np.zeros((NSX, 3, NSZ), dtype=np.float64),
                     np.zeros((NSX, 3, 3), dtype=np.float64)],
                    [np.zeros((NSX, NSY, 3), dtype=np.float64),
                     None,
                     np.zeros((NSX, NSY, 3), dtype=np.float64)],
                    [np.zeros((NSX, 3, 3), dtype=np.float64),
                     np.zeros((NSX, 3, NSZ), dtype=np.float64),
                     np.zeros((NSX, 3, 3), dtype=np.float64)]],
                   [[np.zeros((3, 3, 3), dtype=np.float64),
                     np.zeros((3, 3, NSZ), dtype=np.float64),
                     np.zeros((3, 3, 3), dtype=np.float64)],
                    [np.zeros((3, NSY, 3), dtype=np.float64),
                     np.zeros((3, NSY, NSZ), dtype=np.float64),
                     np.zeros((3, NSY, 3), dtype=np.float64)],
                    [np.zeros((3, 3, 3), dtype=np.float64),
                     np.zeros((3, 3, NSZ), dtype=np.float64),
                     np.zeros((3, 3, 3), dtype=np.float64)]]]

extrap_data_qua = [[[np.zeros((3, 3, 3), dtype=np.float64),
                     np.zeros((3, 3, NSZ), dtype=np.float64),
                     np.zeros((3, 3, 3), dtype=np.float64)],
                    [np.zeros((3, NSY, 3), dtype=np.float64),
                     np.zeros((3, NSY, NSZ), dtype=np.float64),
                     np.zeros((3, NSY, 3), dtype=np.float64)],
                    [np.zeros((3, 3, 3), dtype=np.float64),
                     np.zeros((3, 3, NSZ), dtype=np.float64),
                     np.zeros((3, 3, 3), dtype=np.float64)]],
                   [[np.zeros((NSX, 3, 3), dtype=np.float64),
                     np.zeros((NSX, 3, NSZ), dtype=np.float64),
                     np.zeros((NSX, 3, 3), dtype=np.float64)],
                    [np.zeros((NSX, NSY, 3), dtype=np.float64),
                     None,
                     np.zeros((NSX, NSY, 3), dtype=np.float64)],
                    [np.zeros((NSX, 3, 3), dtype=np.float64),
                     np.zeros((NSX, 3, NSZ), dtype=np.float64),
                     np.zeros((NSX, 3, 3), dtype=np.float64)]],
                   [[np.zeros((3, 3, 3), dtype=np.float64),
                     np.zeros((3, 3, NSZ), dtype=np.float64),
                     np.zeros((3, 3, 3), dtype=np.float64)],
                    [np.zeros((3, NSY, 3), dtype=np.float64),
                     np.zeros((3, NSY, NSZ), dtype=np.float64),
                     np.zeros((3, NSY, 3), dtype=np.float64)],
                    [np.zeros((3, 3, 3), dtype=np.float64),
                     np.zeros((3, 3, NSZ), dtype=np.float64),
                     np.zeros((3, 3, 3), dtype=np.float64)]]]

def f(a, b, c):
    return (a-b)*b*np.cos(10*a)*np.exp(c) + c

data = np.zeros((10, 5, 5), dtype=np.float64)
for i in range(10):
    for j in range(5):
        for k in range(5):
            data[i, j, k] = f(x[i], y[j], z[k])

def write_in_file(data, filename, prefix='    '):

    to_print = ''
    positions = ['inf', 'mid', 'sup']
    tab = '    '
    p = 0

    to_print += prefix + '[\n'
    for x_arrays in data:
        px = p // 9
        to_print += prefix + tab + '[  # x {}\n'.format(positions[px])
        for y_arrays in x_arrays:
            py = (p // 3) % 3
            to_print += prefix + 2 * tab + '[  # y {}\n'.format(positions[py])
            for z_arrays in y_arrays:
                pz = p % 3
                if z_arrays is not None:
                    to_print += prefix + 3 * tab + 'np.array([  # z {}\n'.format(positions[pz])
                    for col in z_arrays:
                        first = True
                        for raw in col:
                            to_print += prefix + tab + '[' if not first else prefix + tab[:-1] + '[['
                            first = False
                            for elt in raw:
                                to_print += '{: 0.12e}, '.format(elt)
                            to_print = to_print[:-2]  # drop last coma
                            to_print += '],\n'
                        to_print = to_print[:-2] + '],\n'  # add a bracket
                    to_print = to_print[:-2] + '\n'  # drop last coma
                    to_print += prefix + 3 * tab + '], dtype=np.float64)'
                    to_print += ',\n' if pz < 2 else '\n'
                else:
                    to_print += '\n' + prefix + 3* tab + 'None,  # z mid (interpolation area)\n\n'
                p += 1
            to_print += prefix + 2 * tab + ']'
            to_print += ',\n' if py < 2 else '\n'
        to_print += prefix + tab + ']'
        to_print += ',\n' if px < 2 else '\n'
    to_print += prefix + ']\n'

    with open(filename, 'w') as file:
        orig_stdout = sys.stdout
        sys.stdout = file

        print(to_print)

        sys.stdout = orig_stdout

# ##### linear ##### #

def lin(d1, d2, delta, pos):
    return d1 + pos * (d2 - d1) / delta

def P(i, j, k):

    return lambda a, b, c: lin(lin(lin(data[i, j, k], data[i+1, j, k], deltax/2, a - x[i]),
                                   lin(data[i, j+1, k], data[i+1, j+1, k], deltax/2, a - x[i]),
                                   deltay/2, b - y[j]),
                               lin(lin(data[i, j, k+1], data[i+1, j, k+1], deltax/2, a - x[i]),
                                   lin(data[i, j+1, k+1], data[i+1, j+1, k+1], deltax/2, a - x[i]),
                                   deltay/2, b - y[j]),
                               deltaz/2, c - z[k])

def interp_func(a, b, c):
    return P(*point_to_indices(a, b, c))(a, b, c)

for i in range(NSX):
    for j in range(NSY):
        for k in range(NSZ):
            interp_data[i, j, k] = interp_func(xsamples[i], ysamples[j], zsamples[k])

def e0(a, b, c):
    a_n = max(min(a, x.max()), x.min())
    b_n = max(min(b, y.max()), y.min())
    c_n = max(min(c, z.max()), z.min())
    return interp_func(a_n, b_n, c_n)

def e1(a, b, c):
    a_n = max(min(a, x.max()), x.min())
    b_n = max(min(b, y.max()), y.min())
    c_n = max(min(c, z.max()), z.min())
    k = max(abs(a - a_n), abs(b - b_n), abs(c - c_n))
    a_ref = a_n - (a - a_n) / k
    b_ref = b_n - (b - b_n) / k
    c_ref = c_n - (c - c_n) / k

    func = P(*point_to_indices(a_n, b_n, c_n))

    da = (func(a_n, b_n, c_n) - func(a_ref, b_n, c_n)) / (a_n - a_ref) if a_n != a_ref else 0.
    db = (func(a_n, b_n, c_n) - func(a_n, b_ref, c_n)) / (b_n - b_ref) if b_n != b_ref else 0.
    dc = (func(a_n, b_n, c_n) - func(a_n, b_n, c_ref)) / (c_n - c_ref) if c_n != c_ref else 0.
    dadb = (func(a_n, b_n, c_n) - func(a_n, b_ref, c_n) - func(a_ref, b_n, c_n) + func(a_ref, b_ref, c_n)) / (a_n - a_ref) / (b_n - b_ref) if a_n != a_ref and b_n != b_ref else 0.
    dbdc = (func(a_n, b_n, c_n) - func(a_n, b_ref, c_n) - func(a_n, b_n, c_ref) + func(a_n, b_ref, c_ref)) / (c_n - c_ref) / (b_n - b_ref) if c_n != c_ref and b_n != b_ref else 0.
    dcda = (func(a_n, b_n, c_n) - func(a_n, b_n, c_ref) - func(a_ref, b_n, c_n) + func(a_ref, b_n, c_ref)) / (a_n - a_ref) / (c_n - c_ref) if a_n != a_ref and c_n != c_ref else 0.
    dadbdc = (func(a_n, b_n, c_n) - func(a_n, b_ref, c_n) - func(a_ref, b_n, c_n) + func(a_ref, b_ref, c_n) - func(a_n, b_n, c_ref) + func(a_n, b_ref, c_ref) + func(a_ref, b_n, c_ref) - func(a_ref, b_ref, c_ref)) / (a_n - a_ref) / (b_n - b_ref) / (c_n - c_ref) if a_n != a_ref and b_n != b_ref and c_n != c_ref else 0.

    return func(a_n, b_n, c_n) + \
           (a - a_n) * da + \
           (b - b_n) * db + \
           (c - c_n) * dc + \
           (a - a_n) * (b - b_n) * dadb + \
           (b - b_n) * (c - c_n) * dbdc + \
           (c - c_n) * (a - a_n) * dcda + \
           (a - a_n) * (b - b_n) * (c - c_n) * dadbdc

whole_calc_data_nea = np.zeros((NSX+6, NSY+6, NSZ+6), dtype=np.float64)
whole_calc_data_nea[3:NSX+3, 3:NSY+3, 3:NSZ+3] = interp_data
whole_calc_data_lin = np.zeros((NSX+6, NSY+6, NSZ+6), dtype=np.float64)
whole_calc_data_lin[3:NSX+3, 3:NSY+3, 3:NSZ+3] = interp_data

for nx in range(3):
    for ny in range(3):
        for nz in range(3):
            if nx != 1 or ny != 1 or nz != 1:
                mini, maxi = extrapol_xdomains[nx]
                minj, maxj = extrapol_ydomains[ny]
                mink, maxk = extrapol_zdomains[nz]
                for iex in range(mini, maxi):
                    for jex in range(minj, maxj):
                        for kex in range(mink, maxk):
                            extrap_data_nea[nx][ny][nz][iex - mini, jex - minj, kex - mink] = e0(xsamples_ex[iex], ysamples_ex[jex], zsamples_ex[kex])
                            extrap_data_lin[nx][ny][nz][iex - mini, jex - minj, kex - mink] = e1(xsamples_ex[iex], ysamples_ex[jex], zsamples_ex[kex])
                whole_calc_data_nea[mini:maxi, minj:maxj, mink:maxk] = extrap_data_nea[nx][ny][nz]
                whole_calc_data_lin[mini:maxi, minj:maxj, mink:maxk] = extrap_data_lin[nx][ny][nz]

# ##### cubic ##### #

# def elt_mat(i, j, k):
#     output = np.zeros((64, 64), dtype=np.float64)
#
#     for ic in range(4):
#         for jc in range(4):
#             for kc in range(4):
#                 l = 0
#                 for u in [i, i+1]:
#                     for v in [j, j+1]:
#                         for w in [k, k+1]:
#                             for dx in [0, 1]:
#                                 for dy in [0, 1]:
#                                     for dz in [0, 1]:
#                                         output[l, ic + 4*jc + 16*kc] = x[u]**(abs(ic-dx))*(ic**dx) * y[v]**(abs(jc-dy))*(jc**dy) * z[w]**(abs(kc-dz))*(kc**dz)
#                                         l += 1
#     return output
#
# def d(i, j, k):
#     ieff = max(min(i, 9), 0)
#     jeff = max(min(j, 4), 0)
#     keff = max(min(k, 4), 0)
#     return data[ieff, jeff, keff]
#
# def xf(i):
#     ieff = max(min(i, 9), 0)
#     return x[ieff]
#
# def yf(j):
#     jeff = max(min(j, 4), 0)
#     return y[jeff]
#
# def zf(k):
#     keff = max(min(k, 4), 0)
#     return z[keff]
#
# def elt_vec(i, j, k):
#
#     output = np.zeros((64,), dtype=np.float64)
#
#     l = 0
#     for u in [i, i+1]:
#         for v in [j, j+1]:
#             for w in [k, k+1]:
#                 for dx in [0, 1]:
#                     for dy in [0, 1]:
#                         for dz in [0, 1]:
#
#                             value = 0.
#                             for a in {-dx, dx}:
#                                 for b in {-dy, dy}:
#                                     for c in {-dz, dz}:
#                                         value += (a+0.5)/abs(a+0.5)*(b+0.5)/abs(b+0.5)*(c+0.5)/abs(c+0.5)*d(u+a, v+b, w+c)
#                             value /= (xf(u+1) - xf(u-1))**dx * (yf(v+1) - yf(v-1))**dy * (zf(w+1) - zf(w-1))**dz
#                             output[l] = value
#                             l += 1
#
#     return output
#
# coeffs = []
#
# for i in range(9):
#     coeffs_i = []
#     for j in range(4):
#         coeffs_ij = []
#         for k in range(4):
#             coeffs_ijk = tensorsolve(elt_mat(i, j, k), elt_vec(i, j, k))
#             coeffs_ij.append(coeffs_ijk)
#         coeffs_i.append(coeffs_ij)
#     coeffs.append(coeffs_i)
#
# def P(i, j, k):
#     if 0 <= i <= 8 and 0 <= j <= 3 and 0 <= k <= 3:
#
#         def output(a, b, c):
#             result = 0.
#             for ic in range(4):
#                 for jc in range(4):
#                     for kc in range(4):
#                         result += coeffs[i][j][k][ic + 4*jc + 16*kc] * a**ic * b**jc * c**kc
#             return result
#
#         return output
#
#     else:
#         raise ValueError("i, j and k must be in the good range")
#
# def dP_dx(i, j, k):
#     if 0 <= i <= 8 and 0 <= j <= 3 and 0 <= k <= 3:
#
#         def output(a, b, c):
#             result = 0.
#             for ic in range(4):
#                 for jc in range(4):
#                     for kc in range(4):
#                         result += coeffs[i][j][k][ic + 4*jc + 16*kc] * ic * a**(abs(ic-1)) * b**jc * c**kc
#             return result
#
#         return output
#
#     else:
#         raise ValueError("i, j and k must be in the good range")
#
# def d2P_dx2(i, j, k):
#     if 0 <= i <= 8 and 0 <= j <= 3 and 0 <= k <= 3:
#
#         def output(a, b, c):
#             result = 0.
#             for ic in range(4):
#                 for jc in range(4):
#                     for kc in range(4):
#                         result += coeffs[i][j][k][ic + 4*jc + 16*kc] * ic*(ic-1) * a**(abs(ic-2)) * b**jc * c**kc
#             return result
#
#         return output
#
#     else:
#         raise ValueError("i, j and k must be in the good range")
#
# def dP_dy(i, j, k):
#     if 0 <= i <= 8 and 0 <= j <= 3 and 0 <= k <= 3:
#
#         def output(a, b, c):
#             result = 0.
#             for ic in range(4):
#                 for jc in range(4):
#                     for kc in range(4):
#                         result += coeffs[i][j][k][ic + 4*jc + 16*kc] * jc * a**ic * b**(abs(jc-1)) * c**kc
#             return result
#
#         return output
#
#     else:
#         raise ValueError("i, j and k must be in the good range")
#
# def d2P_dy2(i, j, k):
#     if 0 <= i <= 8 and 0 <= j <= 3 and 0 <= k <= 3:
#
#         def output(a, b, c):
#             result = 0.
#             for ic in range(4):
#                 for jc in range(4):
#                     for kc in range(4):
#                         result += coeffs[i][j][k][ic + 4*jc + 16*kc] * jc*(jc-1) * a**ic * b**(abs(jc-2)) * c**kc
#             return result
#
#         return output
#
#     else:
#         raise ValueError("i, j and k must be in the good range")
#
# def dP_dz(i, j, k):
#     if 0 <= i <= 8 and 0 <= j <= 3 and 0 <= k <= 3:
#
#         def output(a, b, c):
#             result = 0.
#             for ic in range(4):
#                 for jc in range(4):
#                     for kc in range(4):
#                         result += coeffs[i][j][k][ic + 4*jc + 16*kc] * kc * a**ic * b**jc * c**(abs(kc-1))
#             return result
#
#         return output
#
#     else:
#         raise ValueError("i, j and k must be in the good range")
#
# def d2P_dz2(i, j, k):
#     if 0 <= i <= 8 and 0 <= j <= 3 and 0 <= k <= 3:
#
#         def output(a, b, c):
#             result = 0.
#             for ic in range(4):
#                 for jc in range(4):
#                     for kc in range(4):
#                         result += coeffs[i][j][k][ic + 4*jc + 16*kc] * kc*(kc-1) * a**ic * b**(jc) * c**(abs(kc-2))
#             return result
#
#         return output
#
#     else:
#         raise ValueError("i, j and k must be in the good range")
#
# def d2P_dxdy(i, j, k):
#     if 0 <= i <= 8 and 0 <= j <= 3 and 0 <= k <= 3:
#
#         def output(a, b, c):
#             result = 0.
#             for ic in range(4):
#                 for jc in range(4):
#                     for kc in range(4):
#                         result += coeffs[i][j][k][ic + 4*jc + 16*kc] * ic*jc * a**(abs(ic-1)) * b**(abs(jc-1)) * c**kc
#             return result
#
#         return output
#
#     else:
#         raise ValueError("i, j and k must be in the good range")
#
# def d2P_dydz(i, j, k):
#     if 0 <= i <= 8 and 0 <= j <= 3 and 0 <= k <= 3:
#
#         def output(a, b, c):
#             result = 0.
#             for ic in range(4):
#                 for jc in range(4):
#                     for kc in range(4):
#                         result += coeffs[i][j][k][ic + 4*jc + 16*kc] * kc*jc * a**ic * b**(abs(jc-1)) * c**(abs(kc-1))
#             return result
#
#         return output
#
#     else:
#         raise ValueError("i, j and k must be in the good range")
#
# def d2P_dzdx(i, j, k):
#     if 0 <= i <= 8 and 0 <= j <= 3 and 0 <= k <= 3:
#
#         def output(a, b, c):
#             result = 0.
#             for ic in range(4):
#                 for jc in range(4):
#                     for kc in range(4):
#                         result += coeffs[i][j][k][ic + 4*jc + 16*kc] * kc*ic * a**(abs(ic-1)) * b**jc * c**(abs(kc-1))
#             return result
#
#         return output
#
#     else:
#         raise ValueError("i, j and k must be in the good range")
#
#
# def interp_func(a, b, c):
#     return P(*point_to_indices(a, b, c))(a, b, c)
#
# for i in range(NSX):
#     for j in range(NSY):
#         for k in range(NSZ):
#             interp_data[i, j, k] = interp_func(xsamples[i], ysamples[j], zsamples[k])
#
# # compare_func = interpolators3d.Interpolate3DLinear(x, y, z, data)
# # compare_data = np.zeros((NSX, NSY, NSZ), dtype=np.float64)
# # for i in range(NSX):
# #     for j in range(NSY):
# #         for k in range(NSZ):
# #             compare_data[i, j, k] = compare_func(xsamples[i], ysamples[j], zsamples[k])
#
#
# def e0(a, b, c):
#     a_n = max(min(a, x.max()), x.min())
#     b_n = max(min(b, y.max()), y.min())
#     c_n = max(min(c, z.max()), z.min())
#     return interp_func(a_n, b_n, c_n)
#
# def e1(a, b, c):
#     a_n = max(min(a, x.max()), x.min())
#     b_n = max(min(b, y.max()), y.min())
#     c_n = max(min(c, z.max()), z.min())
#     i, j, k = point_to_indices(a_n, b_n, c_n)
#
#     return P(i, j, k)(a_n, b_n, c_n) + \
#            (a - a_n) * dP_dx(i, j, k)(a_n, b_n, c_n) + \
#            (b - b_n) * dP_dy(i, j, k)(a_n, b_n, c_n) + \
#            (c - c_n) * dP_dz(i, j, k)(a_n, b_n, c_n)
#
# def e2(a, b, c):
#     a_n = max(min(a, x.max()), x.min())
#     b_n = max(min(b, y.max()), y.min())
#     c_n = max(min(c, z.max()), z.min())
#     i, j, k = point_to_indices(a_n, b_n, c_n)
#
#     return e1_cub(a, b, c) + \
#            (a - a_n) * (b - b_n) * d2P_dxdy(i, j, k)(a_n, b_n, c_n) + \
#            (b - b_n) * (c - c_n) * d2P_dydz(i, j, k)(a_n, b_n, c_n) + \
#            (c - c_n) * (a - a_n) * d2P_dzdx(i, j, k)(a_n, b_n, c_n) + \
#            0.5 * (a - a_n)**2 * d2P_dx2(i, j, k)(a_n, b_n, c_n) + \
#            0.5 * (b - b_n)**2 * d2P_dy2(i, j, k)(a_n, b_n, c_n) + \
#            0.5 * (c - c_n)**2 * d2P_dz2(i, j, k)(a_n, b_n, c_n)
#
#
#
# whole_calc_data_nea = np.zeros((NSX+6, NSY+6, NSZ+6), dtype=np.float64)
# whole_calc_data_nea[3:NSX+3, 3:NSY+3, 3:NSZ+3] = interp_data
# whole_calc_data_lin = np.zeros((NSX+6, NSY+6, NSZ+6), dtype=np.float64)
# whole_calc_data_lin[3:NSX+3, 3:NSY+3, 3:NSZ+3] = interp_data
# whole_calc_data_qua = np.zeros((NSX+6, NSY+6, NSZ+6), dtype=np.float64)
# whole_calc_data_qua[3:NSX+3, 3:NSY+3, 3:NSZ+3] = interp_data
#
# for nx in range(3):
#     for ny in range(3):
#         for nz in range(3):
#             if nx != 1 or ny != 1 or nz != 1:
#                 mini, maxi = extrapol_xdomains[nx]
#                 minj, maxj = extrapol_ydomains[ny]
#                 mink, maxk = extrapol_zdomains[nz]
#                 for iex in range(mini, maxi):
#                     for jex in range(minj, maxj):
#                         for kex in range(mink, maxk):
#                             extrap_data_nea[nx][ny][nz][iex - mini, jex - minj, kex - mink] = e0(xsamples_ex[iex], ysamples_ex[jex], zsamples_ex[kex])
#                             extrap_data_lin[nx][ny][nz][iex - mini, jex - minj, kex - mink] = e1(xsamples_ex[iex], ysamples_ex[jex], zsamples_ex[kex])
#                             extrap_data_qua[nx][ny][nz][iex - mini, jex - minj, kex - mink] = e2(xsamples_ex[iex], ysamples_ex[jex], zsamples_ex[kex])
#                 whole_calc_data_nea[mini:maxi, minj:maxj, mink:maxk] = extrap_data_nea[nx][ny][nz]
#                 whole_calc_data_lin[mini:maxi, minj:maxj, mink:maxk] = extrap_data_lin[nx][ny][nz]
#                 whole_calc_data_qua[mini:maxi, minj:maxj, mink:maxk] = extrap_data_qua[nx][ny][nz]
#
#
# # def compare():
# #     for k in range(NSZ):
# #         plt.figure()
# #         plt.contourf(compare_data[:, :, k] - interp_data[:, :, k])
# #         plt.colorbar()
# #         plt.show()
#
