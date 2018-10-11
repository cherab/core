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

import unittest

import numpy as np
from cherab.core.math.interpolators import interpolators3d

from cherab.core.math.interpolators.tests import data_test_interpolators3d as data_file

X_LOWER = 0.
X_UPPER = 1.
NB_X = 10
NB_XSAMPLES = 15

Y_LOWER = 0.
Y_UPPER = 2.
NB_Y = 5
NB_YSAMPLES = 10

Z_LOWER = -1.
Z_UPPER = 1.
NB_Z = 5
NB_ZSAMPLES = 15


class TestInterpolators3D(unittest.TestCase):
    """
    3D interpolators tests.

    NB: every standard interpolator in this package must allow coordinates as
    1D numpy arrays and values as nD numpy arrays with C indexing (meaning
    that the first index is for x coordinate, the second for y coordinate, the
    third for z). This indexing is opposed to the Fortran indexing (1st index
    is y, 2nd is x, 3rd is z) which is the default one in numpy.
    """
    
    def setUp(self):
        """
        Initialise domains and sampled data.
        """
        
        self.x = np.linspace(X_LOWER, X_UPPER, NB_X)
        self.y = np.linspace(Y_LOWER, Y_UPPER, NB_Y)
        self.z = np.linspace(Z_LOWER, Z_UPPER, NB_Z)
        self.xsamples = np.linspace(X_LOWER, X_UPPER, NB_XSAMPLES)
        self.ysamples = np.linspace(Y_LOWER, Y_UPPER, NB_YSAMPLES)
        self.zsamples = np.linspace(Z_LOWER, Z_UPPER, NB_ZSAMPLES)
        self.xsamples_ex = np.array([X_LOWER - 0.08, X_LOWER - 0.05, X_LOWER - 0.02] + list(self.xsamples) + [X_UPPER + 0.02, X_UPPER + 0.05, X_UPPER + 0.08], dtype=np.float64)
        self.ysamples_ex = np.array([Y_LOWER - 0.16, Y_LOWER - 0.10, Y_LOWER - 0.04] + list(self.ysamples) + [Y_UPPER + 0.04, Y_UPPER + 0.10, Y_UPPER + 0.16], dtype=np.float64)
        self.zsamples_ex = np.array([Z_LOWER - 0.16, Z_LOWER - 0.10, Z_LOWER - 0.04] + list(self.zsamples) + [Z_UPPER + 0.04, Z_UPPER + 0.10, Z_UPPER + 0.16], dtype=np.float64)
        self.extrapol_xdomains = [(0, 3), (3, NB_XSAMPLES + 3), (NB_XSAMPLES + 3, NB_XSAMPLES + 6)]
        self.extrapol_ydomains = [(0, 3), (3, NB_YSAMPLES + 3), (NB_YSAMPLES + 3, NB_YSAMPLES + 6)]
        self.extrapol_zdomains = [(0, 3), (3, NB_ZSAMPLES + 3), (NB_ZSAMPLES + 3, NB_ZSAMPLES + 6)]
        self.xsamples_out = np.array([X_LOWER - 1, X_LOWER - 0.05, 0.5, X_UPPER + 0.05, X_UPPER + 1], dtype=np.float64)
        self.ysamples_out = np.array([Y_LOWER - 1, Y_LOWER - 0.10, 1.0, Y_UPPER + 0.10, Y_UPPER + 1], dtype=np.float64)
        self.zsamples_out = np.array([Z_LOWER - 1, Z_LOWER - 0.10, 1.0, Z_UPPER + 0.10, Z_UPPER + 1], dtype=np.float64)

        # data generated from function x, y, z -> (x-y)*y*cos(10*x)*exp(z) + z
        # sampled on self.x, self.y and self.z:
        self.data = np.array( [[[-1.000000000000e+00, -5.000000000000e-01,  0.000000000000e+00,  5.000000000000e-01,  1.000000000000e+00],
                                [-1.091969860293e+00, -6.516326649282e-01, -2.500000000000e-01,  8.781968232497e-02,  3.204295428852e-01],
                                [-1.367879441171e+00, -1.106530659713e+00, -1.000000000000e+00, -1.148721270700e+00, -1.718281828459e+00],
                                [-1.827728742636e+00, -1.864693984353e+00, -2.250000000000e+00, -3.209622859075e+00, -5.116134114033e+00],
                                [-2.471517764686e+00, -2.926122638851e+00, -4.000000000000e+00, -6.094885082801e+00, -9.873127313836e+00]],

                               [[-1.000000000000e+00, -5.000000000000e-01,  0.000000000000e+00,  5.000000000000e-01,  1.000000000000e+00],
                                [-1.031736368248e+00, -5.523244253846e-01, -8.626839310877e-02,  3.577674652925e-01,  7.654981946421e-01],
                                [-1.145080540561e+00, -7.391973731868e-01, -3.943697970686e-01, -1.502058729488e-01, -7.200825306479e-02],
                                [-1.340032516939e+00, -1.060618843407e+00, -9.243042118796e-01, -1.023920014724e+00, -1.512519343121e+00],
                                [-1.616592297382e+00, -1.516588836044e+00, -1.676071637542e+00, -2.263374960032e+00, -3.556035075525e+00]],

                               [[-1.000000000000e+00, -5.000000000000e-01,  0.000000000000e+00,  5.000000000000e-01,  1.000000000000e+00],
                                [-9.690204163759e-01, -4.489233015215e-01,  8.421123921859e-02,  6.388408613317e-01,  1.228909881320e+00],
                                [-8.265143317051e-01, -2.139704885206e-01,  4.715829396241e-01,  1.277508823458e+00,  2.281895335391e+00],
                                [-5.724817459876e-01,  2.048584390029e-01,  1.162115101217e+00,  2.416003886378e+00,  4.158956362215e+00],
                                [-2.069226592233e-01,  8.075634810488e-01,  2.155807723996e+00,  4.054326050092e+00,  6.860092961790e+00]],

                               [[-1.000000000000e+00, -5.000000000000e-01,  0.000000000000e+00,  5.000000000000e-01,  1.000000000000e+00],
                                [-9.699051929779e-01, -4.503820515250e-01,  8.180616705926e-02,  6.348755677050e-01,  1.222372217373e+00],
                                [-7.592415438229e-01, -1.030564121999e-01,  6.544493364741e-01,  1.579004541640e+00,  2.778977738984e+00],
                                [-3.680090525351e-01,  5.419769179753e-01,  1.717929508244e+00,  3.332386921806e+00,  5.669816564834e+00],
                                [ 2.037922808855e-01,  1.484717939001e+00,  3.272246682370e+00,  5.895022708202e+00,  9.894888694922e+00]],

                               [[-1.000000000000e+00, -5.000000000000e-01,  0.000000000000e+00,  5.000000000000e-01,  1.000000000000e+00],
                                [-9.972945545212e-01, -4.955394744924e-01,  7.354163282875e-03,  5.121249654327e-01,  1.019990688415e+00],
                                [-9.458910904243e-01, -4.107894898481e-01,  1.470832656575e-01,  7.424993086535e-01,  1.399813768307e+00],
                                [-8.457896077091e-01, -2.457500460670e-01,  4.191873071238e-01,  1.191123029663e+00,  2.139469239675e+00],
                                [-6.969901063758e-01, -4.211431492131e-04,  8.236662876819e-01,  1.857996128460e+00,  3.238957102520e+00]],

                               [[-1.000000000000e+00, -5.000000000000e-01,  0.000000000000e+00,  5.000000000000e-01,  1.000000000000e+00],
                                [-9.923690261243e-01, -4.874186510550e-01,  2.074313761976e-02,  5.341996522148e-01,  1.056385694057e+00],
                                [-1.122095582011e+00, -7.013015831204e-01, -3.318902019162e-01, -4.719443543620e-02,  9.782889508759e-02],
                                [-1.389179667661e+00, -1.141648796196e+00, -1.057900018608e+00, -1.244182262953e+00, -1.875670396908e+00],
                                [-1.793621283073e+00, -1.808460290283e+00, -2.157286312455e+00, -3.056763830335e+00, -4.864112181931e+00]],

                               [[-1.000000000000e+00, -5.000000000000e-01,  0.000000000000e+00,  5.000000000000e-01,  1.000000000000e+00],
                                [-9.715700406368e-01, -4.531269212727e-01,  7.728064192091e-02,  6.274142381484e-01,  1.210070564625e+00],
                                [-1.113719837453e+00, -6.874923149092e-01, -3.091225676837e-01, -9.656952593488e-03,  1.597177414989e-01],
                                [-1.426449390449e+00, -1.203096180910e+00, -1.159209628814e+00, -1.411213572226e+00, -2.151058469379e+00],
                                [-1.909758699624e+00, -1.999938519274e+00, -2.472980541469e+00, -3.577255620748e+00, -5.722258068009e+00]],

                               [[-1.000000000000e+00, -5.000000000000e-01,  0.000000000000e+00,  5.000000000000e-01,  1.000000000000e+00],
                                [-9.961101794857e-01, -4.935867701789e-01,  1.057362842001e-02,  5.174329660846e-01,  1.028742101995e+00],
                                [-1.006223712823e+00, -5.102611677138e-01, -1.691780547202e-02,  4.721072542647e-01,  9.540126368080e-01],
                                [-1.030340600012e+00, -5.500231926049e-01, -8.247430167608e-02,  3.640228645405e-01,  7.758116044391e-01],
                                [-1.068460841052e+00, -6.128728448522e-01, -1.860958601922e-01,  1.931797969119e-01,  4.941390048881e-01]],

                               [[-1.000000000000e+00, -5.000000000000e-01,  0.000000000000e+00,  5.000000000000e-01,  1.000000000000e+00],
                                [-1.061504384508e+00, -6.014035869796e-01, -1.671862507785e-01,  2.243564721729e-01,  5.455406525406e-01],
                                [-9.648546374240e-01, -4.420550931545e-01,  9.553500044485e-02,  6.575105873298e-01,  1.259691055691e+00],
                                [-7.100507587483e-01, -2.195451852494e-02,  7.881637536700e-01,  1.799462345471e+00,  3.142451209451e+00],
                                [-2.970927484807e-01,  6.588981369092e-01,  1.910700008897e+00,  3.650211746595e+00,  6.193821113821e+00]],

                               [[-1.000000000000e+00, -5.000000000000e-01,  0.000000000000e+00,  5.000000000000e-01,  1.000000000000e+00],
                                [-1.077169291305e+00, -6.272306520192e-01, -2.097678822691e-01,  1.541512305932e-01,  4.297917774335e-01],
                                [-1.000000000000e+00, -5.000000000000e-01,  0.000000000000e+00,  5.000000000000e-01,  1.000000000000e+00],
                                [-7.684921260854e-01, -1.183080439424e-01,  6.293036468073e-01,  1.537546308220e+00,  2.710624667699e+00],
                                [-3.826456695610e-01,  5.178452161537e-01,  1.678143058153e+00,  3.266790155254e+00,  5.561665780532e+00]]],
                             dtype=np.float64)

    def tearDown(self):

        try:
            del self.interp_func
        except AttributeError:
            pass

    def init_3dlinear(self, x=None, y=None, z=None, data=None, extrapolate=False, extrapolation_range=float('inf'),
                      extrapolation_type='nearest', tolerate_single_value=False):
        """Create the interpolating function and reference function. Data is
        assumed sorted and regularly spaced."""

        if x is None:
            x = self.x

        if y is None:
            y = self.y

        if z is None:
            z = self.z

        if data is None:
            data = self.data

        # reference interpolated data:
        self.interp_data = data_file.linear_interpolated_data
        self.extrap_data_nea = data_file.linear_nearest_extrapolated_data
        self.extrap_data_lin = data_file.linear_linear_extrapolated_data

        self.interp_func = interpolators3d.Interpolate3DLinear(x, y, z, data,
                                                               extrapolate=extrapolate,
                                                               extrapolation_range=extrapolation_range,
                                                               extrapolation_type=extrapolation_type,
                                                               tolerate_single_value=tolerate_single_value)

    def init_3dcubic(self, x=None, y=None, z=None, data=None, extrapolate=False, extrapolation_range=float('inf'),
                     extrapolation_type='nearest', tolerate_single_value=False):
        """Create the interpolating function and reference function. Data is
        assumed sorted and regularly spaced."""

        if x is None:
            x = self.x

        if y is None:
            y = self.y

        if z is None:
            z = self.z

        if data is None:
            data = self.data

        # reference interpolated data:
        self.interp_data = data_file.cubic_interpolated_data
        self.extrap_data_nea = data_file.cubic_nearest_extrapolated_data
        self.extrap_data_lin = data_file.cubic_linear_extrapolated_data
        self.extrap_data_qua = data_file.cubic_quadratic_extrapolated_data

        self.interp_func = interpolators3d.Interpolate3DCubic(x, y, z, data,
                                                              extrapolate=extrapolate,
                                                              extrapolation_range=extrapolation_range,
                                                              extrapolation_type=extrapolation_type,
                                                              tolerate_single_value=tolerate_single_value)

    def interpolate_3d_extrapolate_assert(self, i_block, j_block, k_block, ref_data, delta):
        mini, maxi = self.extrapol_xdomains[i_block]
        minj, maxj = self.extrapol_ydomains[j_block]
        mink, maxk = self.extrapol_zdomains[k_block]
        for iex in range(mini, maxi):
            for jex in range(minj, maxj):
                for kex in range(mink, maxk):
                    self.assertAlmostEqual(self.interp_func(self.xsamples_ex[iex], self.ysamples_ex[jex], self.zsamples_ex[kex]), ref_data[iex - mini, jex - minj, kex - mink], delta=delta)

    def interpolate_3d_xboundaries_assert(self, inf, sup, epsilon, y, z):
        with self.assertRaises(ValueError):
            self.interp_func(inf - epsilon, y, z)
        self.assertIsInstance(self.interp_func(inf + epsilon, y, z), float)
        with self.assertRaises(ValueError):
            self.interp_func(sup + epsilon, y, z)
        self.assertIsInstance(self.interp_func(sup - epsilon, y, z), float)

    def interpolate_3d_yboundaries_assert(self, x, inf, sup, epsilon, z):
        with self.assertRaises(ValueError):
            self.interp_func(x, inf - epsilon, z)
        self.assertIsInstance(self.interp_func(x, inf + epsilon, z), float)
        with self.assertRaises(ValueError):
            self.interp_func(x, sup + epsilon, z)
        self.assertIsInstance(self.interp_func(x, sup - epsilon, z), float)

    def interpolate_3d_zboundaries_assert(self, x, y, inf, sup, epsilon):
        with self.assertRaises(ValueError):
            self.interp_func(x, y, inf - epsilon)
        self.assertIsInstance(self.interp_func(x, y, inf + epsilon), float)
        with self.assertRaises(ValueError):
            self.interp_func(x, y, sup + epsilon)
        self.assertIsInstance(self.interp_func(x, y, sup - epsilon), float)

    # General behaviour

    def test_interpolate_3d_invalid_coordinatesx(self):
        """3D interpolation. An error must be raises if coordinates are not an array-like object.
        """
        self.assertRaises(ValueError, interpolators3d._Interpolate3DBase, "blah", [1, 2, 3, 4], [2, 3, 4, 5], np.ones((4, 4, 4)))

    def test_interpolate_3d_invalid_coordinatesy(self):
        """3D interpolation. An error must be raises if coordinates are not an array-like object.
        """
        self.assertRaises(ValueError, interpolators3d._Interpolate3DBase, [1, 2, 3, 4], "blah", [2, 3, 4, 5], np.ones((4, 4, 4)))

    def test_interpolate_3d_invalid_coordinatesz(self):
        """3D interpolation. An error must be raises if coordinates are not an array-like object.
        """
        self.assertRaises(ValueError, interpolators3d._Interpolate3DBase, [1, 2, 3, 4], [2, 3, 4, 5], "blah", np.ones((4, 4, 4)))

    def test_interpolate_3d_invalid_data(self):
        """3D interpolation. An error must be raises if data is not an array-like object.
        """
        self.assertRaises(ValueError, interpolators3d._Interpolate3DBase, [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], "blah")

    def test_interpolate_3d_invalid_data_size(self):
        """3D interpolation. An error must be raises if data has not the same size as the product of the coordinates.
        """
        self.assertRaises(ValueError, interpolators3d._Interpolate3DBase, [1, 2, 3, 4], [2, 3, 4], [3, 4, 5, 6], np.ones((4, 4, 4)))

    def test_interpolate_3d_invalid_data_dimension(self):
        """3D interpolation. An error must be raises if data is not 3D.
        """
        self.assertRaises(ValueError, interpolators3d._Interpolate3DBase, [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], np.ones((4, 4)))

    def test_interpolate_3d_double_coord(self):
        """3D interpolation. An error must be raises if there is a double coordinate.
        """
        self.assertRaises(ValueError, interpolators3d._Interpolate3DBase, [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 6, 6], np.ones((4, 4, 4)))

    def test_interpolate_3d_single_value_invalid_x(self):
        """3D interpolation. By default, a single input value must raise a ValueError.
        """
        self.assertRaises(ValueError, interpolators3d._Interpolate3DBase, [2.], [2, 3, 4, 5], [3, 4, 5, 6], np.ones((1, 4, 4)))

    def test_interpolate_3d_single_value_invalid_y(self):
        """3D interpolation. By default, a single input value must raise a ValueError.
        """
        self.assertRaises(ValueError, interpolators3d._Interpolate3DBase, [1, 2, 3, 4], [4.], [3, 4, 5, 6], np.ones((4, 1, 4)))

    def test_interpolate_3d_single_value_invalid_z(self):
        """3D interpolation. By default, a single input value must raise a ValueError.
        """
        self.assertRaises(ValueError, interpolators3d._Interpolate3DBase, [1, 2, 3, 4], [2, 3, 4, 5], [4.], np.ones((4, 4, 1)))

    def test_interpolate_3d_single_value_invalid_xy(self):
        """3D interpolation. By default, a single input value must raise a ValueError.
        """
        self.assertRaises(ValueError, interpolators3d._Interpolate3DBase, [2.], [4.], [3, 4, 5, 6], np.ones((1, 1, 4)))

    def test_interpolate_3d_single_value_invalid_yz(self):
        """3D interpolation. By default, a single input value must raise a ValueError.
        """
        self.assertRaises(ValueError, interpolators3d._Interpolate3DBase, [1, 2, 3, 4], [4.], [5.], np.ones((4, 1, 1)))

    def test_interpolate_3d_single_value_invalid_xz(self):
        """3D interpolation. By default, a single input value must raise a ValueError.
        """
        self.assertRaises(ValueError, interpolators3d._Interpolate3DBase, [2.], [2, 3, 4, 5], [5.], np.ones((1, 4, 1)))

    def test_interpolate_3d_single_value_invalid_xyz(self):
        """3D interpolation. By default, a single input value must raise a ValueError.
        """
        self.assertRaises(ValueError, interpolators3d._Interpolate3DBase, [2.], [4.], [5.], np.ones((1, 1, 1)))

    # Linear behaviour

    def test_interpolate_3d_linear(self):
        """3D linear interpolation. Test values inside the boundaries"""
        self.init_3dlinear()
        for i in range(len(self.xsamples)):
            for j in range(len(self.ysamples)):
                for k in range(len(self.zsamples)):
                    self.assertAlmostEqual(self.interp_func(self.xsamples[i], self.ysamples[j], self.zsamples[k]),
                                           self.interp_data[i, j, k], delta=1e-8)

    def test_interpolate_3d_linear_bigvalues(self):
        """3D linear interpolation. Test with big values (1e20) inside the boundaries"""
        factor = 1.e20
        self.init_3dlinear(data=factor * self.data)
        for i in range(len(self.xsamples)):
            for j in range(len(self.ysamples)):
                for k in range(len(self.zsamples)):
                    self.assertAlmostEqual(self.interp_func(self.xsamples[i], self.ysamples[j], self.zsamples[k]),
                                           factor * self.interp_data[i, j, k], delta=factor * 1e-8)

    def test_interpolate_3d_linear_lowvalues(self):
        """3D linear interpolation. Test with low values (1e-20) inside the boundaries"""
        factor = 1.e-20
        self.init_3dlinear(data=factor * self.data)
        for i in range(len(self.xsamples)):
            for j in range(len(self.ysamples)):
                for k in range(len(self.zsamples)):
                    self.assertAlmostEqual(self.interp_func(self.xsamples[i], self.ysamples[j], self.zsamples[k]),
                                           factor * self.interp_data[i, j, k], delta=factor * 1e-8)

    def test_interpolate_3d_linear_edge(self):
        """3D linear interpolation. Test edges values"""
        self.init_3dlinear()
        for i in range(len(self.xsamples)):
            for j in range(len(self.ysamples)):
                self.assertAlmostEqual(self.interp_func(self.xsamples[i], self.ysamples[j], self.zsamples[0]),
                                       self.interp_data[i, j, 0], delta=1e-8)
                self.assertAlmostEqual(self.interp_func(self.xsamples[i], self.ysamples[j], self.zsamples[-1]),
                                       self.interp_data[i, j, -1], delta=1e-8)
        for j in range(len(self.ysamples)):
            for k in range(len(self.zsamples)):
                self.assertAlmostEqual(self.interp_func(self.xsamples[0], self.ysamples[j], self.zsamples[k]),
                                       self.interp_data[0, j, k], delta=1e-8)
                self.assertAlmostEqual(self.interp_func(self.xsamples[-1], self.ysamples[j], self.zsamples[k]),
                                       self.interp_data[-1, j, k], delta=1e-8)
        for k in range(len(self.zsamples)):
            for i in range(len(self.xsamples)):
                self.assertAlmostEqual(self.interp_func(self.xsamples[i], self.ysamples[0], self.zsamples[k]),
                                       self.interp_data[i, 0, k], delta=1e-8)
                self.assertAlmostEqual(self.interp_func(self.xsamples[i], self.ysamples[-1], self.zsamples[k]),
                                       self.interp_data[i, -1, k], delta=1e-8)

    def test_interpolate_3d_linear_knot(self):
        """3D linear interpolation. Test knot values"""
        self.init_3dlinear()
        for i in range(len(self.x)):
            for j in range(len(self.y)):
                for k in range(len(self.z)):
                    self.assertAlmostEqual(self.interp_func(self.x[i], self.y[j], self.z[k]),
                                           self.data[i, j, k], delta=1e-8)

    def test_interpolate_3d_linear_out(self):
        """3D linear interpolation. Test values outside the boundaries"""
        self.init_3dlinear(extrapolate=False)
        for i in range(len(self.xsamples_out)):
            for j in range(len(self.ysamples_out)):
                self.assertRaises(ValueError, self.interp_func, self.xsamples_out[i], self.ysamples_out[j], self.zsamples_out[0])
                self.assertRaises(ValueError, self.interp_func, self.xsamples_out[i], self.ysamples_out[j], self.zsamples_out[-1])
        self.interpolate_3d_xboundaries_assert(X_LOWER, X_UPPER, 1e-6, (Y_LOWER + Y_UPPER) / 2, (Z_LOWER + Z_UPPER) / 2)
        for j in range(len(self.ysamples_out)):
            for k in range(len(self.zsamples_out)):
                self.assertRaises(ValueError, self.interp_func, self.xsamples_out[0], self.ysamples_out[j], self.zsamples_out[k])
                self.assertRaises(ValueError, self.interp_func, self.xsamples_out[-1], self.ysamples_out[j], self.zsamples_out[k])
        self.interpolate_3d_yboundaries_assert((Y_LOWER + Y_UPPER) / 2, Y_LOWER, Y_UPPER, 1e-6, (Z_LOWER + Z_UPPER) / 2)
        for k in range(len(self.zsamples_out)):
            for i in range(len(self.xsamples_out)):
                self.assertRaises(ValueError, self.interp_func, self.xsamples_out[i], self.ysamples_out[0], self.zsamples_out[k])
                self.assertRaises(ValueError, self.interp_func, self.xsamples_out[i], self.ysamples_out[-1], self.zsamples_out[k])
        self.interpolate_3d_zboundaries_assert((Y_LOWER + Y_UPPER) / 2, (Z_LOWER + Z_UPPER) / 2, Z_LOWER, Z_UPPER, 1e-6)

    def test_interpolate_3d_linear_extrapolate_nearest_range(self):
        """3D linear interpolation. Tests the size of the extrapolation range."""
        self.init_3dlinear(extrapolate=True, extrapolation_type='nearest', extrapolation_range=1.)
        self.interpolate_3d_xboundaries_assert(X_LOWER - 1, X_UPPER + 1, 1e-6, (Y_LOWER + Y_UPPER) / 2, (Z_LOWER + Z_UPPER) / 2)
        self.interpolate_3d_yboundaries_assert((X_LOWER + X_UPPER) / 2, Y_LOWER - 1, Y_UPPER + 1, 1e-6, (Z_LOWER + Z_UPPER) / 2)
        self.interpolate_3d_zboundaries_assert((X_LOWER + X_UPPER) / 2, (Y_LOWER + Y_UPPER) / 2, Z_LOWER - 1, Z_UPPER + 1, 1e-6)

    def test_interpolate_3d_linear_extrapolate_nearest_xinfyinfzinf(self):
        """3D linear interpolation. Test values in the extrapolation area with x below and y below and z below the interpolation area.
        """
        self.init_3dlinear(extrapolate=True, extrapolation_range=10, extrapolation_type='nearest')
        self.interpolate_3d_extrapolate_assert(0, 0, 0, self.extrap_data_nea[0][0][0], 1e-8)

    def test_interpolate_3d_linear_extrapolate_nearest_xinfyinfzmid(self):
        """3D linear interpolation. Test values in the extrapolation area with x below and y below and z inside the interpolation area.
        """
        self.init_3dlinear(extrapolate=True, extrapolation_range=10, extrapolation_type='nearest')
        self.interpolate_3d_extrapolate_assert(0, 0, 1, self.extrap_data_nea[0][0][1], 1e-8)

    def test_interpolate_3d_linear_extrapolate_nearest_xinfyinfzsup(self):
        """3D linear interpolation. Test values in the extrapolation area with x below and y below and z above the interpolation area.
        """
        self.init_3dlinear(extrapolate=True, extrapolation_range=10, extrapolation_type='nearest')
        self.interpolate_3d_extrapolate_assert(0, 0, 2, self.extrap_data_nea[0][0][2], 1e-8)

    def test_interpolate_3d_linear_extrapolate_nearest_xinfymidzinf(self):
        """3D linear interpolation. Test values in the extrapolation area with x below and y inside and z below the interpolation area.
        """
        self.init_3dlinear(extrapolate=True, extrapolation_range=10, extrapolation_type='nearest')
        self.interpolate_3d_extrapolate_assert(0, 1, 0, self.extrap_data_nea[0][1][0], 1e-8)

    def test_interpolate_3d_linear_extrapolate_nearest_xinfymidzmid(self):
        """3D linear interpolation. Test values in the extrapolation area with x below and y inside and z inside the interpolation area.
        """
        self.init_3dlinear(extrapolate=True, extrapolation_range=10, extrapolation_type='nearest')
        self.interpolate_3d_extrapolate_assert(0, 1, 1, self.extrap_data_nea[0][1][1], 1e-8)

    def test_interpolate_3d_linear_extrapolate_nearest_xinfymidzsup(self):
        """3D linear interpolation. Test values in the extrapolation area with x below and y inside and z above the interpolation area.
        """
        self.init_3dlinear(extrapolate=True, extrapolation_range=10, extrapolation_type='nearest')
        self.interpolate_3d_extrapolate_assert(0, 1, 2, self.extrap_data_nea[0][1][2], 1e-8)

    def test_interpolate_3d_linear_extrapolate_nearest_xinfysupzinf(self):
        """3D linear interpolation. Test values in the extrapolation area with x below and y above and z below the interpolation area.
        """
        self.init_3dlinear(extrapolate=True, extrapolation_range=10, extrapolation_type='nearest')
        self.interpolate_3d_extrapolate_assert(0, 2, 0, self.extrap_data_nea[0][2][0], 1e-8)

    def test_interpolate_3d_linear_extrapolate_nearest_xinfysupzmid(self):
        """3D linear interpolation. Test values in the extrapolation area with x below and y above and z inside the interpolation area.
        """
        self.init_3dlinear(extrapolate=True, extrapolation_range=10, extrapolation_type='nearest')
        self.interpolate_3d_extrapolate_assert(0, 2, 1, self.extrap_data_nea[0][2][1], 1e-8)

    def test_interpolate_3d_linear_extrapolate_nearest_xinfysupzsup(self):
        """3D linear interpolation. Test values in the extrapolation area with x below and y above and z above the interpolation area.
        """
        self.init_3dlinear(extrapolate=True, extrapolation_range=10, extrapolation_type='nearest')
        self.interpolate_3d_extrapolate_assert(0, 2, 2, self.extrap_data_nea[0][2][2], 1e-8)

    def test_interpolate_3d_linear_extrapolate_nearest_xmidyinfzinf(self):
        """3D linear interpolation. Test values in the extrapolation area with x inside and y below and z below the interpolation area.
        """
        self.init_3dlinear(extrapolate=True, extrapolation_range=10, extrapolation_type='nearest')
        self.interpolate_3d_extrapolate_assert(1, 0, 0, self.extrap_data_nea[1][0][0], 1e-8)

    def test_interpolate_3d_linear_extrapolate_nearest_xmidyinfzmid(self):
        """3D linear interpolation. Test values in the extrapolation area with x inside and y below and z inside the interpolation area.
        """
        self.init_3dlinear(extrapolate=True, extrapolation_range=10, extrapolation_type='nearest')
        self.interpolate_3d_extrapolate_assert(1, 0, 1, self.extrap_data_nea[1][0][1], 1e-8)

    def test_interpolate_3d_linear_extrapolate_nearest_xmidyinfzsup(self):
        """3D linear interpolation. Test values in the extrapolation area with x inside and y below and z above the interpolation area.
        """
        self.init_3dlinear(extrapolate=True, extrapolation_range=10, extrapolation_type='nearest')
        self.interpolate_3d_extrapolate_assert(1, 0, 2, self.extrap_data_nea[1][0][2], 1e-8)

    def test_interpolate_3d_linear_extrapolate_nearest_xmidymidzinf(self):
        """3D linear interpolation. Test values in the extrapolation area with x inside and y inside and z below the interpolation area.
        """
        self.init_3dlinear(extrapolate=True, extrapolation_range=10, extrapolation_type='nearest')
        self.interpolate_3d_extrapolate_assert(1, 1, 0, self.extrap_data_nea[1][1][0], 1e-8)

    def test_interpolate_3d_linear_extrapolate_nearest_xmidymidzsup(self):
        """3D linear interpolation. Test values in the extrapolation area with x inside and y inside and z above the interpolation area.
        """
        self.init_3dlinear(extrapolate=True, extrapolation_range=10, extrapolation_type='nearest')
        self.interpolate_3d_extrapolate_assert(1, 1, 2, self.extrap_data_nea[1][1][2], 1e-8)

    def test_interpolate_3d_linear_extrapolate_nearest_xmidysupzinf(self):
        """3D linear interpolation. Test values in the extrapolation area with x inside and y above and z below the interpolation area.
        """
        self.init_3dlinear(extrapolate=True, extrapolation_range=10, extrapolation_type='nearest')
        self.interpolate_3d_extrapolate_assert(1, 2, 0, self.extrap_data_nea[1][2][0], 1e-8)

    def test_interpolate_3d_linear_extrapolate_nearest_xmidysupzmid(self):
        """3D linear interpolation. Test values in the extrapolation area with x inside and y above and z inside the interpolation area.
        """
        self.init_3dlinear(extrapolate=True, extrapolation_range=10, extrapolation_type='nearest')
        self.interpolate_3d_extrapolate_assert(1, 2, 1, self.extrap_data_nea[1][2][1], 1e-8)

    def test_interpolate_3d_linear_extrapolate_nearest_xmidysupzsup(self):
        """3D linear interpolation. Test values in the extrapolation area with x inside and y above and z above the interpolation area.
        """
        self.init_3dlinear(extrapolate=True, extrapolation_range=10, extrapolation_type='nearest')
        self.interpolate_3d_extrapolate_assert(1, 2, 2, self.extrap_data_nea[1][2][2], 1e-8)

    def test_interpolate_3d_linear_extrapolate_nearest_xsupyinfzinf(self):
        """3D linear interpolation. Test values in the extrapolation area with x above and y below and z below the interpolation area.
        """
        self.init_3dlinear(extrapolate=True, extrapolation_range=10, extrapolation_type='nearest')
        self.interpolate_3d_extrapolate_assert(2, 0, 0, self.extrap_data_nea[2][0][0], 1e-8)

    def test_interpolate_3d_linear_extrapolate_nearest_xsupyinfzmid(self):
        """3D linear interpolation. Test values in the extrapolation area with x above and y below and z inside the interpolation area.
        """
        self.init_3dlinear(extrapolate=True, extrapolation_range=10, extrapolation_type='nearest')
        self.interpolate_3d_extrapolate_assert(2, 0, 1, self.extrap_data_nea[2][0][1], 1e-8)

    def test_interpolate_3d_linear_extrapolate_nearest_xsupyinfzsup(self):
        """3D linear interpolation. Test values in the extrapolation area with x above and y below and z above the interpolation area.
        """
        self.init_3dlinear(extrapolate=True, extrapolation_range=10, extrapolation_type='nearest')
        self.interpolate_3d_extrapolate_assert(2, 0, 2, self.extrap_data_nea[2][0][2], 1e-8)

    def test_interpolate_3d_linear_extrapolate_nearest_xsupymidzinf(self):
        """3D linear interpolation. Test values in the extrapolation area with x above and y inside and z below the interpolation area.
        """
        self.init_3dlinear(extrapolate=True, extrapolation_range=10, extrapolation_type='nearest')
        self.interpolate_3d_extrapolate_assert(2, 1, 0, self.extrap_data_nea[2][1][0], 1e-8)

    def test_interpolate_3d_linear_extrapolate_nearest_xsupymidzmid(self):
        """3D linear interpolation. Test values in the extrapolation area with x above and y inside and z inside the interpolation area.
        """
        self.init_3dlinear(extrapolate=True, extrapolation_range=10, extrapolation_type='nearest')
        self.interpolate_3d_extrapolate_assert(2, 1, 1, self.extrap_data_nea[2][1][1], 1e-8)

    def test_interpolate_3d_linear_extrapolate_nearest_xsupymidzsup(self):
        """3D linear interpolation. Test values in the extrapolation area with x above and y inside and z above the interpolation area.
        """
        self.init_3dlinear(extrapolate=True, extrapolation_range=10, extrapolation_type='nearest')
        self.interpolate_3d_extrapolate_assert(2, 1, 2, self.extrap_data_nea[2][1][2], 1e-8)

    def test_interpolate_3d_linear_extrapolate_nearest_xsupysupzinf(self):
        """3D linear interpolation. Test values in the extrapolation area with x above and y above and z below the interpolation area.
        """
        self.init_3dlinear(extrapolate=True, extrapolation_range=10, extrapolation_type='nearest')
        self.interpolate_3d_extrapolate_assert(2, 2, 0, self.extrap_data_nea[2][2][0], 1e-8)

    def test_interpolate_3d_linear_extrapolate_nearest_xsupysupzmid(self):
        """3D linear interpolation. Test values in the extrapolation area with x above and y above and z inside the interpolation area.
        """
        self.init_3dlinear(extrapolate=True, extrapolation_range=10, extrapolation_type='nearest')
        self.interpolate_3d_extrapolate_assert(2, 2, 1, self.extrap_data_nea[2][2][1], 1e-8)

    def test_interpolate_3d_linear_extrapolate_nearest_xsupysupzsup(self):
        """3D linear interpolation. Test values in the extrapolation area with x above and y above and z above the interpolation area.
        """
        self.init_3dlinear(extrapolate=True, extrapolation_range=10, extrapolation_type='nearest')
        self.interpolate_3d_extrapolate_assert(2, 2, 2, self.extrap_data_nea[2][2][2], 1e-8)

    def test_interpolate_3d_linear_extrapolate_linear_range(self):
        """3D linear interpolation. Tests the size of the extrapolation range."""
        self.init_3dlinear(extrapolate=True, extrapolation_type='linear', extrapolation_range=1.)
        self.interpolate_3d_xboundaries_assert(X_LOWER - 1, X_UPPER + 1, 1e-6, (Y_LOWER + Y_UPPER) / 2, (Z_LOWER + Z_UPPER) / 2)
        self.interpolate_3d_yboundaries_assert((X_LOWER + X_UPPER) / 2, Y_LOWER - 1, Y_UPPER + 1, 1e-6, (Z_LOWER + Z_UPPER) / 2)
        self.interpolate_3d_zboundaries_assert((X_LOWER + X_UPPER) / 2, (Y_LOWER + Y_UPPER) / 2, Z_LOWER - 1, Z_UPPER + 1, 1e-6)

    def test_interpolate_3d_linear_extrapolate_linear_xinfyinfzinf(self):
        """3D linear interpolation. Test values in the extrapolation area with x below and y below and z below the interpolation area.
        """
        self.init_3dlinear(extrapolate=True, extrapolation_range=10, extrapolation_type='linear')
        self.interpolate_3d_extrapolate_assert(0, 0, 0, self.extrap_data_lin[0][0][0], 1e-8)

    def test_interpolate_3d_linear_extrapolate_linear_xinfyinfzmid(self):
        """3D linear interpolation. Test values in the extrapolation area with x below and y below and z inside the interpolation area.
        """
        self.init_3dlinear(extrapolate=True, extrapolation_range=10, extrapolation_type='linear')
        self.interpolate_3d_extrapolate_assert(0, 0, 1, self.extrap_data_lin[0][0][1], 1e-8)

    def test_interpolate_3d_linear_extrapolate_linear_xinfyinfzsup(self):
        """3D linear interpolation. Test values in the extrapolation area with x below and y below and z above the interpolation area.
        """
        self.init_3dlinear(extrapolate=True, extrapolation_range=10, extrapolation_type='linear')
        self.interpolate_3d_extrapolate_assert(0, 0, 2, self.extrap_data_lin[0][0][2], 1e-8)

    def test_interpolate_3d_linear_extrapolate_linear_xinfymidzinf(self):
        """3D linear interpolation. Test values in the extrapolation area with x below and y inside and z below the interpolation area.
        """
        self.init_3dlinear(extrapolate=True, extrapolation_range=10, extrapolation_type='linear')
        self.interpolate_3d_extrapolate_assert(0, 1, 0, self.extrap_data_lin[0][1][0], 1e-8)

    def test_interpolate_3d_linear_extrapolate_linear_xinfymidzmid(self):
        """3D linear interpolation. Test values in the extrapolation area with x below and y inside and z inside the interpolation area.
        """
        self.init_3dlinear(extrapolate=True, extrapolation_range=10, extrapolation_type='linear')
        self.interpolate_3d_extrapolate_assert(0, 1, 1, self.extrap_data_lin[0][1][1], 1e-8)

    def test_interpolate_3d_linear_extrapolate_linear_xinfymidzsup(self):
        """3D linear interpolation. Test values in the extrapolation area with x below and y inside and z above the interpolation area.
        """
        self.init_3dlinear(extrapolate=True, extrapolation_range=10, extrapolation_type='linear')
        self.interpolate_3d_extrapolate_assert(0, 1, 2, self.extrap_data_lin[0][1][2], 1e-8)

    def test_interpolate_3d_linear_extrapolate_linear_xinfysupzinf(self):
        """3D linear interpolation. Test values in the extrapolation area with x below and y above and z below the interpolation area.
        """
        self.init_3dlinear(extrapolate=True, extrapolation_range=10, extrapolation_type='linear')
        self.interpolate_3d_extrapolate_assert(0, 2, 0, self.extrap_data_lin[0][2][0], 1e-8)

    def test_interpolate_3d_linear_extrapolate_linear_xinfysupzmid(self):
        """3D linear interpolation. Test values in the extrapolation area with x below and y above and z inside the interpolation area.
        """
        self.init_3dlinear(extrapolate=True, extrapolation_range=10, extrapolation_type='linear')
        self.interpolate_3d_extrapolate_assert(0, 2, 1, self.extrap_data_lin[0][2][1], 1e-8)

    def test_interpolate_3d_linear_extrapolate_linear_xinfysupzsup(self):
        """3D linear interpolation. Test values in the extrapolation area with x below and y above and z above the interpolation area.
        """
        self.init_3dlinear(extrapolate=True, extrapolation_range=10, extrapolation_type='linear')
        self.interpolate_3d_extrapolate_assert(0, 2, 2, self.extrap_data_lin[0][2][2], 1e-8)

    def test_interpolate_3d_linear_extrapolate_linear_xmidyinfzinf(self):
        """3D linear interpolation. Test values in the extrapolation area with x inside and y below and z below the interpolation area.
        """
        self.init_3dlinear(extrapolate=True, extrapolation_range=10, extrapolation_type='linear')
        self.interpolate_3d_extrapolate_assert(1, 0, 0, self.extrap_data_lin[1][0][0], 1e-8)

    def test_interpolate_3d_linear_extrapolate_linear_xmidyinfzmid(self):
        """3D linear interpolation. Test values in the extrapolation area with x inside and y below and z inside the interpolation area.
        """
        self.init_3dlinear(extrapolate=True, extrapolation_range=10, extrapolation_type='linear')
        self.interpolate_3d_extrapolate_assert(1, 0, 1, self.extrap_data_lin[1][0][1], 1e-8)

    def test_interpolate_3d_linear_extrapolate_linear_xmidyinfzsup(self):
        """3D linear interpolation. Test values in the extrapolation area with x inside and y below and z above the interpolation area.
        """
        self.init_3dlinear(extrapolate=True, extrapolation_range=10, extrapolation_type='linear')
        self.interpolate_3d_extrapolate_assert(1, 0, 2, self.extrap_data_lin[1][0][2], 1e-8)

    def test_interpolate_3d_linear_extrapolate_linear_xmidymidzinf(self):
        """3D linear interpolation. Test values in the extrapolation area with x inside and y inside and z below the interpolation area.
        """
        self.init_3dlinear(extrapolate=True, extrapolation_range=10, extrapolation_type='linear')
        self.interpolate_3d_extrapolate_assert(1, 1, 0, self.extrap_data_lin[1][1][0], 1e-8)

    def test_interpolate_3d_linear_extrapolate_linear_xmidymidzsup(self):
        """3D linear interpolation. Test values in the extrapolation area with x inside and y inside and z above the interpolation area.
        """
        self.init_3dlinear(extrapolate=True, extrapolation_range=10, extrapolation_type='linear')
        self.interpolate_3d_extrapolate_assert(1, 1, 2, self.extrap_data_lin[1][1][2], 1e-8)

    def test_interpolate_3d_linear_extrapolate_linear_xmidysupzinf(self):
        """3D linear interpolation. Test values in the extrapolation area with x inside and y above and z below the interpolation area.
        """
        self.init_3dlinear(extrapolate=True, extrapolation_range=10, extrapolation_type='linear')
        self.interpolate_3d_extrapolate_assert(1, 2, 0, self.extrap_data_lin[1][2][0], 1e-8)

    def test_interpolate_3d_linear_extrapolate_linear_xmidysupzmid(self):
        """3D linear interpolation. Test values in the extrapolation area with x inside and y above and z inside the interpolation area.
        """
        self.init_3dlinear(extrapolate=True, extrapolation_range=10, extrapolation_type='linear')
        self.interpolate_3d_extrapolate_assert(1, 2, 1, self.extrap_data_lin[1][2][1], 1e-8)

    def test_interpolate_3d_linear_extrapolate_linear_xmidysupzsup(self):
        """3D linear interpolation. Test values in the extrapolation area with x inside and y above and z above the interpolation area.
        """
        self.init_3dlinear(extrapolate=True, extrapolation_range=10, extrapolation_type='linear')
        self.interpolate_3d_extrapolate_assert(1, 2, 2, self.extrap_data_lin[1][2][2], 1e-8)

    def test_interpolate_3d_linear_extrapolate_linear_xsupyinfzinf(self):
        """3D linear interpolation. Test values in the extrapolation area with x above and y below and z below the interpolation area.
        """
        self.init_3dlinear(extrapolate=True, extrapolation_range=10, extrapolation_type='linear')
        self.interpolate_3d_extrapolate_assert(2, 0, 0, self.extrap_data_lin[2][0][0], 1e-8)

    def test_interpolate_3d_linear_extrapolate_linear_xsupyinfzmid(self):
        """3D linear interpolation. Test values in the extrapolation area with x above and y below and z inside the interpolation area.
        """
        self.init_3dlinear(extrapolate=True, extrapolation_range=10, extrapolation_type='linear')
        self.interpolate_3d_extrapolate_assert(2, 0, 1, self.extrap_data_lin[2][0][1], 1e-8)

    def test_interpolate_3d_linear_extrapolate_linear_xsupyinfzsup(self):
        """3D linear interpolation. Test values in the extrapolation area with x above and y below and z above the interpolation area.
        """
        self.init_3dlinear(extrapolate=True, extrapolation_range=10, extrapolation_type='linear')
        self.interpolate_3d_extrapolate_assert(2, 0, 2, self.extrap_data_lin[2][0][2], 1e-8)

    def test_interpolate_3d_linear_extrapolate_linear_xsupymidzinf(self):
        """3D linear interpolation. Test values in the extrapolation area with x above and y inside and z below the interpolation area.
        """
        self.init_3dlinear(extrapolate=True, extrapolation_range=10, extrapolation_type='linear')
        self.interpolate_3d_extrapolate_assert(2, 1, 0, self.extrap_data_lin[2][1][0], 1e-8)

    def test_interpolate_3d_linear_extrapolate_linear_xsupymidzmid(self):
        """3D linear interpolation. Test values in the extrapolation area with x above and y inside and z inside the interpolation area.
        """
        self.init_3dlinear(extrapolate=True, extrapolation_range=10, extrapolation_type='linear')
        self.interpolate_3d_extrapolate_assert(2, 1, 1, self.extrap_data_lin[2][1][1], 1e-8)

    def test_interpolate_3d_linear_extrapolate_linear_xsupymidzsup(self):
        """3D linear interpolation. Test values in the extrapolation area with x above and y inside and z above the interpolation area.
        """
        self.init_3dlinear(extrapolate=True, extrapolation_range=10, extrapolation_type='linear')
        self.interpolate_3d_extrapolate_assert(2, 1, 2, self.extrap_data_lin[2][1][2], 1e-8)

    def test_interpolate_3d_linear_extrapolate_linear_xsupysupzinf(self):
        """3D linear interpolation. Test values in the extrapolation area with x above and y above and z below the interpolation area.
        """
        self.init_3dlinear(extrapolate=True, extrapolation_range=10, extrapolation_type='linear')
        self.interpolate_3d_extrapolate_assert(2, 2, 0, self.extrap_data_lin[2][2][0], 1e-8)

    def test_interpolate_3d_linear_extrapolate_linear_xsupysupzmid(self):
        """3D linear interpolation. Test values in the extrapolation area with x above and y above and z inside the interpolation area.
        """
        self.init_3dlinear(extrapolate=True, extrapolation_range=10, extrapolation_type='linear')
        self.interpolate_3d_extrapolate_assert(2, 2, 1, self.extrap_data_lin[2][2][1], 1e-8)

    def test_interpolate_3d_linear_extrapolate_linear_xsupysupzsup(self):
        """3D linear interpolation. Test values in the extrapolation area with x above and y above and z above the interpolation area.
        """
        self.init_3dlinear(extrapolate=True, extrapolation_range=10, extrapolation_type='linear')
        self.interpolate_3d_extrapolate_assert(2, 2, 2, self.extrap_data_lin[2][2][2], 1e-8)

    def test_interpolate_3d_linear_type_conversion(self):
        """3D linear interpolation. Whatever the type of input data, the interpolating function must provide float numbers.
        """
        self.init_3dlinear([1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], np.ones((4, 4, 4), dtype=int))
        self.assertIsInstance(self.interp_func(2.5, 4.5, 3.5), float)

    def test_interpolate_3d_linear_single_value_tolerated_x(self):
        """3D linear interpolation. If tolerated, a single value input must be extrapolated to every real value on its axis.
        """
        self.init_3dlinear([2.], [2, 3, 4, 5], [3, 4, 5, 6], np.ones((1, 4, 4)), tolerate_single_value=True)
        self.assertAlmostEqual(self.interp_func(-31946139.346, 2.5, 5.6), 1., delta=1e-8)
        self.assertAlmostEqual(self.interp_func(31946139.346, 2.5, 5.6), 1., delta=1e-8)
        self.assertAlmostEqual(self.interp_func(2., 2.5, 5.6), 1., delta=1e-8)

    def test_interpolate_3d_linear_single_value_tolerated_y(self):
        """3D linear interpolation. If tolerated, a single value input must be extrapolated to every real value on its axis.
        """
        self.init_3dlinear([1, 2, 3, 4], [4.], [3, 4, 5, 6], np.ones((4, 1, 4)), tolerate_single_value=True)
        self.assertAlmostEqual(self.interp_func(1.5, -31946139.346, 5.6), 1., delta=1e-8)
        self.assertAlmostEqual(self.interp_func(1.5, 31946139.346, 5.6), 1., delta=1e-8)
        self.assertAlmostEqual(self.interp_func(1.5, 4., 5.6), 1., delta=1e-8)

    def test_interpolate_3d_linear_single_value_tolerated_z(self):
        """3D linear interpolation. If tolerated, a single value input must be extrapolated to every real value on its axis.
        """
        self.init_3dlinear([1, 2, 3, 4], [2, 3, 4, 5], [4.], np.ones((4, 4, 1)), tolerate_single_value=True)
        self.assertAlmostEqual(self.interp_func(1.5, 2.5, -31946139.346), 1., delta=1e-8)
        self.assertAlmostEqual(self.interp_func(1.5, 2.5, 31946139.346), 1., delta=1e-8)
        self.assertAlmostEqual(self.interp_func(2., 2.5, 4.), 1., delta=1e-8)

    def test_interpolate_3d_linear_single_value_tolerated_xy(self):
        """3D linear interpolation. If tolerated, a single value input must be extrapolated to every real value on its axis.
        """
        self.init_3dlinear([2.], [4.], [3, 4, 5, 6], np.ones((1, 1, 4)), tolerate_single_value=True)
        self.assertAlmostEqual(self.interp_func(-31946139.346, 7856.45, 5.6), 1., delta=1e-8)
        self.assertAlmostEqual(self.interp_func(31946139.346, -7856.45, 5.6), 1., delta=1e-8)
        self.assertAlmostEqual(self.interp_func(2., 4., 5.6), 1., delta=1e-8)

    def test_interpolate_3d_linear_single_value_tolerated_yz(self):
        """3D linear interpolation. If tolerated, a single value input must be extrapolated to every real value on its axis.
        """
        self.init_3dlinear([1, 2, 3, 4], [4.], [5.], np.ones((4, 1, 1)), tolerate_single_value=True)
        self.assertAlmostEqual(self.interp_func(1.5, -31946139.346, 7856.45), 1., delta=1e-8)
        self.assertAlmostEqual(self.interp_func(1.5, 31946139.346, -7856.45), 1., delta=1e-8)
        self.assertAlmostEqual(self.interp_func(1.5, 4., 5.), 1., delta=1e-8)

    def test_interpolate_3d_linear_single_value_tolerated_xz(self):
        """3D linear interpolation. If tolerated, a single value input must be extrapolated to every real value on its axis.
        """
        self.init_3dlinear([2.], [2, 3, 4, 5], [5.], np.ones((1, 4, 1)), tolerate_single_value=True)
        self.assertAlmostEqual(self.interp_func(-31946139.346, 2.5, 7856.45), 1., delta=1e-8)
        self.assertAlmostEqual(self.interp_func(31946139.346, 2.5, -7856.45), 1., delta=1e-8)
        self.assertAlmostEqual(self.interp_func(2., 2.5, 5.), 1., delta=1e-8)

    def test_interpolate_3d_linear_single_value_tolerated_xyz(self):
        """3D linear interpolation. If tolerated, a single value input must be extrapolated to every real value on its axis.
        """
        self.init_3dlinear([2.], [4.], [5.], np.ones((1, 1, 1)), tolerate_single_value=True)
        self.assertAlmostEqual(self.interp_func(-31946139.346, 7856.45, 364646.43), 1., delta=1e-8)
        self.assertAlmostEqual(self.interp_func(31946139.346, -7856.45, 364646.43), 1., delta=1e-8)
        self.assertAlmostEqual(self.interp_func(2., 4., 5.), 1., delta=1e-8)

    # Cubic behaviour

    def test_interpolate_3d_cubic(self):
        """3D cubic interpolation. Test values inside the boundaries"""
        self.init_3dcubic()
        for i in range(len(self.xsamples)):
            for j in range(len(self.ysamples)):
                for k in range(len(self.zsamples)):
                    self.assertAlmostEqual(self.interp_func(self.xsamples[i], self.ysamples[j], self.zsamples[k]),
                                           self.interp_data[i, j, k], delta=1e-8)

    def test_interpolate_3d_cubic_bigvalues(self):
        """3D cubic interpolation. Test with big values (1e20) inside the boundaries"""
        factor = 1.e20
        self.init_3dcubic(data=factor * self.data)
        for i in range(len(self.xsamples)):
            for j in range(len(self.ysamples)):
                for k in range(len(self.zsamples)):
                    self.assertAlmostEqual(self.interp_func(self.xsamples[i], self.ysamples[j], self.zsamples[k]),
                                           factor * self.interp_data[i, j, k], delta=factor * 1e-8)

    def test_interpolate_3d_cubic_lowvalues(self):
        """3D cubic interpolation. Test with low values (1e-20) inside the boundaries"""
        factor = 1.e-20
        self.init_3dcubic(data=factor * self.data)
        for i in range(len(self.xsamples)):
            for j in range(len(self.ysamples)):
                for k in range(len(self.zsamples)):
                    self.assertAlmostEqual(self.interp_func(self.xsamples[i], self.ysamples[j], self.zsamples[k]),
                                           factor * self.interp_data[i, j, k], delta=factor * 1e-8)

    def test_interpolate_3d_cubic_edge(self):
        """3D cubic interpolation. Test edges values"""
        self.init_3dcubic()
        for i in range(len(self.xsamples)):
            for j in range(len(self.ysamples)):
                self.assertAlmostEqual(self.interp_func(self.xsamples[i], self.ysamples[j], self.zsamples[0]),
                                       self.interp_data[i, j, 0], delta=1e-8)
                self.assertAlmostEqual(self.interp_func(self.xsamples[i], self.ysamples[j], self.zsamples[-1]),
                                       self.interp_data[i, j, -1], delta=1e-8)
        for j in range(len(self.ysamples)):
            for k in range(len(self.zsamples)):
                self.assertAlmostEqual(self.interp_func(self.xsamples[0], self.ysamples[j], self.zsamples[k]),
                                       self.interp_data[0, j, k], delta=1e-8)
                self.assertAlmostEqual(self.interp_func(self.xsamples[-1], self.ysamples[j], self.zsamples[k]),
                                       self.interp_data[-1, j, k], delta=1e-8)
        for k in range(len(self.zsamples)):
            for i in range(len(self.xsamples)):
                self.assertAlmostEqual(self.interp_func(self.xsamples[i], self.ysamples[0], self.zsamples[k]),
                                       self.interp_data[i, 0, k], delta=1e-8)
                self.assertAlmostEqual(self.interp_func(self.xsamples[i], self.ysamples[-1], self.zsamples[k]),
                                       self.interp_data[i, -1, k], delta=1e-8)

    def test_interpolate_3d_cubic_knot(self):
        """3D cubic interpolation. Test knot values"""
        self.init_3dcubic()
        for i in range(len(self.x)):
            for j in range(len(self.y)):
                for k in range(len(self.z)):
                    self.assertAlmostEqual(self.interp_func(self.x[i], self.y[j], self.z[k]),
                                           self.data[i, j, k], delta=1e-9)

    def test_interpolate_3d_cubic_out(self):
        """3D cubic interpolation. Test values outside the boundaries"""
        self.init_3dcubic(extrapolate=False)
        for i in range(len(self.xsamples_out)):
            for j in range(len(self.ysamples_out)):
                self.assertRaises(ValueError, self.interp_func, self.xsamples_out[i], self.ysamples_out[j], self.zsamples_out[0])
                self.assertRaises(ValueError, self.interp_func, self.xsamples_out[i], self.ysamples_out[j], self.zsamples_out[-1])
        self.interpolate_3d_xboundaries_assert(X_LOWER, X_UPPER, 1e-6, (Y_LOWER + Y_UPPER) / 2, (Z_LOWER + Z_UPPER) / 2)
        for j in range(len(self.ysamples_out)):
            for k in range(len(self.zsamples_out)):
                self.assertRaises(ValueError, self.interp_func, self.xsamples_out[0], self.ysamples_out[j], self.zsamples_out[k])
                self.assertRaises(ValueError, self.interp_func, self.xsamples_out[-1], self.ysamples_out[j], self.zsamples_out[k])
        self.interpolate_3d_yboundaries_assert((Y_LOWER + Y_UPPER) / 2, Y_LOWER, Y_UPPER, 1e-6, (Z_LOWER + Z_UPPER) / 2)
        for k in range(len(self.zsamples_out)):
            for i in range(len(self.xsamples_out)):
                self.assertRaises(ValueError, self.interp_func, self.xsamples_out[i], self.ysamples_out[0], self.zsamples_out[k])
                self.assertRaises(ValueError, self.interp_func, self.xsamples_out[i], self.ysamples_out[-1], self.zsamples_out[k])
        self.interpolate_3d_zboundaries_assert((Y_LOWER + Y_UPPER) / 2, (Z_LOWER + Z_UPPER) / 2, Z_LOWER, Z_UPPER, 1e-6)

    def test_interpolate_3d_cubic_extrapolate_nearest_range(self):
        """3D cubic interpolation. Tests the size of the extrapolation range."""
        self.init_3dcubic(extrapolate=True, extrapolation_type='nearest', extrapolation_range=1.)
        self.interpolate_3d_xboundaries_assert(X_LOWER - 1, X_UPPER + 1, 1e-6, (Y_LOWER + Y_UPPER) / 2, (Z_LOWER + Z_UPPER) / 2)
        self.interpolate_3d_yboundaries_assert((X_LOWER + X_UPPER) / 2, Y_LOWER - 1, Y_UPPER + 1, 1e-6, (Z_LOWER + Z_UPPER) / 2)
        self.interpolate_3d_zboundaries_assert((X_LOWER + X_UPPER) / 2, (Y_LOWER + Y_UPPER) / 2, Z_LOWER - 1, Z_UPPER + 1, 1e-6)

    def test_interpolate_3d_cubic_extrapolate_nearest_xinfyinfzinf(self):
        """3D cubic interpolation. Test values in the extrapolation area with x below and y below and z below the interpolation area.
        """
        self.init_3dcubic(extrapolate=True, extrapolation_range=10, extrapolation_type='nearest')
        self.interpolate_3d_extrapolate_assert(0, 0, 0, self.extrap_data_nea[0][0][0], 1e-8)

    def test_interpolate_3d_cubic_extrapolate_nearest_xinfyinfzmid(self):
        """3D cubic interpolation. Test values in the extrapolation area with x below and y below and z inside the interpolation area.
        """
        self.init_3dcubic(extrapolate=True, extrapolation_range=10, extrapolation_type='nearest')
        self.interpolate_3d_extrapolate_assert(0, 0, 1, self.extrap_data_nea[0][0][1], 1e-8)

    def test_interpolate_3d_cubic_extrapolate_nearest_xinfyinfzsup(self):
        """3D cubic interpolation. Test values in the extrapolation area with x below and y below and z above the interpolation area.
        """
        self.init_3dcubic(extrapolate=True, extrapolation_range=10, extrapolation_type='nearest')
        self.interpolate_3d_extrapolate_assert(0, 0, 2, self.extrap_data_nea[0][0][2], 1e-8)

    def test_interpolate_3d_cubic_extrapolate_nearest_xinfymidzinf(self):
        """3D cubic interpolation. Test values in the extrapolation area with x below and y inside and z below the interpolation area.
        """
        self.init_3dcubic(extrapolate=True, extrapolation_range=10, extrapolation_type='nearest')
        self.interpolate_3d_extrapolate_assert(0, 1, 0, self.extrap_data_nea[0][1][0], 1e-8)

    def test_interpolate_3d_cubic_extrapolate_nearest_xinfymidzmid(self):
        """3D cubic interpolation. Test values in the extrapolation area with x below and y inside and z inside the interpolation area.
        """
        self.init_3dcubic(extrapolate=True, extrapolation_range=10, extrapolation_type='nearest')
        self.interpolate_3d_extrapolate_assert(0, 1, 1, self.extrap_data_nea[0][1][1], 1e-8)

    def test_interpolate_3d_cubic_extrapolate_nearest_xinfymidzsup(self):
        """3D cubic interpolation. Test values in the extrapolation area with x below and y inside and z above the interpolation area.
        """
        self.init_3dcubic(extrapolate=True, extrapolation_range=10, extrapolation_type='nearest')
        self.interpolate_3d_extrapolate_assert(0, 1, 2, self.extrap_data_nea[0][1][2], 1e-8)

    def test_interpolate_3d_cubic_extrapolate_nearest_xinfysupzinf(self):
        """3D cubic interpolation. Test values in the extrapolation area with x below and y above and z below the interpolation area.
        """
        self.init_3dcubic(extrapolate=True, extrapolation_range=10, extrapolation_type='nearest')
        self.interpolate_3d_extrapolate_assert(0, 2, 0, self.extrap_data_nea[0][2][0], 1e-8)

    def test_interpolate_3d_cubic_extrapolate_nearest_xinfysupzmid(self):
        """3D cubic interpolation. Test values in the extrapolation area with x below and y above and z inside the interpolation area.
        """
        self.init_3dcubic(extrapolate=True, extrapolation_range=10, extrapolation_type='nearest')
        self.interpolate_3d_extrapolate_assert(0, 2, 1, self.extrap_data_nea[0][2][1], 1e-8)

    def test_interpolate_3d_cubic_extrapolate_nearest_xinfysupzsup(self):
        """3D cubic interpolation. Test values in the extrapolation area with x below and y above and z above the interpolation area.
        """
        self.init_3dcubic(extrapolate=True, extrapolation_range=10, extrapolation_type='nearest')
        self.interpolate_3d_extrapolate_assert(0, 2, 2, self.extrap_data_nea[0][2][2], 1e-8)

    def test_interpolate_3d_cubic_extrapolate_nearest_xmidyinfzinf(self):
        """3D cubic interpolation. Test values in the extrapolation area with x inside and y below and z below the interpolation area.
        """
        self.init_3dcubic(extrapolate=True, extrapolation_range=10, extrapolation_type='nearest')
        self.interpolate_3d_extrapolate_assert(1, 0, 0, self.extrap_data_nea[1][0][0], 1e-8)

    def test_interpolate_3d_cubic_extrapolate_nearest_xmidyinfzmid(self):
        """3D cubic interpolation. Test values in the extrapolation area with x inside and y below and z inside the interpolation area.
        """
        self.init_3dcubic(extrapolate=True, extrapolation_range=10, extrapolation_type='nearest')
        self.interpolate_3d_extrapolate_assert(1, 0, 1, self.extrap_data_nea[1][0][1], 1e-8)

    def test_interpolate_3d_cubic_extrapolate_nearest_xmidyinfzsup(self):
        """3D cubic interpolation. Test values in the extrapolation area with x inside and y below and z above the interpolation area.
        """
        self.init_3dcubic(extrapolate=True, extrapolation_range=10, extrapolation_type='nearest')
        self.interpolate_3d_extrapolate_assert(1, 0, 2, self.extrap_data_nea[1][0][2], 1e-8)

    def test_interpolate_3d_cubic_extrapolate_nearest_xmidymidzinf(self):
        """3D cubic interpolation. Test values in the extrapolation area with x inside and y inside and z below the interpolation area.
        """
        self.init_3dcubic(extrapolate=True, extrapolation_range=10, extrapolation_type='nearest')
        self.interpolate_3d_extrapolate_assert(1, 1, 0, self.extrap_data_nea[1][1][0], 1e-8)

    def test_interpolate_3d_cubic_extrapolate_nearest_xmidymidzsup(self):
        """3D cubic interpolation. Test values in the extrapolation area with x inside and y inside and z above the interpolation area.
        """
        self.init_3dcubic(extrapolate=True, extrapolation_range=10, extrapolation_type='nearest')
        self.interpolate_3d_extrapolate_assert(1, 1, 2, self.extrap_data_nea[1][1][2], 1e-8)

    def test_interpolate_3d_cubic_extrapolate_nearest_xmidysupzinf(self):
        """3D cubic interpolation. Test values in the extrapolation area with x inside and y above and z below the interpolation area.
        """
        self.init_3dcubic(extrapolate=True, extrapolation_range=10, extrapolation_type='nearest')
        self.interpolate_3d_extrapolate_assert(1, 2, 0, self.extrap_data_nea[1][2][0], 1e-8)

    def test_interpolate_3d_cubic_extrapolate_nearest_xmidysupzmid(self):
        """3D cubic interpolation. Test values in the extrapolation area with x inside and y above and z inside the interpolation area.
        """
        self.init_3dcubic(extrapolate=True, extrapolation_range=10, extrapolation_type='nearest')
        self.interpolate_3d_extrapolate_assert(1, 2, 1, self.extrap_data_nea[1][2][1], 1e-8)

    def test_interpolate_3d_cubic_extrapolate_nearest_xmidysupzsup(self):
        """3D cubic interpolation. Test values in the extrapolation area with x inside and y above and z above the interpolation area.
        """
        self.init_3dcubic(extrapolate=True, extrapolation_range=10, extrapolation_type='nearest')
        self.interpolate_3d_extrapolate_assert(1, 2, 2, self.extrap_data_nea[1][2][2], 1e-8)

    def test_interpolate_3d_cubic_extrapolate_nearest_xsupyinfzinf(self):
        """3D cubic interpolation. Test values in the extrapolation area with x above and y below and z below the interpolation area.
        """
        self.init_3dcubic(extrapolate=True, extrapolation_range=10, extrapolation_type='nearest')
        self.interpolate_3d_extrapolate_assert(2, 0, 0, self.extrap_data_nea[2][0][0], 1e-8)

    def test_interpolate_3d_cubic_extrapolate_nearest_xsupyinfzmid(self):
        """3D cubic interpolation. Test values in the extrapolation area with x above and y below and z inside the interpolation area.
        """
        self.init_3dcubic(extrapolate=True, extrapolation_range=10, extrapolation_type='nearest')
        self.interpolate_3d_extrapolate_assert(2, 0, 1, self.extrap_data_nea[2][0][1], 1e-8)

    def test_interpolate_3d_cubic_extrapolate_nearest_xsupyinfzsup(self):
        """3D cubic interpolation. Test values in the extrapolation area with x above and y below and z above the interpolation area.
        """
        self.init_3dcubic(extrapolate=True, extrapolation_range=10, extrapolation_type='nearest')
        self.interpolate_3d_extrapolate_assert(2, 0, 2, self.extrap_data_nea[2][0][2], 1e-8)

    def test_interpolate_3d_cubic_extrapolate_nearest_xsupymidzinf(self):
        """3D cubic interpolation. Test values in the extrapolation area with x above and y inside and z below the interpolation area.
        """
        self.init_3dcubic(extrapolate=True, extrapolation_range=10, extrapolation_type='nearest')
        self.interpolate_3d_extrapolate_assert(2, 1, 0, self.extrap_data_nea[2][1][0], 1e-8)

    def test_interpolate_3d_cubic_extrapolate_nearest_xsupymidzmid(self):
        """3D cubic interpolation. Test values in the extrapolation area with x above and y inside and z inside the interpolation area.
        """
        self.init_3dcubic(extrapolate=True, extrapolation_range=10, extrapolation_type='nearest')
        self.interpolate_3d_extrapolate_assert(2, 1, 1, self.extrap_data_nea[2][1][1], 1e-8)

    def test_interpolate_3d_cubic_extrapolate_nearest_xsupymidzsup(self):
        """3D cubic interpolation. Test values in the extrapolation area with x above and y inside and z above the interpolation area.
        """
        self.init_3dcubic(extrapolate=True, extrapolation_range=10, extrapolation_type='nearest')
        self.interpolate_3d_extrapolate_assert(2, 1, 2, self.extrap_data_nea[2][1][2], 1e-8)

    def test_interpolate_3d_cubic_extrapolate_nearest_xsupysupzinf(self):
        """3D cubic interpolation. Test values in the extrapolation area with x above and y above and z below the interpolation area.
        """
        self.init_3dcubic(extrapolate=True, extrapolation_range=10, extrapolation_type='nearest')
        self.interpolate_3d_extrapolate_assert(2, 2, 0, self.extrap_data_nea[2][2][0], 1e-8)

    def test_interpolate_3d_cubic_extrapolate_nearest_xsupysupzmid(self):
        """3D cubic interpolation. Test values in the extrapolation area with x above and y above and z inside the interpolation area.
        """
        self.init_3dcubic(extrapolate=True, extrapolation_range=10, extrapolation_type='nearest')
        self.interpolate_3d_extrapolate_assert(2, 2, 1, self.extrap_data_nea[2][2][1], 1e-8)

    def test_interpolate_3d_cubic_extrapolate_nearest_xsupysupzsup(self):
        """3D cubic interpolation. Test values in the extrapolation area with x above and y above and z above the interpolation area.
        """
        self.init_3dcubic(extrapolate=True, extrapolation_range=10, extrapolation_type='nearest')
        self.interpolate_3d_extrapolate_assert(2, 2, 2, self.extrap_data_nea[2][2][2], 1e-8)

    def test_interpolate_3d_cubic_extrapolate_linear_range(self):
        """3D cubic interpolation. Tests the size of the extrapolation range."""
        self.init_3dcubic(extrapolate=True, extrapolation_type='linear', extrapolation_range=1.)
        self.interpolate_3d_xboundaries_assert(X_LOWER - 1, X_UPPER + 1, 1e-6, (Y_LOWER + Y_UPPER) / 2, (Z_LOWER + Z_UPPER) / 2)
        self.interpolate_3d_yboundaries_assert((X_LOWER + X_UPPER) / 2, Y_LOWER - 1, Y_UPPER + 1, 1e-6, (Z_LOWER + Z_UPPER) / 2)
        self.interpolate_3d_zboundaries_assert((X_LOWER + X_UPPER) / 2, (Y_LOWER + Y_UPPER) / 2, Z_LOWER - 1, Z_UPPER + 1, 1e-6)

    def test_interpolate_3d_cubic_extrapolate_linear_xinfyinfzinf(self):
        """3D cubic interpolation. Test values in the extrapolation area with x below and y below and z below the interpolation area.
        """
        self.init_3dcubic(extrapolate=True, extrapolation_range=10, extrapolation_type='linear')
        self.interpolate_3d_extrapolate_assert(0, 0, 0, self.extrap_data_lin[0][0][0], 1e-8)

    def test_interpolate_3d_cubic_extrapolate_linear_xinfyinfzmid(self):
        """3D cubic interpolation. Test values in the extrapolation area with x below and y below and z inside the interpolation area.
        """
        self.init_3dcubic(extrapolate=True, extrapolation_range=10, extrapolation_type='linear')
        self.interpolate_3d_extrapolate_assert(0, 0, 1, self.extrap_data_lin[0][0][1], 1e-8)

    def test_interpolate_3d_cubic_extrapolate_linear_xinfyinfzsup(self):
        """3D cubic interpolation. Test values in the extrapolation area with x below and y below and z above the interpolation area.
        """
        self.init_3dcubic(extrapolate=True, extrapolation_range=10, extrapolation_type='linear')
        self.interpolate_3d_extrapolate_assert(0, 0, 2, self.extrap_data_lin[0][0][2], 1e-8)

    def test_interpolate_3d_cubic_extrapolate_linear_xinfymidzinf(self):
        """3D cubic interpolation. Test values in the extrapolation area with x below and y inside and z below the interpolation area.
        """
        self.init_3dcubic(extrapolate=True, extrapolation_range=10, extrapolation_type='linear')
        self.interpolate_3d_extrapolate_assert(0, 1, 0, self.extrap_data_lin[0][1][0], 1e-8)

    def test_interpolate_3d_cubic_extrapolate_linear_xinfymidzmid(self):
        """3D cubic interpolation. Test values in the extrapolation area with x below and y inside and z inside the interpolation area.
        """
        self.init_3dcubic(extrapolate=True, extrapolation_range=10, extrapolation_type='linear')
        self.interpolate_3d_extrapolate_assert(0, 1, 1, self.extrap_data_lin[0][1][1], 1e-8)

    def test_interpolate_3d_cubic_extrapolate_linear_xinfymidzsup(self):
        """3D cubic interpolation. Test values in the extrapolation area with x below and y inside and z above the interpolation area.
        """
        self.init_3dcubic(extrapolate=True, extrapolation_range=10, extrapolation_type='linear')
        self.interpolate_3d_extrapolate_assert(0, 1, 2, self.extrap_data_lin[0][1][2], 1e-8)

    def test_interpolate_3d_cubic_extrapolate_linear_xinfysupzinf(self):
        """3D cubic interpolation. Test values in the extrapolation area with x below and y above and z below the interpolation area.
        """
        self.init_3dcubic(extrapolate=True, extrapolation_range=10, extrapolation_type='linear')
        self.interpolate_3d_extrapolate_assert(0, 2, 0, self.extrap_data_lin[0][2][0], 1e-8)

    def test_interpolate_3d_cubic_extrapolate_linear_xinfysupzmid(self):
        """3D cubic interpolation. Test values in the extrapolation area with x below and y above and z inside the interpolation area.
        """
        self.init_3dcubic(extrapolate=True, extrapolation_range=10, extrapolation_type='linear')
        self.interpolate_3d_extrapolate_assert(0, 2, 1, self.extrap_data_lin[0][2][1], 1e-8)

    def test_interpolate_3d_cubic_extrapolate_linear_xinfysupzsup(self):
        """3D cubic interpolation. Test values in the extrapolation area with x below and y above and z above the interpolation area.
        """
        self.init_3dcubic(extrapolate=True, extrapolation_range=10, extrapolation_type='linear')
        self.interpolate_3d_extrapolate_assert(0, 2, 2, self.extrap_data_lin[0][2][2], 1e-8)

    def test_interpolate_3d_cubic_extrapolate_linear_xmidyinfzinf(self):
        """3D cubic interpolation. Test values in the extrapolation area with x inside and y below and z below the interpolation area.
        """
        self.init_3dcubic(extrapolate=True, extrapolation_range=10, extrapolation_type='linear')
        self.interpolate_3d_extrapolate_assert(1, 0, 0, self.extrap_data_lin[1][0][0], 1e-8)

    def test_interpolate_3d_cubic_extrapolate_linear_xmidyinfzmid(self):
        """3D cubic interpolation. Test values in the extrapolation area with x inside and y below and z inside the interpolation area.
        """
        self.init_3dcubic(extrapolate=True, extrapolation_range=10, extrapolation_type='linear')
        self.interpolate_3d_extrapolate_assert(1, 0, 1, self.extrap_data_lin[1][0][1], 1e-8)

    def test_interpolate_3d_cubic_extrapolate_linear_xmidyinfzsup(self):
        """3D cubic interpolation. Test values in the extrapolation area with x inside and y below and z above the interpolation area.
        """
        self.init_3dcubic(extrapolate=True, extrapolation_range=10, extrapolation_type='linear')
        self.interpolate_3d_extrapolate_assert(1, 0, 2, self.extrap_data_lin[1][0][2], 1e-8)

    def test_interpolate_3d_cubic_extrapolate_linear_xmidymidzinf(self):
        """3D cubic interpolation. Test values in the extrapolation area with x inside and y inside and z below the interpolation area.
        """
        self.init_3dcubic(extrapolate=True, extrapolation_range=10, extrapolation_type='linear')
        self.interpolate_3d_extrapolate_assert(1, 1, 0, self.extrap_data_lin[1][1][0], 1e-8)

    def test_interpolate_3d_cubic_extrapolate_linear_xmidymidzsup(self):
        """3D cubic interpolation. Test values in the extrapolation area with x inside and y inside and z above the interpolation area.
        """
        self.init_3dcubic(extrapolate=True, extrapolation_range=10, extrapolation_type='linear')
        self.interpolate_3d_extrapolate_assert(1, 1, 2, self.extrap_data_lin[1][1][2], 1e-8)

    def test_interpolate_3d_cubic_extrapolate_linear_xmidysupzinf(self):
        """3D cubic interpolation. Test values in the extrapolation area with x inside and y above and z below the interpolation area.
        """
        self.init_3dcubic(extrapolate=True, extrapolation_range=10, extrapolation_type='linear')
        self.interpolate_3d_extrapolate_assert(1, 2, 0, self.extrap_data_lin[1][2][0], 1e-8)

    def test_interpolate_3d_cubic_extrapolate_linear_xmidysupzmid(self):
        """3D cubic interpolation. Test values in the extrapolation area with x inside and y above and z inside the interpolation area.
        """
        self.init_3dcubic(extrapolate=True, extrapolation_range=10, extrapolation_type='linear')
        self.interpolate_3d_extrapolate_assert(1, 2, 1, self.extrap_data_lin[1][2][1], 1e-8)

    def test_interpolate_3d_cubic_extrapolate_linear_xmidysupzsup(self):
        """3D cubic interpolation. Test values in the extrapolation area with x inside and y above and z above the interpolation area.
        """
        self.init_3dcubic(extrapolate=True, extrapolation_range=10, extrapolation_type='linear')
        self.interpolate_3d_extrapolate_assert(1, 2, 2, self.extrap_data_lin[1][2][2], 1e-8)

    def test_interpolate_3d_cubic_extrapolate_linear_xsupyinfzinf(self):
        """3D cubic interpolation. Test values in the extrapolation area with x above and y below and z below the interpolation area.
        """
        self.init_3dcubic(extrapolate=True, extrapolation_range=10, extrapolation_type='linear')
        self.interpolate_3d_extrapolate_assert(2, 0, 0, self.extrap_data_lin[2][0][0], 1e-8)

    def test_interpolate_3d_cubic_extrapolate_linear_xsupyinfzmid(self):
        """3D cubic interpolation. Test values in the extrapolation area with x above and y below and z inside the interpolation area.
        """
        self.init_3dcubic(extrapolate=True, extrapolation_range=10, extrapolation_type='linear')
        self.interpolate_3d_extrapolate_assert(2, 0, 1, self.extrap_data_lin[2][0][1], 1e-8)

    def test_interpolate_3d_cubic_extrapolate_linear_xsupyinfzsup(self):
        """3D cubic interpolation. Test values in the extrapolation area with x above and y below and z above the interpolation area.
        """
        self.init_3dcubic(extrapolate=True, extrapolation_range=10, extrapolation_type='linear')
        self.interpolate_3d_extrapolate_assert(2, 0, 2, self.extrap_data_lin[2][0][2], 1e-8)

    def test_interpolate_3d_cubic_extrapolate_linear_xsupymidzinf(self):
        """3D cubic interpolation. Test values in the extrapolation area with x above and y inside and z below the interpolation area.
        """
        self.init_3dcubic(extrapolate=True, extrapolation_range=10, extrapolation_type='linear')
        self.interpolate_3d_extrapolate_assert(2, 1, 0, self.extrap_data_lin[2][1][0], 1e-8)

    def test_interpolate_3d_cubic_extrapolate_linear_xsupymidzmid(self):
        """3D cubic interpolation. Test values in the extrapolation area with x above and y inside and z inside the interpolation area.
        """
        self.init_3dcubic(extrapolate=True, extrapolation_range=10, extrapolation_type='linear')
        self.interpolate_3d_extrapolate_assert(2, 1, 1, self.extrap_data_lin[2][1][1], 1e-8)

    def test_interpolate_3d_cubic_extrapolate_linear_xsupymidzsup(self):
        """3D cubic interpolation. Test values in the extrapolation area with x above and y inside and z above the interpolation area.
        """
        self.init_3dcubic(extrapolate=True, extrapolation_range=10, extrapolation_type='linear')
        self.interpolate_3d_extrapolate_assert(2, 1, 2, self.extrap_data_lin[2][1][2], 1e-8)

    def test_interpolate_3d_cubic_extrapolate_linear_xsupysupzinf(self):
        """3D cubic interpolation. Test values in the extrapolation area with x above and y above and z below the interpolation area.
        """
        self.init_3dcubic(extrapolate=True, extrapolation_range=10, extrapolation_type='linear')
        self.interpolate_3d_extrapolate_assert(2, 2, 0, self.extrap_data_lin[2][2][0], 1e-8)

    def test_interpolate_3d_cubic_extrapolate_linear_xsupysupzmid(self):
        """3D cubic interpolation. Test values in the extrapolation area with x above and y above and z inside the interpolation area.
        """
        self.init_3dcubic(extrapolate=True, extrapolation_range=10, extrapolation_type='linear')
        self.interpolate_3d_extrapolate_assert(2, 2, 1, self.extrap_data_lin[2][2][1], 1e-8)

    def test_interpolate_3d_cubic_extrapolate_linear_xsupysupzsup(self):
        """3D cubic interpolation. Test values in the extrapolation area with x above and y above and z above the interpolation area.
        """
        self.init_3dcubic(extrapolate=True, extrapolation_range=10, extrapolation_type='linear')
        self.interpolate_3d_extrapolate_assert(2, 2, 2, self.extrap_data_lin[2][2][2], 1e-8)

    def test_interpolate_3d_cubic_extrapolate_quadratic_range(self):
        """3D cubic interpolation. Tests the size of the extrapolation range."""
        self.init_3dcubic(extrapolate=True, extrapolation_type='quadratic', extrapolation_range=1.)
        self.interpolate_3d_xboundaries_assert(X_LOWER - 1, X_UPPER + 1, 1e-6, (Y_LOWER + Y_UPPER) / 2, (Z_LOWER + Z_UPPER) / 2)
        self.interpolate_3d_yboundaries_assert((X_LOWER + X_UPPER) / 2, Y_LOWER - 1, Y_UPPER + 1, 1e-6, (Z_LOWER + Z_UPPER) / 2)
        self.interpolate_3d_zboundaries_assert((X_LOWER + X_UPPER) / 2, (Y_LOWER + Y_UPPER) / 2, Z_LOWER - 1, Z_UPPER + 1, 1e-6)

    def test_interpolate_3d_cubic_extrapolate_quadratic_xinfyinfzinf(self):
        """3D cubic interpolation. Test values in the extrapolation area with x below and y below and z below the interpolation area.
        """
        self.init_3dcubic(extrapolate=True, extrapolation_range=10, extrapolation_type='quadratic')
        self.interpolate_3d_extrapolate_assert(0, 0, 0, self.extrap_data_qua[0][0][0], 1e-8)

    def test_interpolate_3d_cubic_extrapolate_quadratic_xinfyinfzmid(self):
        """3D cubic interpolation. Test values in the extrapolation area with x below and y below and z inside the interpolation area.
        """
        self.init_3dcubic(extrapolate=True, extrapolation_range=10, extrapolation_type='quadratic')
        self.interpolate_3d_extrapolate_assert(0, 0, 1, self.extrap_data_qua[0][0][1], 1e-8)

    def test_interpolate_3d_cubic_extrapolate_quadratic_xinfyinfzsup(self):
        """3D cubic interpolation. Test values in the extrapolation area with x below and y below and z above the interpolation area.
        """
        self.init_3dcubic(extrapolate=True, extrapolation_range=10, extrapolation_type='quadratic')
        self.interpolate_3d_extrapolate_assert(0, 0, 2, self.extrap_data_qua[0][0][2], 1e-8)

    def test_interpolate_3d_cubic_extrapolate_quadratic_xinfymidzinf(self):
        """3D cubic interpolation. Test values in the extrapolation area with x below and y inside and z below the interpolation area.
        """
        self.init_3dcubic(extrapolate=True, extrapolation_range=10, extrapolation_type='quadratic')
        self.interpolate_3d_extrapolate_assert(0, 1, 0, self.extrap_data_qua[0][1][0], 1e-8)

    def test_interpolate_3d_cubic_extrapolate_quadratic_xinfymidzmid(self):
        """3D cubic interpolation. Test values in the extrapolation area with x below and y inside and z inside the interpolation area.
        """
        self.init_3dcubic(extrapolate=True, extrapolation_range=10, extrapolation_type='quadratic')
        self.interpolate_3d_extrapolate_assert(0, 1, 1, self.extrap_data_qua[0][1][1], 1e-8)

    def test_interpolate_3d_cubic_extrapolate_quadratic_xinfymidzsup(self):
        """3D cubic interpolation. Test values in the extrapolation area with x below and y inside and z above the interpolation area.
        """
        self.init_3dcubic(extrapolate=True, extrapolation_range=10, extrapolation_type='quadratic')
        self.interpolate_3d_extrapolate_assert(0, 1, 2, self.extrap_data_qua[0][1][2], 1e-8)

    def test_interpolate_3d_cubic_extrapolate_quadratic_xinfysupzinf(self):
        """3D cubic interpolation. Test values in the extrapolation area with x below and y above and z below the interpolation area.
        """
        self.init_3dcubic(extrapolate=True, extrapolation_range=10, extrapolation_type='quadratic')
        self.interpolate_3d_extrapolate_assert(0, 2, 0, self.extrap_data_qua[0][2][0], 1e-8)

    def test_interpolate_3d_cubic_extrapolate_quadratic_xinfysupzmid(self):
        """3D cubic interpolation. Test values in the extrapolation area with x below and y above and z inside the interpolation area.
        """
        self.init_3dcubic(extrapolate=True, extrapolation_range=10, extrapolation_type='quadratic')
        self.interpolate_3d_extrapolate_assert(0, 2, 1, self.extrap_data_qua[0][2][1], 1e-8)

    def test_interpolate_3d_cubic_extrapolate_quadratic_xinfysupzsup(self):
        """3D cubic interpolation. Test values in the extrapolation area with x below and y above and z above the interpolation area.
        """
        self.init_3dcubic(extrapolate=True, extrapolation_range=10, extrapolation_type='quadratic')
        self.interpolate_3d_extrapolate_assert(0, 2, 2, self.extrap_data_qua[0][2][2], 1e-8)

    def test_interpolate_3d_cubic_extrapolate_quadratic_xmidyinfzinf(self):
        """3D cubic interpolation. Test values in the extrapolation area with x inside and y below and z below the interpolation area.
        """
        self.init_3dcubic(extrapolate=True, extrapolation_range=10, extrapolation_type='quadratic')
        self.interpolate_3d_extrapolate_assert(1, 0, 0, self.extrap_data_qua[1][0][0], 1e-8)

    def test_interpolate_3d_cubic_extrapolate_quadratic_xmidyinfzmid(self):
        """3D cubic interpolation. Test values in the extrapolation area with x inside and y below and z inside the interpolation area.
        """
        self.init_3dcubic(extrapolate=True, extrapolation_range=10, extrapolation_type='quadratic')
        self.interpolate_3d_extrapolate_assert(1, 0, 1, self.extrap_data_qua[1][0][1], 1e-8)

    def test_interpolate_3d_cubic_extrapolate_quadratic_xmidyinfzsup(self):
        """3D cubic interpolation. Test values in the extrapolation area with x inside and y below and z above the interpolation area.
        """
        self.init_3dcubic(extrapolate=True, extrapolation_range=10, extrapolation_type='quadratic')
        self.interpolate_3d_extrapolate_assert(1, 0, 2, self.extrap_data_qua[1][0][2], 1e-8)

    def test_interpolate_3d_cubic_extrapolate_quadratic_xmidymidzinf(self):
        """3D cubic interpolation. Test values in the extrapolation area with x inside and y inside and z below the interpolation area.
        """
        self.init_3dcubic(extrapolate=True, extrapolation_range=10, extrapolation_type='quadratic')
        self.interpolate_3d_extrapolate_assert(1, 1, 0, self.extrap_data_qua[1][1][0], 1e-8)

    def test_interpolate_3d_cubic_extrapolate_quadratic_xmidymidzsup(self):
        """3D cubic interpolation. Test values in the extrapolation area with x inside and y inside and z above the interpolation area.
        """
        self.init_3dcubic(extrapolate=True, extrapolation_range=10, extrapolation_type='quadratic')
        self.interpolate_3d_extrapolate_assert(1, 1, 2, self.extrap_data_qua[1][1][2], 1e-8)

    def test_interpolate_3d_cubic_extrapolate_quadratic_xmidysupzinf(self):
        """3D cubic interpolation. Test values in the extrapolation area with x inside and y above and z below the interpolation area.
        """
        self.init_3dcubic(extrapolate=True, extrapolation_range=10, extrapolation_type='quadratic')
        self.interpolate_3d_extrapolate_assert(1, 2, 0, self.extrap_data_qua[1][2][0], 1e-8)

    def test_interpolate_3d_cubic_extrapolate_quadratic_xmidysupzmid(self):
        """3D cubic interpolation. Test values in the extrapolation area with x inside and y above and z inside the interpolation area.
        """
        self.init_3dcubic(extrapolate=True, extrapolation_range=10, extrapolation_type='quadratic')
        self.interpolate_3d_extrapolate_assert(1, 2, 1, self.extrap_data_qua[1][2][1], 1e-8)

    def test_interpolate_3d_cubic_extrapolate_quadratic_xmidysupzsup(self):
        """3D cubic interpolation. Test values in the extrapolation area with x inside and y above and z above the interpolation area.
        """
        self.init_3dcubic(extrapolate=True, extrapolation_range=10, extrapolation_type='quadratic')
        self.interpolate_3d_extrapolate_assert(1, 2, 2, self.extrap_data_qua[1][2][2], 1e-8)

    def test_interpolate_3d_cubic_extrapolate_quadratic_xsupyinfzinf(self):
        """3D cubic interpolation. Test values in the extrapolation area with x above and y below and z below the interpolation area.
        """
        self.init_3dcubic(extrapolate=True, extrapolation_range=10, extrapolation_type='quadratic')
        self.interpolate_3d_extrapolate_assert(2, 0, 0, self.extrap_data_qua[2][0][0], 1e-8)

    def test_interpolate_3d_cubic_extrapolate_quadratic_xsupyinfzmid(self):
        """3D cubic interpolation. Test values in the extrapolation area with x above and y below and z inside the interpolation area.
        """
        self.init_3dcubic(extrapolate=True, extrapolation_range=10, extrapolation_type='quadratic')
        self.interpolate_3d_extrapolate_assert(2, 0, 1, self.extrap_data_qua[2][0][1], 1e-8)

    def test_interpolate_3d_cubic_extrapolate_quadratic_xsupyinfzsup(self):
        """3D cubic interpolation. Test values in the extrapolation area with x above and y below and z above the interpolation area.
        """
        self.init_3dcubic(extrapolate=True, extrapolation_range=10, extrapolation_type='quadratic')
        self.interpolate_3d_extrapolate_assert(2, 0, 2, self.extrap_data_qua[2][0][2], 1e-8)

    def test_interpolate_3d_cubic_extrapolate_quadratic_xsupymidzinf(self):
        """3D cubic interpolation. Test values in the extrapolation area with x above and y inside and z below the interpolation area.
        """
        self.init_3dcubic(extrapolate=True, extrapolation_range=10, extrapolation_type='quadratic')
        self.interpolate_3d_extrapolate_assert(2, 1, 0, self.extrap_data_qua[2][1][0], 1e-8)

    def test_interpolate_3d_cubic_extrapolate_quadratic_xsupymidzmid(self):
        """3D cubic interpolation. Test values in the extrapolation area with x above and y inside and z inside the interpolation area.
        """
        self.init_3dcubic(extrapolate=True, extrapolation_range=10, extrapolation_type='quadratic')
        self.interpolate_3d_extrapolate_assert(2, 1, 1, self.extrap_data_qua[2][1][1], 1e-8)

    def test_interpolate_3d_cubic_extrapolate_quadratic_xsupymidzsup(self):
        """3D cubic interpolation. Test values in the extrapolation area with x above and y inside and z above the interpolation area.
        """
        self.init_3dcubic(extrapolate=True, extrapolation_range=10, extrapolation_type='quadratic')
        self.interpolate_3d_extrapolate_assert(2, 1, 2, self.extrap_data_qua[2][1][2], 1e-8)

    def test_interpolate_3d_cubic_extrapolate_quadratic_xsupysupzinf(self):
        """3D cubic interpolation. Test values in the extrapolation area with x above and y above and z below the interpolation area.
        """
        self.init_3dcubic(extrapolate=True, extrapolation_range=10, extrapolation_type='quadratic')
        self.interpolate_3d_extrapolate_assert(2, 2, 0, self.extrap_data_qua[2][2][0], 1e-8)

    def test_interpolate_3d_cubic_extrapolate_quadratic_xsupysupzmid(self):
        """3D cubic interpolation. Test values in the extrapolation area with x above and y above and z inside the interpolation area.
        """
        self.init_3dcubic(extrapolate=True, extrapolation_range=10, extrapolation_type='quadratic')
        self.interpolate_3d_extrapolate_assert(2, 2, 1, self.extrap_data_qua[2][2][1], 1e-8)

    def test_interpolate_3d_cubic_extrapolate_quadratic_xsupysupzsup(self):
        """3D cubic interpolation. Test values in the extrapolation area with x above and y above and z above the interpolation area.
        """
        self.init_3dcubic(extrapolate=True, extrapolation_range=10, extrapolation_type='quadratic')
        self.interpolate_3d_extrapolate_assert(2, 2, 2, self.extrap_data_qua[2][2][2], 1e-8)

    def test_interpolate_3d_cubic_extrapolate_quadratic(self):
        """3D cubic interpolation. Test values in the extrapolation area"""
        self.init_3dcubic(extrapolate=True, extrapolation_range=10, extrapolation_type='quadratic')

        slopex1 = (self.interp_func(1.01, -0.02, 0.) - self.interp_func(self.x[-1], self.y[0], 0.)) / (1.01 - self.x[-1])
        slopex2 = (self.interp_func(1.03, -0.06, 0.) - self.interp_func(self.x[-1], self.y[0], 0.)) / (1.03 - self.x[-1])
        slopex3 = (self.interp_func(1.05, -0.10, 0.) - self.interp_func(self.x[-1], self.y[0], 0.)) / (1.05 - self.x[-1])
        slopex4 = (self.interp_func(1.07, -0.14, 0.) - self.interp_func(self.x[-1], self.y[0], 0.)) / (1.07 - self.x[-1])
        metaslopex1 = (slopex2 - slopex1) / (1.03 - 1.01)
        metaslopex2 = (slopex3 - slopex2) / (1.05 - 1.03)
        metaslopex3 = (slopex4 - slopex3) / (1.07 - 1.05)
        self.assertAlmostEqual(metaslopex1, metaslopex2, delta=1e-8)
        self.assertAlmostEqual(metaslopex2, metaslopex3, delta=1e-8)

        slopey1 = (self.interp_func(1.01, -0.02, 0.) - self.interp_func(self.x[-1], self.y[0], 0.)) / (-0.02 - self.y[0])
        slopey2 = (self.interp_func(1.03, -0.06, 0.) - self.interp_func(self.x[-1], self.y[0], 0.)) / (-0.06 - self.y[0])
        slopey3 = (self.interp_func(1.05, -0.10, 0.) - self.interp_func(self.x[-1], self.y[0], 0.)) / (-0.10 - self.y[0])
        slopey4 = (self.interp_func(1.07, -0.14, 0.) - self.interp_func(self.x[-1], self.y[0], 0.)) / (-0.14 - self.y[0])
        metaslopey1 = (slopey2 - slopey1) / (-0.06 - -0.02)
        metaslopey2 = (slopey3 - slopey2) / (-0.10 - -0.06)
        metaslopey3 = (slopey4 - slopey3) / (-0.14 - -0.10)
        self.assertAlmostEqual(metaslopey1, metaslopey2, delta=1e-8)
        self.assertAlmostEqual(metaslopey2, metaslopey3, delta=1e-8)

    def test_interpolate_3d_cubic_type_conversion(self):
        """3D cubic interpolation. Whatever the type of input data, the interpolating function must provide float numbers.
        """
        self.init_3dcubic([1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], np.ones((4, 4, 4), dtype=int))
        self.assertIsInstance(self.interp_func(2.5, 4.5, 3.5), float)

    def test_interpolate_3d_cubic_single_value_tolerated_x(self):
        """3D cubic interpolation. If tolerated, a single value input must be extrapolated to every real value on its axis.
        """
        self.init_3dcubic([2.], [2, 3, 4, 5], [3, 4, 5, 6], np.ones((1, 4, 4)), tolerate_single_value=True)
        self.assertAlmostEqual(self.interp_func(-31946139.346, 2.5, 5.6), 1., delta=1e-8)
        self.assertAlmostEqual(self.interp_func(31946139.346, 2.5, 5.6), 1., delta=1e-8)
        self.assertAlmostEqual(self.interp_func(2., 2.5, 5.6), 1., delta=1e-8)

    def test_interpolate_3d_cubic_single_value_tolerated_y(self):
        """3D cubic interpolation. If tolerated, a single value input must be extrapolated to every real value on its axis.
        """
        self.init_3dcubic([1, 2, 3, 4], [4.], [3, 4, 5, 6], np.ones((4, 1, 4)), tolerate_single_value=True)
        self.assertAlmostEqual(self.interp_func(1.5, -31946139.346, 5.6), 1., delta=1e-8)
        self.assertAlmostEqual(self.interp_func(1.5, 31946139.346, 5.6), 1., delta=1e-8)
        self.assertAlmostEqual(self.interp_func(1.5, 4., 5.6), 1., delta=1e-8)

    def test_interpolate_3d_cubic_single_value_tolerated_z(self):
        """3D cubic interpolation. If tolerated, a single value input must be extrapolated to every real value on its axis.
        """
        self.init_3dcubic([1, 2, 3, 4], [2, 3, 4, 5], [4.], np.ones((4, 4, 1)), tolerate_single_value=True)
        self.assertAlmostEqual(self.interp_func(1.5, 2.5, -31946139.346), 1., delta=1e-8)
        self.assertAlmostEqual(self.interp_func(1.5, 2.5, 31946139.346), 1., delta=1e-8)
        self.assertAlmostEqual(self.interp_func(2., 2.5, 4.), 1., delta=1e-8)

    def test_interpolate_3d_cubic_single_value_tolerated_xy(self):
        """3D cubic interpolation. If tolerated, a single value input must be extrapolated to every real value on its axis.
        """
        self.init_3dcubic([2.], [4.], [3, 4, 5, 6], np.ones((1, 1, 4)), tolerate_single_value=True)
        self.assertAlmostEqual(self.interp_func(-31946139.346, 7856.45, 5.6), 1., delta=1e-8)
        self.assertAlmostEqual(self.interp_func(31946139.346, -7856.45, 5.6), 1., delta=1e-8)
        self.assertAlmostEqual(self.interp_func(2., 4., 5.6), 1., delta=1e-8)

    def test_interpolate_3d_cubic_single_value_tolerated_yz(self):
        """3D cubic interpolation. If tolerated, a single value input must be extrapolated to every real value on its axis.
        """
        self.init_3dcubic([1, 2, 3, 4], [4.], [5.], np.ones((4, 1, 1)), tolerate_single_value=True)
        self.assertAlmostEqual(self.interp_func(1.5, -31946139.346, 7856.45), 1., delta=1e-8)
        self.assertAlmostEqual(self.interp_func(1.5, 31946139.346, -7856.45), 1., delta=1e-8)
        self.assertAlmostEqual(self.interp_func(1.5, 4., 5.), 1., delta=1e-8)

    def test_interpolate_3d_cubic_single_value_tolerated_xz(self):
        """3D cubic interpolation. If tolerated, a single value input must be extrapolated to every real value on its axis.
        """
        self.init_3dcubic([2.], [2, 3, 4, 5], [5.], np.ones((1, 4, 1)), tolerate_single_value=True)
        self.assertAlmostEqual(self.interp_func(-31946139.346, 2.5, 7856.45), 1., delta=1e-8)
        self.assertAlmostEqual(self.interp_func(31946139.346, 2.5, -7856.45), 1., delta=1e-8)
        self.assertAlmostEqual(self.interp_func(2., 2.5, 5.), 1., delta=1e-8)

    def test_interpolate_3d_cubic_single_value_tolerated_xyz(self):
        """3D cubic interpolation. If tolerated, a single value input must be extrapolated to every real value on its axis.
        """
        self.init_3dcubic([2.], [4.], [5.], np.ones((1, 1, 1)), tolerate_single_value=True)
        self.assertAlmostEqual(self.interp_func(-31946139.346, 7856.45, 364646.43), 1., delta=1e-8)
        self.assertAlmostEqual(self.interp_func(31946139.346, -7856.45, 364646.43), 1., delta=1e-8)
        self.assertAlmostEqual(self.interp_func(2., 4., 5.), 1., delta=1e-8)


if __name__ == '__main__':
    unittest.main()