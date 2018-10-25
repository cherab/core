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
from cherab.core.math.interpolators import interpolators1d

X_LOWER = 0.
X_UPPER = 1.
NB_X = 10
NB_XSAMPLES = 30


class TestInterpolators1D(unittest.TestCase):
    """
    1D interpolators tests.

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
        self.xsamples = np.linspace(X_LOWER, X_UPPER, NB_XSAMPLES)
        self.xsamples_extrapol = np.array([X_LOWER - 0.08, X_LOWER - 0.04, X_UPPER + 0.04, X_UPPER + 0.08], dtype=np.float64)
        # data from function x -> cos(10*x) sampled on self.x:
        self.data = np.array([ 1.000000000000,  0.443666021702, -0.606320922374, -0.981674004711,
                              -0.264749878183,  0.746752954311,  0.927367703051,  0.076130124624,
                              -0.859815004004, -0.839071529076],
                             dtype=np.float64)

    def tearDown(self):

        try:
            del self.interp_func
        except AttributeError:
            pass

        try:
            del self.interp_data
        except AttributeError:
            pass

    def init_1dlinear(self, x=None, data=None, extrapolate=False, extrapolation_range=float('inf'),
                      extrapolation_type='nearest', tolerate_single_value=False):
        """Create the interpolating function and reference function."""

        if x is None:
            x = self.x

        if data is None:
            data = self.data

        # reference interpolated data:

        # linearly interpolated data sampled on self.xsamples,
        # calculated from linear interpolator scipy.interpolate.interp1d:
        self.interp_data = np.array([ 1.            ,  0.827344627425,  0.65468925485 ,  0.482033882274,  0.19022089727 ,
                                     -0.135637119857, -0.461495136984, -0.671036971053, -0.787525858675, -0.904014746296,
                                     -0.907509439898, -0.685015745458, -0.462522051019, -0.229870470166,  0.084044201987,
                                      0.397958874141,  0.711873546294,  0.796577712584,  0.852630565642,  0.908683418699,
                                      0.751249583376,  0.487072403865,  0.222895224353, -0.052965755187, -0.343431484761,
                                     -0.633897214335, -0.858384419526, -0.851946789376, -0.845509159226, -0.839071529076],
                                    dtype=np.float64)

        # nearestly extrapolated data sampled on self.xsamples_extrapol,
        # calculated from nearest edge value
        self.extrap_data_nea = np.array([1., 1., -0.839071529076, -0.839071529076], dtype=np.float64)

        # linearly extrapolated data sampled on self.xsamples_extrapol,
        # calculated from the nearest linear parts of the interpolation
        self.extrap_data_lin = np.array([1.4005604643743956, 1.2002802321871977, -0.83160387810265701, -0.82413622712886148], dtype=np.float64)

        self.interp_func = interpolators1d.Interpolate1DLinear(x, data,
                                                               extrapolate=extrapolate,
                                                               extrapolation_range=extrapolation_range,
                                                               extrapolation_type=extrapolation_type,
                                                               tolerate_single_value=tolerate_single_value)

    def init_1dcubic_c2(self, x=None, data=None, extrapolate=False, extrapolation_range=float('inf'),
                        extrapolation_type='nearest', tolerate_single_value=False):
        """Create the interpolating function and reference function."""

        if x is None:
            x = self.x

        if data is None:
            data = self.data

        # reference interpolated data:

        # C2 cubic interpolated data sampled on self.xsamples,
        # calculated from solving manually the linear system given by the
        # following constraints on the cubic splines:
        #
        # if P1, P2, ..., P9 are the 9 cubic polynomials defining the interpolation
        # on respectively ranges [x[0], x[1]], [x[1], x[2]], ..., [x[8], x[9]]
        # constraints are: Pi(x[i-1]) = data[i-1]            for i in [1, 9]
        #                  Pi(x[i]) = data[i]                for i in [1, 9]
        #                  dPi/dx(x[i]) = dPi+1/dx(x[i])     for i in [1, 8]
        #                  d2Pi/dx2(x[i]) = d2Pi+1/dx2(x[i]) for i in [1, 8]
        #                  d3P1/dx3(x[0]) = d3P9/dx3(x[9]) = 0
        self.interp_data = np.array([ 1.            ,  0.909386322736,  0.744935119692,  0.506646390868,  0.198359265243,
                                     -0.145597952285, -0.476357515   , -0.745969855864, -0.924576033965, -0.997965016288,
                                     -0.95270426186 , -0.791113531095, -0.539928651551, -0.227988632848,  0.113003806643,
                                      0.443108473075,  0.720913065294,  0.908991353712,  0.992221981868,  0.963237748147,
                                      0.816761597858,  0.571723930259,  0.262672856832, -0.075536667115, -0.406637953849,
                                     -0.692732204577, -0.895899188014, -0.989500940501, -0.970558387522, -0.839071529076],
                                    dtype=np.float64)

        # nearestly extrapolated data sampled on self.xsamples_extrapol,
        # calculated from nearest edge value
        self.extrap_data_nea = np.array([1., 1., -0.839071529076, -0.839071529076], dtype=np.float64)

        # linearly extrapolated data sampled on self.xsamples_extrapol,
        # calculated from P(x_nearest) + (x-x_nearest)*dP/dx(x_nearest)
        # where x_nearest (=x[0] or x[9]) is the nearest coordinate in the
        # interpolated range form x, and P (=P1 or P9) the nearest
        # interpolation polynomial.
        self.extrap_data_lin = np.array([1.1245722013484143, 1.0622861006742075, -0.62127107611011623, -0.40347062314376919], dtype=np.float64)

        # quadraticaly extrapolated data sampled on self.xsamples_extrapol,
        # calculated from P(x_nearest) + (x-x_nearest)*dP/dx(x_nearest)
        # + 0.5*(x-x_nearest)**2*d2P/dx2(x_nearest)
        # where x_nearest (=x[0] or x[9]) is the nearest coordinate in the
        # interpolated range form x, and P (=P1 or P9) the nearest
        # interpolation polynomial.
        self.extrap_data_qua = np.array([0.92586065196970724, 1.0126082133295307, -0.5455512673927978, -0.10059138827449526], dtype=np.float64)

        self.interp_func = interpolators1d.Interpolate1DCubic(x, data,
                                                               extrapolate=extrapolate,
                                                               extrapolation_range=extrapolation_range,
                                                               extrapolation_type=extrapolation_type,
                                                               tolerate_single_value=tolerate_single_value,
                                                               continuity_order=2)

    def init_1dcubic_c1(self, x=None, data=None, extrapolate=False, extrapolation_range=float('inf'),
                        extrapolation_type='nearest', tolerate_single_value=False):
        """Create the interpolating function and reference function."""

        if x is None:
            x = self.x

        if data is None:
            data = self.data

        # reference interpolated data:

        # C1 cubic interpolated data sampled on self.xsamples,
        # calculated from from solving manually the linear system given by the
        # following constraints on the cubic splines:
        #
        # if P1, P2, ..., P9 are the 9 cubic polynomials defining the interpolation
        # on respectively ranges [x[0], x[1]], [x[1], x[2]], ..., [x[8], x[9]]
        # constraints are: Pi(x[i-1]) = data[i-1]            for i in [1, 9]
        #                  Pi(x[i]) = data[i]                for i in [1, 9]
        #                  dPi/dx(x[i]) = (data[i+1]-data[i-1])/(x[i+1]-x[i-1])     for i in [1, 8]
        #                  dPi/dx(x[i-1]) = (data[i]-data[i-2])/(x[i]-x[i-2])       for i in [2, 9]
        #                  d3P1/dx3(x[0]) = d3P9/dx3(x[9]) = 0
        self.interp_data = np.array([ 1.            ,  0.880173125546,  0.712800602783,  0.497882431711,  0.209599324944,
                                     -0.154300121054, -0.492023725037, -0.724305080592, -0.896927467157, -0.986541391474,
                                     -0.954335254748, -0.777457953678, -0.513924204518, -0.234128226514,  0.094607534444,
                                      0.447977457129,  0.725059163866,  0.885104972562,  0.967692230729,  0.955569808692,
                                      0.819830442329,  0.554735584499,  0.240589044126, -0.05646951848 , -0.390685969507,
                                     -0.698474405891, -0.889098560187, -0.964565305133, -0.947889628097, -0.839071529076],
                                    dtype=np.float64)

        # nearestly extrapolated data sampled on self.xsamples_extrapol,
        # calculated from nearest edge value
        self.extrap_data_nea = np.array([1., 1., -0.839071529076, -0.839071529076], dtype=np.float64)

        # linearly extrapolated data sampled on self.xsamples_extrapol,
        # calculated from P(x_nearest) + (x-x_nearest)*dP/dx(x_nearest)
        # where x_nearest (=x[0] or x[9]) is the nearest coordinate in the
        # interpolated range form x, and P (=P1 or P9) the nearest
        # interpolation polynomial.
        self.extrap_data_lin = np.array([ 1.222845396694,  1.111422698347, -0.659399929463, -0.479728329849], dtype=np.float64)

        # quadraticaly extrapolated data sampled on self.xsamples_extrapol,
        # calculated from P(x_nearest) + (x-x_nearest)*dP/dx(x_nearest)
        # + 0.5*(x-x_nearest)**2*d2P/dx2(x_nearest)
        # where x_nearest (=x[0] or x[9]) is the nearest coordinate in the
        # interpolated range form x, and P (=P1 or P9) the nearest
        # interpolation polynomial.
        self.extrap_data_qua = np.array([ 1.094890547964,  1.079433986165, -0.597406507952, -0.231754643808], dtype=np.float64)

        self.interp_func = interpolators1d.Interpolate1DCubic(x, data,
                                                               extrapolate=extrapolate,
                                                               extrapolation_range=extrapolation_range,
                                                               extrapolation_type=extrapolation_type,
                                                               tolerate_single_value=tolerate_single_value,
                                                               continuity_order=1)

    def interpolate_1d_xboundaries_assert(self, inf, sup, epsilon):
        with self.assertRaises(ValueError):
            self.interp_func(inf - epsilon)
        self.assertIsInstance(self.interp_func(inf + epsilon), float)
        with self.assertRaises(ValueError):
            self.interp_func(sup + epsilon)
        self.assertIsInstance(self.interp_func(sup - epsilon), float)

    def derivative(self, f, x, h, order):
        """
        Calculates a numerical derivative at point x.

        Obtains samples f(x - h/2) and (x + h/2) and computes the central
        difference to obtain the first derivative. Method calls itself
        recursively to calculate higher deriviative orders.

        :param f: 1D function object.
        :param x: Sample point.
        :param h: Sample distance.
        :param order: Derivative order.
        :return: Derivative value.
        """

        if order < 1:
            raise ValueError('Derivative order must be > 0.')

        d = 0.5 * h
        if order == 1:
            return (f(x+d) - f(x-d)) / h
        else:
            return (self.derivative(f, x+d, h, order - 1) - self.derivative(f, x-d, h, order - 1)) / h

    # General behaviour

    def test_interpolate_1d_invalid_data_length(self):
        """1D interpolation. An error must be raises if data has not the same length as coordinates.
        """
        self.assertRaises(ValueError, interpolators1d._Interpolate1DBase, [1, 2, 3, 4], [10., 12., 0.7])

    def test_interpolate_1d_invalid_data_dimension(self):
        """1D interpolation. An error must be raises if data is not 1D.
        """
        self.assertRaises(ValueError, interpolators1d._Interpolate1DBase, [1, 2, 3, 4], np.ones((4, 2)))

    def test_interpolate_1d_double_coord(self):
        """1D interpolation. An error must be raises if there is a double coordinate.
        """
        self.assertRaises(ValueError, interpolators1d._Interpolate1DBase, [1, 3, 3, 4], [10, 12, 42, 0])

    def test_interpolate_1d_single_value_invalid(self):
        """1D interpolation. By default, a single input value must raise a ValueError.
        """
        self.assertRaises(ValueError, interpolators1d._Interpolate1DBase, [2.], [4.])

    # Linear behaviour

    def test_interpolate_1d_linear(self):
        """1D linear interpolation. Test values inside the boundaries"""
        self.init_1dlinear(extrapolate=True, extrapolation_type='linear', extrapolation_range=1.)
        for i in range(len(self.xsamples)):
            x = self.xsamples[i]
            self.assertAlmostEqual(self.interp_func(x), self.interp_data[i], delta=1e-8)
            for order in range(1, 4):
                self.assertAlmostEqual(self.interp_func.derivative(x, order), self.derivative(self.interp_func, x, 1e-3, order), delta=1e-6)

    def test_interpolate_1d_linear_bigvalues(self):
        """1D linear interpolation. Test with big values (1e20) inside the boundaries"""
        factor = 1.e20
        self.init_1dlinear(data=factor * self.data, extrapolate=True, extrapolation_type='linear', extrapolation_range=1.)
        for i in range(len(self.xsamples)):
            x = self.xsamples[i]
            self.assertAlmostEqual(self.interp_func(x), factor * self.interp_data[i], delta=factor * 1e-8)
            for order in range(1, 4):
                self.assertAlmostEqual(self.interp_func.derivative(x, order), self.derivative(self.interp_func, x, 1e-3, order), delta=factor * 1e-6)

    def test_interpolate_1d_linear_lowvalues(self):
        """1D linear interpolation. Test with low values (1e-20) inside the boundaries"""
        factor = 1.e-20
        self.init_1dlinear(data=factor * self.data, extrapolate=True, extrapolation_type='linear', extrapolation_range=1.)
        for i in range(len(self.xsamples)):
            x = self.xsamples[i]
            self.assertAlmostEqual(self.interp_func(x), factor * self.interp_data[i], delta=factor * 1e-8)
            for order in range(1, 4):
                self.assertAlmostEqual(self.interp_func.derivative(x, order), self.derivative(self.interp_func, x, 1e-3, order), delta=factor * 1e-6)

    def test_interpolate_1d_linear_edges(self):
        """1D linear interpolation. Test edges values"""
        self.init_1dlinear()
        self.assertAlmostEqual(self.interp_func(self.x[0]), self.data[0], delta=1e-8)
        self.assertAlmostEqual(self.interp_func(self.x[-1]), self.data[-1], delta=1e-8)

    def test_interpolate_1d_linear_knots(self):
        """1D linear interpolation. Test knots values"""
        self.init_1dlinear()
        for i in range(len(self.data)):
            self.assertAlmostEqual(self.interp_func(self.x[i]), self.data[i], delta=1e-8)

    def test_interpolate_1d_linear_out(self):
        """1D linear interpolation. Test values outside the boundaries"""
        self.init_1dlinear()
        self.assertRaises(ValueError, self.interp_func, X_LOWER - 1)
        self.assertRaises(ValueError, self.interp_func, X_UPPER + 1)
        self.interpolate_1d_xboundaries_assert(X_LOWER, X_UPPER, 1e-6)

    def test_interpolate_1d_linear_extrapolate_nearest_range(self):
        """1D linear interpolation. Tests the size of the extrapolation range."""
        self.init_1dlinear(extrapolate=True, extrapolation_type='nearest', extrapolation_range=1.)
        self.interpolate_1d_xboundaries_assert(X_LOWER - 1, X_UPPER + 1, 1e-6)

    def test_interpolate_1d_linear_extrapolate_nearest(self):
        """1D linear interpolation. Test values in the extrapolation areas"""
        self.init_1dlinear(extrapolate=True, extrapolation_type='nearest')
        for i in range(len(self.xsamples_extrapol)):
            x = self.xsamples_extrapol[i]
            self.assertAlmostEqual(self.interp_func(x), self.extrap_data_nea[i], delta=1e-8)
            for order in range(1, 4):
                self.assertAlmostEqual(self.interp_func.derivative(x, order), self.derivative(self.interp_func, x, 1e-3, order), delta=1e-6)

    def test_interpolate_1d_linear_extrapolate_linear_range(self):
        """1D linear interpolation. Tests the size of the extrapolation range."""
        self.init_1dlinear(extrapolate=True, extrapolation_type='linear', extrapolation_range=1.)
        self.interpolate_1d_xboundaries_assert(X_LOWER - 1, X_UPPER + 1, 1e-6)

    def test_interpolate_1d_linear_extrapolate_linear(self):
        """1D linear interpolation. Test values in the extrapolation areas"""
        self.init_1dlinear(extrapolate=True, extrapolation_type='linear')
        for i in range(len(self.xsamples_extrapol)):
            x = self.xsamples_extrapol[i]
            self.assertAlmostEqual(self.interp_func(x), self.extrap_data_lin[i], delta=1e-8)
            for order in range(1, 4):
                self.assertAlmostEqual(self.interp_func.derivative(x, order), self.derivative(self.interp_func, x, 1e-3, order), delta=1e-6)

    def test_interpolate_1d_linear_type_conversion(self):
        """1D linear interpolation. Whatever the type of input data, the interpolating function must provide float numbers.
        """
        self.init_1dlinear([1, 2, 3, 4], [10, 12, 42, 0])
        self.assertIsInstance(self.interp_func(2.5), float)

    def test_interpolate_1d_linear_single_value_tolerated(self):
        """1D linear interpolation. If tolerated, a single value input must be extrapolated to every real value.
        """
        self.init_1dlinear([2.], [4.], tolerate_single_value=True)
        self.assertAlmostEqual(self.interp_func(-31946139.346), 4., delta=1e-8)
        self.assertAlmostEqual(self.interp_func(31946139.346), 4., delta=1e-8)
        self.assertAlmostEqual(self.interp_func(2.), 4., delta=1e-8)

    # Cubic C2 behaviour

    def test_interpolate_1d_cubic_c2(self):
        """1D cubic interpolation. Test values inside the boundaries"""
        self.init_1dcubic_c2(extrapolate=True, extrapolation_type='quadratic')
        for i in range(len(self.xsamples)):
            x = self.xsamples[i]
            self.assertAlmostEqual(self.interp_func(x), self.interp_data[i], delta=1e-8)
            for order in range(1, 3):
                r = self.interp_func.derivative(x, order)
                v = self.derivative(self.interp_func, x, 1e-4, order)
                self.assertAlmostEqual(r, v, delta=1e-5 * abs(v))

    def test_interpolate_1d_cubic_c2_bigvalues(self):
        """1D cubic interpolation. Test with big values (1e20) inside the boundaries"""
        factor = 1.e20
        self.init_1dcubic_c2(data=factor * self.data, extrapolate=True, extrapolation_type='quadratic')
        for i in range(len(self.xsamples)):
            x = self.xsamples[i]
            self.assertAlmostEqual(self.interp_func(x), factor * self.interp_data[i], delta=factor*1e-8)
            for order in range(1, 3):
                r = self.interp_func.derivative(x, order)
                v = self.derivative(self.interp_func, x, 1e-4, order)
                self.assertAlmostEqual(r, v, delta=1e-5 * abs(v))

    def test_interpolate_1d_cubic_c2_lowvalues(self):
        """1D cubic interpolation. Test with low values (1e-20) inside the boundaries"""
        factor = 1.e-20
        self.init_1dcubic_c2(data=factor * self.data, extrapolate=True, extrapolation_type='quadratic')
        for i in range(len(self.xsamples)):
            x = self.xsamples[i]
            self.assertAlmostEqual(self.interp_func(x), factor * self.interp_data[i], delta=factor*1e-8)
            for order in range(1, 3):
                r = self.interp_func.derivative(x, order)
                v = self.derivative(self.interp_func, x, 1e-4, order)
                self.assertAlmostEqual(r, v, delta=1e-5 * abs(v))

    def test_interpolate_1d_cubic_c2_edge(self):
        """1D cubic interpolation. Test edges values"""
        self.init_1dcubic_c2()
        self.assertAlmostEqual(self.interp_func(self.x[0]), self.data[0], delta=1e-8)
        self.assertAlmostEqual(self.interp_func(self.x[-1]), self.data[-1], delta=1e-8)

    def test_interpolate_1d_cubic_c2_knot(self):
        """1D cubic interpolation. Test knots values"""
        self.init_1dcubic_c2()
        for i in range(len(self.data)):
            self.assertAlmostEqual(self.interp_func(self.x[i]), self.data[i], delta=1e-8)

    def test_interpolate_1d_cubic_c2_out(self):
        """1D cubic interpolation. Test values outside the boundaries"""
        self.init_1dcubic_c2()
        self.assertRaises(ValueError, self.interp_func, X_LOWER - 1)
        self.assertRaises(ValueError, self.interp_func, X_UPPER + 1)
        self.interpolate_1d_xboundaries_assert(X_LOWER, X_UPPER, 1e-6)

    def test_interpolate_1d_cubic_c2_extrapolate_nearest_range(self):
        """1D cubic interpolation. Tests the size of the extrapolation range."""
        self.init_1dcubic_c2(extrapolate=True, extrapolation_type='nearest', extrapolation_range=1.)
        self.interpolate_1d_xboundaries_assert(X_LOWER - 1, X_UPPER + 1, 1e-6)

    def test_interpolate_1d_cubic_c2_extrapolate_nearest(self):
        """1D cubic interpolation. Test values in the extrapolation area"""
        self.init_1dcubic_c2(extrapolate=True, extrapolation_type='nearest')
        for i in range(len(self.xsamples_extrapol)):
            x = self.xsamples_extrapol[i]
            self.assertAlmostEqual(self.interp_func(x), self.extrap_data_nea[i], delta=1e-8)
            for order in range(1, 4):
                self.assertAlmostEqual(self.interp_func.derivative(x, order), self.derivative(self.interp_func, x, 1e-3, order), delta=1e-6)

    def test_interpolate_1d_cubic_c2_extrapolate_linear_range(self):
        """1D cubic interpolation. Tests the size of the extrapolation range."""
        self.init_1dcubic_c2(extrapolate=True, extrapolation_type='linear', extrapolation_range=1.)
        self.interpolate_1d_xboundaries_assert(X_LOWER - 1, X_UPPER + 1, 1e-6)

    def test_interpolate_1d_cubic_c2_extrapolate_linear(self):
        """1D cubic interpolation. Test values in the extrapolation area"""
        self.init_1dcubic_c2(extrapolate=True, extrapolation_type='linear')
        for i in range(len(self.xsamples_extrapol)):
            x = self.xsamples_extrapol[i]
            self.assertAlmostEqual(self.interp_func(x), self.extrap_data_lin[i], delta=1e-8)
            for order in range(1, 4):
                self.assertAlmostEqual(self.interp_func.derivative(x, order), self.derivative(self.interp_func, x, 1e-3, order), delta=1e-6)

    def test_interpolate_1d_cubic_c2_extrapolate_quadratic_range(self):
        """1D cubic interpolation. Tests the size of the extrapolation range."""
        self.init_1dcubic_c2(extrapolate=True, extrapolation_type='quadratic', extrapolation_range=1.)
        self.interpolate_1d_xboundaries_assert(X_LOWER - 1, X_UPPER + 1, 1e-6)

    def test_interpolate_1d_cubic_c2_extrapolate_quadratic(self):
        """1D cubic interpolation. Test values in the extrapolation area"""
        self.init_1dcubic_c2(extrapolate=True, extrapolation_type='quadratic')
        for i in range(len(self.xsamples_extrapol)):
            x = self.xsamples_extrapol[i]
            self.assertAlmostEqual(self.interp_func(x), self.extrap_data_qua[i], delta=1e-8)
            for order in range(1, 4):
                self.assertAlmostEqual(self.interp_func.derivative(x, order), self.derivative(self.interp_func, x, 1e-3, order), delta=1e-6)

    def test_interpolate_1d_cubic_c2_type_conversion(self):
        """1D cubic interpolation. Whatever the type of input data, the interpolating function must provide float numbers.
        """
        self.init_1dcubic_c2([1, 2, 3, 4], [10, 12, 42, 0])
        self.assertIsInstance(self.interp_func(2.5), float)

    def test_interpolate_1d_cubic_c2_single_value_tolerated(self):
        """1D cubic interpolation. If tolerated, a single value input must be extrapolated to every real value.
        """
        self.init_1dcubic_c2([2.], [4.], tolerate_single_value=True)
        self.assertAlmostEqual(self.interp_func(-31946139.346), 4., delta=1e-8)
        self.assertAlmostEqual(self.interp_func(31946139.346), 4., delta=1e-8)
        self.assertAlmostEqual(self.interp_func(2.), 4., delta=1e-8)

    # Cubic C1 behaviour

    def test_interpolate_1d_cubic_c1(self):
        """1D cubic interpolation. Test values inside the boundaries"""
        self.init_1dcubic_c1(extrapolate=True, extrapolation_type='quadratic')
        for i in range(len(self.xsamples)):
            x = self.xsamples[i]
            self.assertAlmostEqual(self.interp_func(x), self.interp_data[i], delta=1e-8)
            for order in range(1, 3):
                r = self.interp_func.derivative(x, order)
                v = self.derivative(self.interp_func, x, 1e-4, order)
                self.assertAlmostEqual(r, v, delta=1e-5 * abs(v))

    def test_interpolate_1d_cubic_c1_bigvalues(self):
        """1D cubic interpolation. Test with big values (1e20) inside the boundaries"""
        factor = 1.e20
        self.init_1dcubic_c1(data=factor * self.data, extrapolate=True, extrapolation_type='quadratic')
        for i in range(len(self.xsamples)):
            x = self.xsamples[i]
            self.assertAlmostEqual(self.interp_func(x), factor * self.interp_data[i], delta=factor * 1e-8)
            for order in range(1, 3):
                r = self.interp_func.derivative(x, order)
                v = self.derivative(self.interp_func, x, 1e-4, order)
                self.assertAlmostEqual(r, v, delta=1e-5 * abs(v))

    def test_interpolate_1d_cubic_c1_lowvalues(self):
        """1D cubic interpolation. Test with low values (1e-20) inside the boundaries"""
        factor = 1.e-20
        self.init_1dcubic_c1(data=factor * self.data, extrapolate=True, extrapolation_type='quadratic')
        for i in range(len(self.xsamples)):
            x = self.xsamples[i]
            self.assertAlmostEqual(self.interp_func(x), factor * self.interp_data[i], delta=factor * 1e-8)
            for order in range(1, 3):
                r = self.interp_func.derivative(x, order)
                v = self.derivative(self.interp_func, x, 1e-4, order)
                self.assertAlmostEqual(r, v, delta=1e-5 * abs(v))

    def test_interpolate_1d_cubic_c1_edge(self):
        """1D cubic interpolation. Test edges values"""
        self.init_1dcubic_c1()
        self.assertAlmostEqual(self.interp_func(self.x[0]), self.data[0], delta=1e-8)
        self.assertAlmostEqual(self.interp_func(self.x[-1]), self.data[-1], delta=1e-8)

    def test_interpolate_1d_cubic_c1_knot(self):
        """1D cubic interpolation. Test knots values"""
        self.init_1dcubic_c1()
        for i in range(len(self.data)):
            self.assertAlmostEqual(self.interp_func(self.x[i]), self.data[i], delta=1e-8)

    def test_interpolate_1d_cubic_c1_out(self):
        """1D cubic interpolation. Test values outside the boundaries"""
        self.init_1dcubic_c1()
        self.assertRaises(ValueError, self.interp_func, X_LOWER - 1)
        self.assertRaises(ValueError, self.interp_func, X_UPPER + 1)
        self.interpolate_1d_xboundaries_assert(X_LOWER, X_UPPER, 1e-6)

    def test_interpolate_1d_cubic_c1_extrapolate_nearest_range(self):
        """1D cubic interpolation. Tests the size of the extrapolation range."""
        self.init_1dcubic_c1(extrapolate=True, extrapolation_type='nearest', extrapolation_range=1.)
        self.interpolate_1d_xboundaries_assert(X_LOWER - 1, X_UPPER + 1, 1e-6)

    def test_interpolate_1d_cubic_c1_extrapolate_nearest(self):
        """1D cubic interpolation. Test values in the extrapolation area"""
        self.init_1dcubic_c1(extrapolate=True, extrapolation_type='nearest')
        for i in range(len(self.xsamples_extrapol)):
            x = self.xsamples_extrapol[i]
            self.assertAlmostEqual(self.interp_func(x), self.extrap_data_nea[i], delta=1e-8)
            for order in range(1, 4):
                self.assertAlmostEqual(self.interp_func.derivative(x, order), self.derivative(self.interp_func, x, 1e-3, order), delta=1e-6)

    def test_interpolate_1d_cubic_c1_extrapolate_linear_range(self):
        """1D cubic interpolation. Tests the size of the extrapolation range."""
        self.init_1dcubic_c1(extrapolate=True, extrapolation_type='linear', extrapolation_range=1.)
        self.interpolate_1d_xboundaries_assert(X_LOWER - 1, X_UPPER + 1, 1e-6)

    def test_interpolate_1d_cubic_c1_extrapolate_linear(self):
        """1D cubic interpolation. Test values in the extrapolation area"""
        self.init_1dcubic_c1(extrapolate=True, extrapolation_type='linear')
        for i in range(len(self.xsamples_extrapol)):
            x = self.xsamples_extrapol[i]
            self.assertAlmostEqual(self.interp_func(x), self.extrap_data_lin[i], delta=1e-8)
            for order in range(1, 4):
                self.assertAlmostEqual(self.interp_func.derivative(x, order), self.derivative(self.interp_func, x, 1e-3, order), delta=1e-6)

    def test_interpolate_1d_cubic_c1_extrapolate_quadratic_range(self):
        """1D cubic interpolation. Tests the size of the extrapolation range."""
        self.init_1dcubic_c1(extrapolate=True, extrapolation_type='quadratic', extrapolation_range=1.)
        self.interpolate_1d_xboundaries_assert(X_LOWER - 1, X_UPPER + 1, 1e-6)

    def test_interpolate_1d_cubic_c1_extrapolate_quadratic(self):
        """1D cubic interpolation. Test values in the extrapolation area"""
        self.init_1dcubic_c1(extrapolate=True, extrapolation_type='quadratic')
        for i in range(len(self.xsamples_extrapol)):
            x = self.xsamples_extrapol[i]
            self.assertAlmostEqual(self.interp_func(x), self.extrap_data_qua[i], delta=1e-8)
            for order in range(1, 4):
                self.assertAlmostEqual(self.interp_func.derivative(x, order), self.derivative(self.interp_func, x, 1e-3, order), delta=1e-6)

    def test_interpolate_1d_cubic_c1_type_conversion(self):
        """1D cubic interpolation. Whatever the type of input data, the interpolating function must provide float numbers.
        """
        self.init_1dcubic_c1([1, 2, 3, 4], [10, 12, 42, 0])
        self.assertIsInstance(self.interp_func(2.5), float)

    def test_interpolate_1d_cubic_c1_single_value_tolerated(self):
        """1D cubic interpolation. If tolerated, a single value input must be extrapolated to every real value.
        """
        self.init_1dcubic_c1([2.], [4.], tolerate_single_value=True)
        self.assertAlmostEqual(self.interp_func(-31946139.346), 4., delta=1e-8)
        self.assertAlmostEqual(self.interp_func(31946139.346), 4., delta=1e-8)
        self.assertAlmostEqual(self.interp_func(2.), 4., delta=1e-8)

if __name__ == '__main__':
    unittest.main()