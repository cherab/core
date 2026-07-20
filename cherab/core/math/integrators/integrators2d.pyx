# cython: language_level=3

# Copyright 2016-2025 Euratom
# Copyright 2016-2025 United Kingdom Atomic Energy Authority
# Copyright 2016-2025 Centro de Investigaciones Energéticas, Medioambientales y Tecnológicas
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
from scipy.special import roots_legendre
cimport cython

from raysect.core.math.function.float cimport Function1D, autowrap_function2d, Constant2D

from libc.math cimport INFINITY


cdef class Integrator2D:
    """
    Compute a definite integral of a two-dimensional function.

    :ivar Function2D integrand: A 2D function to integrate.
    """

    @property
    def integrand(self):
        """
        A 2D function to integrate.

        :rtype: Function2D
        """
        return self.function

    @integrand.setter
    def integrand(self, object func not None):

        self.function = autowrap_function2d(func)

    cdef double evaluate(self, double x_lower, double x_upper, Function1D y_lower, Function1D y_upper) except? -1e999:

        raise NotImplementedError("The evaluate() method has not been implemented.")

    def __call__(self, double x_lower, double x_upper, Function1D y_lower, Function1D y_upper):
        """
        Integrates a two-dimensional function over a finite interval.

        :param double x_lower: Lower limit of integration in the x dimension.
        :param double x_upper: Upper limit of integration in the x dimension.
        :param Function1D y_lower: Lower limit of integration in the y dimension.
        :param Function1D y_upper: Upper limit of integration in the y dimension.

        :returns: Definite integral of a two-dimensional function.
        """

        return self.evaluate(x_lower, x_upper, y_lower, y_upper)


cdef class GaussianQuadrature2D(Integrator2D):
    """
    Approximates an integral of a two-dimensional function over a finite interval.

    The integral is approximated with fixed-tolerance Gauss-Legendre quadrature.
    The quadrature approximation is calculated as follows:

    .. math::
        \\int_{x_{\\text{lower}}}^{x_{\\text{upper}}} \\int_{y_{\\text{lower}}(x)}^{y_{\\text{upper}}(x)} f(x, y) \\, dy \\, dx
        \\approx \\sum_{i=1}^{n_x} \\sum_{j=1}^{n_y} w_{i} w_{j} \\,
        f\\left( B \\xi_i + A, \\, D(x) \\eta_j + C(x) \\right) \\,
        B \\cdot D(x)
    where:
        -  :math:`x_{\mathrm{lower}}`: Lower limit of integration for the x-dimension.
        -  :math:`x_{\mathrm{upper}}`: Upper limit of integration for the x-dimension.
        -  :math:`y_{\mathrm{lower}}(x)`: Lower limit of integration for the y-dimension, a function of :math:`x`.
        -  :math:`y_{\mathrm{upper}}(x)`: Upper limit of integration for the y-dimension, a function of :math:`x`.
        -  :math:`f(x, y)`: The function to be integrated over the specified region.
        -  :math:`\\xi`: The transformed variable for the x-dimension, ranging from -1 to 1.
        -  :math:`\\eta`: The transformed variable for the y-dimension, ranging from -1 to 1.
        -  :math:`w_i`: The weight corresponding to the :math:`i`-th root of the Legendre polynomial in the x-dimension.
        -  :math:`w_j`: The weight corresponding to the :math:`j`-th root of the Legendre polynomial in the y-dimension.
        -  :math:`\\xi_i`: The :math:`i`-th root of the Legendre polynomial in the x-dimension.
        -  :math:`\\eta_j`: The :math:`j`-th root of the Legendre polynomial in the y-dimension.
        -  :math:`n_x`: The number of roots (or nodes) in the x-dimension.
        -  :math:`n_y`: The number of roots (or nodes) in the y-dimension.
        -  :math:`A = \\frac{x_{\\mathrm{upper}} + x_{\\mathrm{lower}}}{2}`: Midpoint of the x-interval.
        -  :math:`B = \\frac{x_{\\mathrm{upper}} - x_{\\mathrm{lower}}}{2}`: Half-width of the x-interval.
        -  :math:`D(x) = \\frac{y_{\\mathrm{upper}}(x) - y_{\\mathrm{lower}}(x)}{2}`: Half-width of the y-interval.
        -  :math:`C(x) = \\frac{y_{\\mathrm{upper}}(x) + y_{\\mathrm{lower}}(x)}{2}`: Midpoint of the y-interval.

    :param Function2D integrand: A 2D function to integrate. Default is `Constant2D(0)`.
    :param double relative_tolerance: Iteration stops when relative error between
        last two iterates is less than this value. Default is `1.e-5`.
    :param int x_max_order: Maximum order on Gaussian quadrature in the x dimension. Default is `50`.
    :param int x_min_order: Minimum order on Gaussian quadrature in the x dimension. Default is `1`.
    :param int y_max_order: Maximum order on Gaussian quadrature in the y dimension. Default is `50`.
    :param int y_min_order: Minimum order on Gaussian quadrature in the y dimension. Default is `1`.

    :ivar Function1D integrand: A 1D function to integrate.
    :ivar double relative_tolerance: Iteration stops when relative error between
        last two iterates is less than this value.
    :ivar int x_max_order: Maximum order on Gaussian quadrature in the x dimension.
    :ivar int x_min_order: Minimum order on Gaussian quadrature in the x dimension.
    :ivar int y_max_order: Maximum order on Gaussian quadrature in the y dimension.
    :ivar int y_min_order: Minimum order on Gaussian quadrature in the y dimension.
    """
    def __init__(self, object integrand=Constant2D(0), double relative_tolerance=1.e-5, int x_max_order=50, int x_min_order=1,
                 int y_max_order=50, int y_min_order=1):

        self._check_order(x_min_order, x_max_order, "x")
        self._check_order(y_min_order, y_max_order, "y")

        self._x_min_order = x_min_order
        self._x_max_order = x_max_order

        self._y_min_order = y_min_order
        self._y_max_order = y_max_order

        self._build_cache()

        self.integrand = integrand

        self.relative_tolerance = relative_tolerance
    
    def _check_order(self, int min_order, int max_order, str dimension):
        """
        Check the order of Gaussian quadrature.

        :param int min_order: Minimum order on Gaussian quadrature.
        :param int max_order: Maximum order on Gaussian quadrature.
        :param str dimension: Dimension of Gaussian quadrature.

        :raises ValueError: If the order of Gaussian quadrature is invalid.
        """

        if min_order < 1 or max_order < 1:
            raise ValueError("Order of Gaussian quadrature in the {} dimension must be >= 1.".format(dimension))

        if min_order > max_order:
            raise ValueError("Minimum order of Gaussian quadrature in the {} dimension must be less than or equal to the maximum order.".format(dimension))
    
    @property
    def x_min_order(self):
        """
        Minimum order on Gaussian quadrature in the x dimension.

        :rtype: int
        """
        return self._x_min_order

    @x_min_order.setter
    def x_min_order(self, int value):

        self._check_order(value, self._x_max_order, "x")

        self._x_min_order = value

        self._build_cache()

    @property
    def x_max_order(self):
        """
        Maximum order on Gaussian quadrature in the x dimension.

        :rtype: int
        """
        return self._x_max_order

    @x_max_order.setter
    def x_max_order(self, int value):

        self._check_order(self._x_min_order, value, "x")

        self._x_max_order = value

        self._build_cache()

    @property
    def y_min_order(self):
        """
        Minimum order on Gaussian quadrature in the y dimension.

        :rtype: int
        """
        return self._y_min_order

    @y_min_order.setter
    def y_min_order(self, int value):

        self._check_order(value, self._y_max_order, "y")

        self._y_min_order = value

        self._build_cache()

    @property
    def y_max_order(self):
        """
        Maximum order on Gaussian quadrature in the y dimension.

        :rtype: int
        """
        return self._y_max_order

    @y_max_order.setter
    def y_max_order(self, int value):

        self._check_order(self._y_min_order, value, "y")

        self._y_max_order = value

        self._build_cache()

    @property
    def relative_tolerance(self):
        """
        Iteration stops when relative error between last two iterates is less than this value.

        :rtype: double
        """
        return self._rtol

    @relative_tolerance.setter
    def relative_tolerance(self, double value):

        if value <= 0:
            raise ValueError("Relative tolerance must be positive.")

        self._rtol = value

    cdef _build_cache(self):
        """
        Caches the roots and weights of the Gauss-Legendre quadrature.
        """

        cdef:
            int order, n, i

        # cache roots and weights for x dimension
        n = (self._x_max_order + self._x_min_order) * (self._x_max_order - self._x_min_order + 1) // 2

        self._x_roots = np.zeros(n, dtype=np.float64)
        self._x_weights = np.zeros(n, dtype=np.float64)

        i = 0
        for order in range(self._x_min_order, self._x_max_order + 1):
            self._x_roots[i:i + order], self._x_weights[i:i + order] = roots_legendre(order)
            i += order

        self._x_roots_mv = self._x_roots
        self._x_weights_mv = self._x_weights
        
        # cache roots and weights for y dimension
        n = (self._y_max_order + self._y_min_order) * (self._y_max_order - self._y_min_order + 1) // 2

        self._y_roots = np.zeros(n, dtype=np.float64)
        self._y_weights = np.zeros(n, dtype=np.float64)

        i = 0
        for order in range(self._y_min_order, self._y_max_order + 1):
            self._y_roots[i:i + order], self._y_weights[i:i + order] = roots_legendre(order)
            i += order

        self._y_roots_mv = self._y_roots
        self._y_weights_mv = self._y_weights
    
    cpdef double profile_evaluate(self, int n, double x_lower, double x_upper, Function1D y_lower, Function1D y_upper):
        
        cdef:
            int i

        for i in range(n):
            self.evaluate(x_lower, x_upper, y_lower, y_upper)

        return 0.
    
    cpdef double evaluate_overhead(self, int n):
        cdef:
            int i

        for i in range(n):
            continue
        
        return 0.

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cdef double evaluate(self, double x_lower, double x_upper, Function1D y_lower, Function1D y_upper) except? -1e999:
        """
        Integrates a two-dimensional function over a finite interval.

        :param double x_lower: Lower limit of integration in the x dimension.
        :param double x_upper: Upper limit of integration in the x dimension.
        :param Function1D y_lower: Lower limit of integration in the y dimension as a function of x.
        :param Function1D y_upper: Upper limit of integration in the y dimension as a function of x.

        :returns: Gaussian quadrature approximation to integral.
        """

        cdef:
            int x_order, y_order
            int x_ibegin, y_ibegin
            int i, j
            double current_integral, previous_integral, y_contribution, error
            double x, y, x_offset, y_offset, x_slope, y_slope, scale, rtol, y_lower_val, y_upper_val

        # set initial iteration values
        previous_integral = INFINITY
        x_ibegin = 0
        y_ibegin = 0

        # x variable transformations from [-1, 1] to [x_lower, x_upper] and [y_lower, y_upper]
        x_offset = 0.5 * (x_lower + x_upper)
        x_slope = 0.5 * (x_upper - x_lower)

        # create local caches
        x_order = self._x_min_order
        y_order = self._y_min_order
        
        while True:
            current_integral = 0.
            for i in range(x_ibegin, x_ibegin + x_order):
                x = x_offset + x_slope * self._x_roots_mv[i]
                y_contribution = 0.

                # calculate integration limits and transformation from [-1, 1] to [y_lower, y_upper]
                y_lower_val = y_lower.evaluate(x)
                y_upper_val = y_upper.evaluate(x)
                y_offset = 0.5 * (y_lower_val + y_upper_val)
                y_slope = 0.5 * (y_upper_val - y_lower_val)

                for j in range(y_ibegin, y_ibegin + y_order):
                    y = y_offset + y_slope * self._y_roots_mv[j]
                    y_contribution += self._y_weights_mv[j] * self.function.evaluate(x, y)
            
                current_integral += self._x_weights_mv[i] * y_contribution
            current_integral *= x_slope * y_slope

            error = abs(current_integral - previous_integral)

            # terminate integration if relative error is less than tolerance
            # or if both x and y maximum orders have been reached
            if error < self._rtol * abs(current_integral) or (x_order == self._x_max_order and y_order == self._y_max_order):
                break
            
            previous_integral = current_integral

            # increase the orders of x and y and shift indexes
            if x_order < self._x_max_order:
                x_ibegin += x_order
                x_order += 1

            if y_order < self._y_max_order:
                y_ibegin += y_order
                y_order += 1

        return current_integral
