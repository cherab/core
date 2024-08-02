# cython: language_level=3

# Copyright 2016-2022 Euratom
# Copyright 2016-2022 United Kingdom Atomic Energy Authority
# Copyright 2016-2022 Centro de Investigaciones Energéticas, Medioambientales y Tecnológicas
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

from raysect.core.math.function.float cimport autowrap_function1d, Constant1D

from libc.math cimport INFINITY
cimport cython


cdef class Integrator1D:
    """
    Compute a definite integral of a one-dimensional function.

    :ivar Function1D integrand: A 1D function to integrate.
    """

    @property
    def integrand(self):
        """
        A 1D function to integrate.

        :rtype: int
        """
        return self.function

    @integrand.setter
    def integrand(self, object func not None):

        self.function = autowrap_function1d(func)

    cdef double evaluate(self, double a, double b) except? -1e999:

        raise NotImplementedError("The evaluate() method has not been implemented.")

    def __call__(self, double a, double b):
        """
        Integrates a one-dimensional function over a finite interval.

        :param double a: Lower limit of integration.
        :param double b: Upper limit of integration.

        :returns: Definite integral of a one-dimensional function.
        """

        return self.evaluate(a, b)


cdef class GaussianQuadrature(Integrator1D):
    """
    Compute an integral of a one-dimensional function over a finite interval
    using fixed-tolerance Gaussian quadrature.
    (see Scipy `quadrature <https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.quadrature.html>`).

    :param object integrand: A 1D function to integrate. Default is Constant1D(0).
    :param double relative_tolerance: Iteration stops when relative error between
        last two iterates is less than this value. Default is 1.e-5.
    :param int max_order: Maximum order on Gaussian quadrature. Default is 50.
    :param int min_order: Minimum order on Gaussian quadrature. Default is 1.

    :ivar Function1D integrand: A 1D function to integrate.
    :ivar double relative_tolerance: Iteration stops when relative error between
        last two iterates is less than this value.
    :ivar int max_order: Maximum order on Gaussian quadrature.
    :ivar int min_order: Minimum order on Gaussian quadrature.
    """

    def __init__(self, object integrand=Constant1D(0), double relative_tolerance=1.e-5, int max_order=50, int min_order=1):

        if min_order < 1 or max_order < 1:
            raise ValueError("Order of Gaussian quadrature must be >= 1.")

        if min_order > max_order:
            raise ValueError("Minimum order of Gaussian quadrature must be less than or equal to the maximum order.")

        self._min_order = min_order
        self._max_order = max_order
        self._build_cache()

        self.integrand = integrand

        self.relative_tolerance = relative_tolerance

    @property
    def min_order(self):
        """
        Minimum order on Gaussian quadrature.

        :rtype: int
        """
        return self._min_order

    @min_order.setter
    def min_order(self, int value):

        if value < 1:
            raise ValueError("Order of Gaussian quadrature must be >= 1.")

        if value > self._max_order:
            raise ValueError("Minimum order of Gaussian quadrature must be less than or equal to the maximum order.")

        self._min_order = value

        self._build_cache()

    @property
    def max_order(self):
        """
        Maximum order on Gaussian quadrature.

        :rtype: float
        """
        return self._max_order

    @max_order.setter
    def max_order(self, int value):

        if value < 1:
            raise ValueError("Order of Gaussian quadrature must be >= 1.")

        if value < self._min_order:
            raise ValueError("Maximum order of Gaussian quadrature must be greater than or equal to the minimum order.")

        self._max_order = value

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

        n = (self._max_order + self._min_order) * (self._max_order - self._min_order + 1) // 2

        self._roots = np.zeros(n, dtype=np.float64)
        self._weights = np.zeros(n, dtype=np.float64)

        i = 0
        for order in range(self._min_order, self._max_order + 1):
            self._roots[i:i + order], self._weights[i:i + order] = roots_legendre(order)
            i += order

        self._roots_mv = self._roots
        self._weights_mv = self._weights

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cdef double evaluate(self, double a, double b) except? -1e999:
        """
        Integrates a one-dimensional function over a finite interval.

        :param double a: Lower limit of integration.
        :param double b: Upper limit of integration.

        :returns: Gaussian quadrature approximation to integral.
        """

        cdef:
            int order, i, ibegin
            double newval, oldval, error, x, c, d

        oldval = INFINITY
        newval = 0
        ibegin = 0
        c = 0.5 * (a + b)
        d = 0.5 * (b - a)

        for order in range(self._min_order, self._max_order + 1):
            newval = 0
            for i in range(ibegin, ibegin + order):
                x = c + d * self._roots_mv[i]
                newval += self._weights_mv[i] * self.function.evaluate(x)
            newval *= d

            error = abs(newval - oldval)
            oldval = newval

            ibegin += order

            if error < self._rtol * abs(newval):
                break

        return newval
