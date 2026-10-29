# cython: language_level=3

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

cimport cython
from raysect.core.math.cython.interpolation.cubic cimport calc_coefficients_1d, evaluate_cubic_1d
from cherab.core.math.function cimport autowrap_function1d


cdef class Caching1D(Function1D):
    """
    Precalculate and cache a 1D function on a finite space area.

    The function is sampled and a cubic interpolation is then used to calculate
    a cubic spline approximation of the function. As the spline has a constant
    cost of evaluation, this decreases the evaluation time of functions which
    are very often used.

    The sampling and interpolation are done locally and on demand, so that the
    caching is done progressively when the function is evaluated. Coordinates
    are normalised to the range [0, 1] to avoid float accuracy troubles. The
    values of the function are normalised if their boundaries are given.

    :param object function1d: 1D function to be cached.
    :param tuple space_area: space area where the function has to be cached:
      (minx, maxx).
    :param double resolution: resolution of the sampling
    :param no_boundary_error: Behaviour when evaluated outside the caching area.
      When False a ValueError is raised. When True the function is directly
      evaluated (without caching). Default is False.
    :param function_boundaries: Boundaries of the function values for
      normalisation: (min, max). If None, function values are not normalised.
      Default is None.

    .. code-block:: pycon

       >>> from numpy import sqrt
       >>> from time import sleep
       >>> from cherab.core.math import Caching1D
       >>>
       >>> def expensive_sqrt(x):
       >>>     sleep(5)
       >>>     return sqrt(x)
       >>>
       >>> f1 = Caching1D(expensive_sqrt, (-5, 5), 0.1)
       >>>
       >>> # if you try this, first two executions will be slow, third will be fast
       >>> f1(2.5)
       1.5811388
       >>> f1(2.6)
       1.6124515
       >>> f1(2.55)
       1.5968720
    """

    # The implementation works as follows:
    # - Store a list of which points have already been evaluated, and their values.
    # - For each new value x to evaluate, check if the sample points either side
    #   have already been evaluated.
    #   - If not, evaluate and store them; update the list of already-evaluated points.
    # - Calculate the derivative of the function at these points using the 2nd order
    #   approximation. This requires additionally evaluating the points either side
    #   of the nearest 2 points to x.
    # - Use raysect.core.math.cython.interpolation.cubic's utilities to calculate
    #   the polynomial coefficients for the 1D cubic and then evaluate the cubic
    #   interpolation of the function at the position x.
    # - Note that the sampled points have uniform spacing: this enables a few
    #   optimisations such as analytic calculations of the array index and x
    #   values for the nearest samples to the requested x.

    def __init__(self, function1d, space_area, resolution, no_boundary_error=False, function_boundaries=None):
        self._function = autowrap_function1d(function1d)
        if len(space_area) != 2:
            raise ValueError("Space area should be a tuple (xmin, xmax).")
        self._xmin, self._xmax = space_area
        if resolution <= 0:
            raise ValueError("Resolution must be greater than zero.")
        self._resolution = resolution
        self._nsamples = int((self._xmax - self._xmin) / resolution + 1)
        self._fsamples = np.empty(self._nsamples)
        self._sampled = np.full(self._nsamples, False, dtype='uint8')
        self._no_boundary_error = no_boundary_error
        # TODO: normalise using function_boundaries to avoid numerical issues.

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double _get_and_cache(self, int index) except? -1e999:
        cdef double x, fsamp

        if not self._sampled[index]:
            x = self._xmin + index * self._resolution
            fsamp = self._function.evaluate(x)
            self._sampled[index] = True
            self._fsamples[index] = fsamp
        else:
            fsamp = self._fsamples[index]
        return fsamp

    # TODO: investigate whether to cache the cubic spline coefficients instead
    # of re-calculating them every time. Performance/memory trade-off.
    # Raysect caches the coefficients.
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef double evaluate(self, double x) except? -1e999:
        cdef:
            int lower_index
            double fprev, fnext, feval, xnorm
            double a[4]  # Cubic coefficients
            double f[2]  # Function evaluations
            double dfdx[2]  # Function derivative evaluations

        if self._xmin <= x < self._xmax:
            # Sampling is uniform, so lower_index is determined analytically.
            lower_index = int((x - self._xmin) // self._resolution)
            f[0] = self._get_and_cache(lower_index)
            f[1] = self._get_and_cache(lower_index + 1)
            # Calculate derivatives. calc_coefficiets_1d requires derivatives
            # normalised to the unit interval, i.e. for dx = 1.
            if lower_index == 0:
                dfdx[0] = (f[1] - f[0])
            else:
                fprev = self._get_and_cache(lower_index - 1)
                dfdx[0] = (f[1] - fprev) / 2
            if lower_index == self._nsamples - 2:
                dfdx[1] = (f[1] - f[0])
            else:
                fnext = self._get_and_cache(lower_index + 2)
                dfdx[1] = (fnext - f[0]) / 2
            calc_coefficients_1d(f, dfdx, a)
            # lower_x = xmin + resolution * index
            # xnorm = (x - lower_x) / res = (x - xmin) / resolution - index
            xnorm = (x - self._xmin) / self._resolution - lower_index
            feval = evaluate_cubic_1d(a, xnorm)
            return feval

        # Special case if x is exactly at the maximum allowed value.
        if x == self._xmax:
            return self._get_and_cache(self._nsamples - 1)

        if self._no_boundary_error:
            return self._function.evaluate(x)

        raise ValueError(
            "x is outside the permitted range ({}, {})".format(self._xmin, self._xmax)
        )
