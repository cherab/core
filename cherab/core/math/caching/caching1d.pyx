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

from numpy import array, empty, int8, float64, concatenate, linspace
from numpy.linalg import solve

cimport cython
from libc.math cimport isnan
from numpy cimport ndarray, PyArray_ZEROS, NPY_FLOAT64, npy_intp, import_array
from cherab.core.math.function cimport autowrap_function1d
from cherab.core.math.interpolators.utility cimport find_index, derivatives_array, factorial

# required by numpy c-api
import_array()

EPSILON = 1.e-7

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

    def __init__(self, object function1d, tuple space_area, double resolution, no_boundary_error=False, function_boundaries=None):

        cdef:
            double minx, maxx, deltax

        self.function = autowrap_function1d(function1d)
        self.no_boundary_error = no_boundary_error

        minx, maxx = space_area
        deltax = resolution

        if minx >= maxx:
            raise ValueError('Coordinate range is not consistent, minimum must be less than maximum ({} >= {})!'.format(minx, maxx))
        if deltax <= EPSILON:
            raise ValueError('Resolution must be bigger ({} <= {})!'.format(deltax, EPSILON))

        self.x_np = concatenate((array([minx - deltax]), linspace(minx - EPSILON, maxx + EPSILON, max(int((maxx - minx) / deltax) + 1, 2)), array([maxx + deltax])))

        self.x_domain_view = self.x_np

        self.top_index_x = len(self.x_np) - 1

        # Initialise the caching array
        self.coeffs_view = empty((self.top_index_x - 2, 4), dtype=float64)
        self.coeffs_view[:,::1] = float('NaN')
        self.calculated_view = empty((self.top_index_x - 2,), dtype=int8)
        self.calculated_view[:] = False

        # Normalise coordinates and data
        self.x_delta_inv = 1 / (self.x_np.max() - self.x_np.min())
        self.x_min = self.x_np.min()
        self.x_np = (self.x_np - self.x_min) * self.x_delta_inv
        if function_boundaries is not None:
            self.data_min, self.data_max = function_boundaries
            self.data_delta = self.data_max - self.data_min
            # If data contains only one value (not filtered before) cancel the
            # normalisation scaling by setting data_delta to 1:
            if self.data_delta == 0:
                self.data_delta = 1
        else:
            # no normalisation of data values
            self.data_delta = 1
            self.data_min = 0
            self.data_max = float('NaN')
        self.data_delta_inv = 1 / self.data_delta

        self.data_view = empty((self.top_index_x+1,), dtype=float64)
        self.data_view[::1] = float('NaN')

        # obtain coordinates memory views
        self.x_view = self.x_np
        self.x2_view = self.x_np*self.x_np
        self.x3_view = self.x_np*self.x_np*self.x_np

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double evaluate(self, double px) except? -1e999:
        """
        Evaluate the cached 1D function.

        The function is cached in the vicinity of px if not already done.

        :param double px: coordinate
        :return: The evaluated value
        """

        cdef int i_x

        i_x = find_index(self.x_domain_view, px)

        if 1 <= i_x <= self.top_index_x - 2:
            return self._evaluate(px, i_x)

        # value is outside of permitted limits
        if self.no_boundary_error:
            return self.function.evaluate(px)
        else:
            min_range_x = self.x_domain_view[1]
            max_range_x = self.x_domain_view[self.top_index_x - 1]

            raise ValueError("The specified value (x={}) is outside the range of the supplied data: "
                             "x bounds=({}, {})".format(px, min_range_x, max_range_x))

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double _evaluate(self, double px, int i_x):
        """
        Calculate if not already done then evaluate the polynomial valid in the
        area given by i_x at position px.
        Calculation of the polynomial includes sampling the function where not
        already done and interpolating.

        :param double px: coordinate
        :param int i_x: index of the area of interest
        :return: The evaluated value
        """

        cdef:
            int u, l, i, i_x_p
            double value
            double delta_x
            npy_intp cv_size
            npy_intp cm_size[2]
            double[::1] cv_view, coeffs_view
            double[:, ::1] cm_view

        # If the concerned polynomial has not yet been calculated:
        i_x_p = i_x - 1  # polynomial index
        if not self.calculated_view[i_x_p]:

            # sample the data needed
            for u in range(i_x-1, i_x+3):
                if isnan(self.data_view[u]):
                    value = self.function.evaluate(self.x_domain_view[u])
                    if not isnan(value):
                        # data values are normalised here
                        self.data_view[u] = (value - self.data_min) * self.data_delta_inv

            # Create constraint matrix (un-optimised)
            # cv_view = zeros((4,), dtype=float64)  # constraints vector
            # cm_view = zeros((4, 4), dtype=float64)  # constraints matrix

            # Create constraint matrix (optimised using numpy c-api)
            cv_size = 4
            cv_view = PyArray_ZEROS(1, &cv_size, NPY_FLOAT64, 0)
            cm_size[:] = [4, 4]
            cm_view = PyArray_ZEROS(2, cm_size, NPY_FLOAT64, 0)

            # Fill the constraints matrix
            l = 0
            for u in range(i_x, i_x+2):

                # knot values
                cm_view[l, 0] = 1.
                cm_view[l, 1] = self.x_view[u]
                cm_view[l, 2] = self.x2_view[u]
                cm_view[l, 3] = self.x3_view[u]
                cv_view[l] = self.data_view[u]
                l += 1

                # derivative
                cm_view[l, 1] = 1.
                cm_view[l, 2] = 2.*self.x_view[u]
                cm_view[l, 3] = 3.*self.x2_view[u]
                delta_x = self.x_view[u+1] - self.x_view[u-1]
                cv_view[l] = (self.data_view[u+1] - self.data_view[u-1])/delta_x
                l += 1

            # Solve the linear system and fill the caching coefficients array
            coeffs_view = solve(cm_view, cv_view)
            self.coeffs_view[i_x_p, :] = coeffs_view

            # Denormalisation
            for i in range(4):
                coeffs_view[i] = self.data_delta * (self.x_delta_inv ** i / factorial(i) * self._evaluate_polynomial_derivative(i_x_p, -self.x_delta_inv * self.x_min, i))
            coeffs_view[0] = coeffs_view[0] + self.data_min
            self.coeffs_view[i_x_p, :] = coeffs_view

            self.calculated_view[i_x_p] = True

        return self.coeffs_view[i_x_p, 0] + self.coeffs_view[i_x_p,  1] * px + self.coeffs_view[i_x_p,  2] * px * px + self.coeffs_view[i_x_p, 3] * px * px * px

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double _evaluate_polynomial_derivative(self, int i_x, double px, int der_x):
        """
        Evaluate the derivatives of the polynomial valid in the area given by
        'i_x' at position 'px'. The order of derivative along each axis is
        given by 'der_x'.
        """

        cdef double[::1] x_values

        x_values = derivatives_array(px, der_x)

        return x_values[0]*self.coeffs_view[i_x, 0] + x_values[1]*self.coeffs_view[i_x, 1] + x_values[2]*self.coeffs_view[i_x, 2] + x_values[3]*self.coeffs_view[i_x, 3]

