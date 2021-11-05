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
from cherab.core.math.function cimport autowrap_function2d
from cherab.core.math.interpolators.utility cimport find_index, derivatives_array, factorial

# required by numpy c-api
import_array()

EPSILON = 1.e-7

cdef class Caching2D(Function2D):
    """
    Precalculate and cache a 2D function on a finite space area.

    The function is sampled and a cubic interpolation is then used to calculate
    a cubic spline approximation of the function. As the spline has a constant
    cost of evaluation, this decreases the evaluation time of functions which
    are very often used.

    The sampling and interpolation are done locally and on demand, so that the
    caching is done progressively when the function is evaluated. Coordinates
    are normalised to the range [0, 1] to avoid float accuracy troubles. The
    values of the function are normalised if their boundaries are given.

    :param object function2d: 2D function to be cached.
    :param tuple space_area: space area where the function has to be cached:
      (minx, maxx, miny, maxy).
    :param tuple resolution: resolution of the sampling: (resolutionx, resolutiony).
    :param no_boundary_error: Behaviour when evaluated outside the caching area.
      When False a ValueError is raised. When True the function is directly
      evaluated (without caching). Default is False.
    :param function_boundaries: Boundaries of the function values for
      normalisation: (min, max). If None, function values are not normalised.
      Default is None.

    .. code-block:: pycon

       >>> from numpy import sqrt
       >>> from time import sleep
       >>> from cherab.core.math import Caching2D
       >>>
       >>> def expensive_radius(x, y):
       >>>     sleep(5)
       >>>     return sqrt(x**2 + y**2)
       >>>
       >>> f1 = Caching2D(expensive_radius, (-5, 5, -5, 5), (0.1, 0.1))
       >>>
       >>> # if you try this, first two executions will be slow, third will be fast
       >>> f1(1.5, 1.5)
       2.121320343595476
       >>> f1(1.6, 1.5)
       2.19317121996626
       >>> f1(1.55, 1.5)
       2.156964925578382
    """

    def __init__(self, object function2d, tuple space_area, tuple resolution, no_boundary_error=False, function_boundaries=None):

        cdef:
            double minx, maxx, miny, maxy
            double deltax, deltay

        self.function = autowrap_function2d(function2d)
        self.no_boundary_error = no_boundary_error

        minx, maxx, miny, maxy = space_area
        deltax, deltay = resolution

        if minx >= maxx:
            raise ValueError('Coordinate range is not consistent, minimum must be less than maximum ({} >= {})!'.format(minx, maxx))
        if miny >= maxy:
            raise ValueError('Coordinate range is not consistent, minimum must be less than maximum ({} >= {})!'.format(miny, maxy))
        if deltax <= EPSILON:
            raise ValueError('Resolution must be bigger ({} <= {})!'.format(deltax, EPSILON))
        if deltay <= EPSILON:
            raise ValueError('Resolution must be bigger ({} <= {})!'.format(deltay, EPSILON))

        self.x_np = concatenate((array([minx - deltax]), linspace(minx - EPSILON, maxx + EPSILON, max(int((maxx - minx) / deltax) + 1, 2)), array([maxx + deltax])))
        self.y_np = concatenate((array([miny - deltay]), linspace(miny - EPSILON, maxy + EPSILON, max(int((maxy - miny) / deltay) + 1, 2)), array([maxy + deltay])))

        self.x_domain_view = self.x_np
        self.y_domain_view = self.y_np

        self.top_index_x = len(self.x_np) - 1
        self.top_index_y = len(self.y_np) - 1

        # Initialise the caching array
        self.coeffs_view = empty((self.top_index_x - 2, self.top_index_y - 2, 16), dtype=float64)
        self.coeffs_view[:,:,::1] = float('NaN')
        self.calculated_view = empty((self.top_index_x - 2, self.top_index_y - 2), dtype=int8)
        self.calculated_view[:,:] = False

        # Normalise coordinates and data
        self.x_delta_inv = 1 / (self.x_np.max() - self.x_np.min())
        self.x_min = self.x_np.min()
        self.x_np = (self.x_np - self.x_min) * self.x_delta_inv
        self.y_delta_inv = 1 / (self.y_np.max() - self.y_np.min())
        self.y_min = self.y_np.min()
        self.y_np = (self.y_np - self.y_min) * self.y_delta_inv
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

        self.data_view = empty((self.top_index_x+1, self.top_index_y+1), dtype=float64)
        self.data_view[:,::1] = float('NaN')

        # obtain coordinates memory views
        self.x_view = self.x_np
        self.x2_view = self.x_np*self.x_np
        self.x3_view = self.x_np*self.x_np*self.x_np
        self.y_view = self.y_np
        self.y2_view = self.y_np*self.y_np
        self.y3_view = self.y_np*self.y_np*self.y_np

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double evaluate(self, double px, double py) except? -1e999:
        """
        Evaluate the cached 2D function.

        The function is cached in the vicinity of (px, py) if not already done.

        :param double px: x coordinate
        :param double py: y coordinate
        :return: The evaluated value
        """

        cdef int i_x, i_y

        i_x = find_index(self.x_domain_view, px)
        i_y = find_index(self.y_domain_view, py)

        if 1 <= i_x <= self.top_index_x-2:
            if 1 <= i_y <= self.top_index_y-2:
                return self._evaluate(px, py, i_x, i_y)

        # value is outside of permitted limits
        if self.no_boundary_error:
            return self.function.evaluate(px, py)
        else:
            min_range_x = self.x_domain_view[1]
            max_range_x = self.x_domain_view[self.top_index_x - 1]

            min_range_y = self.y_domain_view[1]
            max_range_y = self.y_domain_view[self.top_index_y - 1]

            raise ValueError("The specified value (x={}, y={}) is outside the range of the supplied data: "
                             "x bounds=({}, {}), y bounds=({}, {})".format(px, py, min_range_x, max_range_x, min_range_y, max_range_y))

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double _evaluate(self, double px, double py, int i_x, int i_y):
        """
        Calculate if not already done then evaluate the polynomial valid in the
        area given by (i_x, i_y) at position (px, py).
        Calculation of the polynomial includes sampling the function where not
        already done and interpolating.

        :param double px: x coordinate
        :param double py: y coordinate
        :param int i_x: x index of the area of interest
        :param int i_y: y index of the area of interest
        :return: The evaluated value
        """

        cdef:
            int u, v, l, i, j, i_x_p, i_y_p
            double value
            double delta_x, delta_y, px2, py2, px3, py3
            npy_intp cv_size
            npy_intp cm_size[2]
            double[::1] cv_view, coeffs_view
            double[:, ::1] cm_view

        # If the concerned polynomial has not yet been calculated:
        i_x_p = i_x - 1  # polynomial index
        i_y_p = i_y - 1  # polynomial index
        if not self.calculated_view[i_x_p, i_y_p]:

            # sample the data needed
            for u in range(i_x-1, i_x+3):
                for v in range(i_y-1, i_y+3):
                    if isnan(self.data_view[u, v]):
                        value = self.function.evaluate(self.x_domain_view[u], self.y_domain_view[v])
                        if not isnan(value):
                            # data values are normalised here
                            self.data_view[u, v] = (value - self.data_min) * self.data_delta_inv

            # Create constraint matrix (un-optimised)
            # cv_view = zeros((16,), dtype=float64)     # constraints vector
            # cm_view = zeros((16, 16), dtype=float64)  # constraints matrix

            # Create constraint matrix (optimised using numpy c-api)
            cv_size = 16
            cv_view = PyArray_ZEROS(1, &cv_size, NPY_FLOAT64, 0)
            cm_size[:] = [16, 16]
            cm_view = PyArray_ZEROS(2, cm_size, NPY_FLOAT64, 0)

            # Fill the constraints matrix
            l = 0
            for u in range(i_x, i_x+2):
                for v in range(i_y, i_y+2):

                    # knot values
                    cm_view[l, 0] = 1.
                    cm_view[l, 1] = self.y_view[v]
                    cm_view[l, 2] = self.y2_view[v]
                    cm_view[l, 3] = self.y3_view[v]
                    cm_view[l, 4] = self.x_view[u]
                    cm_view[l, 5] = self.x_view[u]*self.y_view[v]
                    cm_view[l, 6] = self.x_view[u]*self.y2_view[v]
                    cm_view[l, 7] = self.x_view[u]*self.y3_view[v]
                    cm_view[l, 8] = self.x2_view[u]
                    cm_view[l, 9] = self.x2_view[u]*self.y_view[v]
                    cm_view[l, 10] = self.x2_view[u]*self.y2_view[v]
                    cm_view[l, 11] = self.x2_view[u]*self.y3_view[v]
                    cm_view[l, 12] = self.x3_view[u]
                    cm_view[l, 13] = self.x3_view[u]*self.y_view[v]
                    cm_view[l, 14] = self.x3_view[u]*self.y2_view[v]
                    cm_view[l, 15] = self.x3_view[u]*self.y3_view[v]
                    cv_view[l] = self.data_view[u, v]
                    l += 1

                    # derivative along x
                    cm_view[l, 4] = 1.
                    cm_view[l, 5] = self.y_view[v]
                    cm_view[l, 6] = self.y2_view[v]
                    cm_view[l, 7] = self.y3_view[v]
                    cm_view[l, 8] = 2.*self.x_view[u]
                    cm_view[l, 9] = 2.*self.x_view[u]*self.y_view[v]
                    cm_view[l, 10] = 2.*self.x_view[u]*self.y2_view[v]
                    cm_view[l, 11] = 2.*self.x_view[u]*self.y3_view[v]
                    cm_view[l, 12] = 3.*self.x2_view[u]
                    cm_view[l, 13] = 3.*self.x2_view[u]*self.y_view[v]
                    cm_view[l, 14] = 3.*self.x2_view[u]*self.y2_view[v]
                    cm_view[l, 15] = 3.*self.x2_view[u]*self.y3_view[v]
                    delta_x = self.x_view[u+1] - self.x_view[u-1]
                    cv_view[l] = (self.data_view[u+1, v] - self.data_view[u-1, v])/delta_x
                    l += 1

                    # derivative along y
                    cm_view[l, 1] = 1.
                    cm_view[l, 2] = 2.*self.y_view[v]
                    cm_view[l, 3] = 3.*self.y2_view[v]
                    cm_view[l, 5] = self.x_view[u]
                    cm_view[l, 6] = 2.*self.x_view[u]*self.y_view[v]
                    cm_view[l, 7] = 3.*self.x_view[u]*self.y2_view[v]
                    cm_view[l, 9] = self.x2_view[u]
                    cm_view[l, 10] = 2.*self.x2_view[u]*self.y_view[v]
                    cm_view[l, 11] = 3.*self.x2_view[u]*self.y2_view[v]
                    cm_view[l, 13] = self.x3_view[u]
                    cm_view[l, 14] = 2.*self.x3_view[u]*self.y_view[v]
                    cm_view[l, 15] = 3.*self.x3_view[u]*self.y2_view[v]
                    delta_y = self.y_view[v+1] - self.y_view[v-1]
                    cv_view[l] = (self.data_view[u, v+1] - self.data_view[u, v-1])/delta_y
                    l += 1

                    # cross derivative
                    cm_view[l, 5] = 1.
                    cm_view[l, 6] = 2.*self.y_view[v]
                    cm_view[l, 7] = 3.*self.y2_view[v]
                    cm_view[l, 9] = 2.*self.x_view[u]
                    cm_view[l, 10] = 4.*self.x_view[u]*self.y_view[v]
                    cm_view[l, 11] = 6.*self.x_view[u]*self.y2_view[v]
                    cm_view[l, 13] = 3.*self.x2_view[u]
                    cm_view[l, 14] = 6.*self.x2_view[u]*self.y_view[v]
                    cm_view[l, 15] = 9.*self.x2_view[u]*self.y2_view[v]
                    cv_view[l] = (self.data_view[u+1, v+1] - self.data_view[u+1, v-1] - self.data_view[u-1, v+1] + self.data_view[u-1, v-1])/(delta_x*delta_y)
                    l += 1

            # Solve the linear system and fill the caching coefficients array
            coeffs_view = solve(cm_view, cv_view)
            self.coeffs_view[i_x_p, i_y_p, :] = coeffs_view

            # Denormalisation
            for i in range(4):
                for j in range(4):
                    coeffs_view[4 * i + j] = self.data_delta * (self.x_delta_inv ** i * self.y_delta_inv ** j / (factorial(j) * factorial(i)) * self._evaluate_polynomial_derivative(i_x_p, i_y_p, -self.x_delta_inv * self.x_min, -self.y_delta_inv * self.y_min, i, j))
            coeffs_view[0] = coeffs_view[0] + self.data_min
            self.coeffs_view[i_x_p, i_y_p, :] = coeffs_view

            self.calculated_view[i_x_p, i_y_p] = True

        px2 = px*px
        px3 = px2*px
        py2 = py*py
        py3 = py2*py

        return     (self.coeffs_view[i_x_p, i_y_p,  0] + self.coeffs_view[i_x_p, i_y_p,  1]*py + self.coeffs_view[i_x_p, i_y_p,  2]*py2 + self.coeffs_view[i_x_p, i_y_p,  3]*py3) + \
               px *(self.coeffs_view[i_x_p, i_y_p,  4] + self.coeffs_view[i_x_p, i_y_p,  5]*py + self.coeffs_view[i_x_p, i_y_p,  6]*py2 + self.coeffs_view[i_x_p, i_y_p,  7]*py3) + \
               px2*(self.coeffs_view[i_x_p, i_y_p,  8] + self.coeffs_view[i_x_p, i_y_p,  9]*py + self.coeffs_view[i_x_p, i_y_p, 10]*py2 + self.coeffs_view[i_x_p, i_y_p, 11]*py3) + \
               px3*(self.coeffs_view[i_x_p, i_y_p, 12] + self.coeffs_view[i_x_p, i_y_p, 13]*py + self.coeffs_view[i_x_p, i_y_p, 14]*py2 + self.coeffs_view[i_x_p, i_y_p, 15]*py3)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double _evaluate_polynomial_derivative(self, int i_x, int i_y, double px, double py, int der_x, int der_y):
        """
        Evaluate the derivatives of the polynomial valid in the area given by
        'i_x', 'i_y' at position ('px', 'py'). The order of
        derivative along each axis is given by 'der_x', 'der_y'.
        """

        cdef double[::1] x_values, y_values

        x_values = derivatives_array(px, der_x)
        y_values = derivatives_array(py, der_y)

        return x_values[0]*(y_values[0]*self.coeffs_view[i_x, i_y,  0] + y_values[1]*self.coeffs_view[i_x, i_y,  1] + y_values[2]*self.coeffs_view[i_x, i_y,  2] + y_values[3]*self.coeffs_view[i_x, i_y,  3]) + \
               x_values[1]*(y_values[0]*self.coeffs_view[i_x, i_y,  4] + y_values[1]*self.coeffs_view[i_x, i_y,  5] + y_values[2]*self.coeffs_view[i_x, i_y,  6] + y_values[3]*self.coeffs_view[i_x, i_y,  7]) + \
               x_values[2]*(y_values[0]*self.coeffs_view[i_x, i_y,  8] + y_values[1]*self.coeffs_view[i_x, i_y,  9] + y_values[2]*self.coeffs_view[i_x, i_y, 10] + y_values[3]*self.coeffs_view[i_x, i_y, 11]) + \
               x_values[3]*(y_values[0]*self.coeffs_view[i_x, i_y, 12] + y_values[1]*self.coeffs_view[i_x, i_y, 13] + y_values[2]*self.coeffs_view[i_x, i_y, 14] + y_values[3]*self.coeffs_view[i_x, i_y, 15])

