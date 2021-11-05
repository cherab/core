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
from numpy cimport ndarray, PyArray_ZEROS, PyArray_SimpleNew, NPY_FLOAT64, npy_intp, import_array
from cherab.core.math.function cimport autowrap_function3d
from cherab.core.math.interpolators.utility cimport find_index, derivatives_array, factorial

# required by numpy c-api
import_array()

EPSILON = 1.e-7

cdef class Caching3D(Function3D):
    """
    Precalculate and cache a 3D function on a finite space area.

    The function is sampled and a cubic interpolation is then used to calculate
    a cubic spline approximation of the function. As the spline has a constant
    cost of evaluation, this decreases the evaluation time of functions which
    are very often used.

    The sampling and interpolation are done locally and on demand, so that the
    caching is done progressively when the function is evaluated. Coordinates
    are normalised to the range [0, 1] to avoid float accuracy troubles. The
    values of the function are normalised if their boundaries are given.

    :param object function3d: 3D function to be cached.
    :param tuple space_area: space area where the function has to be cached:
      (minx, maxx, miny, maxy, minz, maxz).
    :param tuple resolution: resolution of the sampling:
      (resolutionx, resolutiony, resolutionz).
    :param no_boundary_error: Behaviour when evaluated outside the caching area.
      When False a ValueError is raised. When True the function is directly
      evaluated (without caching). Default is False.
    :param function_boundaries: Boundaries of the function values for
      normalisation: (min, max). If None, function values are not normalised.
      Default is None.

    .. code-block:: pycon

       >>> from numpy import sqrt
       >>> from time import sleep
       >>> from cherab.core.math import Caching3D
       >>>
       >>> def expensive_radius(x, y, z):
       >>>     sleep(5)
       >>>     return sqrt(x**2 + y**2 + z**2)
       >>>
       >>> f1 = Caching3D(expensive_radius, (-5, 5, -5, 5, -5, 5), (0.1, 0.1, 0.1))
       >>>
       >>> # if you try this, first two executions will be slow, third will be fast
       >>> # Note: the first execution might be particularly slow, this is because it
       >>> # sets up the caching structures on first execution.
       >>> f1(1.5, 1.5, 1.5)
       2.598076
       >>> f1(1.6, 1.5, 1.5)
       2.657066
       >>> f1(1.55, 1.5, 1.5)
       2.627260
    """

    def __init__(self, object function3d, tuple space_area, tuple resolution, no_boundary_error=False, function_boundaries=None):

        cdef:
            double minx, maxx, miny, maxy, minz, maxz
            double deltax, deltay, deltaz

        self.function = autowrap_function3d(function3d)
        self.no_boundary_error = no_boundary_error

        minx, maxx, miny, maxy, minz, maxz = space_area
        deltax, deltay, deltaz = resolution

        if minx >= maxx:
            raise ValueError('Coordinate range is not consistent, minimum must be less than maximum ({} >= {})!'.format(minx, maxx))
        if miny >= maxy:
            raise ValueError('Coordinate range is not consistent, minimum must be less than maximum ({} >= {})!'.format(miny, maxy))
        if minz >= maxz:
            raise ValueError('Coordinate range is not consistent, minimum must be less than maximum ({} >= {})!'.format(minz, maxz))
        if deltax <= EPSILON:
            raise ValueError('Resolution must be bigger ({} <= {})!'.format(deltax, EPSILON))
        if deltay <= EPSILON:
            raise ValueError('Resolution must be bigger ({} <= {})!'.format(deltay, EPSILON))
        if deltaz <= EPSILON:
            raise ValueError('Resolution must be bigger ({} <= {})!'.format(deltaz, EPSILON))

        self.x_np = concatenate((array([minx - deltax]), linspace(minx - EPSILON, maxx + EPSILON, max(int((maxx - minx) / deltax) + 1, 2)), array([maxx + deltax])))
        self.y_np = concatenate((array([miny - deltay]), linspace(miny - EPSILON, maxy + EPSILON, max(int((maxy - miny) / deltay) + 1, 2)), array([maxy + deltay])))
        self.z_np = concatenate((array([minz - deltaz]), linspace(minz - EPSILON, maxz + EPSILON, max(int((maxz - minz) / deltaz) + 1, 2)), array([maxz + deltaz])))

        self.x_domain_view = self.x_np
        self.y_domain_view = self.y_np
        self.z_domain_view = self.z_np

        self.top_index_x = len(self.x_np) - 1
        self.top_index_y = len(self.y_np) - 1
        self.top_index_z = len(self.z_np) - 1

        # Initialise the caching array
        self.coeffs_view = empty((self.top_index_x - 2, self.top_index_y - 2, self.top_index_z - 2, 64), dtype=float64)
        self.coeffs_view[:,:,:,::1] = float('NaN')
        self.calculated_view = empty((self.top_index_x - 2, self.top_index_y - 2, self.top_index_z - 2), dtype=int8)
        self.calculated_view[:,:,:] = False

        # Normalise coordinates and data
        self.x_delta_inv = 1 / (self.x_np.max() - self.x_np.min())
        self.x_min = self.x_np.min()
        self.x_np = (self.x_np - self.x_min) * self.x_delta_inv
        self.y_delta_inv = 1 / (self.y_np.max() - self.y_np.min())
        self.y_min = self.y_np.min()
        self.y_np = (self.y_np - self.y_min) * self.y_delta_inv
        self.z_delta_inv = 1 / (self.z_np.max() - self.z_np.min())
        self.z_min = self.z_np.min()
        self.z_np = (self.z_np - self.z_min) * self.z_delta_inv
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

        self.data_view = empty((self.top_index_x+1, self.top_index_y+1, self.top_index_z+1), dtype=float64)
        self.data_view[:,:,::1] = float('NaN')

        # obtain coordinates memory views
        self.x_view = self.x_np
        self.x2_view = self.x_np*self.x_np
        self.x3_view = self.x_np*self.x_np*self.x_np
        self.y_view = self.y_np
        self.y2_view = self.y_np*self.y_np
        self.y3_view = self.y_np*self.y_np*self.y_np
        self.z_view = self.z_np
        self.z2_view = self.z_np*self.z_np
        self.z3_view = self.z_np*self.z_np*self.z_np

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double evaluate(self, double px, double py, double pz) except? -1e999:
        """
        Evaluate the cached 3D function.

        The function is cached in the vicinity of (px, py, pz) if not already done.

        :param double px: x coordinate
        :param double py: y coordinate
        :param double pz: z coordinate
        :return: The evaluated value
        """

        cdef int i_x, i_y, i_z

        i_x = find_index(self.x_domain_view, px)
        i_y = find_index(self.y_domain_view, py)
        i_z = find_index(self.z_domain_view, pz)

        if 1 <= i_x <= self.top_index_x-2:
            if 1 <= i_y <= self.top_index_y-2:
                if 1 <= i_z <= self.top_index_z-2:
                    return self._evaluate(px, py, pz, i_x, i_y, i_z)

        # value is outside of permitted limits
        if self.no_boundary_error:
            return self.function.evaluate(px, py, pz)
        else:
            min_range_x = self.x_domain_view[1]
            max_range_x = self.x_domain_view[self.top_index_x - 1]

            min_range_y = self.y_domain_view[1]
            max_range_y = self.y_domain_view[self.top_index_y - 1]

            min_range_z = self.z_domain_view[1]
            max_range_z = self.z_domain_view[self.top_index_z - 1]

            raise ValueError("The specified value (x={}, y={}, z={}) is outside the range of the supplied data: "
                             "x bounds=({}, {}), y bounds=({}, {}), z bounds=({}, {})".format(px, py, pz, min_range_x, max_range_x, min_range_y, max_range_y, min_range_z, max_range_z))

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double _evaluate(self, double px, double py, double pz, int i_x, int i_y, int i_z):
        """
        Calculate if not already done then evaluate the polynomial valid in the
        area given by (i_x, i_y, i_z) at position (px, py, pz).
        Calculation of the polynomial includes sampling the function where not
        already done and interpolating.

        :param double px: x coordinate
        :param double py: y coordinate
        :param double pz: z coordinate
        :param int i_x: x index of the area of interest
        :param int i_y: y index of the area of interest
        :param int i_z: z index of the area of interest
        :return: The evaluated value
        """

        cdef:
            int u, v, w, l, i, j, k, i_x_p, i_y_p, i_z_p
            double value
            double delta_x, delta_y, delta_z, px2, py2, pz2, px3, py3, pz3
            npy_intp cv_size
            npy_intp cm_size[2]
            double[::1] cv_view, coeffs_view
            double[:, ::1] cm_view

        # If the concerned polynomial has not yet been calculated:
        i_x_p = i_x - 1  # polynomial index
        i_y_p = i_y - 1  # polynomial index
        i_z_p = i_z - 1  # polynomial index
        if not self.calculated_view[i_x_p, i_y_p, i_z_p]:

            # sample the data needed
            for u in range(i_x-1, i_x+3):
                for v in range(i_y-1, i_y+3):
                    for w in range(i_z-1, i_z+3):
                        if isnan(self.data_view[u, v, w]):
                            value = self.function.evaluate(self.x_domain_view[u], self.y_domain_view[v], self.z_domain_view[w])
                            if not isnan(value):
                                # data values are normalised here
                                self.data_view[u, v, w] = (value - self.data_min) * self.data_delta_inv

            # Create constraint matrix (un-optimised)
            # cv_view = zeros((64,), dtype=float64)       # constraints vector
            # cm_view = zeros((64, 64), dtype=float64)    # constraints matrix

            # Create constraint matrix (optimised using numpy c-api)
            cv_size = 64
            cv_view = PyArray_ZEROS(1, &cv_size, NPY_FLOAT64, 0)
            cm_size[:] = [64, 64]
            cm_view = PyArray_ZEROS(2, cm_size, NPY_FLOAT64, 0)

            # Fill the constraints matrix
            l = 0
            for u in range(i_x, i_x+2):
                for v in range(i_y, i_y+2):
                    for w in range(i_z, i_z+2):

                        # knot values

                        cm_view[l, :] = self._constraints3d(u, v, w, False, False, False)
                        cv_view[l] = self.data_view[u, v, w]
                        l += 1

                        # derivatives along x, y, z

                        cm_view[l, :] = self._constraints3d(u, v, w, True, False, False)
                        delta_x = self.x_view[u+1] - self.x_view[u-1]
                        cv_view[l] = (self.data_view[u+1, v, w] - self.data_view[u-1, v, w])/delta_x
                        l += 1

                        cm_view[l, :] = self._constraints3d(u, v, w, False ,True , False)
                        delta_y = self.y_view[v+1] - self.y_view[v-1]
                        cv_view[l] = (self.data_view[u, v+1, w] - self.data_view[u, v-1, w])/delta_y
                        l += 1

                        cm_view[l, :] = self._constraints3d(u, v, w, False, False, True)
                        delta_z = self.z_view[w+1] - self.z_view[w-1]
                        cv_view[l] = (self.data_view[u, v, w+1] - self.data_view[u, v, w-1])/delta_z
                        l += 1

                        # cross derivatives xy, xz, yz

                        cm_view[l, :] = self._constraints3d(u, v, w, True, True, False)
                        cv_view[l] = (self.data_view[u+1, v+1, w] - self.data_view[u+1, v-1, w] - self.data_view[u-1, v+1, w] + self.data_view[u-1, v-1, w])/(delta_x*delta_y)
                        l += 1

                        cm_view[l, :] = self._constraints3d(u, v, w, True, False, True)
                        cv_view[l] = (self.data_view[u+1, v, w+1] - self.data_view[u+1, v, w-1] - self.data_view[u-1, v, w+1] + self.data_view[u-1, v, w-1])/(delta_x*delta_z)
                        l += 1

                        cm_view[l, :] = self._constraints3d(u, v, w, False, True, True)
                        cv_view[l] = (self.data_view[u, v+1, w+1] - self.data_view[u, v-1, w+1] - self.data_view[u, v+1, w-1] + self.data_view[u, v-1, w-1])/(delta_y*delta_z)
                        l += 1

                        # cross derivative xyz

                        cm_view[l, :] = self._constraints3d(u, v, w, True, True, True)
                        cv_view[l] = (self.data_view[u+1, v+1, w+1] - self.data_view[u+1, v+1, w-1] - self.data_view[u+1, v-1, w+1] + self.data_view[u+1, v-1, w-1] - self.data_view[u-1, v+1, w+1] + self.data_view[u-1, v+1, w-1] + self.data_view[u-1, v-1, w+1] - self.data_view[u-1, v-1, w-1])/(delta_x*delta_y*delta_z)
                        l += 1

            # Solve the linear system and fill the caching coefficients array
            coeffs_view = solve(cm_view, cv_view)
            self.coeffs_view[i_x_p, i_y_p, i_z_p, :] = coeffs_view

            # Denormalisation
            for i in range(4):
                for j in range(4):
                    for k in range(4):
                        coeffs_view[16 * i + 4 * j + k] = self.data_delta * self.x_delta_inv ** i * self.y_delta_inv ** j * self.z_delta_inv ** k / (factorial(i) * factorial(j) * factorial(k)) \
                                                          * self._evaluate_polynomial_derivative(i_x_p, i_y_p, i_z_p, -self.x_delta_inv * self.x_min, -self.y_delta_inv * self.y_min, -self.z_delta_inv * self.z_min, i, j, k)
            coeffs_view[0] = coeffs_view[0] + self.data_min
            self.coeffs_view[i_x_p, i_y_p, i_z_p, :] = coeffs_view

            self.calculated_view[i_x_p, i_y_p, i_z_p] = True

        px2 = px*px
        px3 = px2*px
        py2 = py*py
        py3 = py2*py
        pz2 = pz*pz
        pz3 = pz2*pz

        return         (self.coeffs_view[i_x_p, i_y_p, i_z_p,  0] + self.coeffs_view[i_x_p, i_y_p, i_z_p,  1]*pz + self.coeffs_view[i_x_p, i_y_p, i_z_p,  2]*pz2 + self.coeffs_view[i_x_p, i_y_p, i_z_p,  3]*pz3) + \
                   py *(self.coeffs_view[i_x_p, i_y_p, i_z_p,  4] + self.coeffs_view[i_x_p, i_y_p, i_z_p,  5]*pz + self.coeffs_view[i_x_p, i_y_p, i_z_p,  6]*pz2 + self.coeffs_view[i_x_p, i_y_p, i_z_p,  7]*pz3) + \
                   py2*(self.coeffs_view[i_x_p, i_y_p, i_z_p,  8] + self.coeffs_view[i_x_p, i_y_p, i_z_p,  9]*pz + self.coeffs_view[i_x_p, i_y_p, i_z_p, 10]*pz2 + self.coeffs_view[i_x_p, i_y_p, i_z_p, 11]*pz3) + \
                   py3*(self.coeffs_view[i_x_p, i_y_p, i_z_p, 12] + self.coeffs_view[i_x_p, i_y_p, i_z_p, 13]*pz + self.coeffs_view[i_x_p, i_y_p, i_z_p, 14]*pz2 + self.coeffs_view[i_x_p, i_y_p, i_z_p, 15]*pz3) \
               + px*( \
                       (self.coeffs_view[i_x_p, i_y_p, i_z_p, 16] + self.coeffs_view[i_x_p, i_y_p, i_z_p, 17]*pz + self.coeffs_view[i_x_p, i_y_p, i_z_p, 18]*pz2 + self.coeffs_view[i_x_p, i_y_p, i_z_p, 19]*pz3) + \
                   py *(self.coeffs_view[i_x_p, i_y_p, i_z_p, 20] + self.coeffs_view[i_x_p, i_y_p, i_z_p, 21]*pz + self.coeffs_view[i_x_p, i_y_p, i_z_p, 22]*pz2 + self.coeffs_view[i_x_p, i_y_p, i_z_p, 23]*pz3) + \
                   py2*(self.coeffs_view[i_x_p, i_y_p, i_z_p, 24] + self.coeffs_view[i_x_p, i_y_p, i_z_p, 25]*pz + self.coeffs_view[i_x_p, i_y_p, i_z_p, 26]*pz2 + self.coeffs_view[i_x_p, i_y_p, i_z_p, 27]*pz3) + \
                   py3*(self.coeffs_view[i_x_p, i_y_p, i_z_p, 28] + self.coeffs_view[i_x_p, i_y_p, i_z_p, 29]*pz + self.coeffs_view[i_x_p, i_y_p, i_z_p, 30]*pz2 + self.coeffs_view[i_x_p, i_y_p, i_z_p, 31]*pz3) \
               ) \
               + px2*( \
                       (self.coeffs_view[i_x_p, i_y_p, i_z_p, 32] + self.coeffs_view[i_x_p, i_y_p, i_z_p, 33]*pz + self.coeffs_view[i_x_p, i_y_p, i_z_p, 34]*pz2 + self.coeffs_view[i_x_p, i_y_p, i_z_p, 35]*pz3) + \
                   py *(self.coeffs_view[i_x_p, i_y_p, i_z_p, 36] + self.coeffs_view[i_x_p, i_y_p, i_z_p, 37]*pz + self.coeffs_view[i_x_p, i_y_p, i_z_p, 38]*pz2 + self.coeffs_view[i_x_p, i_y_p, i_z_p, 39]*pz3) + \
                   py2*(self.coeffs_view[i_x_p, i_y_p, i_z_p, 40] + self.coeffs_view[i_x_p, i_y_p, i_z_p, 41]*pz + self.coeffs_view[i_x_p, i_y_p, i_z_p, 42]*pz2 + self.coeffs_view[i_x_p, i_y_p, i_z_p, 43]*pz3) + \
                   py3*(self.coeffs_view[i_x_p, i_y_p, i_z_p, 44] + self.coeffs_view[i_x_p, i_y_p, i_z_p, 45]*pz + self.coeffs_view[i_x_p, i_y_p, i_z_p, 46]*pz2 + self.coeffs_view[i_x_p, i_y_p, i_z_p, 47]*pz3) \
               ) \
               + px3*( \
                       (self.coeffs_view[i_x_p, i_y_p, i_z_p, 48] + self.coeffs_view[i_x_p, i_y_p, i_z_p, 49]*pz + self.coeffs_view[i_x_p, i_y_p, i_z_p, 50]*pz2 + self.coeffs_view[i_x_p, i_y_p, i_z_p, 51]*pz3) + \
                   py *(self.coeffs_view[i_x_p, i_y_p, i_z_p, 52] + self.coeffs_view[i_x_p, i_y_p, i_z_p, 53]*pz + self.coeffs_view[i_x_p, i_y_p, i_z_p, 54]*pz2 + self.coeffs_view[i_x_p, i_y_p, i_z_p, 55]*pz3) + \
                   py2*(self.coeffs_view[i_x_p, i_y_p, i_z_p, 56] + self.coeffs_view[i_x_p, i_y_p, i_z_p, 57]*pz + self.coeffs_view[i_x_p, i_y_p, i_z_p, 58]*pz2 + self.coeffs_view[i_x_p, i_y_p, i_z_p, 59]*pz3) + \
                   py3*(self.coeffs_view[i_x_p, i_y_p, i_z_p, 60] + self.coeffs_view[i_x_p, i_y_p, i_z_p, 61]*pz + self.coeffs_view[i_x_p, i_y_p, i_z_p, 62]*pz2 + self.coeffs_view[i_x_p, i_y_p, i_z_p, 63]*pz3) \
               )

    cdef double _evaluate_polynomial_derivative(self, int i_x, int i_y, int i_z, double px, double py, double pz, int der_x, int der_y, der_z):
        """
        Evaluate the derivatives of the polynomial valid in the area given by
        'i_x', 'i_y' and 'i_z' at position ('px', 'py', 'pz'). The order of
        derivative along each axis is given by 'der_x', 'der_y' and 'der_z'.
        """

        cdef double[::1] x_values, y_values, z_values

        x_values = derivatives_array(px, der_x)
        y_values = derivatives_array(py, der_y)
        z_values = derivatives_array(pz, der_z)

        return   x_values[0]*( \
                   y_values[0]*(z_values[0]*self.coeffs_view[i_x, i_y, i_z,  0] + z_values[1]*self.coeffs_view[i_x, i_y, i_z,  1] + z_values[2]*self.coeffs_view[i_x, i_y, i_z,  2] + z_values[3]*self.coeffs_view[i_x, i_y, i_z,  3]) + \
                   y_values[1]*(z_values[0]*self.coeffs_view[i_x, i_y, i_z,  4] + z_values[1]*self.coeffs_view[i_x, i_y, i_z,  5] + z_values[2]*self.coeffs_view[i_x, i_y, i_z,  6] + z_values[3]*self.coeffs_view[i_x, i_y, i_z,  7]) + \
                   y_values[2]*(z_values[0]*self.coeffs_view[i_x, i_y, i_z,  8] + z_values[1]*self.coeffs_view[i_x, i_y, i_z,  9] + z_values[2]*self.coeffs_view[i_x, i_y, i_z, 10] + z_values[3]*self.coeffs_view[i_x, i_y, i_z, 11]) + \
                   y_values[3]*(z_values[0]*self.coeffs_view[i_x, i_y, i_z, 12] + z_values[1]*self.coeffs_view[i_x, i_y, i_z, 13] + z_values[2]*self.coeffs_view[i_x, i_y, i_z, 14] + z_values[3]*self.coeffs_view[i_x, i_y, i_z, 15]) \
               ) \
               + x_values[1]*( \
                   y_values[0]*(z_values[0]*self.coeffs_view[i_x, i_y, i_z, 16] + z_values[1]*self.coeffs_view[i_x, i_y, i_z, 17] + z_values[2]*self.coeffs_view[i_x, i_y, i_z, 18] + z_values[3]*self.coeffs_view[i_x, i_y, i_z, 19]) + \
                   y_values[1]*(z_values[0]*self.coeffs_view[i_x, i_y, i_z, 20] + z_values[1]*self.coeffs_view[i_x, i_y, i_z, 21] + z_values[2]*self.coeffs_view[i_x, i_y, i_z, 22] + z_values[3]*self.coeffs_view[i_x, i_y, i_z, 23]) + \
                   y_values[2]*(z_values[0]*self.coeffs_view[i_x, i_y, i_z, 24] + z_values[1]*self.coeffs_view[i_x, i_y, i_z, 25] + z_values[2]*self.coeffs_view[i_x, i_y, i_z, 26] + z_values[3]*self.coeffs_view[i_x, i_y, i_z, 27]) + \
                   y_values[3]*(z_values[0]*self.coeffs_view[i_x, i_y, i_z, 28] + z_values[1]*self.coeffs_view[i_x, i_y, i_z, 29] + z_values[2]*self.coeffs_view[i_x, i_y, i_z, 30] + z_values[3]*self.coeffs_view[i_x, i_y, i_z, 31]) \
               ) \
               + x_values[2]*( \
                   y_values[0]*(z_values[0]*self.coeffs_view[i_x, i_y, i_z, 32] + z_values[1]*self.coeffs_view[i_x, i_y, i_z, 33] + z_values[2]*self.coeffs_view[i_x, i_y, i_z, 34] + z_values[3]*self.coeffs_view[i_x, i_y, i_z, 35]) + \
                   y_values[1]*(z_values[0]*self.coeffs_view[i_x, i_y, i_z, 36] + z_values[1]*self.coeffs_view[i_x, i_y, i_z, 37] + z_values[2]*self.coeffs_view[i_x, i_y, i_z, 38] + z_values[3]*self.coeffs_view[i_x, i_y, i_z, 39]) + \
                   y_values[2]*(z_values[0]*self.coeffs_view[i_x, i_y, i_z, 40] + z_values[1]*self.coeffs_view[i_x, i_y, i_z, 41] + z_values[2]*self.coeffs_view[i_x, i_y, i_z, 42] + z_values[3]*self.coeffs_view[i_x, i_y, i_z, 43]) + \
                   y_values[3]*(z_values[0]*self.coeffs_view[i_x, i_y, i_z, 44] + z_values[1]*self.coeffs_view[i_x, i_y, i_z, 45] + z_values[2]*self.coeffs_view[i_x, i_y, i_z, 46] + z_values[3]*self.coeffs_view[i_x, i_y, i_z, 47]) \
               ) \
               + x_values[3]*( \
                   y_values[0]*(z_values[0]*self.coeffs_view[i_x, i_y, i_z, 48] + z_values[1]*self.coeffs_view[i_x, i_y, i_z, 49] + z_values[2]*self.coeffs_view[i_x, i_y, i_z, 50] + z_values[3]*self.coeffs_view[i_x, i_y, i_z, 51]) + \
                   y_values[1]*(z_values[0]*self.coeffs_view[i_x, i_y, i_z, 52] + z_values[1]*self.coeffs_view[i_x, i_y, i_z, 53] + z_values[2]*self.coeffs_view[i_x, i_y, i_z, 54] + z_values[3]*self.coeffs_view[i_x, i_y, i_z, 55]) + \
                   y_values[2]*(z_values[0]*self.coeffs_view[i_x, i_y, i_z, 56] + z_values[1]*self.coeffs_view[i_x, i_y, i_z, 57] + z_values[2]*self.coeffs_view[i_x, i_y, i_z, 58] + z_values[3]*self.coeffs_view[i_x, i_y, i_z, 59]) + \
                   y_values[3]*(z_values[0]*self.coeffs_view[i_x, i_y, i_z, 60] + z_values[1]*self.coeffs_view[i_x, i_y, i_z, 61] + z_values[2]*self.coeffs_view[i_x, i_y, i_z, 62] + z_values[3]*self.coeffs_view[i_x, i_y, i_z, 63]) \
               )

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double[::1] _constraints3d(self, int u, int v, int w, bint x_der, bint y_der, bint z_der):
        """
        Return the coefficients of a given constraints and at a given point.

        This method is used to easily build the constraint matrix. It only
        handles constraints on P, dP/dx, dP/dy, dP/z, d2P/dxdy, d2P/dydz,
        d2P/dxdz and d3P/dxdydz (where P is the concerned polynomial).

        :param int u, int v, int w: indices of the point where the constraints apply.
        :param bint x_der, bint y_der, bint z_der: set to True or False in order to chose
        what constraint is returned. For each axis, True means the constraint
        considered a derivative along this axis.
        For example:
        x_der=False, y_der=True, z_der=False means the constraint returned is
        on the derivative along y (dP/dy).
        x_der=True, y_der=True, z_der=False means the constraint returned is
        on the cross derivative along x and y (d2P/dxdy).
        :return: a memory view of a 1x64 array filled with the coefficients
        corresponding to the requested constraint.
        """

        cdef:
            double x_components[4]
            double y_components[4]
            double z_components[4]
            npy_intp result_size
            double[::1] result_view
            double x_component, y_component, z_component
            int l

        if x_der:
            x_components[0] = 0.
            x_components[1] = 1.
            x_components[2] = 2.*self.x_view[u]
            x_components[3] = 3.*self.x2_view[u]
        else:
            x_components[0] = 1.
            x_components[1] = self.x_view[u]
            x_components[2] = self.x2_view[u]
            x_components[3] = self.x3_view[u]

        if y_der:
            y_components[0] = 0.
            y_components[1] = 1.
            y_components[2] = 2.*self.y_view[v]
            y_components[3] = 3.*self.y2_view[v]
        else:
            y_components[0] = 1.
            y_components[1] = self.y_view[v]
            y_components[2] = self.y2_view[v]
            y_components[3] = self.y3_view[v]

        if z_der:
            z_components[0] = 0.
            z_components[1] = 1.
            z_components[2] = 2.*self.z_view[w]
            z_components[3] = 3.*self.z2_view[w]
        else:
            z_components[0] = 1.
            z_components[1] = self.z_view[w]
            z_components[2] = self.z2_view[w]
            z_components[3] = self.z3_view[w]

        # create an empty (uninitialised) ndarray via numpy c-api
        result_size = 64
        result_view = PyArray_SimpleNew(1, &result_size, NPY_FLOAT64)

        l = 0
        for x_component in x_components:
            for y_component in y_components:
                for z_component in z_components:
                    result_view[l] = x_component * y_component * z_component
                    l += 1

        return result_view