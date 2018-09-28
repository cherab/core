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

from numpy import array, empty, int8, float64, shape, concatenate, argsort, arange
from numpy.linalg import solve

cimport cython
from numpy cimport ndarray, PyArray_ZEROS, NPY_FLOAT64, npy_intp, import_array
from cherab.core.math.interpolators.utility cimport find_index, lerp, derivatives_array, factorial

# required by numpy c-api
import_array()

# internal constants used to represent the different extrapolation options
DEF EXT_NEAREST = 0
DEF EXT_LINEAR = 1
DEF EXT_QUADRATIC = 2

# map extrapolation string to relevant internal constant
_EXTRAPOLATION_TYPES = {
    'nearest': EXT_NEAREST,
    'linear': EXT_LINEAR,
    'quadratic': EXT_QUADRATIC
}

cdef class _Interpolate2DBase(Function2D):
    """
    Base class for 2D interpolators. Coordinates and data arrays are here
    sorted and transformed into numpy arrays.

    :param object x: An array-like object containing real values.
    :param object y: An array-like object containing real values.
    :param object data: A 2D array-like object of sample values corresponding to the
    `x` and `y` array points.
    :param bint extrapolate: optional
    If True, the extrapolation of data is enabled outside the range of the
    data set. The default is False. A ValueError is raised if extrapolation
    is disabled and a point is requested outside the data set.
    :param object extrapolation_type: optional
    Sets the method of extrapolation. The options are: 'nearest' (default)
    and other options given by the subclass.
    :param double extrapolation_range: optional
    The attribute can be set to limit the range beyond the data set bounds
    that extrapolation is permitted. The default range is set to infinity.
    Requesting data beyond the extrapolation range will result in a
    ValueError being raised.
    :param tolerate_single_value: optional
    If True, single-value arrays will be tolerated as inputs. If a single
    value is supplied, that value will be extrapolated over the entire
    real range. If False (default), supplying a single value will result
    in a ValueError being raised.
    """

    def __init__(self, object x, object y, object data, bint extrapolate=False, str extrapolation_type='nearest',
                 double extrapolation_range=float('inf'), bint tolerate_single_value=False):

        cdef ndarray mask_x, mask_y

        # check the shapes of data and coordinates are consistent
        if shape(data) != tuple(list(shape(x))+list(shape(y))):
            raise ValueError("Data and coordinates must have the same shapes.")

        # extrapolation is controlled internally by setting a positive extrapolation_range
        self.extrapolate = extrapolate
        if extrapolate:
            self.extrapolation_range = max(0, extrapolation_range)
        else:
            self.extrapolation_range = 0

        # map extrapolation type name to internal constant
        if extrapolation_type in _EXTRAPOLATION_TYPES:
            self.extrapolation_type = _EXTRAPOLATION_TYPES[extrapolation_type]
        else:
            raise ValueError("Extrapolation type {} does not exist.".format(extrapolation_type))

        # copies the arguments converted into double arrays and sort x and y
        mask_x = argsort(x)
        mask_y = argsort(y)
        self.x_np = array(x, dtype=float64)[mask_x]
        self.y_np = array(y, dtype=float64)[mask_y]
        self.data_np = array(data, dtype=float64)[mask_x, :][:, mask_y]

        self.x_domain_view = self.x_np
        self.y_domain_view = self.y_np
        self.top_index_x = len(x) - 1
        self.top_index_y = len(y) - 1

        # Check for single value in x input
        if len(self.x_np) == 1:
            if tolerate_single_value:
                # single value tolerated, set constant
                self._set_constant_x()
            else:
                raise ValueError("There is only a single value in the x input. "
                    "Consider turning on the 'tolerate_single_value' argument.")

        # if x is not a single value, check for duplicate values
        else:
            if (self.x_np == self.x_np[arange(len(self.x_np))-1]).any():
                raise ValueError("The x coordinates array has a duplicate value.")

        # Check for single value in y input
        if len(self.y_np) == 1:
            if tolerate_single_value:
                # single value tolerated, set constant
                self._set_constant_y()
            else:
                raise ValueError("There is only a single value in the y input. "
                    "Consider turning on the 'tolerate_single_value' argument.")

        # if y is not a single value, check for duplicate values
        else:
            if (self.y_np == self.y_np[arange(len(self.y_np))-1]).any():
                raise ValueError("The y coordinates array has a duplicate value.")

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double evaluate(self, double px, double py) except? -1e999:
        """
        Evaluate the interpolating function.

        :param double px, double py: coordinates
        :return: the interpolated value
        """

        cdef int i_x, i_y

        i_x = find_index(self.x_domain_view, self.top_index_x+1, px, self.extrapolation_range)
        i_y = find_index(self.y_domain_view, self.top_index_y+1, py, self.extrapolation_range)

        if 0 <= i_x <= self.top_index_x-1:
            if 0 <= i_y <= self.top_index_y-1:
                return self._evaluate(px, py, i_x, i_y)
            elif i_y == -1:
                return self._extrapolate(px, py, i_x, 0, px, self.y_domain_view[0])
            elif i_y == self.top_index_y:
                return self._extrapolate(px, py, i_x, self.top_index_y-1, px, self.y_domain_view[self.top_index_y])
        elif i_x == -1:
            if 0 <= i_y <= self.top_index_y-1:
                return self._extrapolate(px, py, 0, i_y, self.x_domain_view[0], py)
            elif i_y == -1:
                return self._extrapolate(px, py, 0, 0, self.x_domain_view[0], self.y_domain_view[0])
            elif i_y == self.top_index_y:
                return self._extrapolate(px, py, 0, self.top_index_y-1, self.x_domain_view[0], self.y_domain_view[self.top_index_y])
        elif i_x == self.top_index_x:
            if 0 <= i_y <= self.top_index_y-1:
                return self._extrapolate(px, py, self.top_index_x-1, i_y, self.x_domain_view[self.top_index_x], py)
            elif i_y == -1:
                return self._extrapolate(px, py, self.top_index_x-1, 0, self.x_domain_view[self.top_index_x], self.y_domain_view[0])
            elif i_y == self.top_index_y:
                return self._extrapolate(px, py, self.top_index_x-1, self.top_index_y-1, self.x_domain_view[self.top_index_x], self.y_domain_view[self.top_index_y])

        # value is outside of permitted limits
        min_range_x = self.x_domain_view[0] - self.extrapolation_range
        max_range_x = self.x_domain_view[self.top_index_x] + self.extrapolation_range

        min_range_y = self.y_domain_view[0] - self.extrapolation_range
        max_range_y = self.y_domain_view[self.top_index_y] + self.extrapolation_range

        raise ValueError("The specified value (x={}, y={}) is outside the range of the supplied data and/or extrapolation range: "
                         "x bounds=({}, {}), y bounds=({}, {})".format(px, py, min_range_x, max_range_x, min_range_y, max_range_y))

    cdef double _evaluate(self, double px, double py, int i_x, int i_y) except? -1e999:
        """
        Evaluate the interpolating function which is valid in the area given
        by 'i_x' and 'i_y' at any position ('px', 'py').

        :param double px, double py: coordinates
        :param int i_x, int i_y: indices of the area of interest
        :return: the interpolated value
        """
        raise NotImplementedError("This abstract method has not been implemented yet.")

    cdef double _extrapolate(self, double px, double py, int i_x, int i_y, double nearest_px, double nearest_py) except? -1e999:
        """
        Extrapolate the interpolation function valid on area given by
        'i_x' and 'i_y' to position ('px', 'py').

        :param double px, double py: coordinates
        :param int i_x, int i_y: indices of the area of interest
        :param double nearest_px, nearest_py: the nearest position from
        ('px', 'py') in the interpolation domain.
        :return: the extrapolated value
        """

        if self.extrapolation_type == EXT_NEAREST:
            return self._evaluate(nearest_px, nearest_py, i_x, i_y)
        elif self.extrapolation_type == EXT_LINEAR:
            return self._extrapol_linear(px, py, i_x, i_y, nearest_px, nearest_py)
        elif self.extrapolation_type == EXT_QUADRATIC:
            return self._extrapol_quadratic(px, py, i_x, i_y, nearest_px, nearest_py)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double _extrapol_linear(self, double px, double py, int i_x, int i_y, double nearest_px, double nearest_py) except? -1e999:
        """
        Extrapolate linearly the interpolation function valid on area given by
        'i_x' and 'i_y' to position ('px', 'py').

        :param double px, double py: coordinates
        :param int i_x, int i_y: indices of the area of interest
        :param double nearest_px, nearest_py: the nearest position from
        ('px', 'py') in the interpolation domain.
        :return: the extrapolated value
        """
        raise NotImplementedError("There is no linear extrapolation available for this interpolation.")

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double _extrapol_quadratic(self, double px, double py, int i_x, int i_y, double nearest_px, double nearest_py) except? -1e999:
        """
        Extrapolate quadratically the interpolation function valid on area given by
        'i_x' and 'i_y' to position ('px', 'py').

        :param double px, double py: coordinates
        :param int i_x, int i_y: indices of the area of interest
        :param double nearest_px, nearest_py: the nearest position from
        ('px', 'py') in the interpolation domain.
        :return: the extrapolated value
        """
        raise NotImplementedError("There is no quadratic extrapolation available for this interpolation.")

    cdef void _set_constant_x(self):
        """
        Set the interpolation function constant on the x axis, and extend the
        domain to all the reals.
        """

        cdef ndarray data

        self.x_domain_view = array([-float('Inf'), +float('Inf')], dtype=float64)
        self.top_index_x = 1

        self.x_np = array([-1., +1.], dtype=float64)
        data = self.data_np
        self.data_np = empty((2, shape(data)[1]), dtype=float64)
        self.data_np[:,:] = data

    cdef void _set_constant_y(self):
        """
        Set the interpolation function constant on the y axis, and extend the
        domain to all the reals.
        """

        cdef ndarray data

        self.y_domain_view = array([-float('Inf'), +float('Inf')], dtype=float64)
        self.top_index_y = 1

        self.y_np = array([-1., +1.], dtype=float64)
        data = self.data_np
        self.data_np = empty((shape(data)[0], 2), dtype=float64)
        self.data_np[:,:] = data


cdef class Interpolate2DLinear(_Interpolate2DBase):
    """
    Interpolates 2D data using linear interpolation.

    :param object x: An array-like object containing real values.
    :param object y: An array-like object containing real values.
    :param object data: A 2D array-like object of sample values corresponding to the
    `x` and `y` array points.
    :param bint extrapolate: optional
    If True, the extrapolation of data is enabled outside the range of the
    data set. The default is False. A ValueError is raised if extrapolation
    is disabled and a point is requested outside the data set.
    :param object extrapolation_type: optional
    Sets the method of extrapolation. The options are: 'nearest' (default),
     'linear'
    :param double extrapolation_range: optional
    The attribute can be set to limit the range beyond the data set bounds
    that extrapolation is permitted. The default range is set to infinity.
    Requesting data beyond the extrapolation range will result in a
    ValueError being raised.
    :param tolerate_single_value: optional
    If True, single-value arrays will be tolerated as inputs. If a single
    value is supplied, that value will be extrapolated over the entire
    real range. If False (default), supplying a single value will result
    in a ValueError being raised.
    """

    def __init__(self, object x, object y, object data, bint extrapolate=False, str extrapolation_type='nearest',
                 double extrapolation_range=float('inf'), bint tolerate_single_value=False):

        supported_extrapolations = ['nearest', 'linear']

        extrapolation_type = extrapolation_type.lower()
        if extrapolation_type not in supported_extrapolations:
            raise ValueError("Unsupported extrapolation type: {}".format(extrapolation_type))

        super().__init__(x, y, data, extrapolate, extrapolation_type, extrapolation_range, tolerate_single_value)

        # obtain memory views
        self.x_view = self.x_np
        self.y_view = self.y_np
        self.data_view = self.data_np

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double _evaluate(self, double px, double py, int i_x, int i_y) except? -1e999:
        """
        Evaluate the interpolating function which is valid in the area given
        by 'i_x' and 'i_y' at any position ('px', 'py').

        :param double px, double py: coordinates
        :param int i_x, int i_y: indices of the area of interest
        :return: the interpolated value
        """

        cdef double interm_value_up, interm_value_down

        interm_value_up = lerp(self.x_view[i_x], self.x_view[i_x+1], self.data_view[i_x, i_y+1], self.data_view[i_x+1, i_y+1], px)
        interm_value_down = lerp(self.x_view[i_x], self.x_view[i_x+1], self.data_view[i_x, i_y], self.data_view[i_x+1, i_y], px)

        return lerp(self.y_view[i_y], self.y_view[i_y+1], interm_value_down, interm_value_up, py)

    cdef double _extrapol_linear(self, double px, double py, int i_x, int i_y, double nearest_px, double nearest_py) except? -1e999:
        """
        Extrapolate linearly the interpolation function valid on area given by
        'i_x' and 'i_y' to position ('px', 'py').

        :param double px, double py: coordinates
        :param int i_x, int i_y: indices of the area of interest
        :param double nearest_px, nearest_py: the nearest position from
        ('px', 'py') in the interpolation domain.
        :return: the extrapolated value
        """

        return self._evaluate(px, py, i_x, i_y)

cdef class Interpolate2DCubic(_Interpolate2DBase):
    """
    Interpolates 2D data using cubic interpolation.

    Data and coordinates are first normalised to the range [0, 1] so as to
    prevent inaccuracy from float numbers.
    A local calculation based on finite differences is used. The
    splines coefficients are not calculated before evaluation but on demand
    only and are cached as they are calculated. Plus, only one polynomial
    is calculated at each evaluation. The first derivatives and the cross
    derivative are imposed by the finite differences. The resulting
    function is C1.

    :param object x: An array-like object containing real values.
    :param object y: An array-like object containing real values.
    :param object data: A 2D array-like object of sample values corresponding to the
    `x` and `y` array points.
    :param bint extrapolate: optional
    If True, the extrapolation of data is enabled outside the range of the
    data set. The default is False. A ValueError is raised if extrapolation
    is disabled and a point is requested outside the data set.
    :param object extrapolation_type: optional
    Sets the method of extrapolation. The options are: 'nearest' (default),
     'linear', 'quadratic'
    :param double extrapolation_range: optional
    The attribute can be set to limit the range beyond the data set bounds
    that extrapolation is permitted. The default range is set to infinity.
    Requesting data beyond the extrapolation range will result in a
    ValueError being raised.
    :param tolerate_single_value: optional
    If True, single-value arrays will be tolerated as inputs. If a single
    value is supplied, that value will be extrapolated over the entire
    real range. If False (default), supplying a single value will result
    in a ValueError being raised.
    """

    def __init__(self, object x, object y, object data, bint extrapolate=False, double extrapolation_range=float('inf'),
                 str extrapolation_type='nearest', bint tolerate_single_value=False):

        cdef int i, j, i_narrowed, j_narrowed

        supported_extrapolations = ['nearest', 'linear', 'quadratic']

        extrapolation_type = extrapolation_type.lower()
        if extrapolation_type not in supported_extrapolations:
            raise ValueError("Unsupported extrapolation type: {}".format(extrapolation_type))

        super().__init__(x, y, data, extrapolate, extrapolation_type, extrapolation_range, tolerate_single_value)

        # Initialise the caching array
        self.coeffs_view = empty((self.top_index_x, self.top_index_y, 16), dtype=float64)
        self.coeffs_view[:,:,:] = float('NaN')
        self.calculated_view = empty((self.top_index_x, self.top_index_y), dtype=int8)
        self.calculated_view[:,:] = False

        # Normalise coordinates and data arrays
        self.x_delta_inv = 1 / (self.x_np.max() - self.x_np.min())
        self.x_min = self.x_np.min()
        self.x_np = (self.x_np - self.x_min) * self.x_delta_inv
        self.y_delta_inv = 1 / (self.y_np.max() - self.y_np.min())
        self.y_min = self.y_np.min()
        self.y_np = (self.y_np - self.y_min) * self.y_delta_inv
        self.data_delta = self.data_np.max() - self.data_np.min()
        self.data_min = self.data_np.min()
        # If data contains only one value (not filtered before) cancel the
        # normalisation scaling by setting data_delta to 1:
        if self.data_delta == 0:
            self.data_delta = 1
        self.data_np = (self.data_np - self.data_min) * (1 / self.data_delta)

        # widen arrays for automatic handling of boundaries polynomials and get memory views
        self.x_np = concatenate(([self.x_np[0]], self.x_np, [self.x_np[-1]]))
        self.y_np = concatenate(([self.y_np[0]], self.y_np, [self.y_np[-1]]))

        self.data_view = empty((self.top_index_x+3, self.top_index_y+3), dtype=float64)

        for i in range(self.top_index_x+3):
            for j in range(self.top_index_y+3):
                i_narrowed = min(max(0, i-1), self.top_index_x)
                j_narrowed = min(max(0, j-1), self.top_index_y)
                self.data_view[i, j] = self.data_np[i_narrowed, j_narrowed]

        # obtain coordinates memory views
        self.x_view = self.x_np
        self.x2_view = self.x_np*self.x_np
        self.x3_view = self.x_np*self.x_np*self.x_np
        self.y_view = self.y_np
        self.y2_view = self.y_np*self.y_np
        self.y3_view = self.y_np*self.y_np*self.y_np

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double _evaluate(self, double px, double py, int i_x, int i_y) except? -1e999:
        """
        Evaluate the interpolating function which is valid in the area given
        by 'i_x' and 'i_y' at any position ('px', 'py').

        :param double px, double py: coordinates
        :param int i_x, int i_y: indices of the area of interest
        :return: the interpolated value
        """

        cdef double px2, py2, px3, py3

        # If the concerned polynomial has not yet been calculated:
        if not self.calculated_view[i_x, i_y]:
            self._calculate_polynomial(i_x, i_y)

        px2 = px*px
        px3 = px2*px
        py2 = py*py
        py3 = py2*py

        return     (self.coeffs_view[i_x, i_y,  0] + self.coeffs_view[i_x, i_y,  1]*py + self.coeffs_view[i_x, i_y,  2]*py2 + self.coeffs_view[i_x, i_y,  3]*py3) + \
               px *(self.coeffs_view[i_x, i_y,  4] + self.coeffs_view[i_x, i_y,  5]*py + self.coeffs_view[i_x, i_y,  6]*py2 + self.coeffs_view[i_x, i_y,  7]*py3) + \
               px2*(self.coeffs_view[i_x, i_y,  8] + self.coeffs_view[i_x, i_y,  9]*py + self.coeffs_view[i_x, i_y, 10]*py2 + self.coeffs_view[i_x, i_y, 11]*py3) + \
               px3*(self.coeffs_view[i_x, i_y, 12] + self.coeffs_view[i_x, i_y, 13]*py + self.coeffs_view[i_x, i_y, 14]*py2 + self.coeffs_view[i_x, i_y, 15]*py3)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef int _calculate_polynomial(self, int i_x, int i_y) except -1:
        """
        Calculates and caches the polynomial coefficients for area given by
        'i_x', 'i_y'. Declares this area as already calculated.

        :param int i_x, int i_y: indices of the area of interest
        """

        cdef:
            int u, v, l, i, j
            double delta_x, delta_y
            npy_intp cv_size
            npy_intp cm_size[2]
            double[::1] cv_view, coeffs_view
            double[:, ::1] cm_view

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
        for u in range(i_x+1, i_x+3):
            for v in range(i_y+1, i_y+3):

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
        self.coeffs_view[i_x, i_y, :] = coeffs_view

        # Denormalisation
        for i in range(4):
            for j in range(4):
                coeffs_view[4 * i + j] = self.data_delta * (self.x_delta_inv ** i * self.y_delta_inv ** j / (factorial(j) * factorial(i)) * self._evaluate_polynomial_derivative(i_x, i_y, -self.x_delta_inv * self.x_min, -self.y_delta_inv * self.y_min, i, j))
        coeffs_view[0] = coeffs_view[0] + self.data_min
        self.coeffs_view[i_x, i_y, :] = coeffs_view

        self.calculated_view[i_x, i_y] = True

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double _extrapol_linear(self, double px, double py, int i_x, int i_y, double nearest_px, double nearest_py) except? -1e999:
        """
        Extrapolate linearly the interpolation function valid on area given by
        'i_x' and 'i_y' to position ('px', 'py').

        :param double px, double py: coordinates
        :param int i_x, int i_y: indices of the area of interest
        :param double nearest_px, nearest_py: the nearest position from
        ('px', 'py') in the interpolation domain.
        :return: the extrapolated value
        """

        cdef double delta_x, delta_y, nx, nx2, nx3, ny, ny2, ny3, result

        delta_x = px - nearest_px
        delta_y = py - nearest_py

        nx = nearest_px
        nx2 = nx*nx
        nx3 = nx2*nx
        ny = nearest_py
        ny2 = ny*ny
        ny3 = ny2*ny

        result = self._evaluate(nx, ny, i_x, i_y)

        if delta_x != 0.:

            result += delta_x * (       (self.coeffs_view[i_x, i_y,  4] + self.coeffs_view[i_x, i_y,  5]*ny + self.coeffs_view[i_x, i_y,  6]*ny2 + self.coeffs_view[i_x, i_y,  7]*ny3) + \
                                 2.*nx *(self.coeffs_view[i_x, i_y,  8] + self.coeffs_view[i_x, i_y,  9]*ny + self.coeffs_view[i_x, i_y, 10]*ny2 + self.coeffs_view[i_x, i_y, 11]*ny3) + \
                                 3.*nx2*(self.coeffs_view[i_x, i_y, 12] + self.coeffs_view[i_x, i_y, 13]*ny + self.coeffs_view[i_x, i_y, 14]*ny2 + self.coeffs_view[i_x, i_y, 15]*ny3))

        if delta_y != 0.:

            result += delta_y * (    (self.coeffs_view[i_x, i_y,  1] + 2.*self.coeffs_view[i_x, i_y,  2]*ny + 3.*self.coeffs_view[i_x, i_y,  3]*ny2) + \
                                 nx *(self.coeffs_view[i_x, i_y,  5] + 2.*self.coeffs_view[i_x, i_y,  6]*ny + 3.*self.coeffs_view[i_x, i_y,  7]*ny2) + \
                                 nx2*(self.coeffs_view[i_x, i_y,  9] + 2.*self.coeffs_view[i_x, i_y, 10]*ny + 3.*self.coeffs_view[i_x, i_y, 11]*ny2) + \
                                 nx3*(self.coeffs_view[i_x, i_y, 13] + 2.*self.coeffs_view[i_x, i_y, 14]*ny + 3.*self.coeffs_view[i_x, i_y, 15]*ny2))

        return result

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double _extrapol_quadratic(self, double px, double py, int i_x, int i_y, double nearest_px, double nearest_py) except? -1e999:
        """
        Extrapolate quadratically the interpolation function valid on area given by
        'i_x' and 'i_y' to position ('px', 'py').

        :param double px, double py: coordinates
        :param int i_x, int i_y: indices of the area of interest
        :param double nearest_px, nearest_py: the nearest position from
        ('px', 'py') in the interpolation domain.
        :return: the extrapolated value
        """

        cdef double delta_x, delta_y, nx, nx2, nx3, ny, ny2, ny3, result

        delta_x = px - nearest_px
        delta_y = py - nearest_py

        nx = nearest_px
        nx2 = nx*nx
        nx3 = nx2*nx
        ny = nearest_py
        ny2 = ny*ny
        ny3 = ny2*ny

        result = self._evaluate(nx, ny, i_x, i_y)

        if delta_x != 0.:

            result += delta_x * (       (self.coeffs_view[i_x, i_y,  4] + self.coeffs_view[i_x, i_y,  5]*ny + self.coeffs_view[i_x, i_y,  6]*ny2 + self.coeffs_view[i_x, i_y,  7]*ny3) + \
                                 2.*nx *(self.coeffs_view[i_x, i_y,  8] + self.coeffs_view[i_x, i_y,  9]*ny + self.coeffs_view[i_x, i_y, 10]*ny2 + self.coeffs_view[i_x, i_y, 11]*ny3) + \
                                 3.*nx2*(self.coeffs_view[i_x, i_y, 12] + self.coeffs_view[i_x, i_y, 13]*ny + self.coeffs_view[i_x, i_y, 14]*ny2 + self.coeffs_view[i_x, i_y, 15]*ny3))

            result += delta_x*delta_x*0.5 * (2.   *(self.coeffs_view[i_x, i_y,  8] + self.coeffs_view[i_x, i_y,  9]*ny + self.coeffs_view[i_x, i_y, 10]*ny2 + self.coeffs_view[i_x, i_y, 11]*ny3) + \
                                             6.*nx*(self.coeffs_view[i_x, i_y, 12] + self.coeffs_view[i_x, i_y, 13]*ny + self.coeffs_view[i_x, i_y, 14]*ny2 + self.coeffs_view[i_x, i_y, 15]*ny3))

        if delta_y != 0.:

            result += delta_y * (    (self.coeffs_view[i_x, i_y,  1] + 2.*self.coeffs_view[i_x, i_y,  2]*ny + 3.*self.coeffs_view[i_x, i_y,  3]*ny2) + \
                                 nx *(self.coeffs_view[i_x, i_y,  5] + 2.*self.coeffs_view[i_x, i_y,  6]*ny + 3.*self.coeffs_view[i_x, i_y,  7]*ny2) + \
                                 nx2*(self.coeffs_view[i_x, i_y,  9] + 2.*self.coeffs_view[i_x, i_y, 10]*ny + 3.*self.coeffs_view[i_x, i_y, 11]*ny2) + \
                                 nx3*(self.coeffs_view[i_x, i_y, 13] + 2.*self.coeffs_view[i_x, i_y, 14]*ny + 3.*self.coeffs_view[i_x, i_y, 15]*ny2))

            result += delta_y*delta_y*0.5 * (    (2.*self.coeffs_view[i_x, i_y,  2] + 6.*self.coeffs_view[i_x, i_y,  3]*ny) + \
                                             nx *(2.*self.coeffs_view[i_x, i_y,  6] + 6.*self.coeffs_view[i_x, i_y,  7]*ny) + \
                                             nx2*(2.*self.coeffs_view[i_x, i_y, 10] + 6.*self.coeffs_view[i_x, i_y, 11]*ny) + \
                                             nx3*(2.*self.coeffs_view[i_x, i_y, 14] + 6.*self.coeffs_view[i_x, i_y, 15]*ny))

            if delta_x != 0.:

                result += delta_x*delta_y * (       (self.coeffs_view[i_x, i_y,  5] + 2.*self.coeffs_view[i_x, i_y,  6]*ny + 3.*self.coeffs_view[i_x, i_y,  7]*ny2) + \
                                             2.*nx *(self.coeffs_view[i_x, i_y,  9] + 2.*self.coeffs_view[i_x, i_y, 10]*ny + 3.*self.coeffs_view[i_x, i_y, 11]*ny2) + \
                                             3.*nx2*(self.coeffs_view[i_x, i_y, 13] + 2.*self.coeffs_view[i_x, i_y, 14]*ny + 3.*self.coeffs_view[i_x, i_y, 15]*ny2))

        return result

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double _evaluate_polynomial_derivative(self, int i_x, int i_y, double px, double py, int der_x, int der_y):
        """
        Evaluate the derivatives of the polynomial valid in the area given by
        'i_x', 'i_y' at position ('px', 'py'). The order of
        derivative along each axis is given by 'der_x', 'der_y'.

        :param int i_x, int i_y: indices of the area of interest
        :param double px, double py: coordinates
        :param int der_x, int der_y: orders of derivative along each axis
        :return: value evaluated from the derivated polynomial
        """

        cdef double[::1] x_values, y_values

        x_values = derivatives_array(px, der_x)
        y_values = derivatives_array(py, der_y)

        return x_values[0]*(y_values[0]*self.coeffs_view[i_x, i_y,  0] + y_values[1]*self.coeffs_view[i_x, i_y,  1] + y_values[2]*self.coeffs_view[i_x, i_y,  2] + y_values[3]*self.coeffs_view[i_x, i_y,  3]) + \
               x_values[1]*(y_values[0]*self.coeffs_view[i_x, i_y,  4] + y_values[1]*self.coeffs_view[i_x, i_y,  5] + y_values[2]*self.coeffs_view[i_x, i_y,  6] + y_values[3]*self.coeffs_view[i_x, i_y,  7]) + \
               x_values[2]*(y_values[0]*self.coeffs_view[i_x, i_y,  8] + y_values[1]*self.coeffs_view[i_x, i_y,  9] + y_values[2]*self.coeffs_view[i_x, i_y, 10] + y_values[3]*self.coeffs_view[i_x, i_y, 11]) + \
               x_values[3]*(y_values[0]*self.coeffs_view[i_x, i_y, 12] + y_values[1]*self.coeffs_view[i_x, i_y, 13] + y_values[2]*self.coeffs_view[i_x, i_y, 14] + y_values[3]*self.coeffs_view[i_x, i_y, 15])
