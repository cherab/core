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
from numpy.linalg import solve

cimport cython
cimport numpy as np
from libc.math cimport INFINITY
from cherab.core.math.interpolators.utility cimport find_index, lerp, derivatives_array, factorial


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


cdef class _Interpolate1DBase(Function1D):
    """
    Base class for 1D interpolators. Coordinate and data arrays are here
    sorted and transformed into numpy arrays.

    :param object x: A 1D array-like object of real values.
    :param object data: A 1D array-like object of real values. The length
     of `data` must be equal to the length of `x`.
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

    def __init__(self, object x, object data, bint extrapolate=False, str extrapolation_type='nearest',
                 double extrapolation_range=INFINITY, bint tolerate_single_value=False):

        cdef ndarray mask

        # convert data to numpy arrays
        x = np.array(x, dtype=np.float64)
        data = np.array(data, dtype=np.float64)

        # check data dimensions are 1D
        if x.ndim != 1:
            raise ValueError("The x array must be 1D.")

        if data.ndim != 1:
            raise ValueError("The data array must be 1D.")

        # check the shapes of data and coordinates are consistent
        if x.shape != data.shape:
            raise ValueError("The x and data arrays must have the same shape (x={}, data={}).".format(x.shape, data.shape))

        # check the x array is monotonically increasing
        if (np.diff(x) <= 0).any():
            raise ValueError("The x array must be monotonically increasing.")

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

        # populate internal arrays and memory views
        self.x = x
        self.x_mv = x

        self.data = data
        self.data_mv = data

        self.top_index = len(x) - 1

        # Check for single value in x input
        if len(self.x) == 1:
            if tolerate_single_value:
                self.constant = True
            else:
                raise ValueError("There is only a single value in the input arrays. "
                    "Consider turning on the 'tolerate_single_value' argument.")

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double evaluate(self, double px) except? -1e999:
        """
        Evaluate the interpolating function.

        :param double px: x coordinate
        :return: the interpolated value
        """

        cdef int index

        if self.constant:
            return self.data_mv[0]

        index = find_index(self.x_mv, self.top_index+1, px, self.extrapolation_range)

        if 0 <= index <= self.top_index-1:
            return self._evaluate(px, index)
        elif index == -1:
            return self._extrapolate(px, 0, self.x_mv[0])
        elif index == self.top_index:
            return self._extrapolate(px, self.top_index-1, self.x_mv[self.top_index])

        # value is outside of permitted limits
        min_range = self.x_mv[0] - self.extrapolation_range
        max_range = self.x_mv[self.top_index] + self.extrapolation_range

        raise ValueError("The specified value (x={}) is outside the range of the supplied data and/or extrapolation range: "
                         "x bounds=({}, {})".format(px, min_range, max_range))

    cdef double _evaluate(self, double px, int index) except? -1e999:
        """
        Evaluate the interpolating function which is valid in the area given
        by 'index' at any position 'px'.

        :param double px: x coordinate
        :param int index: index of the area of interest
        :return: the interpolated value
        """
        raise NotImplementedError("This abstract method has not been implemented yet.")

    cdef double _extrapolate(self, double px, int index, double nearest_px) except? -1e999:
        """
        Extrapolate the interpolation function valid on area given by
        'index' to position 'px'.

        :param double px: x coordinate
        :param int index: index of the area of interest
        :param double nearest_px: the nearest position from 'px' in the
        interpolation domain.
        :return: the extrapolated value
        """

        if self.extrapolation_type == EXT_NEAREST:
            return self._evaluate(nearest_px, index)
        elif self.extrapolation_type == EXT_LINEAR:
            return self._extrapol_linear(px, index, nearest_px)
        elif self.extrapolation_type == EXT_QUADRATIC:
            return self._extrapol_quadratic(px, index, nearest_px)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double _extrapol_linear(self, double px, int index, double nearest_px) except? -1e999:
        """
        Extrapolate linearly the interpolation function valid on area given by
        'index' to position 'px'.

        :param double px: x coordinate
        :param int index: index of the area of interest
        :param double nearest_px: the nearest position from 'px' in the
        interpolation domain.
        :return: the extrapolated value
        """
        raise NotImplementedError("There is no linear extrapolation available for this interpolation.")

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double _extrapol_quadratic(self, double px, int index, double nearest_px) except? -1e999:
        """
        Extrapolate quadratically the interpolation function valid on area given by
        'index' to position 'px'.

        :param double px: x coordinate
        :param int index: index of the area of interest
        :param double nearest_px: the nearest position from 'px' in the
        interpolation domain.
        :return: the extrapolated value
        """
        raise NotImplementedError("There is no quadratic extrapolation available for this interpolation.")


cdef class Interpolate1DLinear(_Interpolate1DBase):
    """
    Interpolates 1D data using linear interpolation.

    :param object x_data: A 1D array-like object of real values.
    :param object f_data: A 1D array-like object of real values. The length
     of `f_data` must be equal to the length of `x_data`.
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

    def __init__(self, object x_data, object f_data, bint extrapolate=False, str extrapolation_type='nearest',
                 double extrapolation_range=INFINITY, bint tolerate_single_value=False):

        supported_extrapolations = ['nearest', 'linear']

        extrapolation_type = extrapolation_type.lower()
        if extrapolation_type not in supported_extrapolations:
            raise ValueError("Unsupported extrapolation type: {}".format(extrapolation_type))

        super().__init__(x_data, f_data, extrapolate, extrapolation_type, extrapolation_range, tolerate_single_value)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double _evaluate(self, double px, int index) except? -1e999:
        """
        Evaluate the interpolating function which is valid in the area given
        by 'index' at any position 'px'.

        :param double px: x coordinate
        :param int index: index of the area of interest
        :return: the interpolated value
        """

        return lerp(self.x_mv[index], self.x_mv[index + 1], self.data_mv[index], self.data_mv[index + 1], px)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double _extrapol_linear(self, double px, int index, double nearest_px) except? -1e999:
        """
        Extrapolate linearly the interpolation function valid on area given by
        'index' to position 'px'.

        :param double px: x coordinate
        :param int index: index of the area of interest
        :param double nearest_px: the nearest position from 'px' in the
        interpolation domain.
        :return: the extrapolated value
        """

        return lerp(self.x_mv[index], self.x_mv[index + 1], self.data_mv[index], self.data_mv[index + 1], px)


cdef class Interpolate1DCubic(_Interpolate1DBase):
    """
    Interpolates 1D data using cubic interpolation.

    Data and coordinates are first normalised to the range [0, 1] so as to
    prevent inaccuracy from float numbers.
    Spline coefficients are cached so they have to be calculated at
    initialisation only.

    :param object x_data: A 1D array-like object of real values.
    :param object f_data: A 1D array-like object of real values. The length
     of `f_data` must be equal to the length of `x_data`.
    :param int continuity_order: optional
    Sets the continuity of the cubic spline.
    When set to 1 the cubic spline second derivatives are estimated from
    the data samples and is not continuous. Here, the first derivative is
    free and forced to be continuous, but the second derivative is imposed
    from finite differences estimation.
    When set to 2 the cubic spline second derivatives are free but are
    forced to be continuous.
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

    def __init__(self, object x_data, object f_data, int continuity_order=2,
                 bint extrapolate=False, double extrapolation_range=INFINITY,
                 str extrapolation_type='nearest', bint tolerate_single_value=False):

        supported_extrapolations = ['nearest', 'linear', 'quadratic']

        extrapolation_type = extrapolation_type.lower()
        if extrapolation_type not in supported_extrapolations:
            raise ValueError("Unsupported extrapolation type: {}".format(extrapolation_type))

        super().__init__(x_data, f_data, extrapolate, extrapolation_type, extrapolation_range, tolerate_single_value)

        if not self.constant:
            self._calc_coefficients(continuity_order)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef object _calc_coefficients(self, int continuity_order):
        """
        Calculate the spline coefficients.
        
        :param continuity_order: Order of spline continuity (1 or 2). 
        """

        cdef:
            int k, n, l, i_x, i
            double[::1] x_view, x2_view, x3_view, data_view
            double[::1] cv_view
            double [:, ::1] cm_view, coeffs_view
            double d

        # Normalise coordinates and data arrays
        self.x_delta_inv = 1 / (self.x.max() - self.x.min())
        self.x_min = self.x.min()
        self.x = (self.x - self.x_min) * self.x_delta_inv
        self.data_delta = self.data.max() - self.data.min()
        self.data_min = self.data.min()
        # If data contains only one value (not filtered before) cancel the
        # normalisation scaling by setting data_delta to 1:
        if self.data_delta == 0:
            self.data_delta = 1
        self.data = (self.data - self.data_min) * (1 / self.data_delta)

        x_view = self.x
        x2_view = self.x * self.x
        x3_view = self.x * self.x * self.x
        data_view = self.data

        n = len(self.x) - 1
        l = 0

        # Fill the constraints matrix and vector
        cv_view = np.zeros((4*n,), dtype=np.float64)  # constraints_vector
        cm_view = np.zeros((4*n, 4*n), dtype=np.float64)  # constraints_matrix

        # Knots values constraints:
        for k in range(n):

            cm_view[l, 4*k] = 1.
            cm_view[l, 4*k+1] = x_view[k]
            cm_view[l, 4*k+2] = x2_view[k]
            cm_view[l, 4*k+3] = x3_view[k]
            cv_view[l] = data_view[k]
            l += 1

            cm_view[l, 4*k] = 1.
            cm_view[l, 4*k+1] = x_view[k+1]
            cm_view[l, 4*k+2] = x2_view[k+1]
            cm_view[l, 4*k+3] = x3_view[k+1]
            cv_view[l] = data_view[k+1]
            l += 1

        # first and/or second derivatives constraints:
        if continuity_order == 1:

            for k in range(n-1):

                d = (data_view[k+2] - data_view[k]) / (x_view[k+2] - x_view[k])
                cm_view[l, 4*k+1] = 1.
                cm_view[l, 4*k+2] = 2*x_view[k+1]
                cm_view[l, 4*k+3] = 3*x2_view[k+1]
                cv_view[l] = d
                l += 1

                cm_view[l, 4*k+5] = 1.
                cm_view[l, 4*k+6] = 2*x_view[k+1]
                cm_view[l, 4*k+7] = 3*x2_view[k+1]
                cv_view[l] = d
                l += 1

        elif continuity_order == 2:

            for k in range(n-1):

                # first derivative
                cm_view[l, 4*k+1] = -1.
                cm_view[l, 4*k+2] = -2*x_view[k+1]
                cm_view[l, 4*k+3] = -3*x2_view[k+1]
                cm_view[l, 4*k+5] = 1.
                cm_view[l, 4*k+6] = 2*x_view[k+1]
                cm_view[l, 4*k+7] = 3*x2_view[k+1]
                l += 1

                # second derivative
                cm_view[l, 4*k+2] = -2.
                cm_view[l, 4*k+3] = -6*x_view[k+1]
                cm_view[l, 4*k+6] = 2.
                cm_view[l, 4*k+7] = 6*x_view[k+1]
                l += 1

        else:
            raise NotImplementedError("'continuity_order' must be 1 or 2.")

        # two last constraints: the choice is here to set the third derivative
        # to zero at the edges because it is the less constraining condition.
        # Other possibilities are to impose values to the first or the second
        # derivatives. (comment/uncomment to change the constraints)

        # Third derivative
        cm_view[l, 3] = 6.
        l += 1
        cm_view[l, 4*n -1] = 6.
        l += 1

        # Second derivative
        # cm_view[l, 2] = 2.
        # cm_view[l, 3] = 6*x_view[0]
        # l = l + 1
        # cm_view[l, 4*n -2] = 2.
        # cm_view[l, 4*n -1] = 6*x_view[n]
        # l = l + 1

        # First derivative
        # cm_view[l, 1] = 1.
        # cm_view[l, 2] = 2*x_view[0]
        # cm_view[l, 3] = 3*x2_view[0]
        # l = l + 1
        # cm_view[l, 4*n -3] = 1.
        # cm_view[l, 4*n -2] = 2*x_view[n]
        # cm_view[l, 4*n -1] = 3*x2_view[n]
        # l = l + 1

        # Solve the linear system
        coeffs_view = solve(cm_view, cv_view).reshape((n, 4))
        self.coeffs_view = coeffs_view

        # Denormalisation
        for i_x in range(n):
            for i in range(4):
                coeffs_view[i_x, i] = self.data_delta * (self.x_delta_inv ** i / factorial(i) * self._evaluate_polynomial_derivative(i_x, -self.x_delta_inv * self.x_min, i))
            coeffs_view[i_x, 0] = coeffs_view[i_x, 0] + self.data_min
        self.coeffs_view = coeffs_view

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double _evaluate(self, double px, int index) except? -1e999:
        """
        Evaluate the interpolating function which is valid in the area given
        by 'index' at any position 'px'.

        :param double px: x coordinate
        :param int index: index of the area of interest
        :return: the interpolated value
        """

        cdef double px2, px3

        px2 = px*px
        px3 = px2*px

        return self.coeffs_view[index, 0] + self.coeffs_view[index, 1]*px + self.coeffs_view[index, 2]*px2 + self.coeffs_view[index, 3]*px3

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double _extrapol_linear(self, double px, int index, double nearest_px) except? -1e999:
        """
        Extrapolate linearly the interpolation function valid on area given by
        'index' to position 'px'.

        :param double px: x coordinate
        :param int index: index of the area of interest
        :param double nearest_px: the nearest position from 'px' in the
        interpolation domain.
        :return: the extrapolated value
        """

        return self._evaluate(nearest_px, index) \
               + (px - nearest_px) * (3.*self.coeffs_view[index, 3]*nearest_px*nearest_px + 2.*self.coeffs_view[index, 2]*nearest_px + self.coeffs_view[index, 1])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double _extrapol_quadratic(self, double px, int index, double nearest_px) except? -1e999:
        """
        Extrapolate quadratically the interpolation function valid on area given by
        'index' to position 'px'.

        :param double px: x coordinate
        :param int index: index of the area of interest
        :param double nearest_px: the nearest position from 'px' in the
        interpolation domain.
        :return: the extrapolated value
        """

        cdef double delta = px - nearest_px

        return self._evaluate(nearest_px, index) \
               + delta * (3.*self.coeffs_view[index, 3]*nearest_px*nearest_px + 2.*self.coeffs_view[index, 2]*nearest_px + self.coeffs_view[index, 1]) \
               + delta*delta*0.5 * (6.*self.coeffs_view[index, 3]*nearest_px + 2.*self.coeffs_view[index, 2])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double _evaluate_polynomial_derivative(self, int i_x, double px, int der_x):
        """
        Evaluate the derivatives of the polynomial valid in the area given by
        'i_x' at position 'px'. The order of derivative is given by 'der_x'.

        :param int i_x: index of the area of interest
        :param double px: x coordinate
        :param int der_x: order of derivative
        :return: value evaluated from the derivated polynomial
        """

        cdef double[::1] x_values
        x_values = derivatives_array(px, der_x)
        return x_values[0]*self.coeffs_view[i_x, 0] + x_values[1]*self.coeffs_view[i_x, 1] + x_values[2]*self.coeffs_view[i_x, 2] + x_values[3]*self.coeffs_view[i_x, 3]
