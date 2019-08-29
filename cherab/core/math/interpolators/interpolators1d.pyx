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
    :param object f: A 1D array-like object of real values. The length
     of `f` must be equal to the length of `x`.
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

    def __init__(self, object x, object f, bint extrapolate=False, str extrapolation_type='nearest',
                 double extrapolation_range=INFINITY, bint tolerate_single_value=False):

        # convert data to numpy arrays
        x = np.array(x, dtype=np.float64)
        f = np.array(f, dtype=np.float64)

        # check data dimensions are 1D
        if x.ndim != 1:
            raise ValueError("The x array must be 1D.")

        if f.ndim != 1:
            raise ValueError("The f array must be 1D.")

        # check the shapes of data and coordinates are consistent
        if x.shape != f.shape:
            raise ValueError("The x and data arrays must have the same shape (x={}, f={}).".format(x.shape, f.shape))

        # check the x array is monotonically increasing
        if (np.diff(x) <= 0).any():
            raise ValueError("The x array must be monotonically increasing.")

        # extrapolation is controlled internally by setting a positive extrapolation_range
        if extrapolate:
            self._extrapolation_range = max(0, extrapolation_range)
        else:
            self._extrapolation_range = 0

        # map extrapolation type name to internal constant
        if extrapolation_type in _EXTRAPOLATION_TYPES:
            self._extrapolation_type = _EXTRAPOLATION_TYPES[extrapolation_type]
        else:
            raise ValueError("Extrapolation type {} does not exist.".format(extrapolation_type))

        # populate internal arrays and memory views
        self._x = x
        self._f = f

        # Check for single value in x input
        if len(x) == 1:
            if tolerate_single_value:
                self._constant = True
            else:
                raise ValueError("There is only a single value in the input arrays. "
                    "Consider turning on the 'tolerate_single_value' argument.")
        else:
            # build internal state of interpolator
            self._build(x, f)

    cdef object _build(self, ndarray x, ndarray f):
        """
        Build additional internal state.
        
        Implement in sub-classes that require additional state to be build
        from the source arrays.        
            
        :param x: x array 
        :param f: f array 
        """
        pass

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double evaluate(self, double px) except? -1e999:
        """
        Evaluate the interpolating function.

        :param double px: x coordinate.
        :return: the interpolated value
        """

        cdef int nx, index

        if self._constant:
            return self._f[0]

        nx = self._x.shape[0]
        index = find_index(self._x, px, self._extrapolation_range)

        if 0 <= index < nx - 1:
            return self._evaluate(px, 0, index)

        elif index == -1:
            return self._extrapolate(px, 0, 0, self._x[0])

        elif index == nx - 1:
            return self._extrapolate(px, 0, nx - 2, self._x[nx - 1])

        # value is outside of permitted limits
        min_range = self._x[0] - self._extrapolation_range
        max_range = self._x[nx - 1] + self._extrapolation_range

        raise ValueError("The specified value (x={}) is outside the range of the supplied data and/or extrapolation range: "
                         "x bounds=({}, {})".format(px, min_range, max_range))

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef double derivative(self, double px, int order) except? -1e999:
        """
        Returns the derivative of the interpolating function to the specified order.
        
        :param px: The x coordinate.
        :param order: The order of the derivative. 
        :return: The interpolated derivative.
        """

        if order < 1:
            raise ValueError('Derivative order must be greater than zero.')
        cdef int nx, index

        if self._constant:
            return self._f[0]

        nx = self._x.shape[0]
        index = find_index(self._x, px, self._extrapolation_range)

        if 0 <= index < nx - 1:
            return self._evaluate(px, order, index)

        elif index == -1:
            return self._extrapolate(px, order, 0, self._x[0])

        elif index == nx - 1:
            return self._extrapolate(px, order, nx - 2, self._x[nx - 1])

        # value is outside of permitted limits
        min_range = self._x[0] - self._extrapolation_range
        max_range = self._x[nx - 1] + self._extrapolation_range

        raise ValueError("The specified value (x={}) is outside the range of the supplied data and/or extrapolation range: "
                         "x bounds=({}, {})".format(px, min_range, max_range))

    cdef double _evaluate(self, double px, int order, int index) except? -1e999:
        """
        Evaluate the interpolating function or derivative which is valid in the area given
        by 'index' at any position 'px'.

        :param double px: x coordinate
        :param int order: the derivative order
        :param int index: index of the area of interest
        :return: the interpolated value
        """
        raise NotImplementedError("This abstract method has not been implemented yet.")

    cdef double _extrapolate(self, double px, int order, int index, double rx) except? -1e999:
        """
        Extrapolate the interpolation function or derivative valid on area given by
        'index' to position 'px'.

        :param double px: x coordinate
        :param int index: index of the area of interest
        :param double rx: the nearest position from 'px' in the interpolation domain.
        :return: the extrapolated value
        """

        if self._extrapolation_type == EXT_NEAREST:
            if order == 0:
                return self._evaluate(rx, order, index)
            return 0.0

        elif self._extrapolation_type == EXT_LINEAR:
            return self._extrapol_linear(px, order, index, rx)

        elif self._extrapolation_type == EXT_QUADRATIC:
            return self._extrapol_quadratic(px, order, index, rx)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double _extrapol_linear(self, double px, int order, int index, double nearest_px) except? -1e999:
        """
        Extrapolate linearly the interpolation function or derivative valid on area given by
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
    cdef double _extrapol_quadratic(self, double px, int order, int index, double nearest_px) except? -1e999:
        """
        Extrapolate quadratically the interpolation function or derivative valid on area given by
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

    Inherits from Function1D, implements `__call__(x)`.

    :param object x: A 1D array-like object of real values.
    :param object f: A 1D array-like object of real values. The length
      of `f_data` must be equal to the length of `x_data`.
    :param bint extrapolate: If True, the extrapolation of data is enabled
      outside the range of the data set. The default is False. A ValueError
      is raised if extrapolation is disabled and a point is requested outside
      the data set.
    :param object extrapolation_type: Sets the method of extrapolation. The
      options are: 'nearest' (default), 'linear'.
    :param double extrapolation_range: The attribute can be set to limit the
      range beyond the data set bounds that extrapolation is permitted. The
      default range is set to infinity. Requesting data beyond the extrapolation
      range will result in a ValueError being raised.
    :param tolerate_single_value: If True, single-value arrays will be tolerated
      as inputs. If a single value is supplied, that value will be extrapolated
      over the entire real range. If False (default), supplying a single value
      will result in a ValueError being raised.

    .. code-block:: pycon

       >>> from cherab.core.math import Interpolate1DLinear
       >>>
       >>> f1d = Interpolate1DLinear([0, 0.5, 0.9, 1.0], [2500, 2000, 1000, 0])
       >>>
       >>> f1d(0.2)
       2300.0
       >>> f1d(0.875)
       1062.5
       >>> f1d(1.2)
       ValueError: The specified value (x=1.2) is outside the range of the supplied
       data and/or extrapolation range: x bounds=(0.0, 1.0)
    """

    def __init__(self, object x, object f, bint extrapolate=False, str extrapolation_type='nearest',
                 double extrapolation_range=INFINITY, bint tolerate_single_value=False):

        supported_extrapolations = ['nearest', 'linear']

        extrapolation_type = extrapolation_type.lower()
        if extrapolation_type not in supported_extrapolations:
            raise ValueError("Unsupported extrapolation type: {}".format(extrapolation_type))

        super().__init__(x, f, extrapolate, extrapolation_type, extrapolation_range, tolerate_single_value)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @cython.cdivision(True)
    cdef double _evaluate(self, double px, int order, int index) except? -1e999:
        """
        Evaluate the interpolating function which is valid in the area given
        by 'index' at any position 'px'.

        :param double px: x coordinate
        :param int index: index of the area of interest
        :return: the interpolated value
        """

        if order == 0:
            return lerp(self._x[index], self._x[index + 1], self._f[index], self._f[index + 1], px)
        elif order == 1:
            return (self._f[index + 1] - self._f[index]) / (self._x[index + 1] - self._x[index])
        return 0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @cython.cdivision(True)
    cdef double _extrapol_linear(self, double px, int order, int index, double nearest_px) except? -1e999:
        """
        Extrapolate linearly the interpolation function valid on area given by
        'index' to position 'px'.

        :param double px: x coordinate
        :param int index: index of the area of interest
        :param double nearest_px: the nearest position from 'px' in the
        interpolation domain.
        :return: the extrapolated value
        """

        if order == 0:
            return lerp(self._x[index], self._x[index + 1], self._f[index], self._f[index + 1], px)
        elif order == 1:
            return (self._f[index + 1] - self._f[index]) / (self._x[index + 1] - self._x[index])
        return 0


cdef class Interpolate1DCubic(_Interpolate1DBase):
    """
    Interpolates 1D data using cubic interpolation.

    Inherits from Function1D, implements `__call__(x)`.

    Data and coordinates are first normalised to the range [0, 1] so as to
    prevent inaccuracy from float numbers. Spline coefficients are cached
    so they have to be calculated at initialisation only.

    :param object x: A 1D array-like object of real values.
    :param object f: A 1D array-like object of real values. The length
      of `f_data` must be equal to the length of `x_data`.
    :param int continuity_order: Sets the continuity of the cubic spline.
      When set to 1 the cubic spline second derivatives are estimated from
      the data samples and is not continuous. Here, the first derivative is
      free and forced to be continuous, but the second derivative is imposed
      from finite differences estimation. When set to 2 the cubic spline
      second derivatives are free but are forced to be continuous. Defaults
      to `continuity_order = 2`.
    :param bint extrapolate: If True, the extrapolation of data is enabled
      outside the range of the data set. The default is False. A ValueError
      is raised if extrapolation is disabled and a point is requested
      outside the data set.
    :param object extrapolation_type: Sets the method of extrapolation.
      The options are: 'nearest' (default), 'linear', 'quadratic'
    :param double extrapolation_range: The attribute can be set to limit
      the range beyond the data set bounds that extrapolation is permitted.
      The default range is set to infinity. Requesting data beyond the
      extrapolation range will result in a ValueError being raised.
    :param tolerate_single_value: If True, single-value arrays will be
      tolerated as inputs. If a single value is supplied, that value
      will be extrapolated over the entire real range. If False (default),
      supplying a single value will result in a ValueError being raised.

    .. code-block:: pycon

       >>> from cherab.core.math import Interpolate1DCubic
       >>>
       >>> f1d = Interpolate1DCubic([0, 0.5, 0.9, 1.0], [2500, 2000, 1000, 0])
       >>>
       >>> f1d(0.2)
       2197.4683
       >>> f1d(0.875)
       1184.4343
       >>> f1d(1.2)
       ValueError: The specified value (x=1.2) is outside the range of the supplied
       data and/or extrapolation range: x bounds=(0.0, 1.0)
    """

    def __init__(self, object x, object f, int continuity_order=2,
                 bint extrapolate=False, double extrapolation_range=INFINITY,
                 str extrapolation_type='nearest', bint tolerate_single_value=False):

        supported_extrapolations = ['nearest', 'linear', 'quadratic']

        extrapolation_type = extrapolation_type.lower()
        if extrapolation_type not in supported_extrapolations:
            raise ValueError("Unsupported extrapolation type: {}".format(extrapolation_type))

        self._continuity_order = continuity_order

        super().__init__(x, f, extrapolate, extrapolation_type, extrapolation_range, tolerate_single_value)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef object _build(self, ndarray x, ndarray f):
        """
        Calculate the spline coefficients.
        
        :param continuity_order: Order of spline continuity (1 or 2). 
        """

        cdef:
            int n, k, l, i
            double[::1] xv, x2v, x3v, fv
            double[::1] cv
            double [:, ::1] cm, coeffs
            double d

        n = len(x) - 1

        # Normalise coordinates and data arrays
        self._sx = 1 / (x.max() - x.min())
        self._ox = x.min()
        x = (x - self._ox) * self._sx

        # normalise data array
        self._of = f.min()
        self._sf = f.max() - f.min()
        if self._sf == 0:
            # zero data range, all values the same, disable scaling
            self._sf = 1
        f = (f - self._of) * (1 / self._sf)

        # obtain memory views
        xv = x
        x2v = x*x
        x3v = x*x*x
        fv = f

        # Fill the constraints matrix and vector
        cv = np.zeros((4*n,), dtype=np.float64)      # constraints_vector
        cm = np.zeros((4*n, 4*n), dtype=np.float64)  # constraints_matrix

        # Knots values constraints:
        l = 0
        for k in range(n):

            cm[l, 4*k] = 1.
            cm[l, 4*k+1] = xv[k]
            cm[l, 4*k+2] = x2v[k]
            cm[l, 4*k+3] = x3v[k]
            cv[l] = fv[k]
            l += 1

            cm[l, 4*k] = 1.
            cm[l, 4*k+1] = xv[k+1]
            cm[l, 4*k+2] = x2v[k+1]
            cm[l, 4*k+3] = x3v[k+1]
            cv[l] = fv[k+1]
            l += 1

        # first and/or second derivatives constraints:
        if self._continuity_order == 1:

            for k in range(n-1):

                d = (fv[k+2] - fv[k]) / (xv[k+2] - xv[k])
                cm[l, 4*k+1] = 1.
                cm[l, 4*k+2] = 2*xv[k+1]
                cm[l, 4*k+3] = 3*x2v[k+1]
                cv[l] = d
                l += 1

                cm[l, 4*k+5] = 1.
                cm[l, 4*k+6] = 2*xv[k+1]
                cm[l, 4*k+7] = 3*x2v[k+1]
                cv[l] = d
                l += 1

        elif self._continuity_order == 2:

            for k in range(n-1):

                # first derivative
                cm[l, 4*k+1] = -1.
                cm[l, 4*k+2] = -2*xv[k+1]
                cm[l, 4*k+3] = -3*x2v[k+1]
                cm[l, 4*k+5] = 1.
                cm[l, 4*k+6] = 2*xv[k+1]
                cm[l, 4*k+7] = 3*x2v[k+1]
                l += 1

                # second derivative
                cm[l, 4*k+2] = -2.
                cm[l, 4*k+3] = -6*xv[k+1]
                cm[l, 4*k+6] = 2.
                cm[l, 4*k+7] = 6*xv[k+1]
                l += 1

        else:
            raise NotImplementedError("'continuity_order' must be 1 or 2.")

        # two last constraints: the choice is here to set the third derivative
        # to zero at the edges because it is the less constraining condition.
        # Other possibilities are to impose values to the first or the second
        # derivatives. (comment/uncomment to change the constraints)

        # Third derivative
        cm[l, 3] = 6.
        l += 1
        cm[l, 4*n -1] = 6.
        l += 1

        # Second derivative
        # cm_view[l, 2] = 2.
        # cm_view[l, 3] = 6*xv[0]
        # l = l + 1
        # cm_view[l, 4*n -2] = 2.
        # cm_view[l, 4*n -1] = 6*xv[n]
        # l = l + 1

        # First derivative
        # cm_view[l, 1] = 1.
        # cm_view[l, 2] = 2*xv[0]
        # cm_view[l, 3] = 3*x2v[0]
        # l = l + 1
        # cm_view[l, 4*n -3] = 1.
        # cm_view[l, 4*n -2] = 2*xv[n]
        # cm_view[l, 4*n -1] = 3*x2v[n]
        # l = l + 1

        # Solve the linear system
        coeffs = solve(cm, cv).reshape((n, 4))
        self._k = coeffs

        # Denormalisation
        for i in range(n):
            for k in range(4):
                coeffs[i, k] = self._sf * self._sx ** k / factorial(k) * self._calc_polynomial_derivative(i, -self._sx * self._ox, k)
            coeffs[i, 0] = coeffs[i, 0] + self._of
        self._k = coeffs

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double _evaluate(self, double px, int order, int index) except? -1e999:
        """
        Evaluate the interpolating function which is valid in the area given
        by 'index' at any position 'px'.

        :param double px: x coordinate
        :param int index: index of the area of interest
        :return: the interpolated value
        """

        cdef double px2, px3

        # f(x)
        if order == 0:
            px2 = px*px
            px3 = px2*px
            return self._k[index, 0] + self._k[index, 1]*px + self._k[index, 2]*px2 + self._k[index, 3]*px3

        # df(x)/dx
        elif order == 1:
            px2 = px*px
            return self._k[index, 1] + 2*self._k[index, 2]*px + 3*self._k[index, 3]*px2

        # d2f(x)/dx2
        elif order == 2:
            return 2*self._k[index, 2] + 6*self._k[index, 3]*px

        # d3f(x)/dx3
        elif order == 3:
            return 6*self._k[index, 3]

        return 0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double _extrapol_linear(self, double px, int order, int index, double rx) except? -1e999:
        """
        Extrapolate linearly the interpolation function valid on area given by
        'index' to position 'px'.

        :param double px: x coordinate
        :param int index: index of the area of interest
        :param double rx: the nearest position from 'px' in the
        interpolation domain.
        :return: the extrapolated value
        """

        # f(x)
        if order == 0:
            return self._evaluate(rx, order, index) + (px - rx) * (3. * self._k[index, 3] * rx * rx + 2. * self._k[index, 2] * rx + self._k[index, 1])

        # df(x)/dx
        elif order == 1:
            return 3. * self._k[index, 3] * rx * rx + 2. * self._k[index, 2] * rx + self._k[index, 1]

        return 0


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double _extrapol_quadratic(self, double px, int order, int index, double rx) except? -1e999:
        """
        Extrapolate quadratically the interpolation function valid on area given by
        'index' to position 'px'.

        :param double px: x coordinate
        :param int index: index of the area of interest
        :param double rx: the nearest position from 'px' in the interpolation domain.
        :return: the extrapolated value
        """

        cdef:
            double a0, a1, a2
            double d = px - rx

        # f(x)
        if order == 0:
            a0 = self._evaluate(rx, order, index)
            a1 = 3 * self._k[index, 3] * rx * rx + 2 * self._k[index, 2] * rx + self._k[index, 1]
            a2 = 6 * self._k[index, 3] * rx + 2 * self._k[index, 2]
            return a0 + d * a1 + 0.5*d*d * a2

        # df(x)/dx
        elif order == 1:
            a1 = 3 * self._k[index, 3] * rx * rx + 2 * self._k[index, 2] * rx + self._k[index, 1]
            a2 = 6 * self._k[index, 3] * rx + 2 * self._k[index, 2]
            return a1 + d * a2

        # d2f(x)/dx2
        elif order == 2:
            return 6 * self._k[index, 3] * rx + 2 * self._k[index, 2]

        return 0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double _calc_polynomial_derivative(self, int ix, double px, int order_x):
        """
        Evaluate the derivatives of the polynomial valid in the area given by
        'ix' at position 'px'. The order of derivative is given by 'der_x'.

        :param int ix: index of the area of interest
        :param double px: x coordinate
        :param int der_x: order of derivative
        :return: value evaluated from the derivated polynomial
        """

        cdef double[::1] a = derivatives_array(px, order_x)
        return a[0]*self._k[ix, 0] + a[1]*self._k[ix, 1] + a[2]*self._k[ix, 2] + a[3]*self._k[ix, 3]
