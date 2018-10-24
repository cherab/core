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

from numpy import array, empty, int8, float64, concatenate, diff
from numpy.linalg import solve

cimport cython
from numpy cimport ndarray, npy_intp
from cherab.core.math.interpolators.utility cimport find_index, lerp, derivatives_array, factorial
from libc.math cimport INFINITY, NAN

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
    :param object f: A 2D array-like object of sample values corresponding to the
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

    def __init__(self, object x, object y, object f, bint extrapolate=False, str extrapolation_type='nearest',
                 double extrapolation_range=INFINITY, bint tolerate_single_value=False):

        # convert data to c-contiguous numpy arrays
        x = array(x, dtype=float64, order='c')
        y = array(y, dtype=float64, order='c')
        f = array(f, dtype=float64, order='c')

        # check dimensions are 1D
        if x.ndim != 1:
            raise ValueError("The x array must be 1D.")

        if y.ndim != 1:
            raise ValueError("The y array must be 1D.")

        # check data is 2D
        if f.ndim != 2:
            raise ValueError("The f array must be 2D.")

        # check the shapes of data and coordinates are consistent
        shape = (x.shape[0], y.shape[0])
        if f.shape != shape:
            raise ValueError("The dimension and data arrays must have consistent shapes ((x, y)={}, f={}).".format(shape, f.shape))

        # check the dimension arrays must be monotonically increasing
        if (diff(x) <= 0).any():
            raise ValueError("The x array must be monotonically increasing.")

        if (diff(y) <= 0).any():
            raise ValueError("The y array must be monotonically increasing.")

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
        self._y = y

        # Check for single value in x input
        if x.shape[0] == 1 and not tolerate_single_value:
            raise ValueError("There is only a single value in the x array. "
                             "Consider turning on the 'tolerate_single_value' argument.")

        # Check for single value in y input
        if y.shape[0] == 1 and not tolerate_single_value:
            raise ValueError("There is only a single value in the y array. "
                             "Consider turning on the 'tolerate_single_value' argument.")

        # build internal state of interpolator
        self._build(x, y, f)

    cdef object _build(self, ndarray x, ndarray y, ndarray f):
        """
        Build additional internal state.
        
        Implement in sub-classes that require additional state to be build
        from the source arrays.        
            
        :param x: x array 
        :param y: y array
        :param f: f array 
        """
        pass

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double evaluate(self, double px, double py) except? -1e999:
        """
        Evaluate the interpolating function.

        :param double px, double py: coordinates
        :return: the interpolated value
        """

        return self._dispatch(px, py, x_order=0, y_order=0)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef double derivative(self, double px, double py, int x_order, int y_order) except? -1e999:
        """
        Evaluate the interpolating function.

        :param double px, double py: coordinates
        :return: the interpolated value
        """

        cdef:
            int ix, iy, nx, ny
            double[::1] x, y

        if x_order < 1 and y_order < 1:
            raise ValueError('At least one derivative order must be > 0.')

        if x_order < 0:
            raise ValueError('The y derivative order cannot be less than zero.')

        if y_order < 0:
            raise ValueError('The y derivative order cannot be less than zero.')

        return self._dispatch(px, py, x_order, y_order)

    cdef double _dispatch(self, double px, double py, int x_order, int y_order) except? -1e999:
        """
        Identifies the region the sample point lies in and calls the relevant evaluator.
        """

        cdef:
            int ix, iy, nx, ny
            double[::1] x, y

        x = self._x
        y = self._y

        nx = x.shape[0]
        ny = y.shape[0]

        ix = find_index(x, px, self._extrapolation_range)
        iy = find_index(y, py, self._extrapolation_range)

        if 0 <= ix < nx - 1:
            if 0 <= iy < ny - 1:
                return self._evaluate(px, py, x_order, y_order, ix, iy)
            elif iy == -1:
                return self._extrapolate(px, py, x_order, y_order, ix, 0, px, y[0])
            elif iy == ny - 1:
                return self._extrapolate(px, py, x_order, y_order, ix, ny - 2, px, y[ny - 1])

        elif ix == -1:
            if 0 <= iy < ny - 1:
                return self._extrapolate(px, py, x_order, y_order, 0, iy, x[0], py)
            elif iy == -1:
                return self._extrapolate(px, py, x_order, y_order, 0, 0, x[0], y[0])
            elif iy == ny - 1:
                return self._extrapolate(px, py, x_order, y_order, 0, ny - 2, x[0], y[ny - 1])

        elif ix == nx - 1:
            if 0 <= iy < ny - 1:
                return self._extrapolate(px, py, x_order, y_order, nx - 2, iy, x[nx - 1], py)
            elif iy == -1:
                return self._extrapolate(px, py, x_order, y_order, nx - 2, 0, x[nx - 1], y[0])
            elif iy == ny - 1:
                return self._extrapolate(px, py, x_order, y_order, nx - 2, ny - 2, x[nx - 1], y[ny - 1])

        # value is outside of permitted limits
        min_range_x = x[0] - self._extrapolation_range
        max_range_x = x[nx - 1] + self._extrapolation_range

        min_range_y = y[0] - self._extrapolation_range
        max_range_y = y[ny - 1] + self._extrapolation_range

        raise ValueError("The specified value (x={}, y={}) is outside the range of the supplied data and/or extrapolation range: "
                         "x bounds=({}, {}), y bounds=({}, {})".format(px, py, min_range_x, max_range_x, min_range_y, max_range_y))

    cdef double _evaluate(self, double px, double py, int x_order, int y_order, int ix, int iy) except? -1e999:
        """
        Evaluate the interpolating function which is valid in the area given
        by 'ix' and 'iy' at any position ('px', 'py').

        :param double px, double py: coordinates
        :param int ix, int iy: indices of the area of interest
        :return: the interpolated value
        """
        raise NotImplementedError("This abstract method has not been implemented yet.")

    cdef double _extrapolate(self, double px, double py, int x_order, int y_order, int ix, int iy, double nearest_px, double nearest_py) except? -1e999:
        """
        Extrapolate the interpolation function valid on area given by
        'ix' and 'iy' to position ('px', 'py').

        :param double px, double py: coordinates
        :param int ix, int iy: indices of the area of interest
        :param double nearest_px, nearest_py: the nearest position from
        ('px', 'py') in the interpolation domain.
        :return: the extrapolated value
        """

        if self._extrapolation_type == EXT_NEAREST:
            if x_order == 0 and y_order == 0:
                return self._evaluate(nearest_px, nearest_py, 0, 0, ix, iy)
            else:
                # todo: implement extrapolation_nearest
                raise NotImplementedError

        elif self._extrapolation_type == EXT_LINEAR:
            return self._extrapol_linear(px, py, x_order, y_order, ix, iy, nearest_px, nearest_py)

        elif self._extrapolation_type == EXT_QUADRATIC:
            return self._extrapol_quadratic(px, py, x_order, y_order, ix, iy, nearest_px, nearest_py)

    cdef double _extrapol_linear(self, double px, double py, int x_order, int y_order, int ix, int iy, double nearest_px, double nearest_py) except? -1e999:
        """
        Extrapolate linearly the interpolation function valid on area given by
        'ix' and 'iy' to position ('px', 'py').

        :param double px, double py: coordinates
        :param int ix, int iy: indices of the area of interest
        :param double nearest_px, nearest_py: the nearest position from
        ('px', 'py') in the interpolation domain.
        :return: the extrapolated value
        """
        raise NotImplementedError("There is no linear extrapolation available for this interpolation.")

    cdef double _extrapol_quadratic(self, double px, double py, int x_order, int y_order, int ix, int iy, double nearest_px, double nearest_py) except? -1e999:
        """
        Extrapolate quadratically the interpolation function valid on area given by
        'ix' and 'iy' to position ('px', 'py').

        :param double px, double py: coordinates
        :param int ix, int iy: indices of the area of interest
        :param double nearest_px, nearest_py: the nearest position from
        ('px', 'py') in the interpolation domain.
        :return: the extrapolated value
        """
        raise NotImplementedError("There is no quadratic extrapolation available for this interpolation.")


cdef class Interpolate2DLinear(_Interpolate2DBase):
    """
    Interpolates 2D data using linear interpolation.

    :param object x: An array-like object containing real values.
    :param object y: An array-like object containing real values.
    :param object f: A 2D array-like object of sample values corresponding to the
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

    def __init__(self, object x, object y, object f, bint extrapolate=False, str extrapolation_type='nearest',
                 double extrapolation_range=float('inf'), bint tolerate_single_value=False):

        supported_extrapolations = ['nearest', 'linear']

        extrapolation_type = extrapolation_type.lower()
        if extrapolation_type not in supported_extrapolations:
            raise ValueError("Unsupported extrapolation type: {}".format(extrapolation_type))

        super().__init__(x, y, f, extrapolate, extrapolation_type, extrapolation_range, tolerate_single_value)

    cdef object _build(self, ndarray x, ndarray y, ndarray f):

        cdef ndarray temp

        # if x array is single valued, expand x array and data along x axis to simplify interpolation
        if x.shape[0] == 1:

            # set x array to full real range
            self._x = array([-INFINITY, INFINITY], dtype=float64)

            # duplicate data to provide a pseudo range for interpolation coefficient calculation
            temp = f
            x = array([-1., +1.], dtype=float64)
            f = empty((2, f.shape[1]), dtype=float64)
            f[:, :] = temp

        # if y array is single valued, expand y array and data along y axis to simplify interpolation
        if y.shape[0] == 1:

            # set y array to full real range
            self._y = array([-INFINITY, INFINITY], dtype=float64)

            # duplicate data to provide a pseudo range for interpolation coefficient calculation
            temp = f
            y = array([-1., +1.], dtype=float64)
            f = empty((f.shape[0], 2), dtype=float64)
            f[:, :] = temp

        # create memory views
        self._wx = x
        self._wy = y
        self._wf = f

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double _evaluate(self, double px, double py, int x_order, int y_order, int ix, int iy) except? -1e999:
        """
        Evaluate the interpolating function which is valid in the area given
        by 'ix' and 'iy' at any position ('px', 'py').

        :param double px, double py: coordinates
        :param int ix, int iy: indices of the area of interest
        :return: the interpolated value
        """
        cdef:
            double[::1] x, y
            double[:,::1] f
            double t0, t1

        x = self._wx
        y = self._wy
        f = self._wf

        if x_order == 0:

            # f(x, y)
            if y_order == 0:

                t0 = lerp(x[ix], x[ix+1], f[ix, iy],   f[ix+1, iy],   px)
                t1 = lerp(x[ix], x[ix+1], f[ix, iy+1], f[ix+1, iy+1], px)
                return lerp(y[iy], y[iy+1], t0, t1, py)

            # df(x, y) / dy
            elif y_order == 1:

                t0 = (f[ix, iy+1] - f[ix, iy]) / (y[iy+1] - y[iy])
                t1 = (f[ix+1, iy+1] - f[ix+1, iy]) / (y[iy+1] - y[iy])
                return lerp(x[ix], x[ix+1], t0, t1, px)

        elif x_order == 1:

            # df(x, y) / dx
            if y_order == 0:

                t0 = (f[ix+1, iy] - f[ix, iy]) / (x[ix+1] - x[ix])
                t1 = (f[ix+1, iy+1] - f[ix, iy+1]) / (x[ix+1] - x[ix])
                return lerp(y[iy], y[iy+1], t0, t1, py)

            # d2f(x, y) / dxdy
            elif y_order == 1:

                t0 = self._evaluate(px, y[iy], 1, 0, ix, iy)
                t1 = self._evaluate(px, y[iy+1], 1, 0, ix, iy)
                return (t1 - t0) / (y[iy+1] - y[iy])

        # higher order derivatives
        return 0

    cdef double _extrapol_linear(self, double px, double py, int x_order, int y_order, int ix, int iy, double nearest_px, double nearest_py) except? -1e999:
        """
        Extrapolate linearly the interpolation function valid on area given by
        'ix' and 'iy' to position ('px', 'py').

        :param double px, double py: coordinates
        :param int ix, int iy: indices of the area of interest
        :param double nearest_px, nearest_py: the nearest position from
        ('px', 'py') in the interpolation domain.
        :return: the extrapolated value
        """

        return self._evaluate(px, py, x_order, y_order, ix, iy)


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
    :param object f: A 2D array-like object of sample values corresponding to the
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

    def __init__(self, object x, object y, object f, bint extrapolate=False, double extrapolation_range=float('inf'),
                 str extrapolation_type='nearest', bint tolerate_single_value=False):

        cdef int i, j, i_narrowed, j_narrowed

        supported_extrapolations = ['nearest', 'linear', 'quadratic']

        extrapolation_type = extrapolation_type.lower()
        if extrapolation_type not in supported_extrapolations:
            raise ValueError("Unsupported extrapolation type: {}".format(extrapolation_type))

        super().__init__(x, y, f, extrapolate, extrapolation_type, extrapolation_range, tolerate_single_value)


    cdef object _build(self, ndarray x, ndarray y, ndarray f):

        cdef:
            ndarray temp
            ndarray wx, wy, wf
            double[:,::1] f_mv
            int nx, ny
            int i, j, i_mapped, j_mapped

        # if x array is single valued, expand x array and data along x axis to simplify interpolation
        if x.shape[0] == 1:

            # set x array to full real range
            self._x = array([-INFINITY, INFINITY], dtype=float64)

            # duplicate data to provide a pseudo range for interpolation coefficient calculation
            temp = f
            x = array([-1., +1.], dtype=float64)
            f = empty((2, f.shape[1]), dtype=float64)
            f[:, :] = temp

        # if y array is single valued, expand y array and data along y axis to simplify interpolation
        if y.shape[0] == 1:

            # set y array to full real range
            self._y = array([-INFINITY, INFINITY], dtype=float64)

            # duplicate data to provide a pseudo range for interpolation coefficient calculation
            temp = f
            y = array([-1., +1.], dtype=float64)
            f = empty((f.shape[0], 2), dtype=float64)
            f[:, :] = temp

        nx = x.shape[0]
        ny = y.shape[0]

        # initialise the spline coefficient cache arrays
        self._k = empty((nx - 1, ny - 1, 16), dtype=float64)
        self._k[:,:,:] = NAN

        self._available = empty((nx - 1, ny - 1), dtype=int8)
        self._available[:,:] = False

        # normalise coordinate arrays
        self._ox = x.min()
        self._oy = y.min()

        self._sx = 1 / (x.max() - x.min())
        self._sy = 1 / (y.max() - y.min())

        x = (x - self._ox) * self._sx
        y = (y - self._oy) * self._sy

        # normalise data array
        self._of = f.min()
        self._sf = f.max() - f.min()
        if self._sf == 0:
            # zero data range, all values the same, disable scaling
            self._sf = 1
        f = (f - self._of) * (1 / self._sf)

        # widen arrays for automatic handling of boundaries polynomials
        wx = concatenate(([x[0]], x, [x[-1]]))
        wy = concatenate(([y[0]], y, [y[-1]]))
        wf = empty((nx + 2, ny + 2), dtype=float64)

        # store memory views to widened data
        self._wx = wx
        self._wy = wy
        self._wf = wf

        # populate expanded f array by duplicating the edges of the array
        f_mv = f
        for i in range(nx + 2):
            for j in range(ny + 2):
                i_mapped = min(max(0, i - 1), nx - 1)
                j_mapped = min(max(0, j - 1), ny - 1)
                self._wf[i, j] = f_mv[i_mapped, j_mapped]

        # calculate and cache higher powers of the dimension array data
        self._wx2 = wx * wx
        self._wx3 = wx * wx * wx

        self._wy2 = wy * wy
        self._wy3 = wy * wy * wy

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double _evaluate(self, double px, double py, int x_order, int y_order, int ix, int iy) except? -1e999:
        """
        Evaluate the interpolating function which is valid in the area given
        by 'ix' and 'iy' at any position ('px', 'py').

        :param double px, double py: coordinates
        :param int ix, int iy: indices of the area of interest
        :return: the interpolated value
        """

        cdef:
            double px2, py2, px3, py3
            double[:,:,::1] k

        k = self._k

        # If the concerned polynomial has not yet been calculated:
        if not self._available[ix, iy]:
            self._calc_polynomial(ix, iy)

        # f(x,y)
        if x_order == 0 and y_order == 0:

            px2 = px*px
            px3 = px2*px

            py2 = py*py
            py3 = py2*py

            return     (k[ix, iy,  0] + k[ix, iy,  1]*py + k[ix, iy,  2]*py2 + k[ix, iy,  3]*py3) + \
                   px *(k[ix, iy,  4] + k[ix, iy,  5]*py + k[ix, iy,  6]*py2 + k[ix, iy,  7]*py3) + \
                   px2*(k[ix, iy,  8] + k[ix, iy,  9]*py + k[ix, iy, 10]*py2 + k[ix, iy, 11]*py3) + \
                   px3*(k[ix, iy, 12] + k[ix, iy, 13]*py + k[ix, iy, 14]*py2 + k[ix, iy, 15]*py3)

        raise NotImplementedError('Derivative of x order {} and y order {} is not implemented.'.format(x_order, y_order))

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef int _calc_polynomial(self, int ix, int iy) except -1:
        """
        Calculates and caches the polynomial coefficients for area given by
        'ix', 'iy'. Declares this area as already calculated.

        :param int ix, int iy: indices of the area of interest
        """

        cdef:
            int u, v, l, i, j
            double delta_x, delta_y
            double cv_buffer[16]
            double cm_buffer[16][16]
            double[::1] cv, coeffs
            double[:,::1] cm
            double s

        # create memory views to constraint vector and matrix buffers
        cv = cv_buffer
        cm = cm_buffer

        # Fill the constraints matrix
        l = 0
        for u in range(ix+1, ix+3):
            for v in range(iy+1, iy+3):

                delta_x = self._wx[u+1] - self._wx[u-1]
                delta_y = self._wy[v+1] - self._wy[v-1]

                # knot values
                cm[l, 0] = 1.
                cm[l, 1] = self._wy[v]
                cm[l, 2] = self._wy2[v]
                cm[l, 3] = self._wy3[v]
                cm[l, 4] = self._wx[u]
                cm[l, 5] = self._wx[u]*self._wy[v]
                cm[l, 6] = self._wx[u]*self._wy2[v]
                cm[l, 7] = self._wx[u]*self._wy3[v]
                cm[l, 8] = self._wx2[u]
                cm[l, 9] = self._wx2[u]*self._wy[v]
                cm[l, 10] = self._wx2[u]*self._wy2[v]
                cm[l, 11] = self._wx2[u]*self._wy3[v]
                cm[l, 12] = self._wx3[u]
                cm[l, 13] = self._wx3[u]*self._wy[v]
                cm[l, 14] = self._wx3[u]*self._wy2[v]
                cm[l, 15] = self._wx3[u]*self._wy3[v]
                cv[l] = self._wf[u, v]
                l += 1

                # derivative along x
                cm[l, 0] = 0.
                cm[l, 1] = 0.
                cm[l, 2] = 0.
                cm[l, 3] = 0.
                cm[l, 4] = 1.
                cm[l, 5] = self._wy[v]
                cm[l, 6] = self._wy2[v]
                cm[l, 7] = self._wy3[v]
                cm[l, 8] = 2.*self._wx[u]
                cm[l, 9] = 2.*self._wx[u]*self._wy[v]
                cm[l, 10] = 2.*self._wx[u]*self._wy2[v]
                cm[l, 11] = 2.*self._wx[u]*self._wy3[v]
                cm[l, 12] = 3.*self._wx2[u]
                cm[l, 13] = 3.*self._wx2[u]*self._wy[v]
                cm[l, 14] = 3.*self._wx2[u]*self._wy2[v]
                cm[l, 15] = 3.*self._wx2[u]*self._wy3[v]
                cv[l] = (self._wf[u+1, v] - self._wf[u-1, v])/delta_x
                l += 1

                # derivative along y
                cm[l, 0] = 0.
                cm[l, 1] = 1.
                cm[l, 2] = 2.*self._wy[v]
                cm[l, 3] = 3.*self._wy2[v]
                cm[l, 4] = 0.
                cm[l, 5] = self._wx[u]
                cm[l, 6] = 2.*self._wx[u]*self._wy[v]
                cm[l, 7] = 3.*self._wx[u]*self._wy2[v]
                cm[l, 8] = 0.
                cm[l, 9] = self._wx2[u]
                cm[l, 10] = 2.*self._wx2[u]*self._wy[v]
                cm[l, 11] = 3.*self._wx2[u]*self._wy2[v]
                cm[l, 12] = 0.
                cm[l, 13] = self._wx3[u]
                cm[l, 14] = 2.*self._wx3[u]*self._wy[v]
                cm[l, 15] = 3.*self._wx3[u]*self._wy2[v]
                cv[l] = (self._wf[u, v+1] - self._wf[u, v-1])/delta_y
                l += 1

                # cross derivative
                cm[l, 0] = 0.
                cm[l, 1] = 0.
                cm[l, 2] = 0.
                cm[l, 3] = 0.
                cm[l, 4] = 0.
                cm[l, 5] = 1.
                cm[l, 6] = 2.*self._wy[v]
                cm[l, 7] = 3.*self._wy2[v]
                cm[l, 8] = 0.
                cm[l, 9] = 2.*self._wx[u]
                cm[l, 10] = 4.*self._wx[u]*self._wy[v]
                cm[l, 11] = 6.*self._wx[u]*self._wy2[v]
                cm[l, 12] = 0.
                cm[l, 13] = 3.*self._wx2[u]
                cm[l, 14] = 6.*self._wx2[u]*self._wy[v]
                cm[l, 15] = 9.*self._wx2[u]*self._wy2[v]
                cv[l] = (self._wf[u+1, v+1] - self._wf[u+1, v-1] - self._wf[u-1, v+1] + self._wf[u-1, v-1])/(delta_x*delta_y)
                l += 1

        # Solve the linear system and fill the caching coefficients array
        coeffs = solve(cm, cv)
        self._k[ix, iy, :] = coeffs

        # Denormalisation
        for i in range(4):
            for j in range(4):
                s = self._sf * self._sx**i * self._sy**j / (factorial(j) * factorial(i))
                coeffs[4*i + j] = s * self._calc_polynomial_derivative(ix, iy, -self._sx * self._ox, -self._sy * self._oy, i, j)
        coeffs[0] = coeffs[0] + self._of

        # populate coefficients and set cell as calculated
        self._k[ix, iy, :] = coeffs
        self._available[ix, iy] = True

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double _extrapol_linear(self, double px, double py, int x_order, int y_order, int ix, int iy, double rx, double ry) except? -1e999:
        """
        Extrapolate linearly the interpolation function valid on area given by
        'ix' and 'iy' to position ('px', 'py').

        :param double px, double py: coordinates
        :param int ix, int iy: indices of the area of interest
        :param double rx, ry: the nearest position from ('px', 'py') in the interpolation domain.
        :return: the extrapolated value
        """

        cdef:
            double ex, ey, nx, nx2, nx3, ny, ny2, ny3, result
            double[:,:,::1] k

        k = self._k

        if x_order == 0 and y_order == 0:

            # calculate extrapolation distances from end of array
            ex = px - rx
            ey = py - ry

            nx = rx
            nx2 = nx*nx
            nx3 = nx2*nx

            ny = ry
            ny2 = ny*ny
            ny3 = ny2*ny

            result = self._evaluate(nx, ny, 0, 0, ix, iy)
            if ex != 0.:
                result += ex * (       (k[ix, iy,  4] + k[ix, iy,  5]*ny + k[ix, iy,  6]*ny2 + k[ix, iy,  7]*ny3) + \
                                2.*nx *(k[ix, iy,  8] + k[ix, iy,  9]*ny + k[ix, iy, 10]*ny2 + k[ix, iy, 11]*ny3) + \
                                3.*nx2*(k[ix, iy, 12] + k[ix, iy, 13]*ny + k[ix, iy, 14]*ny2 + k[ix, iy, 15]*ny3))

            if ey != 0.:
                result += ey * (    (k[ix, iy,  1] + 2.*k[ix, iy,  2]*ny + 3.*k[ix, iy,  3]*ny2) + \
                                nx *(k[ix, iy,  5] + 2.*k[ix, iy,  6]*ny + 3.*k[ix, iy,  7]*ny2) + \
                                nx2*(k[ix, iy,  9] + 2.*k[ix, iy, 10]*ny + 3.*k[ix, iy, 11]*ny2) + \
                                nx3*(k[ix, iy, 13] + 2.*k[ix, iy, 14]*ny + 3.*k[ix, iy, 15]*ny2))

            return result

        raise NotImplementedError('Derivative of x order {} and y order {} is not implemented.'.format(x_order, y_order))

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double _extrapol_quadratic(self, double px, double py, int x_order, int y_order, int ix, int iy, double rx, double ry) except? -1e999:
        """
        Extrapolate quadratically the interpolation function valid on area given by
        'ix' and 'iy' to position ('px', 'py').

        :param double px, double py: coordinates
        :param int ix, int iy: indices of the area of interest
        :param double rx, ry: the nearest position from ('px', 'py') in the interpolation domain.
        :return: the extrapolated value
        """

        cdef:
            double ex, ey, nx, nx2, nx3, ny, ny2, ny3, result
            double[:,:,::1] k

        k = self._k

        if x_order == 0 and y_order == 0:

            ex = px - rx
            ey = py - ry

            nx = rx
            nx2 = nx*nx
            nx3 = nx2*nx

            ny = ry
            ny2 = ny*ny
            ny3 = ny2*ny

            result = self._evaluate(nx, ny, 0, 0, ix, iy)
            if ex != 0.:
                result += ex * (       (k[ix, iy,  4] + k[ix, iy,  5]*ny + k[ix, iy,  6]*ny2 + k[ix, iy,  7]*ny3) + \
                                2.*nx *(k[ix, iy,  8] + k[ix, iy,  9]*ny + k[ix, iy, 10]*ny2 + k[ix, iy, 11]*ny3) + \
                                3.*nx2*(k[ix, iy, 12] + k[ix, iy, 13]*ny + k[ix, iy, 14]*ny2 + k[ix, iy, 15]*ny3))

                result += ex*ex*0.5 * (2.   *(k[ix, iy,  8] + k[ix, iy,  9]*ny + k[ix, iy, 10]*ny2 + k[ix, iy, 11]*ny3) + \
                                       6.*nx*(k[ix, iy, 12] + k[ix, iy, 13]*ny + k[ix, iy, 14]*ny2 + k[ix, iy, 15]*ny3))

            if ey != 0.:
                result += ey * (    (k[ix, iy,  1] + 2.*k[ix, iy,  2]*ny + 3.*k[ix, iy,  3]*ny2) + \
                                nx *(k[ix, iy,  5] + 2.*k[ix, iy,  6]*ny + 3.*k[ix, iy,  7]*ny2) + \
                                nx2*(k[ix, iy,  9] + 2.*k[ix, iy, 10]*ny + 3.*k[ix, iy, 11]*ny2) + \
                                nx3*(k[ix, iy, 13] + 2.*k[ix, iy, 14]*ny + 3.*k[ix, iy, 15]*ny2))

                result += ey*ey*0.5 * (    (2.*k[ix, iy,  2] + 6.*k[ix, iy,  3]*ny) + \
                                       nx *(2.*k[ix, iy,  6] + 6.*k[ix, iy,  7]*ny) + \
                                       nx2*(2.*k[ix, iy, 10] + 6.*k[ix, iy, 11]*ny) + \
                                       nx3*(2.*k[ix, iy, 14] + 6.*k[ix, iy, 15]*ny))

                if ex != 0.:
                    result += ex*ey * (       (k[ix, iy,  5] + 2.*k[ix, iy,  6]*ny + 3.*k[ix, iy,  7]*ny2) + \
                                       2.*nx *(k[ix, iy,  9] + 2.*k[ix, iy, 10]*ny + 3.*k[ix, iy, 11]*ny2) + \
                                       3.*nx2*(k[ix, iy, 13] + 2.*k[ix, iy, 14]*ny + 3.*k[ix, iy, 15]*ny2))

            return result

        raise NotImplementedError('Derivative of x order {} and y order {} is not implemented.'.format(x_order, y_order))

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double _calc_polynomial_derivative(self, int ix, int iy, double px, double py, int order_x, int order_y):
        """
        Evaluate the derivatives of the polynomial valid in the area given by
        'ix', 'iy' at position ('px', 'py'). The order of
        derivative along each axis is given by 'der_x', 'der_y'.

        :param int ix, int iy: indices of the area of interest
        :param double px, double py: coordinates
        :param int der_x, int order_y: orders of derivative along each axis
        :return: value evaluated from the derivated polynomial
        """

        cdef:
            double[::1] ax, ay
            double[:,:,::1] k

        k = self._k

        ax = derivatives_array(px, order_x)
        ay = derivatives_array(py, order_y)

        return ax[0]*(ay[0]*k[ix, iy,  0] + ay[1]*k[ix, iy,  1] + ay[2]*k[ix, iy,  2] + ay[3]*k[ix, iy,  3]) + \
               ax[1]*(ay[0]*k[ix, iy,  4] + ay[1]*k[ix, iy,  5] + ay[2]*k[ix, iy,  6] + ay[3]*k[ix, iy,  7]) + \
               ax[2]*(ay[0]*k[ix, iy,  8] + ay[1]*k[ix, iy,  9] + ay[2]*k[ix, iy, 10] + ay[3]*k[ix, iy, 11]) + \
               ax[3]*(ay[0]*k[ix, iy, 12] + ay[1]*k[ix, iy, 13] + ay[2]*k[ix, iy, 14] + ay[3]*k[ix, iy, 15])
