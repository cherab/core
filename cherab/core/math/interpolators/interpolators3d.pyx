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

cdef class _Interpolate3DBase(Function3D):
    """
    Base class for 3D interpolators. Coordinates and data arrays are here
    sorted and transformed into numpy arrays.

    :param object x: An array-like object containing real values.
    :param object y: An array-like object containing real values.
    :param object z: An array-like object containing real values.
    :param object f: A 3D array-like object of sample values corresponding to the
    `x`, `y` and `z` array points.
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

    def __init__(self, object x, object y, object z, object f, bint extrapolate=False, str extrapolation_type='nearest',
                 double extrapolation_range=INFINITY, bint tolerate_single_value=False):

        # convert data to c-contiguous numpy arrays
        x = array(x, dtype=float64, order='c')
        y = array(y, dtype=float64, order='c')
        z = array(z, dtype=float64, order='c')
        f = array(f, dtype=float64, order='c')

        # check dimensions are 1D
        if x.ndim != 1:
            raise ValueError("The x array must be 1D.")

        if y.ndim != 1:
            raise ValueError("The y array must be 1D.")

        if z.ndim != 1:
            raise ValueError("The z array must be 1D.")

        # check data is 3D
        if f.ndim != 3:
            raise ValueError("The f array must be 3D.")

        # check the shapes of data and coordinates are consistent
        shape = (x.shape[0], y.shape[0], z.shape[0])
        if f.shape != shape:
            raise ValueError("The dimension and data arrays must have consistent shapes ((x, y, z)={}, f={}).".format(shape, f.shape))

        # check the dimension arrays must be monotonically increasing
        if (diff(x) <= 0).any():
            raise ValueError("The x array must be monotonically increasing.")

        if (diff(y) <= 0).any():
            raise ValueError("The y array must be monotonically increasing.")

        if (diff(z) <= 0).any():
            raise ValueError("The z array must be monotonically increasing.")

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
        self._z = z

        # Check for single value in x input
        if x.shape[0] == 1 and not tolerate_single_value:
            raise ValueError("There is only a single value in the x array. "
                             "Consider turning on the 'tolerate_single_value' argument.")

        # Check for single value in y input
        if y.shape[0] == 1 and not tolerate_single_value:
            raise ValueError("There is only a single value in the y array. "
                             "Consider turning on the 'tolerate_single_value' argument.")

        # Check for single value in z input
        if z.shape[0] == 1 and not tolerate_single_value:
            raise ValueError("There is only a single value in the z array. "
                             "Consider turning on the 'tolerate_single_value' argument.")

        # build internal state of interpolator
        self._build(x, y, z, f)

    cdef object _build(self, ndarray x, ndarray y, ndarray z, ndarray f):
        """
        Build additional internal state.
        
        Implement in sub-classes that require additional state to be build
        from the source arrays.        
            
        :param x: x array 
        :param y: y array
        :param z: z array
        :param f: f array 
        """
        pass

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double evaluate(self, double px, double py, double pz) except? -1e999:
        """
        Evaluate the interpolating function.

        :param double px, double py, double pz: coordinates
        :return: the interpolated value
        """

        cdef:
            int ix, iy, iz, nx, ny, nz
            double[::1] x, y, z

        x = self._x
        y = self._y
        z = self._z

        nx = x.shape[0]
        ny = y.shape[0]
        nz = z.shape[0]

        ix = find_index(x, px, self._extrapolation_range)
        iy = find_index(y, py, self._extrapolation_range)
        iz = find_index(z, pz, self._extrapolation_range)

        if 0 <= ix < nx - 1:
            if 0 <= iy < ny - 1:
                if 0 <= iz < nz - 1:
                    return self._evaluate(px, py, pz, ix, iy, iz)
                elif iz == -1:
                    return self._extrapolate(px, py, pz, ix, iy, 0, px, py, z[0])
                elif iz == nz - 1:
                    return self._extrapolate(px, py, pz, ix, iy, nz - 2, px, py, z[nz - 1])

            elif iy == -1:
                if 0 <= iz < nz - 1:
                    return self._extrapolate(px, py, pz, ix, 0, iz, px, y[0], pz)
                elif iz == -1:
                    return self._extrapolate(px, py, pz, ix, 0, 0, px, y[0], z[0])
                elif iz == nz - 1:
                    return self._extrapolate(px, py, pz, ix, 0, nz - 2, px, y[0], z[nz - 1])

            elif iy == ny - 1:
                if 0 <= iz < nz - 1:
                    return self._extrapolate(px, py, pz, ix, ny - 2, iz, px, y[ny - 1], pz)
                elif iz == -1:
                    return self._extrapolate(px, py, pz, ix, ny - 2, 0, px, y[ny - 1], z[0])
                elif iz == nz - 1:
                    return self._extrapolate(px, py, pz, ix, ny - 2, nz - 2, px, y[ny - 1], z[nz - 1])

        elif ix == -1:
            if 0 <= iy < ny - 1:
                if 0 <= iz < nz - 1:
                    return self._extrapolate(px, py, pz, 0, iy, iz, x[0], py, pz)
                elif iz == -1:
                    return self._extrapolate(px, py, pz, 0, iy, 0, x[0], py, z[0])
                elif iz == nz - 1:
                    return self._extrapolate(px, py, pz, 0, iy, nz - 2, x[0], py, z[nz - 1])

            elif iy == -1:
                if 0 <= iz < nz - 1:
                    return self._extrapolate(px, py, pz, 0, 0, iz, x[0], y[0], pz)
                elif iz == -1:
                    return self._extrapolate(px, py, pz, 0, 0, 0, x[0], y[0], z[0])
                elif iz == nz - 1:
                    return self._extrapolate(px, py, pz, 0, 0, nz - 2, x[0], y[0], z[nz - 1])

            elif iy == ny - 1:
                if 0 <= iz < nz - 1:
                    return self._extrapolate(px, py, pz, 0, ny - 2, iz, x[0], y[ny - 1], pz)
                elif iz == -1:
                    return self._extrapolate(px, py, pz, 0, ny - 2, 0, x[0], y[ny - 1], z[0])
                elif iz == nz - 1:
                    return self._extrapolate(px, py, pz, 0, ny - 2, nz - 2, x[0], y[ny - 1], z[nz - 1])

        elif ix == nx - 1:
            if 0 <= iy < ny - 1:
                if 0 <= iz < nz - 1:
                    return self._extrapolate(px, py, pz, nx - 2, iy, iz, x[nx - 1], py, pz)
                elif iz == -1:
                    return self._extrapolate(px, py, pz, nx - 2, iy, 0, x[nx - 1], py, z[0])
                elif iz == nz - 1:
                    return self._extrapolate(px, py, pz, nx - 2, iy, nz - 2, x[nx - 1], py, z[nz - 1])

            elif iy == -1:
                if 0 <= iz < nz - 1:
                    return self._extrapolate(px, py, pz, nx - 2, 0, iz, x[nx - 1], y[0], pz)
                elif iz == -1:
                    return self._extrapolate(px, py, pz, nx - 2, 0, 0, x[nx - 1], y[0], z[0])
                elif iz == nz - 1:
                    return self._extrapolate(px, py, pz, nx - 2, 0, nz - 2, x[nx - 1], y[0], z[nz - 1])

            elif iy == ny - 1:
                if 0 <= iz < nz - 1:
                    return self._extrapolate(px, py, pz, nx - 2, ny - 2, iz, x[nx - 1], y[ny - 1], pz)
                elif iz == -1:
                    return self._extrapolate(px, py, pz, nx - 2, ny - 2, 0, x[nx - 1], y[ny - 1], z[0])
                elif iz == nz - 1:
                    return self._extrapolate(px, py, pz, nx - 2, ny - 2, nz - 2, x[nx - 1], y[ny - 1], z[nz - 1])

        # value is outside of permitted limits
        min_range_x = x[0] - self._extrapolation_range
        max_range_x = x[nx - 1] + self._extrapolation_range

        min_range_y = y[0] - self._extrapolation_range
        max_range_y = y[ny - 1] + self._extrapolation_range

        min_range_z = z[0] - self._extrapolation_range
        max_range_z = z[nz - 1] + self._extrapolation_range

        raise ValueError("The specified value (x={}, y={}, z={}) is outside the range of the supplied data and/or extrapolation range: "
                         "x bounds=({}, {}), y bounds=({}, {}), z bounds=({}, {})".format(px, py, pz, min_range_x, max_range_x, min_range_y, max_range_y, min_range_z, max_range_z))

    cdef double _evaluate(self, double px, double py, double pz, int ix, int iy, int iz) except? -1e999:
        """
        Evaluate the interpolating function which is valid in the area given
        by 'ix', 'iy' and 'iz' at any position ('px', 'py', 'pz').

        :param double px, double py, double pz: coordinates
        :param int ix, int iy, int iz: indices of the area of interest
        :return: the interpolated value
        """
        raise NotImplementedError("This abstract method has not been implemented yet.")

    cdef double _extrapolate(self, double px, double py, double pz, int ix, int iy, int iz, double nearest_px, double nearest_py, double nearest_pz) except? -1e999:
        """
        Extrapolate the interpolation function valid on area given by
        'ix', 'iy' and 'iz' to position ('px', 'py', 'pz').

        :param double px, double py, double pz: coordinates
        :param int ix, int iy, int iz: indices of the area of interest
        :param double nearest_px, nearest_py, nearest_pz: the nearest position from
        ('px', 'py', 'pz') in the interpolation domain.
        :return: the extrapolated value
        """

        if self._extrapolation_type == EXT_NEAREST:
            return self._evaluate(nearest_px, nearest_py, nearest_pz, ix, iy, iz)
        
        elif self._extrapolation_type == EXT_LINEAR:
            return self._extrapol_linear(px, py, pz, ix, iy, iz, nearest_px, nearest_py, nearest_pz)
        
        elif self._extrapolation_type == EXT_QUADRATIC:
            return self._extrapol_quadratic(px, py, pz, ix, iy, iz, nearest_px, nearest_py, nearest_pz)

    cdef double _extrapol_linear(self, double px, double py, double pz, int ix, int iy, int iz, double nearest_px, double nearest_py, double nearest_pz) except? -1e999:
        """
        Extrapolate linearly the interpolation function valid on area given by
        'ix', 'iy' and 'iz' to position ('px', 'py', 'pz').

        :param double px, double py, double pz: coordinates
        :param int ix, int iy, int iz: indices of the area of interest
        :param double nearest_px, nearest_py, nearest_pz: the nearest position from
        ('px', 'py', 'pz') in the interpolation domain.
        :return: the extrapolated value
        """
        raise NotImplementedError("There is no linear extrapolation available for this interpolation.")

    cdef double _extrapol_quadratic(self, double px, double py, double pz, int ix, int iy, int iz, double nearest_px, double nearest_py, double nearest_pz) except? -1e999:
        """
        Extrapolate quadratically the interpolation function valid on area given by
        'ix', 'iy' and 'iz' to position ('px', 'py', 'pz').

        :param double px, double py, double pz: coordinates
        :param int ix, int iy, int iz: indices of the area of interest
        :param double nearest_px, nearest_py, nearest_pz: the nearest position from
        ('px', 'py', 'pz') in the interpolation domain.
        :return: the extrapolated value
        """
        raise NotImplementedError("There is no quadratic extrapolation available for this interpolation.")


cdef class Interpolate3DLinear(_Interpolate3DBase):
    """
    Interpolates 3D data using linear interpolation.

    Inherits from Function3D, implements `__call__(x, y, z)`.

    :param object x: An array-like object containing real values.
    :param object y: An array-like object containing real values.
    :param object z: An array-like object containing real values.
    :param object f: A 3D array-like object of sample values corresponding to the
      `x`, `y` and `z` array points.
    :param bint extrapolate: If True, the extrapolation of data is enabled outside
      the range of the data set. The default is False. A ValueError is raised if
      extrapolation is disabled and a point is requested outside the data set.
    :param object extrapolation_type: Sets the method of extrapolation.
      The options are: 'nearest' (default), 'linear'.
    :param double extrapolation_range: The attribute can be set to limit the range
      beyond the data set bounds that extrapolation is permitted. The default range
      is set to infinity. Requesting data beyond the extrapolation range will result
      in a ValueError being raised.
    :param tolerate_single_value: If True, single-value arrays will be tolerated as
      inputs. If a single value is supplied, that value will be extrapolated over
      the entire real range. If False (default), supplying a single value will
      result in a ValueError being raised.

    .. code-block:: pycon

       >>> import numpy as np
       >>> from cherab.core.math import Interpolate3DLinear
       >>>
       >>> # implements x**3 + y**2 + z
       >>> drange = np.linspace(-2.5, 2.5, 100)
       >>> values = np.zeros((100, 100, 100))
       >>> for i in range(100):
       >>>     for j in range(100):
       >>>         for k in range(100):
       >>>             values[i, j, k] = drange[i]**3 + drange[j]**2 + drange[k]
       >>>
       >>> f3d = Interpolate3DLinear(drange, drange, drange, values)
       >>>
       >>> f3d(0, 0, 0)
       0.00063769
       >>> f3d(-2, 1, 1.5)
       -5.50085102
       >>> f3d(-3, 1, 0)
       ValueError: The specified value (x=-3.0, y=1.0, z=0.0) is outside the range
       of the supplied data and/or extrapolation range: x bounds=(-2.5, 2.5),
       y bounds=(-2.5, 2.5), z bounds=(-2.5, 2.5)
    """

    def __init__(self, object x, object y, object z, object f, bint extrapolate=False, str extrapolation_type='nearest',
                 double extrapolation_range=INFINITY, bint tolerate_single_value=False):

        supported_extrapolations = ['nearest', 'linear']

        extrapolation_type = extrapolation_type.lower()
        if extrapolation_type not in supported_extrapolations:
            raise ValueError("Unsupported extrapolation type: {}".format(extrapolation_type))

        super().__init__(x, y, z, f, extrapolate, extrapolation_type, extrapolation_range, tolerate_single_value)

    cdef object _build(self, ndarray x, ndarray y, ndarray z, ndarray f):

        cdef ndarray temp

        # if x array is single valued, expand x array and data along x axis to simplify interpolation
        if x.shape[0] == 1:

            # set x array to full real range
            self._x = array([-INFINITY, INFINITY], dtype=float64)

            # duplicate data to provide a pseudo range for interpolation coefficient calculation
            temp = f
            x = array([-1., +1.], dtype=float64)
            f = empty((2, f.shape[1], f.shape[2]), dtype=float64)
            f[:, :] = temp

        # if y array is single valued, expand y array and data along y axis to simplify interpolation
        if y.shape[0] == 1:

            # set y array to full real range
            self._y = array([-INFINITY, INFINITY], dtype=float64)

            # duplicate data to provide a pseudo range for interpolation coefficient calculation
            temp = f
            y = array([-1., +1.], dtype=float64)
            f = empty((f.shape[0], 2, f.shape[2]), dtype=float64)
            f[:, :] = temp

        # if z array is single valued, expand z array and data along z axis to simplify interpolation
        if z.shape[0] == 1:

            # set z array to full real range
            self._z = array([-INFINITY, INFINITY], dtype=float64)

            # duplicate data to provide a pseudo range for interpolation coefficient calculation
            temp = f
            z = array([-1., +1.], dtype=float64)
            f = empty((f.shape[0], f.shape[1], 2), dtype=float64)
            f[:, :] = temp

        # create memory views
        self._wx = x
        self._wy = y
        self._wz = z
        self._wf = f

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double _evaluate(self, double px, double py, double pz, int ix, int iy, int iz) except? -1e999:
        """
        Evaluate the interpolating function which is valid in the area given
        by 'ix', 'iy' and 'iz' at any position ('px', 'py', 'pz').

        :param double px, double py, double pz: coordinates
        :param int ix, int iy, int iz: indices of the area of interest
        :return: the interpolated value
        """

        cdef:
            double[::1] x, y, z
            double[:,:,::1] f
            double a0, a1, b0, b1, c0, c1

        x = self._wx
        y = self._wy
        z = self._wz
        f = self._wf

        # interpolate along y
        a0 = lerp(y[iy], y[iy+1], f[ix, iy, iz],     f[ix, iy+1, iz],     py)
        a1 = lerp(y[iy], y[iy+1], f[ix, iy, iz+1],   f[ix, iy+1, iz+1],   py)

        b0 = lerp(y[iy], y[iy+1], f[ix+1, iy, iz],   f[ix+1, iy+1, iz],   py)
        b1 = lerp(y[iy], y[iy+1], f[ix+1, iy, iz+1], f[ix+1, iy+1, iz+1], py)

        # interpolate along z
        c0 = lerp(z[iz], z[iz+1], a0, a1, pz)
        c1 = lerp(z[iz], z[iz+1], b0, b1, pz)

        # interpolate along x
        return lerp(x[ix], x[ix+1], c0, c1, px)

    cdef double _extrapol_linear(self, double px, double py, double pz, int ix, int iy, int iz, double nearest_px, double nearest_py, double nearest_pz) except? -1e999:
        """
        Extrapolate linearly the interpolation function valid on area given by
        'ix', 'iy' and 'iz' to position ('px', 'py', 'pz').

        :param double px, double py, double pz: coordinates
        :param int ix, int iy, int iz: indices of the area of interest
        :param double nearest_px, nearest_py, nearest_pz: the nearest position from
        ('px', 'py', 'pz') in the interpolation domain.
        :return: the extrapolated value
        """

        return self._evaluate(px, py, pz, ix, iy, iz)


cdef class Interpolate3DCubic(_Interpolate3DBase):
    """
    Interpolates 3D data using cubic interpolation.

    Inherits from Function3D, implements `__call__(x, y, z)`.

    Data and coordinates are first normalised to the range [0, 1] so as to
    prevent inaccuracy from float numbers. A local calculation based on
    finite differences is used. The splines coefficients are calculated
    on demand and are cached as they are calculated. Plus, no more than
    one polynomial is calculated at each evaluation. The first derivatives
    and the cross derivatives (xy, xz, yz and xyz) are imposed by the
    finite differences approximation, and the resulting function is C1
    (first derivatives are continuous).

    :param object x: An array-like object containing real values.
    :param object y: An array-like object containing real values.
    :param object z: An array-like object containing real values.
    :param object f: A 3D array-like object of sample values corresponding to the
      `x`, `y` and `z` array points.
    :param bint extrapolate: If True, the extrapolation of data is enabled
      outside the range of the data set. The default is False. A ValueError
      is raised if extrapolation is disabled and a point is requested
      outside the data set.
    :param object extrapolation_type: Sets the method of extrapolation.
      The options are: 'nearest' (default), 'linear', 'quadratic'.
    :param double extrapolation_range: The attribute can be set to limit
      the range beyond the data set bounds that extrapolation is permitted.
      The default range is set to infinity. Requesting data beyond the
      extrapolation range will result in a ValueError being raised.
    :param tolerate_single_value: If True, single-value arrays will be
      tolerated as inputs. If a single value is supplied, that value will
      be extrapolated over the entire real range. If False (default),
      supplying a single value will result in a ValueError being raised.

    .. code-block:: pycon

       >>> import numpy as np
       >>> from cherab.core.math import Interpolate3DCubic
       >>>
       >>> # implements x**3 + y**2 + z
       >>> drange = np.linspace(-2.5, 2.5, 100)
       >>> values = np.zeros((100, 100, 100))
       >>> for i in range(100):
       >>>     for j in range(100):
       >>>         for k in range(100):
       >>>             values[i, j, k] = drange[i]**3 + drange[j]**2 + drange[k]
       >>>
       >>> f3d = Interpolate3DCubic(drange, drange, drange, values)
       >>>
       >>> f3d(0, 0, 0)
       -1.7763e-14
       >>> f3d(-2, 1, 1.5)
       -5.50000927
       >>> f3d(-3, 1, 0)
       ValueError: The specified value (x=-3.0, y=1.0, z=0.0) is outside the range
       of the supplied data and/or extrapolation range: x bounds=(-2.5, 2.5),
       y bounds=(-2.5, 2.5), z bounds=(-2.5, 2.5)
    """

    def __init__(self, object x, object y, object z, object f, bint extrapolate=False, double extrapolation_range=INFINITY,
                 str extrapolation_type='nearest', bint tolerate_single_value=False):

        supported_extrapolations = ['nearest', 'linear', 'quadratic']

        extrapolation_type = extrapolation_type.lower()
        if extrapolation_type not in supported_extrapolations:
            raise ValueError("Unsupported extrapolation type: {}".format(extrapolation_type))

        super().__init__(x, y, z, f, extrapolate, extrapolation_type, extrapolation_range, tolerate_single_value)

    cdef object _build(self, ndarray x, ndarray y, ndarray z, ndarray f):

        cdef:
            ndarray temp
            ndarray wx, wy, wz, wf
            double[:,:,::1] f_mv
            int nx, ny, nz
            int i, j, k, i_mapped, j_mapped, k_mapped

        # if x array is single valued, expand x array and data along x axis to simplify interpolation
        if x.shape[0] == 1:

            # set x array to full real range
            self._x = array([-INFINITY, INFINITY], dtype=float64)

            # duplicate data to provide a pseudo range for interpolation coefficient calculation
            temp = f
            x = array([-1., +1.], dtype=float64)
            f = empty((2, f.shape[1], f.shape[2]), dtype=float64)
            f[:, :] = temp

        # if y array is single valued, expand y array and data along y axis to simplify interpolation
        if y.shape[0] == 1:

            # set y array to full real range
            self._y = array([-INFINITY, INFINITY], dtype=float64)

            # duplicate data to provide a pseudo range for interpolation coefficient calculation
            temp = f
            y = array([-1., +1.], dtype=float64)
            f = empty((f.shape[0], 2, f.shape[2]), dtype=float64)
            f[:, :] = temp

        # if z array is single valued, expand z array and data along z axis to simplify interpolation
        if z.shape[0] == 1:

            # set z array to full real range
            self._z = array([-INFINITY, INFINITY], dtype=float64)

            # duplicate data to provide a pseudo range for interpolation coefficient calculation
            temp = f
            z = array([-1., +1.], dtype=float64)
            f = empty((f.shape[0], f.shape[1], 2), dtype=float64)
            f[:, :] = temp

        nx = x.shape[0]
        ny = y.shape[0]
        nz = z.shape[0]

        # initialise the spline coefficient cache arrays
        self._k = empty((nx - 1, ny - 1, nz - 1, 64), dtype=float64)
        self._k[:,:,:,:] = NAN

        self._available = empty((nx - 1, ny - 1, nz - 1), dtype=int8)
        self._available[:,:,:] = False

        # normalise coordinate arrays
        self._ox = x.min()
        self._oy = y.min()
        self._oz = z.min()

        self._sx = 1 / (x.max() - x.min())
        self._sy = 1 / (y.max() - y.min())
        self._sz = 1 / (z.max() - z.min())

        x = (x - self._ox) * self._sx
        y = (y - self._oy) * self._sy
        z = (z - self._oz) * self._sz

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
        wz = concatenate(([z[0]], z, [z[-1]]))
        wf = empty((nx + 2, ny + 2, nz + 2), dtype=float64)

        # store memory views to widened data
        self._wx = wx
        self._wy = wy
        self._wz = wz
        self._wf = wf

        # populate expanded f array by duplicating the edges of the array
        f_mv = f
        for i in range(nx + 2):
            for j in range(ny + 2):
                for k in range(nz + 2):
                    i_mapped = min(max(0, i - 1), nx - 1)
                    j_mapped = min(max(0, j - 1), ny - 1)
                    k_mapped = min(max(0, k - 1), nz - 1)
                    self._wf[i, j, k] = f_mv[i_mapped, j_mapped, k_mapped]

        # calculate and cache higher powers of the dimension array data
        self._wx2 = wx * wx
        self._wx3 = wx * wx * wx

        self._wy2 = wy * wy
        self._wy3 = wy * wy * wy

        self._wz2 = wz * wz
        self._wz3 = wz * wz * wz

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double _evaluate(self, double px, double py, double pz, int ix, int iy, int iz) except? -1e999:
        """
        Evaluate the interpolating function which is valid in the area given
        by 'ix', 'iy' and 'iz' at any position ('px', 'py', 'pz').

        :param double px, double py, double pz: coordinates
        :param int ix, int iy, int iz: indices of the area of interest
        :return: the interpolated value
        """

        cdef:
            int u, v, w, l, i, j, k
            double delta_x, delta_y, delta_z, px2, py2, pz2, px3, py3, pz3
            double cv_buffer[64]
            double cm_buffer[64][64]
            double[::1] cv, coeffs
            double[:,::1] cm
            double s

        # If the concerned polynomial has not yet been calculated:
        if not self._available[ix, iy, iz]:

            # create memory views to constraint vector and matrix buffers
            cv = cv_buffer
            cm = cm_buffer

            # Fill the constraints matrix
            l = 0
            for u in range(ix+1, ix+3):
                for v in range(iy+1, iy+3):
                    for w in range(iz+1, iz+3):

                        delta_x = self._wx[u+1] - self._wx[u-1]
                        delta_y = self._wy[v+1] - self._wy[v-1]
                        delta_z = self._wz[w+1] - self._wz[w-1]

                        # knot values
                        self._constraints3d(cm[l + 0, :], u, v, w, False, False, False)
                        cv[l] = self._wf[u, v, w]

                        # derivatives along x, y, z
                        self._constraints3d(cm[l + 1, :], u, v, w, True, False, False)
                        self._constraints3d(cm[l + 2, :], u, v, w, False ,True , False)
                        self._constraints3d(cm[l + 3, :], u, v, w, False, False, True)

                        cv[l + 1] = (self._wf[u+1, v, w] - self._wf[u-1, v, w])/delta_x
                        cv[l + 2] = (self._wf[u, v+1, w] - self._wf[u, v-1, w])/delta_y
                        cv[l + 3] = (self._wf[u, v, w+1] - self._wf[u, v, w-1])/delta_z

                        # cross derivatives xy, xz, yz
                        self._constraints3d(cm[l + 4, :], u, v, w, True, True, False)
                        self._constraints3d(cm[l + 5, :], u, v, w, True, False, True)
                        self._constraints3d(cm[l + 6, :], u, v, w, False, True, True)

                        cv[l + 4] = (self._wf[u+1, v+1, w] - self._wf[u+1, v-1, w] - self._wf[u-1, v+1, w] + self._wf[u-1, v-1, w])/(delta_x*delta_y)
                        cv[l + 5] = (self._wf[u+1, v, w+1] - self._wf[u+1, v, w-1] - self._wf[u-1, v, w+1] + self._wf[u-1, v, w-1])/(delta_x*delta_z)
                        cv[l + 6] = (self._wf[u, v+1, w+1] - self._wf[u, v-1, w+1] - self._wf[u, v+1, w-1] + self._wf[u, v-1, w-1])/(delta_y*delta_z)

                        # cross derivative xyz
                        self._constraints3d(cm[l + 7, :], u, v, w, True, True, True)
                        cv[l + 7] = (self._wf[u+1, v+1, w+1] - self._wf[u+1, v+1, w-1] - self._wf[u+1, v-1, w+1] + self._wf[u+1, v-1, w-1] - self._wf[u-1, v+1, w+1] + self._wf[u-1, v+1, w-1] + self._wf[u-1, v-1, w+1] - self._wf[u-1, v-1, w-1])/(delta_x*delta_y*delta_z)

                        l += 8

            # Solve the linear system and fill the caching coefficients array
            coeffs = solve(cm, cv)
            self._k[ix, iy, iz, :] = coeffs

            # Denormalisation
            for i in range(4):
                for j in range(4):
                    for k in range(4):
                        s = self._sf * self._sx**i * self._sy**j * self._sz**k / (factorial(k) * factorial(j) * factorial(i))
                        coeffs[16*i + 4*j + k] = s * self._calc_polynomial_derivative(ix, iy, iz, -self._sx * self._ox, -self._sy * self._oy, -self._sz * self._oz, i, j, k)
            coeffs[0] = coeffs[0] + self._of

            # populate coefficients and set cell as calculated
            self._k[ix, iy, iz, :] = coeffs
            self._available[ix, iy, iz] = True

        px2 = px*px
        px3 = px2*px
        py2 = py*py
        py3 = py2*py
        pz2 = pz*pz
        pz3 = pz2*pz

        return         (self._k[ix, iy, iz,  0] + self._k[ix, iy, iz,  1]*pz + self._k[ix, iy, iz,  2]*pz2 + self._k[ix, iy, iz,  3]*pz3) + \
                   py *(self._k[ix, iy, iz,  4] + self._k[ix, iy, iz,  5]*pz + self._k[ix, iy, iz,  6]*pz2 + self._k[ix, iy, iz,  7]*pz3) + \
                   py2*(self._k[ix, iy, iz,  8] + self._k[ix, iy, iz,  9]*pz + self._k[ix, iy, iz, 10]*pz2 + self._k[ix, iy, iz, 11]*pz3) + \
                   py3*(self._k[ix, iy, iz, 12] + self._k[ix, iy, iz, 13]*pz + self._k[ix, iy, iz, 14]*pz2 + self._k[ix, iy, iz, 15]*pz3) \
               + px*( \
                       (self._k[ix, iy, iz, 16] + self._k[ix, iy, iz, 17]*pz + self._k[ix, iy, iz, 18]*pz2 + self._k[ix, iy, iz, 19]*pz3) + \
                   py *(self._k[ix, iy, iz, 20] + self._k[ix, iy, iz, 21]*pz + self._k[ix, iy, iz, 22]*pz2 + self._k[ix, iy, iz, 23]*pz3) + \
                   py2*(self._k[ix, iy, iz, 24] + self._k[ix, iy, iz, 25]*pz + self._k[ix, iy, iz, 26]*pz2 + self._k[ix, iy, iz, 27]*pz3) + \
                   py3*(self._k[ix, iy, iz, 28] + self._k[ix, iy, iz, 29]*pz + self._k[ix, iy, iz, 30]*pz2 + self._k[ix, iy, iz, 31]*pz3) \
               ) \
               + px2*( \
                       (self._k[ix, iy, iz, 32] + self._k[ix, iy, iz, 33]*pz + self._k[ix, iy, iz, 34]*pz2 + self._k[ix, iy, iz, 35]*pz3) + \
                   py *(self._k[ix, iy, iz, 36] + self._k[ix, iy, iz, 37]*pz + self._k[ix, iy, iz, 38]*pz2 + self._k[ix, iy, iz, 39]*pz3) + \
                   py2*(self._k[ix, iy, iz, 40] + self._k[ix, iy, iz, 41]*pz + self._k[ix, iy, iz, 42]*pz2 + self._k[ix, iy, iz, 43]*pz3) + \
                   py3*(self._k[ix, iy, iz, 44] + self._k[ix, iy, iz, 45]*pz + self._k[ix, iy, iz, 46]*pz2 + self._k[ix, iy, iz, 47]*pz3) \
               ) \
               + px3*( \
                       (self._k[ix, iy, iz, 48] + self._k[ix, iy, iz, 49]*pz + self._k[ix, iy, iz, 50]*pz2 + self._k[ix, iy, iz, 51]*pz3) + \
                   py *(self._k[ix, iy, iz, 52] + self._k[ix, iy, iz, 53]*pz + self._k[ix, iy, iz, 54]*pz2 + self._k[ix, iy, iz, 55]*pz3) + \
                   py2*(self._k[ix, iy, iz, 56] + self._k[ix, iy, iz, 57]*pz + self._k[ix, iy, iz, 58]*pz2 + self._k[ix, iy, iz, 59]*pz3) + \
                   py3*(self._k[ix, iy, iz, 60] + self._k[ix, iy, iz, 61]*pz + self._k[ix, iy, iz, 62]*pz2 + self._k[ix, iy, iz, 63]*pz3) \
               )

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef object _constraints3d(self, double[::1] c, int u, int v, int w, bint dx, bint dy, bint dz):
        """
        Return the coefficients of a given constraints and at a given point.

        This method is used to easily build the constraint matrix. It only
        handles constraints on P, dP/dx, dP/dy, dP/z, d2P/dxdy, d2P/dydz,
        d2P/dxdz and d3P/dxdydz (where P is the concerned polynomial).

        :param int u, int v, int w: indices of the point where the constraints apply.
        :param bint x_der, bint y_der, bint dz: set to True or False in order to chose
        what constraint is returned. For each axis, True means the constraint
        considered a derivative along this axis.
        For example:
        dx=False, dy=True, dz=False means the constraint returned is
        on the derivative along y (dP/dy).
        dx=True, dy=True, dz=False means the constraint returned is
        on the cross derivative along x and y (d2P/dxdy).
        :return: a memory view of a 1x64 array filled with the coefficients
        corresponding to the requested constraint.
        """

        cdef:
            double hx[4]
            double hy[4]
            double hz[4]
            int i, j, k

        if dx:
            hx[0] = 0.
            hx[1] = 1.
            hx[2] = 2.*self._wx[u]
            hx[3] = 3.*self._wx2[u]
        else:
            hx[0] = 1.
            hx[1] = self._wx[u]
            hx[2] = self._wx2[u]
            hx[3] = self._wx3[u]

        if dy:
            hy[0] = 0.
            hy[1] = 1.
            hy[2] = 2.*self._wy[v]
            hy[3] = 3.*self._wy2[v]
        else:
            hy[0] = 1.
            hy[1] = self._wy[v]
            hy[2] = self._wy2[v]
            hy[3] = self._wy3[v]

        if dz:
            hz[0] = 0.
            hz[1] = 1.
            hz[2] = 2.*self._wz[w]
            hz[3] = 3.*self._wz2[w]
        else:
            hz[0] = 1.
            hz[1] = self._wz[w]
            hz[2] = self._wz2[w]
            hz[3] = self._wz3[w]

        for i in range(4):
            for j in range(4):
                for k in range(4):
                    c[16*i + 4*j + k] = hx[i] * hy[j] * hz[k]

    cdef double _extrapol_linear(self, double px, double py, double pz, int ix, int iy, int iz, double rx, double ry, double rz) except? -1e999:
        """
        Extrapolate linearly the interpolation function valid on area given by
        'ix', 'iy' and 'iz' to position ('px', 'py', 'pz').

        :param double px, double py, double pz: coordinates
        :param int ix, int iy, int iz: indices of the area of interest
        :param double rx, ry, rz: the nearest position to ('px', 'py', 'pz') in the interpolation domain.
        :return: the extrapolated value
        """

        cdef double ex, ey, ez, result

        # calculate extrapolation distances from end of array
        ex = px - rx
        ey = py - ry
        ez = pz - rz

        result = self._evaluate(rx, ry, rz, ix, iy, iz)
        if ex != 0.:
            result += ex * self._calc_polynomial_derivative(ix, iy, iz, rx, ry, rz, 1, 0, 0)

        if ey != 0.:
            result += ey * self._calc_polynomial_derivative(ix, iy, iz, rx, ry, rz, 0, 1, 0)

        if ez != 0.:
            result += ez * self._calc_polynomial_derivative(ix, iy, iz, rx, ry, rz, 0, 0, 1)

        return result

    cdef double _extrapol_quadratic(self, double px, double py, double pz, int ix, int iy, int iz, double rx, double ry, double rz) except? -1e999:
        """
        Extrapolate quadratically the interpolation function valid on area given by
        'ix', 'iy' and 'iz' to position ('px', 'py', 'pz').

        :param double px, double py, double pz: coordinates
        :param int ix, int iy, int iz: indices of the area of interest
        :param double rx, ry, rz: the nearest position to ('px', 'py', 'pz') in the interpolation domain.
        :return: the extrapolated value
        """

        cdef double ex, ey, ez, result

        # calculate extrapolation distances from end of array
        ex = px - rx
        ey = py - ry
        ez = pz - rz

        result = self._evaluate(rx, ry, rz, ix, iy, iz)

        if ex != 0.:
            result += ex * self._calc_polynomial_derivative(ix, iy, iz, rx, ry, rz, 1, 0, 0)
            result += ex*ex*0.5 * self._calc_polynomial_derivative(ix, iy, iz, rx, ry, rz, 2, 0, 0)

            if ey != 0.:
                result += ex*ey * self._calc_polynomial_derivative(ix, iy, iz, rx, ry, rz, 1, 1, 0)

        if ey != 0.:
            result += ey * self._calc_polynomial_derivative(ix, iy, iz, rx, ry, rz, 0, 1, 0)
            result += ey*ey*0.5 * self._calc_polynomial_derivative(ix, iy, iz, rx, ry, rz, 0, 2, 0)

            if ez != 0.:
                result += ey*ez * self._calc_polynomial_derivative(ix, iy, iz, rx, ry, rz, 0, 1, 1)

        if ez != 0.:
            result += ez * self._calc_polynomial_derivative(ix, iy, iz, rx, ry, rz, 0, 0, 1)
            result += ez*ez*0.5 * self._calc_polynomial_derivative(ix, iy, iz, rx, ry, rz, 0, 0, 2)

            if ex != 0.:
                result += ez*ex * self._calc_polynomial_derivative(ix, iy, iz, rx, ry, rz, 1, 0, 1)

        return result

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double _calc_polynomial_derivative(self, int ix, int iy, int iz, double px, double py, double pz, int order_x, int order_y, int order_z):
        """
        Evaluate the derivatives of the polynomial valid in the area given by
        'ix', 'iy' and 'iz' at position ('px', 'py', 'pz'). The order of
        derivative along each axis is given by 'der_x', 'der_y' and 'der_z'.

        :param int ix, int iy, int iz: indices of the area of interest
        :param double px, double py, double pz: coordinates
        :param int der_x, int der_y, int order_z: orders of derivative along each axis
        :return: value evaluated from the derivated polynomial
        """

        cdef:
            double[::1] ax, ay, az
            double[:,:,:,::1] k

        k = self._k

        ax = derivatives_array(px, order_x)
        ay = derivatives_array(py, order_y)
        az = derivatives_array(pz, order_z)

        return   ax[0]*( \
                   ay[0]*(az[0]*k[ix, iy, iz,  0] + az[1]*k[ix, iy, iz,  1] + az[2]*k[ix, iy, iz,  2] + az[3]*k[ix, iy, iz,  3]) + \
                   ay[1]*(az[0]*k[ix, iy, iz,  4] + az[1]*k[ix, iy, iz,  5] + az[2]*k[ix, iy, iz,  6] + az[3]*k[ix, iy, iz,  7]) + \
                   ay[2]*(az[0]*k[ix, iy, iz,  8] + az[1]*k[ix, iy, iz,  9] + az[2]*k[ix, iy, iz, 10] + az[3]*k[ix, iy, iz, 11]) + \
                   ay[3]*(az[0]*k[ix, iy, iz, 12] + az[1]*k[ix, iy, iz, 13] + az[2]*k[ix, iy, iz, 14] + az[3]*k[ix, iy, iz, 15]) \
               ) \
               + ax[1]*( \
                   ay[0]*(az[0]*k[ix, iy, iz, 16] + az[1]*k[ix, iy, iz, 17] + az[2]*k[ix, iy, iz, 18] + az[3]*k[ix, iy, iz, 19]) + \
                   ay[1]*(az[0]*k[ix, iy, iz, 20] + az[1]*k[ix, iy, iz, 21] + az[2]*k[ix, iy, iz, 22] + az[3]*k[ix, iy, iz, 23]) + \
                   ay[2]*(az[0]*k[ix, iy, iz, 24] + az[1]*k[ix, iy, iz, 25] + az[2]*k[ix, iy, iz, 26] + az[3]*k[ix, iy, iz, 27]) + \
                   ay[3]*(az[0]*k[ix, iy, iz, 28] + az[1]*k[ix, iy, iz, 29] + az[2]*k[ix, iy, iz, 30] + az[3]*k[ix, iy, iz, 31]) \
               ) \
               + ax[2]*( \
                   ay[0]*(az[0]*k[ix, iy, iz, 32] + az[1]*k[ix, iy, iz, 33] + az[2]*k[ix, iy, iz, 34] + az[3]*k[ix, iy, iz, 35]) + \
                   ay[1]*(az[0]*k[ix, iy, iz, 36] + az[1]*k[ix, iy, iz, 37] + az[2]*k[ix, iy, iz, 38] + az[3]*k[ix, iy, iz, 39]) + \
                   ay[2]*(az[0]*k[ix, iy, iz, 40] + az[1]*k[ix, iy, iz, 41] + az[2]*k[ix, iy, iz, 42] + az[3]*k[ix, iy, iz, 43]) + \
                   ay[3]*(az[0]*k[ix, iy, iz, 44] + az[1]*k[ix, iy, iz, 45] + az[2]*k[ix, iy, iz, 46] + az[3]*k[ix, iy, iz, 47]) \
               ) \
               + ax[3]*( \
                   ay[0]*(az[0]*k[ix, iy, iz, 48] + az[1]*k[ix, iy, iz, 49] + az[2]*k[ix, iy, iz, 50] + az[3]*k[ix, iy, iz, 51]) + \
                   ay[1]*(az[0]*k[ix, iy, iz, 52] + az[1]*k[ix, iy, iz, 53] + az[2]*k[ix, iy, iz, 54] + az[3]*k[ix, iy, iz, 55]) + \
                   ay[2]*(az[0]*k[ix, iy, iz, 56] + az[1]*k[ix, iy, iz, 57] + az[2]*k[ix, iy, iz, 58] + az[3]*k[ix, iy, iz, 59]) + \
                   ay[3]*(az[0]*k[ix, iy, iz, 60] + az[1]*k[ix, iy, iz, 61] + az[2]*k[ix, iy, iz, 62] + az[3]*k[ix, iy, iz, 63]) \
               )


