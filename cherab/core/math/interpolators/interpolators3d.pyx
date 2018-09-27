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
from numpy cimport ndarray, PyArray_ZEROS, PyArray_SimpleNew, NPY_FLOAT64, npy_intp, import_array
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

cdef class _Interpolate3DBase(Function3D):
    """
    Base class for 3D interpolators. Coordinates and data arrays are here
    sorted and transformed into numpy arrays.

    :param object x: An array-like object containing real values.
    :param object y: An array-like object containing real values.
    :param object z: An array-like object containing real values.
    :param object data: A 3D array-like object of sample values corresponding to the
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

    def __init__(self, object x, object y, object z, object data, bint extrapolate=False, str extrapolation_type='nearest',
                 double extrapolation_range=float('inf'), bint tolerate_single_value=False):

        cdef ndarray mask_x, mask_y, mask_z

        # check the shapes of data and coordinates are consistent
        if shape(data) != tuple(list(shape(x))+list(shape(y))+list(shape(z))):
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

        # copies the arguments converted into double arrays and sort x, y and z
        mask_x = argsort(x)
        mask_y = argsort(y)
        mask_z = argsort(z)
        self.x_np = array(x, dtype=float64)[mask_x]
        self.y_np = array(y, dtype=float64)[mask_y]
        self.z_np = array(z, dtype=float64)[mask_z]
        self.data_np = array(data, dtype=float64)[mask_x,:,:][:,mask_y,:][:,:,mask_z]

        self.x_domain_view = self.x_np
        self.y_domain_view = self.y_np
        self.z_domain_view = self.z_np
        self.top_index_x = len(x) - 1
        self.top_index_y = len(y) - 1
        self.top_index_z = len(z) - 1

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

        # Check for single value in z input
        if len(self.z_np) == 1:
            if tolerate_single_value:
                # single value tolerated, set constant
                self._set_constant_z()
            else:
                raise ValueError("There is only a single value in the z input. "
                    "Consider turning on the 'tolerate_single_value' argument.")

        # if z is not a single value, check for duplicate values
        else:
            if (self.z_np == self.z_np[arange(len(self.z_np))-1]).any():
                raise ValueError("The z coordinates array has a duplicate value.")

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double evaluate(self, double px, double py, double pz) except? -1e999:
        """
        Evaluate the interpolating function.

        :param double px, double py, double pz: coordinates
        :return: the interpolated value
        """

        cdef int i_x, i_y, i_z

        i_x = find_index(self.x_domain_view, self.top_index_x+1, px, self.extrapolation_range)
        i_y = find_index(self.y_domain_view, self.top_index_y+1, py, self.extrapolation_range)
        i_z = find_index(self.z_domain_view, self.top_index_z+1, pz, self.extrapolation_range)

        if 0 <= i_x <= self.top_index_x-1:
            if 0 <= i_y <= self.top_index_y-1:
                if 0 <= i_z <= self.top_index_z-1:
                    return self._evaluate(px, py, pz, i_x, i_y, i_z)
                elif i_z == -1:
                    return self._extrapolate(px, py, pz, i_x               , i_y               , 0                 , px                                  , py                                  , self.z_domain_view[0])
                elif i_z == self.top_index_z:
                    return self._extrapolate(px, py, pz, i_x               , i_y               , self.top_index_z-1, px                                  , py                                  , self.z_domain_view[self.top_index_z])
            elif i_y == -1:
                if 0 <= i_z <= self.top_index_z-1:
                    return self._extrapolate(px, py, pz, i_x               , 0                 , i_z               , px                                  , self.y_domain_view[0]               , pz)
                elif i_z == -1:
                    return self._extrapolate(px, py, pz, i_x               , 0                 , 0                 , px                                  , self.y_domain_view[0]               , self.z_domain_view[0])
                elif i_z == self.top_index_z:
                    return self._extrapolate(px, py, pz, i_x               , 0                 , self.top_index_z-1, px                                  , self.y_domain_view[0]               , self.z_domain_view[self.top_index_z])
            elif i_y == self.top_index_y:
                if 0 <= i_z <= self.top_index_z-1:
                    return self._extrapolate(px, py, pz, i_x               , self.top_index_y-1, i_z               , px                                  , self.y_domain_view[self.top_index_y], pz)
                elif i_z == -1:
                    return self._extrapolate(px, py, pz, i_x               , self.top_index_y-1, 0                 , px                                  , self.y_domain_view[self.top_index_y], self.z_domain_view[0])
                elif i_z == self.top_index_z:
                    return self._extrapolate(px, py, pz, i_x               , self.top_index_y-1, self.top_index_z-1, px                                  , self.y_domain_view[self.top_index_y], self.z_domain_view[self.top_index_z])

        elif i_x == -1:
            if 0 <= i_y <= self.top_index_y-1:
                if 0 <= i_z <= self.top_index_z-1:
                    return self._extrapolate(px, py, pz, 0                 , i_y               , i_z               , self.x_domain_view[0]               , py                                  , pz)
                elif i_z == -1:
                    return self._extrapolate(px, py, pz, 0                 , i_y               , 0                 , self.x_domain_view[0]               , py                                  , self.z_domain_view[0])
                elif i_z == self.top_index_z:
                    return self._extrapolate(px, py, pz, 0                 , i_y               , self.top_index_z-1, self.x_domain_view[0]               , py                                  , self.z_domain_view[self.top_index_z])
            elif i_y == -1:
                if 0 <= i_z <= self.top_index_z-1:
                    return self._extrapolate(px, py, pz, 0                 , 0                 , i_z               , self.x_domain_view[0]               , self.y_domain_view[0]               , pz)
                elif i_z == -1:
                    return self._extrapolate(px, py, pz, 0                 , 0                 , 0                 , self.x_domain_view[0]               , self.y_domain_view[0]               , self.z_domain_view[0])
                elif i_z == self.top_index_z:
                    return self._extrapolate(px, py, pz, 0                 , 0                 , self.top_index_z-1, self.x_domain_view[0]               , self.y_domain_view[0]               , self.z_domain_view[self.top_index_z])
            elif i_y == self.top_index_y:
                if 0 <= i_z <= self.top_index_z-1:
                    return self._extrapolate(px, py, pz, 0                 , self.top_index_y-1, i_z               , self.x_domain_view[0]               , self.y_domain_view[self.top_index_y], pz)
                elif i_z == -1:
                    return self._extrapolate(px, py, pz, 0                 , self.top_index_y-1, 0                 , self.x_domain_view[0]               , self.y_domain_view[self.top_index_y], self.z_domain_view[0])
                elif i_z == self.top_index_z:
                    return self._extrapolate(px, py, pz, 0                 , self.top_index_y-1, self.top_index_z-1, self.x_domain_view[0]               , self.y_domain_view[self.top_index_y], self.z_domain_view[self.top_index_z])

        elif i_x == self.top_index_x:
            if 0 <= i_y <= self.top_index_y-1:
                if 0 <= i_z <= self.top_index_z-1:
                    return self._extrapolate(px, py, pz, self.top_index_x-1, i_y               , i_z               , self.x_domain_view[self.top_index_x], py                                  , pz)
                elif i_z == -1:
                    return self._extrapolate(px, py, pz, self.top_index_x-1, i_y               , 0                 , self.x_domain_view[self.top_index_x], py                                  , self.z_domain_view[0])
                elif i_z == self.top_index_z:
                    return self._extrapolate(px, py, pz, self.top_index_x-1, i_y               , self.top_index_z-1, self.x_domain_view[self.top_index_x], py                                  , self.z_domain_view[self.top_index_z])
            elif i_y == -1:
                if 0 <= i_z <= self.top_index_z-1:
                    return self._extrapolate(px, py, pz, self.top_index_x-1, 0                 , i_z               , self.x_domain_view[self.top_index_x], self.y_domain_view[0]               , pz)
                elif i_z == -1:
                    return self._extrapolate(px, py, pz, self.top_index_x-1, 0                 , 0                 , self.x_domain_view[self.top_index_x], self.y_domain_view[0]               , self.z_domain_view[0])
                elif i_z == self.top_index_z:
                    return self._extrapolate(px, py, pz, self.top_index_x-1, 0                 , self.top_index_z-1, self.x_domain_view[self.top_index_x], self.y_domain_view[0]               , self.z_domain_view[self.top_index_z])
            elif i_y == self.top_index_y:
                if 0 <= i_z <= self.top_index_z-1:
                    return self._extrapolate(px, py, pz, self.top_index_x-1, self.top_index_y-1, i_z               , self.x_domain_view[self.top_index_x], self.y_domain_view[self.top_index_y], pz)
                elif i_z == -1:
                    return self._extrapolate(px, py, pz, self.top_index_x-1, self.top_index_y-1, 0                 , self.x_domain_view[self.top_index_x], self.y_domain_view[self.top_index_y], self.z_domain_view[0])
                elif i_z == self.top_index_z:
                    return self._extrapolate(px, py, pz, self.top_index_x-1, self.top_index_y-1, self.top_index_z-1, self.x_domain_view[self.top_index_x], self.y_domain_view[self.top_index_y], self.z_domain_view[self.top_index_z])

        # value is outside of permitted limits
        min_range_x = self.x_domain_view[0] - self.extrapolation_range
        max_range_x = self.x_domain_view[self.top_index_x] + self.extrapolation_range

        min_range_y = self.y_domain_view[0] - self.extrapolation_range
        max_range_y = self.y_domain_view[self.top_index_y] + self.extrapolation_range

        min_range_z = self.z_domain_view[0] - self.extrapolation_range
        max_range_z = self.z_domain_view[self.top_index_z] + self.extrapolation_range

        raise ValueError("The specified value (x={}, y={}, z={}) is outside the range of the supplied data and/or extrapolation range: "
                         "x bounds=({}, {}), y bounds=({}, {}), z bounds=({}, {})".format(px, py, pz, min_range_x, max_range_x, min_range_y, max_range_y, min_range_z, max_range_z))

    cdef double _evaluate(self, double px, double py, double pz, int i_x, int i_y, int i_z) except? -1e999:
        """
        Evaluate the interpolating function which is valid in the area given
        by 'i_x', 'i_y' and 'i_z' at any position ('px', 'py', 'pz').

        :param double px, double py, double pz: coordinates
        :param int i_x, int i_y, int i_z: indices of the area of interest
        :return: the interpolated value
        """
        raise NotImplementedError("This abstract method has not been implemented yet.")

    cdef double _extrapolate(self, double px, double py, double pz, int i_x, int i_y, int i_z, double nearest_px, double nearest_py, double nearest_pz) except? -1e999:
        """
        Extrapolate the interpolation function valid on area given by
        'i_x', 'i_y' and 'i_z' to position ('px', 'py', 'pz').

        :param double px, double py, double pz: coordinates
        :param int i_x, int i_y, int i_z: indices of the area of interest
        :param double nearest_px, nearest_py, nearest_pz: the nearest position from
        ('px', 'py', 'pz') in the interpolation domain.
        :return: the extrapolated value
        """

        if self.extrapolation_type == EXT_NEAREST:
            return self._evaluate(nearest_px, nearest_py, nearest_pz, i_x, i_y, i_z)
        elif self.extrapolation_type == EXT_LINEAR:
            return self._extrapol_linear(px, py, pz, i_x, i_y, i_z, nearest_px, nearest_py, nearest_pz)
        elif self.extrapolation_type == EXT_QUADRATIC:
            return self._extrapol_quadratic(px, py, pz, i_x, i_y, i_z, nearest_px, nearest_py, nearest_pz)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double _extrapol_linear(self, double px, double py, double pz, int i_x, int i_y, int i_z, double nearest_px, double nearest_py, double nearest_pz) except? -1e999:
        """
        Extrapolate linearly the interpolation function valid on area given by
        'i_x', 'i_y' and 'i_z' to position ('px', 'py', 'pz').

        :param double px, double py, double pz: coordinates
        :param int i_x, int i_y, int i_z: indices of the area of interest
        :param double nearest_px, nearest_py, nearest_pz: the nearest position from
        ('px', 'py', 'pz') in the interpolation domain.
        :return: the extrapolated value
        """
        raise NotImplementedError("There is no linear extrapolation available for this interpolation.")

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double _extrapol_quadratic(self, double px, double py, double pz, int i_x, int i_y, int i_z, double nearest_px, double nearest_py, double nearest_pz) except? -1e999:
        """
        Extrapolate quadratically the interpolation function valid on area given by
        'i_x', 'i_y' and 'i_z' to position ('px', 'py', 'pz').

        :param double px, double py, double pz: coordinates
        :param int i_x, int i_y, int i_z: indices of the area of interest
        :param double nearest_px, nearest_py, nearest_pz: the nearest position from
        ('px', 'py', 'pz') in the interpolation domain.
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
        self.data_np = empty((2, shape(data)[1], shape(data)[2]), dtype=float64)
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
        self.data_np = empty((shape(data)[0], 2, shape(data)[2]), dtype=float64)
        self.data_np[:,:] = data

    cdef void _set_constant_z(self):
        """
        Set the interpolation function constant on the z axis, and extend the
        domain to all the reals.
        """

        cdef ndarray data

        self.z_domain_view = array([-float('Inf'), +float('Inf')], dtype=float64)
        self.top_index_z = 1

        self.z_np = array([-1., +1.], dtype=float64)
        data = self.data_np
        self.data_np = empty((shape(data)[0], shape(data)[1], 2), dtype=float64)
        self.data_np[:,:,:] = data


cdef class Interpolate3DLinear(_Interpolate3DBase):
    """
    Interpolates 3D data using linear interpolation.

    :param object x: An array-like object containing real values.
    :param object y: An array-like object containing real values.
    :param object z: An array-like object containing real values.
    :param object data: A 3D array-like object of sample values corresponding to the
    `x`, `y` and `z` array points.
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

    def __init__(self, object x, object y, object z, object data, bint extrapolate=False, str extrapolation_type='nearest',
                 double extrapolation_range=float('inf'), bint tolerate_single_value=False):

        supported_extrapolations = ['nearest', 'linear']

        extrapolation_type = extrapolation_type.lower()
        if extrapolation_type not in supported_extrapolations:
            raise ValueError("Unsupported extrapolation type: {}".format(extrapolation_type))

        super().__init__(x, y, z, data, extrapolate, extrapolation_type, extrapolation_range, tolerate_single_value)

        # obtain memory views
        self.x_view = self.x_np
        self.y_view = self.y_np
        self.z_view = self.z_np
        self.data_view = self.data_np

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double _evaluate(self, double px, double py, double pz, int i_x, int i_y, int i_z) except? -1e999:
        """
        Evaluate the interpolating function which is valid in the area given
        by 'i_x', 'i_y' and 'i_z' at any position ('px', 'py', 'pz').

        :param double px, double py, double pz: coordinates
        :param int i_x, int i_y, int i_z: indices of the area of interest
        :return: the interpolated value
        """

        cdef:
            double interm_value_00, interm_value_01, interm_value_0
            double interm_value_10, interm_value_11, interm_value_1

        interm_value_00 = lerp(self.y_view[i_y], self.y_view[i_y+1], self.data_view[i_x, i_y, i_z], self.data_view[i_x, i_y+1, i_z], py)
        interm_value_01 = lerp(self.y_view[i_y], self.y_view[i_y+1], self.data_view[i_x, i_y, i_z+1], self.data_view[i_x, i_y+1, i_z+1], py)
        interm_value_10 = lerp(self.y_view[i_y], self.y_view[i_y+1], self.data_view[i_x+1, i_y, i_z], self.data_view[i_x+1, i_y+1, i_z], py)
        interm_value_11 = lerp(self.y_view[i_y], self.y_view[i_y+1], self.data_view[i_x+1, i_y, i_z+1], self.data_view[i_x+1, i_y+1, i_z+1], py)

        interm_value_0 = lerp(self.z_view[i_z], self.z_view[i_z+1], interm_value_00, interm_value_01, pz)
        interm_value_1 = lerp(self.z_view[i_z], self.z_view[i_z+1], interm_value_10, interm_value_11, pz)

        return lerp(self.x_view[i_x], self.x_view[i_x+1], interm_value_0, interm_value_1, px)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double _extrapol_linear(self, double px, double py, double pz, int i_x, int i_y, int i_z, double nearest_px, double nearest_py, double nearest_pz) except? -1e999:
        """
        Extrapolate linearly the interpolation function valid on area given by
        'i_x', 'i_y' and 'i_z' to position ('px', 'py', 'pz').

        :param double px, double py, double pz: coordinates
        :param int i_x, int i_y, int i_z: indices of the area of interest
        :param double nearest_px, nearest_py, nearest_pz: the nearest position from
        ('px', 'py', 'pz') in the interpolation domain.
        :return: the extrapolated value
        """

        return self._evaluate(px, py, pz, i_x, i_y, i_z)

cdef class Interpolate3DCubic(_Interpolate3DBase):
    """
    Interpolates 3D data using cubic interpolation.

    Data and coordinates are first normalised to the range [0, 1] so as to
    prevent inaccuracy from float numbers.
    A local calculation based on finite differences is used. The
    splines coefficients are calculated on demand and are cached as they
    are calculated. Plus, no more than one polynomial is calculated at
    each evaluation. The first derivatives and the cross
    derivatives (xy, xz, yz and xyz) are imposed by the finite differences
    approximation, and the resulting function is C1 (first derivatives are
    continuous).

    :param object x: An array-like object containing real values.
    :param object y: An array-like object containing real values.
    :param object z: An array-like object containing real values.
    :param object data: A 3D array-like object of sample values corresponding to the
    `x`, `y` and `z` array points.
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

    def __init__(self, object x, object y, object z, object data, bint extrapolate=False, double extrapolation_range=float('inf'),
                 str extrapolation_type='nearest', bint tolerate_single_value=False):

        cdef int i, j, k, i_narrowed, j_narrowed, k_narrowed

        supported_extrapolations = ['nearest', 'linear', 'quadratic']

        extrapolation_type = extrapolation_type.lower()
        if extrapolation_type not in supported_extrapolations:
            raise ValueError("Unsupported extrapolation type: {}".format(extrapolation_type))

        super().__init__(x, y, z, data, extrapolate, extrapolation_type, extrapolation_range, tolerate_single_value)

        # Initialise the caching array
        self.coeffs_view = empty((self.top_index_x, self.top_index_y, self.top_index_z, 64), dtype=float64)
        self.coeffs_view[:,:,:,::1] = float('NaN')
        self.calculated_view = empty((self.top_index_x, self.top_index_y, self.top_index_z), dtype=int8)
        self.calculated_view[:,:,:] = False

        # Normalise coordinates and data arrays
        self.x_delta_inv = 1 / (self.x_np.max() - self.x_np.min())
        self.x_min = self.x_np.min()
        self.x_np = (self.x_np - self.x_min) * self.x_delta_inv
        self.y_delta_inv = 1 / (self.y_np.max() - self.y_np.min())
        self.y_min = self.y_np.min()
        self.y_np = (self.y_np - self.y_min) * self.y_delta_inv
        self.z_delta_inv = 1 / (self.z_np.max() - self.z_np.min())
        self.z_min = self.z_np.min()
        self.z_np = (self.z_np - self.z_min) * self.z_delta_inv
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
        self.z_np = concatenate(([self.z_np[0]], self.z_np, [self.z_np[-1]]))

        self.data_view = empty((self.top_index_x+3, self.top_index_y+3, self.top_index_z+3), dtype=float64)

        for i in range(self.top_index_x+3):
            for j in range(self.top_index_y+3):
                for k in range(self.top_index_z+3):
                    i_narrowed = min(max(0, i-1), self.top_index_x)
                    j_narrowed = min(max(0, j-1), self.top_index_y)
                    k_narrowed = min(max(0, k-1), self.top_index_z)
                    self.data_view[i, j, k] = self.data_np[i_narrowed, j_narrowed, k_narrowed]

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

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double _evaluate(self, double px, double py, double pz, int i_x, int i_y, int i_z) except? -1e999:
        """
        Evaluate the interpolating function which is valid in the area given
        by 'i_x', 'i_y' and 'i_z' at any position ('px', 'py', 'pz').

        :param double px, double py, double pz: coordinates
        :param int i_x, int i_y, int i_z: indices of the area of interest
        :return: the interpolated value
        """

        cdef:
            int u, v, w, l, i, j, k
            double delta_x, delta_y, delta_z, px2, py2, pz2, px3, py3, pz3
            npy_intp cv_size
            npy_intp cm_size[2]
            double[::1] cv_view, coeffs_view
            double[:, ::1] cm_view

        # If the concerned polynomial has not yet been calculated:
        if not self.calculated_view[i_x, i_y, i_z]:

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
            for u in range(i_x+1, i_x+3):
                for v in range(i_y+1, i_y+3):
                    for w in range(i_z+1, i_z+3):

                        # knot values

                        cm_view[l, :] = self._constraints3d(u, v, w, False, False, False)
                        cv_view[l] = self.data_view[u, v, w]
                        l = l+1

                        # derivatives along x, y, z

                        cm_view[l, :] = self._constraints3d(u, v, w, True, False, False)
                        delta_x = self.x_view[u+1] - self.x_view[u-1]
                        cv_view[l] = (self.data_view[u+1, v, w] - self.data_view[u-1, v, w])/delta_x
                        l = l+1

                        cm_view[l, :] = self._constraints3d(u, v, w, False ,True , False)
                        delta_y = self.y_view[v+1] - self.y_view[v-1]
                        cv_view[l] = (self.data_view[u, v+1, w] - self.data_view[u, v-1, w])/delta_y
                        l = l+1

                        cm_view[l, :] = self._constraints3d(u, v, w, False, False, True)
                        delta_z = self.z_view[w+1] - self.z_view[w-1]
                        cv_view[l] = (self.data_view[u, v, w+1] - self.data_view[u, v, w-1])/delta_z
                        l = l+1

                        # cross derivatives xy, xz, yz

                        cm_view[l, :] = self._constraints3d(u, v, w, True, True, False)
                        cv_view[l] = (self.data_view[u+1, v+1, w] - self.data_view[u+1, v-1, w] - self.data_view[u-1, v+1, w] + self.data_view[u-1, v-1, w])/(delta_x*delta_y)
                        l = l+1

                        cm_view[l, :] = self._constraints3d(u, v, w, True, False, True)
                        cv_view[l] = (self.data_view[u+1, v, w+1] - self.data_view[u+1, v, w-1] - self.data_view[u-1, v, w+1] + self.data_view[u-1, v, w-1])/(delta_x*delta_z)
                        l = l+1

                        cm_view[l, :] = self._constraints3d(u, v, w, False, True, True)
                        cv_view[l] = (self.data_view[u, v+1, w+1] - self.data_view[u, v-1, w+1] - self.data_view[u, v+1, w-1] + self.data_view[u, v-1, w-1])/(delta_y*delta_z)
                        l = l+1

                        # cross derivative xyz

                        cm_view[l, :] = self._constraints3d(u, v, w, True, True, True)
                        cv_view[l] = (self.data_view[u+1, v+1, w+1] - self.data_view[u+1, v+1, w-1] - self.data_view[u+1, v-1, w+1] + self.data_view[u+1, v-1, w-1] - self.data_view[u-1, v+1, w+1] + self.data_view[u-1, v+1, w-1] + self.data_view[u-1, v-1, w+1] - self.data_view[u-1, v-1, w-1])/(delta_x*delta_y*delta_z)
                        l = l+1

            # Solve the linear system and fill the caching coefficients array
            coeffs_view = solve(cm_view, cv_view)
            self.coeffs_view[i_x, i_y, i_z, :] = coeffs_view

            # Denormalisation
            for i in range(4):
                for j in range(4):
                    for k in range(4):
                        coeffs_view[16 * i + 4 * j + k] = self.data_delta * self.x_delta_inv ** i * self.y_delta_inv ** j * self.z_delta_inv ** k / (factorial(k) * factorial(j) * factorial(i)) \
                                                          * self._evaluate_polynomial_derivative(i_x, i_y, i_z, -self.x_delta_inv * self.x_min, -self.y_delta_inv * self.y_min, -self.z_delta_inv * self.z_min, i, j, k)
            coeffs_view[0] = coeffs_view[0] + self.data_min
            self.coeffs_view[i_x, i_y, i_z, :] = coeffs_view

            self.calculated_view[i_x, i_y, i_z] = True

        px2 = px*px
        px3 = px2*px
        py2 = py*py
        py3 = py2*py
        pz2 = pz*pz
        pz3 = pz2*pz

        return         (self.coeffs_view[i_x, i_y, i_z,  0] + self.coeffs_view[i_x, i_y, i_z,  1]*pz + self.coeffs_view[i_x, i_y, i_z,  2]*pz2 + self.coeffs_view[i_x, i_y, i_z,  3]*pz3) + \
                   py *(self.coeffs_view[i_x, i_y, i_z,  4] + self.coeffs_view[i_x, i_y, i_z,  5]*pz + self.coeffs_view[i_x, i_y, i_z,  6]*pz2 + self.coeffs_view[i_x, i_y, i_z,  7]*pz3) + \
                   py2*(self.coeffs_view[i_x, i_y, i_z,  8] + self.coeffs_view[i_x, i_y, i_z,  9]*pz + self.coeffs_view[i_x, i_y, i_z, 10]*pz2 + self.coeffs_view[i_x, i_y, i_z, 11]*pz3) + \
                   py3*(self.coeffs_view[i_x, i_y, i_z, 12] + self.coeffs_view[i_x, i_y, i_z, 13]*pz + self.coeffs_view[i_x, i_y, i_z, 14]*pz2 + self.coeffs_view[i_x, i_y, i_z, 15]*pz3) \
               + px*( \
                       (self.coeffs_view[i_x, i_y, i_z, 16] + self.coeffs_view[i_x, i_y, i_z, 17]*pz + self.coeffs_view[i_x, i_y, i_z, 18]*pz2 + self.coeffs_view[i_x, i_y, i_z, 19]*pz3) + \
                   py *(self.coeffs_view[i_x, i_y, i_z, 20] + self.coeffs_view[i_x, i_y, i_z, 21]*pz + self.coeffs_view[i_x, i_y, i_z, 22]*pz2 + self.coeffs_view[i_x, i_y, i_z, 23]*pz3) + \
                   py2*(self.coeffs_view[i_x, i_y, i_z, 24] + self.coeffs_view[i_x, i_y, i_z, 25]*pz + self.coeffs_view[i_x, i_y, i_z, 26]*pz2 + self.coeffs_view[i_x, i_y, i_z, 27]*pz3) + \
                   py3*(self.coeffs_view[i_x, i_y, i_z, 28] + self.coeffs_view[i_x, i_y, i_z, 29]*pz + self.coeffs_view[i_x, i_y, i_z, 30]*pz2 + self.coeffs_view[i_x, i_y, i_z, 31]*pz3) \
               ) \
               + px2*( \
                       (self.coeffs_view[i_x, i_y, i_z, 32] + self.coeffs_view[i_x, i_y, i_z, 33]*pz + self.coeffs_view[i_x, i_y, i_z, 34]*pz2 + self.coeffs_view[i_x, i_y, i_z, 35]*pz3) + \
                   py *(self.coeffs_view[i_x, i_y, i_z, 36] + self.coeffs_view[i_x, i_y, i_z, 37]*pz + self.coeffs_view[i_x, i_y, i_z, 38]*pz2 + self.coeffs_view[i_x, i_y, i_z, 39]*pz3) + \
                   py2*(self.coeffs_view[i_x, i_y, i_z, 40] + self.coeffs_view[i_x, i_y, i_z, 41]*pz + self.coeffs_view[i_x, i_y, i_z, 42]*pz2 + self.coeffs_view[i_x, i_y, i_z, 43]*pz3) + \
                   py3*(self.coeffs_view[i_x, i_y, i_z, 44] + self.coeffs_view[i_x, i_y, i_z, 45]*pz + self.coeffs_view[i_x, i_y, i_z, 46]*pz2 + self.coeffs_view[i_x, i_y, i_z, 47]*pz3) \
               ) \
               + px3*( \
                       (self.coeffs_view[i_x, i_y, i_z, 48] + self.coeffs_view[i_x, i_y, i_z, 49]*pz + self.coeffs_view[i_x, i_y, i_z, 50]*pz2 + self.coeffs_view[i_x, i_y, i_z, 51]*pz3) + \
                   py *(self.coeffs_view[i_x, i_y, i_z, 52] + self.coeffs_view[i_x, i_y, i_z, 53]*pz + self.coeffs_view[i_x, i_y, i_z, 54]*pz2 + self.coeffs_view[i_x, i_y, i_z, 55]*pz3) + \
                   py2*(self.coeffs_view[i_x, i_y, i_z, 56] + self.coeffs_view[i_x, i_y, i_z, 57]*pz + self.coeffs_view[i_x, i_y, i_z, 58]*pz2 + self.coeffs_view[i_x, i_y, i_z, 59]*pz3) + \
                   py3*(self.coeffs_view[i_x, i_y, i_z, 60] + self.coeffs_view[i_x, i_y, i_z, 61]*pz + self.coeffs_view[i_x, i_y, i_z, 62]*pz2 + self.coeffs_view[i_x, i_y, i_z, 63]*pz3) \
               )

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double _extrapol_linear(self, double px, double py, double pz, int i_x, int i_y, int i_z, double nearest_px, double nearest_py, double nearest_pz) except? -1e999:
        """
        Extrapolate linearly the interpolation function valid on area given by
        'i_x', 'i_y' and 'i_z' to position ('px', 'py', 'pz').

        :param double px, double py, double pz: coordinates
        :param int i_x, int i_y, int i_z: indices of the area of interest
        :param double nearest_px, nearest_py, nearest_pz: the nearest position from
        ('px', 'py', 'pz') in the interpolation domain.
        :return: the extrapolated value
        """

        cdef double delta_x, delta_y, delta_z, result

        delta_x = px - nearest_px
        delta_y = py - nearest_py
        delta_z = pz - nearest_pz

        result = self._evaluate(nearest_px, nearest_py, nearest_pz, i_x, i_y, i_z)

        if delta_x != 0.:
            result += delta_x * self._evaluate_polynomial_derivative(i_x, i_y, i_z, nearest_px, nearest_py, nearest_pz, 1, 0, 0)

        if delta_y != 0.:
            result += delta_y * self._evaluate_polynomial_derivative(i_x, i_y, i_z, nearest_px, nearest_py, nearest_pz, 0, 1, 0)

        if delta_z != 0.:
            result += delta_z * self._evaluate_polynomial_derivative(i_x, i_y, i_z, nearest_px, nearest_py, nearest_pz, 0, 0, 1)

        return result

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double _extrapol_quadratic(self, double px, double py, double pz, int i_x, int i_y, int i_z, double nearest_px, double nearest_py, double nearest_pz) except? -1e999:
        """
        Extrapolate quadratically the interpolation function valid on area given by
        'i_x', 'i_y' and 'i_z' to position ('px', 'py', 'pz').

        :param double px, double py, double pz: coordinates
        :param int i_x, int i_y, int i_z: indices of the area of interest
        :param double nearest_px, nearest_py, nearest_pz: the nearest position from
        ('px', 'py', 'pz') in the interpolation domain.
        :return: the extrapolated value
        """

        cdef double delta_x, delta_y, delta_z, result

        delta_x = px - nearest_px
        delta_y = py - nearest_py
        delta_z = pz - nearest_pz

        result = self._evaluate(nearest_px, nearest_py, nearest_pz, i_x, i_y, i_z)

        if delta_x != 0.:
            result += delta_x * self._evaluate_polynomial_derivative(i_x, i_y, i_z, nearest_px, nearest_py, nearest_pz, 1, 0, 0)

            result += delta_x*delta_x*0.5 * self._evaluate_polynomial_derivative(i_x, i_y, i_z, nearest_px, nearest_py, nearest_pz, 2, 0, 0)

            if delta_y != 0.:
                result += delta_x*delta_y * self._evaluate_polynomial_derivative(i_x, i_y, i_z, nearest_px, nearest_py, nearest_pz, 1, 1, 0)

        if delta_y != 0.:
            result += delta_y * self._evaluate_polynomial_derivative(i_x, i_y, i_z, nearest_px, nearest_py, nearest_pz, 0, 1, 0)

            result += delta_y*delta_y*0.5 * self._evaluate_polynomial_derivative(i_x, i_y, i_z, nearest_px, nearest_py, nearest_pz, 0, 2, 0)

            if delta_z != 0.:
                result += delta_y*delta_z * self._evaluate_polynomial_derivative(i_x, i_y, i_z, nearest_px, nearest_py, nearest_pz, 0, 1, 1)

        if delta_z != 0.:
            result += delta_z * self._evaluate_polynomial_derivative(i_x, i_y, i_z, nearest_px, nearest_py, nearest_pz, 0, 0, 1)

            result += delta_z*delta_z*0.5 * self._evaluate_polynomial_derivative(i_x, i_y, i_z, nearest_px, nearest_py, nearest_pz, 0, 0, 2)

            if delta_x != 0.:
                result += delta_z*delta_x * self._evaluate_polynomial_derivative(i_x, i_y, i_z, nearest_px, nearest_py, nearest_pz, 1, 0, 1)

        return result

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double _evaluate_polynomial_derivative(self, int i_x, int i_y, int i_z, double px, double py, double pz, int der_x, int der_y, int der_z):
        """
        Evaluate the derivatives of the polynomial valid in the area given by
        'i_x', 'i_y' and 'i_z' at position ('px', 'py', 'pz'). The order of
        derivative along each axis is given by 'der_x', 'der_y' and 'der_z'.

        :param int i_x, int i_y, int i_z: indices of the area of interest
        :param double px, double py, double pz: coordinates
        :param int der_x, int der_y, int der_z: orders of derivative along each axis
        :return: value evaluated from the derivated polynomial
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
