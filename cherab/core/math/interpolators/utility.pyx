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

cimport cython
from numpy cimport ndarray, PyArray_SimpleNew, NPY_FLOAT64, npy_intp, import_array

# required by numpy c-api
import_array()

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef int find_index(double[::1] x, double v, double padding=0.):
    """
    Locates the lower index or the range that contains the specified value.

    This function performs a fast bisection search to identify the index range
    (bin) that encloses the specified value. The lower index of the range is
    returned. This function expects the memory view of a monotonically
    increasing ndarray for x. The array type must be double and may not be
    empty.

    .. WARNING:: For speed, this function does not perform any type or bounds
       checking. Supplying malformed data may result in data corruption or a
       segmentation fault.

    :param double[::1] x_view: The memory view of an array containing
    monotonically increasing values.
    :param int size: The size of the x array
    :param double v: The value to search for.
    :param double padding: defines the range in which extrapolation is allowed.
    Optional, default is 0.
    :return: The lower index f the bin containing the search value.
    :rtype: int
    """

    cdef:
        int bottom_index
        int top_index
        int bisection_index

    top_index = x.shape[0] - 1

    # on array ends?
    if v == x[0]:
        return 0

    if v == x[top_index]:
        return top_index - 1

    # beyond extrapolation range?
    if v < x[0] - padding:

        # value is lower than the lowest value in the array and outside
        # extrapolation range
        return -2

    if v > x[top_index] + padding:

        # value is above the highest value in the array and outside
        # extrapolation range
        return top_index + 1

    # inside extrapolation region?
    if v < x[0]:

        # value is lower than the lowest value in the array but inside
        # extrapolation range
        return -1

    if v > x[top_index]:

        # value is above the highest value in the array but inside
        # extrapolation range
        return top_index

    # bisection search inside array range
    bottom_index = 0
    bisection_index = top_index / 2
    while (top_index - bottom_index) != 1:
        if v >= x[bisection_index]:
            bottom_index = bisection_index
        else:
            top_index = bisection_index
        bisection_index = (top_index + bottom_index) / 2

    return bottom_index


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double[::1] derivatives_array(double v, int deriv):
    """
    Return, in the form of a memory view, the array [1, v, v^2, v^3] derivated
    'deriv' times along v.
    """

    cdef:
        npy_intp size
        double[::1] result

    # create an empty (uninitialised) ndarray via numpy c-api
    size = 4
    result = PyArray_SimpleNew(1, &size, NPY_FLOAT64)

    if deriv == 0:
        result[0] = 1.
        result[1] = v
        result[2] = v*v
        result[3] = v*v*v
    elif deriv == 1:
        result[0] = 0.
        result[1] = 1.
        result[2] = 2.*v
        result[3] = 3.*v*v
    elif deriv == 2:
        result[0] = 0.
        result[1] = 0.
        result[2] = 2.
        result[3] = 6.*v
    elif deriv == 3:
        result[0] = 0.
        result[1] = 0.
        result[2] = 0.
        result[3] = 6.
    else:
        result[:] = 0.

    return result


cdef int factorial(int n):
    if n <= 0:
        return 1
    else:
        return n * factorial(n-1)