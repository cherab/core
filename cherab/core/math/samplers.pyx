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

from numpy import empty, linspace
from cherab.core.math.function cimport Function1D, Function2D, Function3D, VectorFunction2D, VectorFunction3D
from cherab.core.math.function cimport autowrap_function1d, autowrap_function2d, autowrap_function3d, autowrap_vectorfunction2d, autowrap_vectorfunction3d
from raysect.core cimport Vector3D
cimport cython

"""
This module provides a set of sampling functions for rapidly generating samples
of 1D, 2D and 3D functions.

These functions use C calls when sampling Function1D, Function2D and Function3D
objects and are therefore considerably faster than the equivalent Python code.
"""

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple sample1d(object function1d, tuple x_range):
    """
    Samples a 1D function over the specified range.

    :param function1d: a Python function or Function1D object
    :param x_range: a tuple defining the sample range: (min, max, samples)
    :return: a tuple containing the sampled values: (x_points, function_samples)
    """

    cdef:
        int i, samples
        Function1D f1d
        double[::1] x_view, v_view

    if len(x_range) != 3:
        raise ValueError("Range must be a tuple containing: (min range, max range, no. of samples).")

    if x_range[0] > x_range[1]:
        raise ValueError("Minimum range can not be greater than maximum range.")

    if x_range[2] < 1:
        raise ValueError("The number of samples must be >= 1.")

    f1d = autowrap_function1d(function1d)
    samples = x_range[2]

    x = linspace(x_range[0], x_range[1], samples)
    v = empty(samples)

    # obtain memoryviews for fast, direct memory access
    x_view = x
    v_view = v

    for i in range(samples):
        v_view[i] = f1d.evaluate(x_view[i])

    return x, v


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple sample2d(object function2d, tuple x_range, tuple y_range):
    """
    Samples a 2D function over the specified range.

    :param function2d: a Python function or Function2D object
    :param x_range: a tuple defining the x sample range: (x_min, x_max, x_samples)
    :param y_range: a tuple defining the y sample range: (y_min, y_max, y_samples)
    :return: a tuple containing the sampled values: (x_points, y_points, function_samples)
    """

    cdef:
        int i, j
        Function2D f2d
        int x_samples, y_samples
        double[::1] x_view, y_view
        double[:, ::1] v_view

    if len(x_range) != 3:
        raise ValueError("X range must be a tuple containing: (min range, max range, no. of samples).")

    if len(y_range) != 3:
        raise ValueError("Y range must be a tuple containing: (min range, max range, no. of samples).")

    if x_range[0] > x_range[1]:
        raise ValueError("Minimum x range can not be greater than maximum x range.")

    if y_range[0] > y_range[1]:
        raise ValueError("Minimum y range can not be greater than maximum y range.")

    if x_range[2] < 1:
        raise ValueError("The number of x samples must be >= 1.")

    if y_range[2] < 1:
        raise ValueError("The number of y samples must be >= 1.")

    f2d = autowrap_function2d(function2d)
    x_samples = x_range[2]
    y_samples = y_range[2]

    x = linspace(x_range[0], x_range[1], x_samples)
    y = linspace(y_range[0], y_range[1], y_samples)
    v = empty((x_samples, y_samples))

    # obtain memoryviews for fast, direct memory access
    x_view = x
    y_view = y
    v_view = v

    for i in range(x_samples):
        for j in range(y_samples):
            v_view[i, j] = f2d.evaluate(x_view[i], y_view[j])

    return x, y, v


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple sample3d(object function3d, tuple x_range, tuple y_range, tuple z_range):
    """
    Samples a 3D function over the specified range.

    :param function3d: a Python function or Function2D object
    :param x_range: a tuple defining the x sample range: (x_min, x_max, x_samples)
    :param y_range: a tuple defining the y sample range: (y_min, y_max, y_samples)
    :param z_range: a tuple defining the z sample range: (z_min, z_max, z_samples)
    :return: a tuple containing the sampled values: (x_points, y_points, z_points, function_samples)
    """

    cdef:
        int i, j, k
        Function3D f3d
        int x_samples, y_samples, z_samples
        double[::1] x_view, y_view, z_view
        double[:, :, ::1] v_view

    if len(x_range) != 3:
        raise ValueError("X range must be a tuple containing: (min range, max range, no. of samples).")

    if len(y_range) != 3:
        raise ValueError("Y range must be a tuple containing: (min range, max range, no. of samples).")

    if len(z_range) != 3:
        raise ValueError("Z range must be a tuple containing: (min range, max range, no. of samples).")

    if x_range[0] > x_range[1]:
        raise ValueError("Minimum x range can not be greater than maximum x range.")

    if y_range[0] > y_range[1]:
        raise ValueError("Minimum y range can not be greater than maximum y range.")

    if z_range[0] > z_range[1]:
        raise ValueError("Minimum z range can not be greater than maximum z range.")

    if x_range[2] < 1:
        raise ValueError("The number of x samples must be >= 1.")

    if y_range[2] < 1:
        raise ValueError("The number of y samples must be >= 1.")

    if z_range[2] < 1:
        raise ValueError("The number of z samples must be >= 1.")

    f3d = autowrap_function3d(function3d)
    x_samples = x_range[2]
    y_samples = y_range[2]
    z_samples = z_range[2]

    x = linspace(x_range[0], x_range[1], x_samples)
    y = linspace(y_range[0], y_range[1], y_samples)
    z = linspace(z_range[0], z_range[1], z_samples)
    v = empty((x_samples, y_samples, z_samples))

    # obtain memoryviews for fast, direct memory access
    x_view = x
    y_view = y
    z_view = z
    v_view = v

    for i in range(x_samples):
        for j in range(y_samples):
            for k in range(z_samples):
                v_view[i, j, k] = f3d.evaluate(x_view[i], y_view[j], z_view[k])

    return x, y, z, v


# todo: add test
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple samplevector2d(object function2d, tuple x_range, tuple y_range):
    """
    Samples a 2D vector function over the specified range.

    The function samples returns are an NxMx3 array where the last axis are the
    x, y, and z components of the vector respectively.

    :param function2d: a Python function or Function2D object
    :param x_range: a tuple defining the x sample range: (x_min, x_max, x_samples)
    :param y_range: a tuple defining the y sample range: (y_min, y_max, y_samples)
    :return: a tuple containing the sampled values: (x_points, y_points, function_samples)
    """

    cdef:
        int i, j
        VectorFunction2D f2d
        int x_samples, y_samples
        double[::1] x_view, y_view
        double[:, :, ::1] v_view
        Vector3D vector

    if len(x_range) != 3:
        raise ValueError("X range must be a tuple containing: (min range, max range, no. of samples).")

    if len(y_range) != 3:
        raise ValueError("Y range must be a tuple containing: (min range, max range, no. of samples).")

    if x_range[0] > x_range[1]:
        raise ValueError("Minimum x range can not be greater than maximum x range.")

    if y_range[0] > y_range[1]:
        raise ValueError("Minimum y range can not be greater than maximum y range.")

    if x_range[2] < 1:
        raise ValueError("The number of x samples must be >= 1.")

    if y_range[2] < 1:
        raise ValueError("The number of y samples must be >= 1.")

    f2d = autowrap_vectorfunction2d(function2d)
    x_samples = x_range[2]
    y_samples = y_range[2]

    x = linspace(x_range[0], x_range[1], x_samples)
    y = linspace(y_range[0], y_range[1], y_samples)
    v = empty((x_samples, y_samples, 3))

    # obtain memoryviews for fast, direct memory access
    x_view = x
    y_view = y
    v_view = v

    for i in range(x_samples):
        for j in range(y_samples):
            vector = f2d.evaluate(x_view[i], y_view[j])
            v_view[i, j, 0] = vector.x
            v_view[i, j, 1] = vector.y
            v_view[i, j, 2] = vector.z

    return x, y, v


# todo: add test
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple samplevector3d(object function3d, tuple x_range, tuple y_range, tuple z_range):
    """
    Samples a 3D vector function over the specified range.

    The function samples returns are an NxMxKx3 array where the last axis are the
    x, y, and z components of the vector respectively.

    :param function3d: a Python function or Function2D object
    :param x_range: a tuple defining the x sample range: (x_min, x_max, x_samples)
    :param y_range: a tuple defining the y sample range: (y_min, y_max, y_samples)
    :param z_range: a tuple defining the z sample range: (z_min, z_max, z_samples)
    :return: a tuple containing the sampled values: (x_points, y_points, z_points, function_samples)
    """

    cdef:
        int i, j, k
        VectorFunction3D f3d
        int x_samples, y_samples, z_samples
        double[::1] x_view, y_view, z_view
        double[:, :, :, ::1] v_view
        Vector3D vector

    if len(x_range) != 3:
        raise ValueError("X range must be a tuple containing: (min range, max range, no. of samples).")

    if len(y_range) != 3:
        raise ValueError("Y range must be a tuple containing: (min range, max range, no. of samples).")

    if len(z_range) != 3:
        raise ValueError("Z range must be a tuple containing: (min range, max range, no. of samples).")

    if x_range[0] > x_range[1]:
        raise ValueError("Minimum x range can not be greater than maximum x range.")

    if y_range[0] > y_range[1]:
        raise ValueError("Minimum y range can not be greater than maximum y range.")

    if z_range[0] > z_range[1]:
        raise ValueError("Minimum z range can not be greater than maximum z range.")

    if x_range[2] < 1:
        raise ValueError("The number of x samples must be >= 1.")

    if y_range[2] < 1:
        raise ValueError("The number of y samples must be >= 1.")

    if z_range[2] < 1:
        raise ValueError("The number of z samples must be >= 1.")

    f3d = autowrap_vectorfunction3d(function3d)
    x_samples = x_range[2]
    y_samples = y_range[2]
    z_samples = z_range[2]

    x = linspace(x_range[0], x_range[1], x_samples)
    y = linspace(y_range[0], y_range[1], y_samples)
    z = linspace(z_range[0], z_range[1], z_samples)
    v = empty((x_samples, y_samples, z_samples, 3))

    # obtain memoryviews for fast, direct memory access
    x_view = x
    y_view = y
    z_view = z
    v_view = v

    for i in range(x_samples):
        for j in range(y_samples):
            for k in range(z_samples):
                vector = f3d.evaluate(x_view[i], y_view[j], z_view[k])
                v_view[i, j, k, 0] = vector.x
                v_view[i, j, k, 1] = vector.y
                v_view[i, j, k, 2] = vector.z

    return x, y, z, v
