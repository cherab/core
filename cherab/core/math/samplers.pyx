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

from numpy import asarray, ascontiguousarray, empty, linspace
from cherab.core.math.function cimport Function1D, Function2D, Function3D, VectorFunction2D, VectorFunction3D
from cherab.core.math.function cimport autowrap_function1d, autowrap_function2d, autowrap_function3d, autowrap_vectorfunction2d, autowrap_vectorfunction3d
from raysect.core cimport Vector3D
cimport cython
cimport numpy as np

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
    
    .. code-block:: pycon

       >>> from cherab.core.math import sample1d
       >>>
       >>> def f1(x):
       >>>     return x**2
       >>>
       >>> x_pts, f_vals = sample1d(f1, (0, 3, 5))
       >>> x_pts
       array([0.  , 0.75, 1.5 , 2.25, 3.  ])
       >>> f_vals
       array([0.    , 0.5625, 2.25  , 5.0625, 9.    ])
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
cpdef np.ndarray sample1d_points(object function1d, object x_points):
    """
    Sample a 1D function at the specified points

    :param function1d: a Python function or Function1D object
    :param x_points: an array of points at which to sample the function
    :return: an array containing the sampled values

    .. code-block:: pycon

       >>> from cherab.core.math import sample1d_points
       >>>
       >>> def f1(x):
       >>>     return x**2
       >>>
       >>> f_vals = sample1d_points(f1, [1, 2, 3])
       >>> f_vals
       array([1., 4., 9.])
    """
    cdef:
        int i, nsamples
        Function1D f1d
        double[::1] x_view, v_view

    x_points = ascontiguousarray(x_points, dtype=float)

    f1d = autowrap_function1d(function1d)
    nsamples = len(x_points)

    v = empty(nsamples)

    x_view = x_points
    v_view = v

    for i in range(nsamples):
        v_view[i] = f1d.evaluate(x_view[i])

    return v


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple sample2d(object function2d, tuple x_range, tuple y_range):
    """
    Samples a 2D function over the specified range.

    :param function2d: a Python function or Function2D object
    :param x_range: a tuple defining the x sample range: (x_min, x_max, x_samples)
    :param y_range: a tuple defining the y sample range: (y_min, y_max, y_samples)
    :return: a tuple containing the sampled values: (x_points, y_points, function_samples)

    .. code-block:: pycon

       >>> from cherab.core.math import sample2d
       >>>
       >>> def f1(x, y):
       >>>     return x**2 + y
       >>>
       >>> x_pts, y_pts, f_vals = sample2d(f1, (1, 3, 5), (1, 3, 5))
       >>> x_pts
       array([1. , 1.5, 2. , 2.5, 3. ])
       >>> y_pts
       array([1. , 1.5, 2. , 2.5, 3. ])
       >>> f_vals
       array([[ 2.  ,  2.5 ,  3.  ,  3.5 ,  4.  ],
              [ 3.25,  3.75,  4.25,  4.75,  5.25],
              [ 5.  ,  5.5 ,  6.  ,  6.5 ,  7.  ],
              [ 7.25,  7.75,  8.25,  8.75,  9.25],
              [10.  , 10.5 , 11.  , 11.5 , 12.  ]])
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
cpdef np.ndarray sample2d_points(object function2d, object points):
    """
    Sample a 2D function at the specified points.

    This function is for sampling at an unstructured sequence of points.
    For sampling over a regular grid, consider sample2d or sample2d_grid
    instead.

    :param function2d: a Python function or Function2D object
    :param points: an Nx2 array of points at which to sample the function
    :return: a 1D array containing the sampled values at each point

    .. code-block:: pycon

       >>> from cherab.core.math import sample2d
       >>>
       >>> def f1(x, y):
       >>>     return x**2 + y
       >>>
       >>> f_vals = sample2d_points(f1, [[1, 1], [2, 2], [3, 3]])
       >>> f_vals
       array([ 2.,  6., 12.])
    """
    cdef:
        int i, j, nsamples
        Function2D f2d
        double[::1] x_view, y_view, v_view

    points = asarray(points)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points should be an Nx2 array of points.")

    f2d = autowrap_function2d(function2d)
    x = ascontiguousarray(points[:, 0], dtype=float)
    y = ascontiguousarray(points[:, 1], dtype=float)
    nsamples = points.shape[0]
    v = empty(nsamples)

    x_view = x
    y_view = y
    v_view = v

    for i in range(nsamples):
        v_view[i] = f2d.evaluate(x_view[i], y_view[i])

    return v


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray sample2d_grid(object function2d, object x, object y):
    """
    Sample a 2D function on a rectilinear grid

    Note that v[i, j] = f(x[i], y[j])

    :param function2d: a Python function or Function2D object
    :param x: the x coordinates of each column in the grid
    :param y: the y coordinates of each row in the grid
    :return v: a 2D array containing the sampled values at each grid point

    .. code-block:: pycon

       >>> from cherab.core.math import sample2d_grid
       >>>
       >>> def f1(x, y):
       >>>     return x**2 + y
       >>>
       >>> f_vals = sample2d_grid(f1, [1, 2], [1, 2, 3])
       >>> f_vals
       array([[2., 3., 4.],
              [5., 6., 7.]])
    """
    cdef:
        int i, j, x_samples, y_samples
        Function2D f2d
        double[::1] x_view, y_view
        double[:, ::1] v_view

    x = ascontiguousarray(x, dtype=float)
    y = ascontiguousarray(y, dtype=float)
    if x.ndim != 1:
        raise ValueError("x should be a 1D sequence of coordinates")
    if y.ndim != 1:
        raise ValueError("y should be a 1D sequence of coordinates")

    f2d = autowrap_function2d(function2d)

    x_samples = x.shape[0]
    y_samples = y.shape[0]
    v = empty((x_samples, y_samples))

    x_view = x
    y_view = y
    v_view = v

    for i in range(x_samples):
        for j in range(y_samples):
            v_view[i, j] = f2d.evaluate(x_view[i], y_view[j])

    return v


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

    .. code-block:: pycon

       >>> from cherab.core.math import sample3d
       >>>
       >>> def f1(x, y, z):
       >>>     return x**3 + y**2 + z
       >>>
       >>> x_pts, y_pts, z_pts, f_vals = sample3d(f1, (1, 3, 3), (1, 3, 3), (1, 3, 3))
       >>> x_pts
       array([1., 2., 3.])
       >>> f_vals
       array([[[ 3.,  4.,  5.],
               [ 6.,  7.,  8.],
               [11., 12., 13.]],       
              [[10., 11., 12.],
               [13., 14., 15.],
               [18., 19., 20.]],
              [[29., 30., 31.],
               [32., 33., 34.],
               [37., 38., 39.]]])
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


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray sample3d_points(object function3d, object points):
    """
    Sample a 3D function at the specified points.

    This function is for sampling at an unstructured sequence of points.
    For sampling over a regular grid, consider sample3d or sample3d_grid
    instead.

    :param function3d: a Python function or Function3D object
    :param points: an Nx3 array of points at which to sample the function
    :return: a 1D array containing the sampled values at each point
    
    .. code-block:: pycon

       >>> from cherab.core.math import sample3d_points
       >>>
       >>> def f1(x, y, z):
       >>>     return x**3 + y**2 + z
       >>>
       >>> f_vals = sample3d_points(f1, [[1,1,1], [2,2,2], [3,3,3]])
       >>> f_vals
       array([ 3., 14., 39.])
    """
    cdef:
        int i, j, nsamples
        Function3D f3d
        double[::1] x_view, y_view, z_view, v_view

    points = asarray(points)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points should be an Nx3 array of points.")

    f3d = autowrap_function3d(function3d)
    x = ascontiguousarray(points[:, 0], dtype=float)
    y = ascontiguousarray(points[:, 1], dtype=float)
    z = ascontiguousarray(points[:, 2], dtype=float)
    nsamples = points.shape[0]
    v = empty(nsamples)

    x_view = x
    y_view = y
    z_view = z
    v_view = v

    for i in range(nsamples):
        v_view[i] = f3d.evaluate(x_view[i], y_view[i], z_view[i])

    return v


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray sample3d_grid(object function3d, object x, object y, object z):
    """
    Sample a 3D function on a rectilinear grid

    Note that v[i, j, k] = f(x[i], y[j], z[k])

    :param function3d: a Python function or Function3D object
    :param x: the x coordinates of each column in the grid
    :param y: the y coordinates of each row in the grid
    :param z: the z coordinates of each plane in the grid
    :return v: a 3D array containing the sampled values at each grid point

    .. code-block:: pycon

       >>> from cherab.core.math import sample3d_grid
       >>>
       >>> def f1(x, y, z):
       >>>     return x**3 + y**2 + z
       >>>
       >>> f_vals = sample3d_grid(f1, (1, 3, 3), (1, 3, 3), (1, 3, 3))
       >>> f_vals
       array([[[ 3.,  5.,  5.],
               [11., 13., 13.],
               [11., 13., 13.]],
              [[29., 31., 31.],
               [37., 39., 39.],
               [37., 39., 39.]],
              [[29., 31., 31.],
               [37., 39., 39.],
               [37., 39., 39.]]])
    """
    cdef:
        int i, j, k, x_samples, y_samples, z_samples
        Function3D f3d
        double[::1] x_view, y_view, z_view
        double[:, :, ::1] v_view

    x = ascontiguousarray(x, dtype=float)
    y = ascontiguousarray(y, dtype=float)
    z = ascontiguousarray(z, dtype=float)
    if x.ndim != 1:
        raise ValueError("x should be a 1D sequence of coordinates")
    if y.ndim != 1:
        raise ValueError("y should be a 1D sequence of coordinates")
    if z.ndim != 1:
        raise ValueError("z should be a 1D sequence of coordinates")

    f3d = autowrap_function3d(function3d)

    x_samples = x.shape[0]
    y_samples = y.shape[0]
    z_samples = z.shape[0]
    v = empty((x_samples, y_samples, z_samples))

    x_view = x
    y_view = y
    z_view = z
    v_view = v

    for i in range(x_samples):
        for j in range(y_samples):
            for k in range(z_samples):
                v_view[i, j, k] = f3d.evaluate(x_view[i], y_view[j], z_view[k])

    return v


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

    .. code-block:: pycon

       >>> from raysect.core.math import Vector3D
       >>> from cherab.core.math import samplevector2d
       >>>
       >>> def my_func(x, y):
       >>>     return Vector3D(x, y, 0)
       >>>
       >>> x_pts, y_pts, f_vals = samplevector2d(my_func, (1, 3, 3), (1, 3, 3))
       >>> x_pts
       array([1., 2., 3.])
       >>> y_pts
       array([1., 2., 3.])
       >>> f_vals
       array([[[1., 1., 0.],
               [1., 2., 0.],
               [1., 3., 0.]],
              [[2., 1., 0.],
               [2., 2., 0.],
               [2., 3., 0.]],
              [[3., 1., 0.],
               [3., 2., 0.],
               [3., 3., 0.]]])
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


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray samplevector2d_points(object function2d, object points):
    """
    Sample a 2D vector function at the specified points.

    This function is for sampling at an unstructured sequence of points.
    For sampling over a regular grid, consider samplevector2d or
    samplevector2d_grid instead.

    :param function2d: a Python function or Function2D object
    :param points: an Nx2 array of points at which to sample the function
    :return: a Nx3 array containing the sampled values at each point

    .. code-block:: pycon

       >>> from raysect.core.math import Vector3D
       >>> from cherab.core.math import samplevector2d_points
       >>>
       >>> def my_func(x, y):
       >>>     return Vector3D(x, y, 0)
       >>>
       >>> f_vals = samplevector2d_points(my_func, [[1, 1], [2, 2], [3, 3]])
       >>> f_vals
       array([[1., 1., 0.],
              [2., 2., 0.],
              [3., 3., 0.]])
    """
    cdef:
        int i, j, nsamples
        VectorFunction2D f2d
        double[::1] x_view, y_view,
        double [:, ::1] v_view
        Vector3D vector

    points = asarray(points)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points should be an Nx2 array of points.")

    f2d = autowrap_vectorfunction2d(function2d)
    x = ascontiguousarray(points[:, 0], dtype=float)
    y = ascontiguousarray(points[:, 1], dtype=float)
    nsamples = points.shape[0]
    v = empty((nsamples, 3))

    x_view = x
    y_view = y
    v_view = v

    for i in range(nsamples):
        vector = f2d.evaluate(x_view[i], y_view[i])
        v_view[i, 0] = vector.x
        v_view[i, 1] = vector.y
        v_view[i, 2] = vector.z

    return v


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray samplevector2d_grid(object function2d, object x, object y):
    """
    Sample a 2D vector function on a rectilinear grid

    :param function2d: a Python function or Function2D object
    :param x: the x coordinates of each column in the grid
    :param y: the y coordinates of each row in the grid
    :return v: a 3D array containing the sampled values at each grid point

    Note that v[i, j] = f(x[i], y[j])

    .. code-block:: pycon

       >>> from raysect.core.math import Vector3D
       >>> from cherab.core.math import samplevector2d_grid
       >>>
       >>> def my_func(x, y):
       >>>     return Vector3D(x, y, 0)
       >>>
       >>> f_vals = samplevector2d_grid(my_func, [1, 2], [1, 2, 3])
       >>> f_vals
       array([[[1., 1., 0.],
               [1., 2., 0.],
               [1., 3., 0.]],
              [[2., 1., 0.],
               [2., 2., 0.],
               [2., 3., 0.]]])
    """
    cdef:
        int i, j, x_samples, y_samples
        VectorFunction2D f2d
        double[::1] x_view, y_view
        double[:, :, ::1] v_view
        Vector3D vector

    x = ascontiguousarray(x, dtype=float)
    y = ascontiguousarray(y, dtype=float)
    if x.ndim != 1:
        raise ValueError("x should be a 1D sequence of coordinates")
    if y.ndim != 1:
        raise ValueError("y should be a 1D sequence of coordinates")

    f2d = autowrap_vectorfunction2d(function2d)

    x_samples = x.shape[0]
    y_samples = y.shape[0]
    v = empty((x_samples, y_samples, 3))

    x_view = x
    y_view = y
    v_view = v

    for i in range(x_samples):
        for j in range(y_samples):
            vector = f2d.evaluate(x_view[i], y_view[j])
            v_view[i, j, 0] = vector.x
            v_view[i, j, 1] = vector.y
            v_view[i, j, 2] = vector.z

    return v


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

    .. code-block:: pycon

       >>> from raysect.core.math import Vector3D
       >>> from cherab.core.math import samplevector3d
       >>>
       >>> def my_func(x, y, z):
       >>>     return Vector3D(x, y, z)
       >>>
       >>> x_pts, y_pts, z_pts, f_vals = samplevector3d(my_func, (1, 2, 2), (1, 3, 3), (1, 3, 3))
       >>> x_pts
       array([1., 2.])
       >>> f_vals
       array([[[[1., 1., 1.],
                [1., 1., 2.],
                [1., 1., 3.]],
               [[1., 2., 1.],
                [1., 2., 2.],
                [1., 2., 3.]],
               [[1., 3., 1.],
                [1., 3., 2.],
                [1., 3., 3.]]],

              [[[2., 1., 1.],
                [2., 1., 2.],
                [2., 1., 3.]],
               [[2., 2., 1.],
                [2., 2., 2.],
                [2., 2., 3.]],
               [[2., 3., 1.],
                [2., 3., 2.],
                [2., 3., 3.]]]])
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


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray samplevector3d_points(object function3d, object points):
    """
    Sample a 3D vector function at the specified points.

    This function is for sampling at an unstructured sequence of points.
    For sampling over a regular grid, consider samplevector3d or
    samplevector3d_grid instead.

    :param function3d: a Python function or Function3D object
    :param points: an Nx3 array of points at which to sample the function
    :return: an Nx3 array containing the sampled values at each point

    .. code-block:: pycon

       >>> from raysect.core.math import Vector3D
       >>> from cherab.core.math import samplevector3d_points
       >>>
       >>> def my_func(x, y, z):
       >>>     return Vector3D(x, y, z)
       >>>
       >>> f_vals = samplevector3d_points(my_func, [[1,1,1], [2,2,2], [3,3,3]])
       >>> f_vals
       array([[1., 1., 1.],
              [2., 2., 2.],
              [3., 3., 3.]])
    """
    cdef:
        int i, j, nsamples
        VectorFunction3D f3d
        double[::1] x_view, y_view, z_view,
        double[:, ::1] v_view
        Vector3D vector

    points = asarray(points)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points should be an Nx3 array of points.")

    f3d = autowrap_vectorfunction3d(function3d)
    x = ascontiguousarray(points[:, 0], dtype=float)
    y = ascontiguousarray(points[:, 1], dtype=float)
    z = ascontiguousarray(points[:, 2], dtype=float)
    nsamples = points.shape[0]
    v = empty((nsamples, 3))

    x_view = x
    y_view = y
    z_view = z
    v_view = v

    for i in range(nsamples):
        vector = f3d.evaluate(x_view[i], y_view[i], z_view[i])
        v_view[i, 0] = vector.x
        v_view[i, 1] = vector.y
        v_view[i, 2] = vector.z

    return v


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray samplevector3d_grid(object function3d, object x, object y, object z):
    """
    Sample a 3D vector function on a rectilinear grid

    Note that v[i, j, k] = f(x[i], y[j], z[k])

    :param function3d: a Python function or Function3D object
    :param x: the x coordinates of each column in the grid
    :param y: the y coordinates of each row in the grid
    :param z: the z coordinates of each plane in the grid
    :return v: an NxMxkx3 array containing the sampled values at each grid point

    .. code-block:: pycon

       >>> from raysect.core.math import Vector3D
       >>> from cherab.core.math import samplevector3d_grid
       >>>
       >>> def my_func(x, y, z):
       >>>     return Vector3D(x, y, z)
       >>>
       >>> f_vals = samplevector3d_grid(my_func, [1,2], [1,2,3], [1,2,3])
       >>> f_vals
       array([[[[1., 1., 1.],
                [1., 1., 2.],
                [1., 1., 3.]],
               [[1., 2., 1.],
                [1., 2., 2.],
                [1., 2., 3.]],
               [[1., 3., 1.],
                [1., 3., 2.],
                [1., 3., 3.]]],

              [[[2., 1., 1.],
                [2., 1., 2.],
                [2., 1., 3.]],
               [[2., 2., 1.],
                [2., 2., 2.],
                [2., 2., 3.]],
               [[2., 3., 1.],
                [2., 3., 2.],
                [2., 3., 3.]]]])
    """
    cdef:
        int i, j, k, x_samples, y_samples, z_samples
        VectorFunction3D f3d
        double[::1] x_view, y_view, z_view
        double[:, :, :, ::1] v_view
        Vector3D vector

    x = ascontiguousarray(x, dtype=float)
    y = ascontiguousarray(y, dtype=float)
    z = ascontiguousarray(z, dtype=float)
    if x.ndim != 1:
        raise ValueError("x should be a 1D sequence of coordinates")
    if y.ndim != 1:
        raise ValueError("y should be a 1D sequence of coordinates")
    if z.ndim != 1:
        raise ValueError("z should be a 1D sequence of coordinates")

    f3d = autowrap_vectorfunction3d(function3d)

    x_samples = x.shape[0]
    y_samples = y.shape[0]
    z_samples = z.shape[0]
    v = empty((x_samples, y_samples, z_samples, 3))

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

    return v
