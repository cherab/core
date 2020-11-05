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

from libc.math cimport INFINITY

from cherab.core.math.function cimport autowrap_function1d, autowrap_function2d, autowrap_function3d
from raysect.core.math.cython cimport clamp


cdef class ClampOutput1D(Function1D):
    """
    Clamps the output of a Function1D to the range [min, max].

    :param object f: A Function1D instance or a callable python object that takes one argument.
    :param float min: the lower bound, default=-INFINITY.
    :param float max: the upper bound, default=+INFINITY.

    .. code-block:: pycon

       >>> import numpy as np
       >>> from cherab.core.math import ClampOutput1D
       >>>
       >>> clamped_func = ClampOutput1D(np.exp, min=0.5, max=3)
       >>> clamped_func(-3)
       0.5
       >>> clamped_func(0)
       1.0
       >>> clamped_func(3)
       3.0
    """

    def __init__(self, object f, double min=-INFINITY, double max=INFINITY):

        if min >= max:
            raise ValueError('The minimum clamp value must be less than the maximum.')

        self._f = autowrap_function1d(f)
        self._min = min
        self._max = max

    cdef double evaluate(self, double x) except? -1e999:
        return clamp(self._f.evaluate(x), self._min, self._max)


cdef class ClampOutput2D(Function2D):
    """
    Clamps the output of a Function2D to the range [min, max].

    :param object f: A Function2D instance or a callable python object that takes two arguments.
    :param float min: the lower bound, default=-INFINITY.
    :param float max: the upper bound, default=+INFINITY.

    .. code-block:: pycon

       >>> import numpy as np
       >>> from cherab.core.math import ClampOutput2D
       >>>
       >>> clamped_func = ClampOutput2D(np.arctan2, min=-1, max=1)
       >>> clamped_func(-1, -1)
       -1.0
       >>> clamped_func(1, -1)
       1.0
    """

    def __init__(self, object f, double min=-INFINITY, double max=INFINITY):

        if min >= max:
            raise ValueError('The minimum clamp value must be less than the maximum.')

        self._f = autowrap_function2d(f)
        self._min = min
        self._max = max

    cdef double evaluate(self, double x, double y) except? -1e999:
        return clamp(self._f.evaluate(x, y), self._min, self._max)


cdef class ClampOutput3D(Function3D):
    """
    Clamps the output of a Function3D to the range [min, max].

    :param object f: A Function3D instance or a callable python object that takes three arguments.
    :param float min: the lower bound, default=-INFINITY.
    :param float max: the upper bound, default=+INFINITY.

    .. code-block:: pycon

       >>> import numpy as np
       >>> from cherab.core.math import ClampOutput3D
       >>>
       >>> def my_func(x, y, z):
       >>>     return x**2 + y**2 + z**2
       >>>
       >>> clamped_func = ClampOutput3D(my_func, max=10)
       >>>
       >>> my_func(1, 2, 3)
       14
       >>> clamped_func(1, 2, 3)
       10
    """

    def __init__(self, object f, double min=-INFINITY, double max=INFINITY):

        if min >= max:
            raise ValueError('The minimum clamp value must be less than the maximum.')

        self._f = autowrap_function3d(f)
        self._min = min
        self._max = max

    cdef double evaluate(self, double x, double y, double z) except? -1e999:
        return clamp(self._f.evaluate(x, y, z), self._min, self._max)


cdef class ClampInput1D(Function1D):
    """
    Clamps the x input of a Function1D to the range [xmin, xmax].

    :param object f: A Function1D instance or a callable python object that takes one argument.
    :param float xmin: the lower bound, default=-INFINITY.
    :param float xmax: the upper bound, default=+INFINITY.

    .. code-block:: pycon

       >>> import numpy as np
       >>> from cherab.core.math import ClampInput1D
       >>>
       >>> clamped_func = ClampInput1D(np.exp, xmin=0)
       >>> clamped_func(1)
       2.718281828459045
       >>> clamped_func(-1)
       1.0
    """

    def __init__(self, object f, double xmin=-INFINITY, double xmax=INFINITY):

        if xmin >= xmax:
            raise ValueError('The minimum clamp value must be less than the maximum.')

        self._f = autowrap_function1d(f)
        self._xmin = xmin
        self._xmax = xmax

    cdef double evaluate(self, double x) except? -1e999:
        x = clamp(x, self._xmin, self._xmax)
        return self._f.evaluate(x)


cdef class ClampInput2D(Function2D):
    """
    Clamps the [x, y] inputs of a Function2D to the ranges [xmin, xmax], [ymin, ymax].

    :param object f: A Function2D instance or a callable python object that takes two arguments.
    :param float xmin: the x lower bound, default=-INFINITY.
    :param float xmax: the x upper bound, default=+INFINITY.
    :param float ymin: the y lower bound, default=-INFINITY.
    :param float ymax: the y upper bound, default=+INFINITY.

    .. code-block:: pycon

       >>> import numpy as np
       >>> from cherab.core.math import ClampInput2D
       >>>
       >>> def my_func(x, y):
       >>>     return x**2 + y**2
       >>>
       >>> my_func(1, 1)
       2
       >>> clamped_func = ClampInput2D(my_func, xmax=0, ymax=0)
       >>> clamped_func(1, 1)
       0.0
    """

    def __init__(self, object f, double xmin=-INFINITY, double xmax=INFINITY, double ymin=-INFINITY, double ymax=INFINITY):

        if xmin >= xmax:
            raise ValueError('The minimum x clamp value must be less than the maximum.')

        if ymin >= ymax:
            raise ValueError('The minimum y clamp value must be less than the maximum.')

        self._f = autowrap_function2d(f)
        self._xmin = xmin
        self._xmax = xmax
        self._ymin = ymin
        self._ymax = ymax

    cdef double evaluate(self, double x, double y) except? -1e999:
        x = clamp(x, self._xmin, self._xmax)
        y = clamp(y, self._ymin, self._ymax)
        return self._f.evaluate(x, y)


cdef class ClampInput3D(Function3D):
    """
    Clamps the [x, y, z] inputs of a Function3D to the ranges [xmin, xmax], [ymin, ymax], [zmin, zmax].

    :param object f: A Function3D instance or a callable python object that takes three arguments.
    :param float xmin: the x lower bound, default=-INFINITY.
    :param float xmax: the x upper bound, default=+INFINITY.
    :param float ymin: the y lower bound, default=-INFINITY.
    :param float ymax: the y upper bound, default=+INFINITY.
    :param float zmin: the z lower bound, default=-INFINITY.
    :param float zmax: the z upper bound, default=+INFINITY.

    .. code-block:: pycon

       >>> import numpy as np
       >>> from cherab.core.math import ClampInput3D
       >>>
       >>> def my_func(x, y, z):
       >>>     return x**2 + y**2 + z**2
       >>>
       >>> my_func(-1, -1, -1)
       3
       >>> clamped_func = ClampInput3D(my_func, xmin=0, ymin=0, zmin=0)
       >>> clamped_func(-1, -1, -1)
       0.0
    """

    def __init__(self, object f, double xmin=-INFINITY, double xmax=INFINITY, double ymin=-INFINITY, double ymax=INFINITY, double zmin=-INFINITY, double zmax=INFINITY):

        if xmin >= xmax:
            raise ValueError('The minimum x clamp value must be less than the maximum.')

        if ymin >= ymax:
            raise ValueError('The minimum y clamp value must be less than the maximum.')

        if zmin >= zmax:
            raise ValueError('The minimum z clamp value must be less than the maximum.')

        self._f = autowrap_function3d(f)
        self._xmin = xmin
        self._xmax = xmax
        self._ymin = ymin
        self._ymax = ymax
        self._zmin = zmin
        self._zmax = zmax

    cdef double evaluate(self, double x, double y, double z) except? -1e999:
        x = clamp(x, self._xmin, self._xmax)
        y = clamp(y, self._ymin, self._ymax)
        z = clamp(z, self._zmin, self._zmax)
        return self._f.evaluate(x, y, z)

