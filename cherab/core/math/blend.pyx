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

from cherab.core.math.function cimport autowrap_function1d, autowrap_function2d, autowrap_function3d
from raysect.core.math.cython cimport clamp


cdef class Blend1D(Function1D):
    """
    Performs a linear interpolation between two 1D functions controlled by a third.

    This function is a functional equivalent to the computer graphics Lerp
    function. The value returned when this function is evaluated is a linear
    interpolation between the sampled values of f1 and f2. The interpolation
    weighting is supplied by a third 'mask' function that must return a value
    in the range [0, 1].

    Mathematically the value returned by this function is as follows:

    .. math::
        r = (1 - f_m(x)) f_1(x) + f_m(x) f_2(x)

    The value of the mask function is clamped to the range [0, 1] if the
    sampled value exceeds the required range.

    :param Function1D f1: the first 1D function.
    :param Function1D f2: the second 1D function.
    :param Function1D mask: the masking 1D function.

    .. code-block:: pycon

       >>> from cherab.core.math import Blend1D
       >>> from raysect.core.math.function.function1d import PythonFunction1D
       >>>
       >>> def my_f1(x):
       >>>     return 0.5
       >>> f1 = PythonFunction1D(my_f1)
       >>>
       >>> def my_f2(x):
       >>>     return x**3
       >>> f2 = PythonFunction1D(my_f2)
       >>>
       >>> def my_fmask(x):
       >>>     if x <= 0:
       >>>         return 0
       >>>     elif 0< x <= 1:
       >>>         return x
       >>>     else:
       >>>         return 1
       >>> f_mask = PythonFunction1D(my_fmask)
       >>>
       >>> fb = Blend1D(f1, f2, f_mask)
       >>> fb(-3)
       0.5
       >>> fb(0.3)
       0.3581
       >>> fb(1.5)
       3.375
    """

    def __init__(self, object f1, object f2, object mask):
        self._f1 = autowrap_function1d(f1)
        self._f2 = autowrap_function1d(f2)
        self._mask = autowrap_function1d(mask)

    cdef double evaluate(self, double x) except? -1e999:
        cdef w = clamp(self._mask.evaluate(x), 0.0, 1.0)

        # only evaluate single function is at end of mask range
        if w == 0:
            return self._f1.evaluate(x)

        if w == 1:
            return self._f2.evaluate(x)

        # perform lerp
        cdef f1 = self._f1.evaluate(x)
        cdef f2 = self._f2.evaluate(x)
        return (1 - w) * f1 + w * f2


cdef class Blend2D(Function2D):
    """
    Performs a linear interpolation between two 2D functions controlled by a third.

    This function is a functional equivalent to the computer graphics Lerp
    function. The value returned when this function is evaluated is a linear
    interpolation between the sampled values of f1 and f2. The interpolation
    weighting is supplied by a third 'mask' function that must return a value
    in the range [0, 1].

    Mathematically the value returned by this function is as follows:

    .. math::
        r = (1 - f_m(x,y)) f_1(x,y) + f_m(x,y) f_2(x,y)

    The value of the mask function is clamped to the range [0, 1] if the
    sampled value exceeds the required range.

    :param Function2D f1: the first 2D function.
    :param Function2D f2: the second 2D function.
    :param Function2D mask: the masking 2D function.

    .. code-block:: pycon

       >>> from numpy import sqrt
       >>> from cherab.core.math import Blend2D
       >>> from raysect.core.math.function.function2d import PythonFunction2D
       >>>
       >>> def my_f1(x, y):
       >>>     return 0.5
       >>> f1 = PythonFunction2D(my_f1)
       >>>
       >>> def my_f2(x, y):
       >>>     return x**2 + y
       >>> f2 = PythonFunction2D(my_f2)
       >>>
       >>> def my_fmask(x, y):
       >>>    radius = sqrt(x**2 + y**2)
       >>>    if radius <= 1:
       >>>        return 0
       >>>    elif 1 < radius <= 2:
       >>>        return radius - 1
       >>>    else:
       >>>        return 1
       >>> f_mask = PythonFunction2D(my_fmask)
       >>>
       >>> fb = Blend2D(f1, f2, f_mask)
       >>> fb(0, 0)
       0.5
       >>> fb(1, 1)
       1.121320
       >>> fb(2, 2)
       6.0
    """

    def __init__(self, object f1, object f2, object mask):
        self._f1 = autowrap_function2d(f1)
        self._f2 = autowrap_function2d(f2)
        self._mask = autowrap_function2d(mask)

    cdef double evaluate(self, double x, double y) except? -1e999:
        cdef w = clamp(self._mask.evaluate(x, y), 0.0, 1.0)

        # only evaluate single function is at end of mask range
        if w == 0:
            return self._f1.evaluate(x, y)

        if w == 1:
            return self._f2.evaluate(x, y)

        # perform lerp
        cdef f1 = self._f1.evaluate(x, y)
        cdef f2 = self._f2.evaluate(x, y)
        return (1 - w) * f1 + w * f2


cdef class Blend3D(Function3D):
    """
    Performs a linear interpolation between two 3D functions controlled by a third.

    This function is a functional equivalent to the computer graphics Lerp
    function. The value returned when this function is evaluated is a linear
    interpolation between the sampled values of f1 and f2. The interpolation
    weighting is supplied by a third 'mask' function that must return a value
    in the range [0, 1].

    Mathematically the value returned by this function is as follows:

    .. math::
        r = (1 - f_m(x,y,z)) f_1(x,y,z) + f_m(x,y,z) f_2(x,y,z)

    The value of the mask function is clamped to the range [0, 1] if the
    sampled value exceeds the required range.

    :param Function3D f1: the first 3D function.
    :param Function3D f2: the second 3D function.
    :param Function3D mask: the masking 3D function.

    .. code-block:: pycon

       >>> from numpy import sqrt
       >>> from cherab.core.math import Blend3D
       >>> from raysect.core.math.function.function3d import PythonFunction3D
       >>>
       >>> def my_f1(x, y, z):
       >>>     return 0.5
       >>> f1 = PythonFunction3D(my_f1)
       >>>
       >>> def my_f2(x, y, z):
       >>>     return x**3 + y**2 + z
       >>> f2 = PythonFunction3D(my_f2)
       >>>
       >>> def my_fmask(x, y, z):
       >>>    radius = sqrt(x**2 + y**2 + z**2)
       >>>    if radius <= 1:
       >>>        return 0
       >>>    elif 1 < radius <= 2:
       >>>        return radius - 1
       >>>    else:
       >>>        return 1
       >>> f_mask = PythonFunction3D(my_fmask)
       >>>
       >>> fb = Blend3D(f1, f2, f_mask)
       >>> fb(0, 0, 0)
       0.5
       >>> fb(1, 1, 1)
       2.3301270189221928
       >>> fb(2, 2, 2)
       14.0
    """

    def __init__(self, object f1, object f2, object mask):
        self._f1 = autowrap_function3d(f1)
        self._f2 = autowrap_function3d(f2)
        self._mask = autowrap_function3d(mask)

    cdef double evaluate(self, double x, double y, double z) except? -1e999:
        cdef w = clamp(self._mask.evaluate(x, y, z), 0.0, 1.0)

        # only evaluate single function is at end of mask range
        if w == 0:
            return self._f1.evaluate(x, y, z)

        if w == 1:
            return self._f2.evaluate(x, y, z)

        # perform lerp
        cdef f1 = self._f1.evaluate(x, y, z)
        cdef f2 = self._f2.evaluate(x, y, z)
        return (1 - w) * f1 + w * f2