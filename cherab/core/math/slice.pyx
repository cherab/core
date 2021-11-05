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

from cherab.core.math.function cimport autowrap_function2d, autowrap_function3d


cdef class Slice2D(Function1D):
    """
    Exposes a slice of a Function2D as a Function1D.

    :param Function2D function: the function to be sliced.
    :param axis: the axis to be sliced. Must be ['x', 'y'] or [0, 1].
    :param float value: the axis value at which to return a slice

    .. code-block:: pycon

       >>> from cherab.core.math import Slice2D
       >>>
       >>> def f1(x, y):
       >>>     return x**2 + y
       >>>
       >>> f2 = Slice2D(f1, 'x', 1.5)
       >>>
       >>> f2(0)
       2.25
       >>> f2(1)
       3.25
    """

    def __init__(self, object function, object axis, double value):

        # convert string axis to numerical axis value
        if isinstance(axis, str):
            map = {'x': 0, 'y': 1}
            try:
                axis = map[axis.lower()]
            except KeyError:
                raise ValueError('The axis must be either the string \'x\' or \'y\', or the value 0 or 1.')

        # check numerical value
        if axis not in [0, 1]:
            raise ValueError('The axis must be either the string \'x\' or \'y\', or the value 0 or 1.')

        self.axis = axis
        self.value = value
        self._function = autowrap_function2d(function)

    cdef double evaluate(self, double x) except? -1e999:

        if self.axis == 0:
            return self._function.evaluate(self.value, x)
        else:
            return self._function.evaluate(x, self.value)


cdef class Slice3D(Function2D):
    """
    Exposes a slice of a Function3D as a Function2D.

    :param Function3D function: the function to be sliced.
    :param axis: the axis to be sliced. Must be ['x', 'y', 'z'] or [0, 1, 2].
    :param float value: the axis value at which to return a slice

    .. code-block:: pycon

       >>> from cherab.core.math import Slice3D
       >>>
       >>> def f1(x, y, z):
       >>>     return x**3 + y**2 + z
       >>>
       >>> f2 = Slice3D(f1, 'x', 1.5)
       >>>
       >>> f2(0, 0)
       3.375
       >>> f2(1, 0)
       4.375
    """

    def __init__(self, object function, object axis, double value):

        # convert string axis to numerical axis value
        if isinstance(axis, str):
            map = {'x': 0, 'y': 1, 'z': 2}
            try:
                axis = map[axis.lower()]
            except KeyError:
                raise ValueError('The axis must be either the string \'x\', \'y\' or \'z\', or the value 0, 1 or 2.')

        # check numerical value
        if axis not in [0, 1, 2]:
            raise ValueError('The axis must be either the string \'x\', \'y\' or \'z\', or the value 0, 1 or 2.')

        self.axis = axis
        self.value = value
        self._function = autowrap_function3d(function)

    cdef double evaluate(self, double x, double y) except? -1e999:

        if self.axis == 0:
            return self._function.evaluate(self.value, x, y)
        if self.axis == 1:
            return self._function.evaluate(x, self.value, y)
        else:
            return self._function.evaluate(x, y, self.value)