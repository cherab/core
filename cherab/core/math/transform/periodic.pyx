# Copyright 2016-2022 Euratom
# Copyright 2016-2022 United Kingdom Atomic Energy Authority
# Copyright 2016-2022 Centro de Investigaciones Energéticas, Medioambientales y Tecnológicas
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

from raysect.core.math cimport Vector3D
from raysect.core.math.function.float cimport autowrap_function1d, autowrap_function2d, autowrap_function3d
from raysect.core.math.function.vector3d cimport autowrap_function1d as autowrap_vectorfunction1d
from raysect.core.math.function.vector3d cimport autowrap_function2d as autowrap_vectorfunction2d
from raysect.core.math.function.vector3d cimport autowrap_function3d as autowrap_vectorfunction3d


cdef class PeriodicTransform1D(Function1D):
    """
    Extends a periodic 1D function to an infinite 1D space.

    :param Function1D function1d: The periodic 1D function defined
                                  in the [0, period) interval.
    :param double period: The period of the function.

    .. code-block:: pycon

       >>> from cherab.core.math import PeriodicTransform1D
       >>>
       >>> def f1(x):
       >>>     return x
       >>>
       >>> f2 = PeriodicTransform1D(f1, 1.)
       >>>
       >>> f2(1.5)
       0.5
       >>> f2(-0.3)
       0.7
    """

    def __init__(self, object function1d, double period):

        if not callable(function1d):
            raise TypeError("function1d is not callable.")

        self.function1d = autowrap_function1d(function1d)

        if period <= 0:
            raise ValueError("Argument period must be positive.")

        self.period = period

    cdef double evaluate(self, double x) except? -1e999:
        """Return the value of periodic function."""

        return self.function1d.evaluate(remainder(x, self.period))


cdef class PeriodicTransform2D(Function2D):
    """
    Extends a periodic 2D function to an infinite 2D space.

    Set period_x/period_y to 0 if the function is not periodic along x/y axis.

    :param Function2D function2d: The periodic 2D function defined
                                  in the ([0, period_x), [0, period_y)) intervals.
    :param double period_x: The period of the function along x-axis.
                            0 if not periodic.
    :param double period_y: The period of the function along y-axis.
                            0 if not periodic.

    .. code-block:: pycon

       >>> from cherab.core.math import PeriodicTransform2D
       >>>
       >>> def f1(x, y):
       >>>     return x * y
       >>>
       >>> f2 = PeriodicTransform2D(f1, 1., 1.)
       >>>
       >>> f2(1.5, 1.5)
       0.25
       >>> f2(-0.3, -1.3)
       0.49
       >>>
       >>> f3 = PeriodicTransform2D(f1, 1., 0)
       >>>
       >>> f3(1.5, 1.5)
       0.75
       >>> f3(-0.3, -1.3)
       -0.91
    """

    def __init__(self, object function2d, double period_x, double period_y):

        if not callable(function2d):
            raise TypeError("function2d is not callable.")

        self.function2d = autowrap_function2d(function2d)

        if period_x < 0:
            raise ValueError("Argument period_x must be >= 0.")
        if period_y < 0:
            raise ValueError("Argument period_y must be >= 0.")

        self.period_x = period_x
        self.period_y = period_y

    cdef double evaluate(self, double x, double y) except? -1e999:
        """Return the value of periodic function."""

        x = remainder(x, self.period_x)
        y = remainder(y, self.period_y)

        return self.function2d.evaluate(x, y)


cdef class PeriodicTransform3D(Function3D):
    """
    Extends a periodic 3D function to an infinite 3D space.

    Set period_x/period_y/period_z to 0 if the function is not periodic along x/y/z axis.

    :param Function3D function3d: The periodic 3D function defined in the
                                  ([0, period_x), [0, period_y), [0, period_z)) intervals.
    :param double period_x: The period of the function along x-axis.
                            0 if not periodic.
    :param double period_y: The period of the function along y-axis.
                            0 if not periodic.
    :param double period_z: The period of the function along z-axis.
                            0 if not periodic.

    .. code-block:: pycon

       >>> from cherab.core.math import PeriodicTransform3D
       >>>
       >>> def f1(x, y, z):
       >>>     return x * y * z
       >>>
       >>> f2 = PeriodicTransform3D(f1, 1., 1., 1.)
       >>>
       >>> f2(1.5, 1.5, 1.5)
       0.125
       >>> f2(-0.3, -1.3, -2.3)
       0.343
       >>>
       >>> f3 = PeriodicTransform3D(f1, 0, 1., 0)
       >>>
       >>> f3(1.5, 1.5, 1.5)
       1.125
       >>> f3(-0.3, -1.3, -0.3)
       0.063
    """

    def __init__(self, object function3d, double period_x, double period_y, double period_z):

        if not callable(function3d):
            raise TypeError("function2d is not callable.")

        self.function3d = autowrap_function3d(function3d)

        if period_x < 0:
            raise ValueError("Argument period_x must be >= 0.")
        if period_y < 0:
            raise ValueError("Argument period_y must be >= 0.")
        if period_z < 0:
            raise ValueError("Argument period_z must be >= 0.")

        self.period_x = period_x
        self.period_y = period_y
        self.period_z = period_z

    cdef double evaluate(self, double x, double y, double z) except? -1e999:
        """Return the value of periodic function."""

        x = remainder(x, self.period_x)
        y = remainder(y, self.period_y)
        z = remainder(z, self.period_z)

        return self.function3d.evaluate(x, y, z)


cdef class VectorPeriodicTransform1D(VectorFunction1D):
    """
    Extends a periodic 1D vector function to an infinite 1D space.

    :param VectorFunction1D function1d: The periodic 1D vector function
                                        defined in the [0, period) interval.
    :param double period: The period of the function.

    .. code-block:: pycon

       >>> from raysect.core.math import Vector3D
       >>> from cherab.core.math import VectorPeriodicTransform1D
       >>>
       >>> def f1(x):
       >>>     return Vector3D(x, 0, 0)
       >>>
       >>> f2 = VectorPeriodicTransform1D(f1, 1.)
       >>>
       >>> f2(1.5)
       Vector3D(0.5, 0, 0)
       >>> f2(-0.3)
       Vector3D(0.7, 0, 0)
    """

    def __init__(self, object function1d, double period):

        if not callable(function1d):
            raise TypeError("function1d is not callable.")

        self.function1d = autowrap_vectorfunction1d(function1d)

        if period <= 0:
            raise ValueError("Argument period must be positive.")

        self.period = period

    cdef Vector3D evaluate(self, double x):
        """Return the value of periodic function."""

        return self.function1d.evaluate(remainder(x, self.period))


cdef class VectorPeriodicTransform2D(VectorFunction2D):
    """
    Extends a periodic 2D vector function to an infinite 2D space.

    Set period_x/period_y to 0 if the function is not periodic along x/y axis.

    :param VectorFunction2D function2d: The periodic 2D vector function defined in
                                        the ([0, period_x), [0, period_y)) intervals.
    :param double period_x: The period of the function along x-axis.
                            0 if not periodic.
    :param double period_y: The period of the function along y-axis.
                            0 if not periodic.

    .. code-block:: pycon

       >>> from cherab.core.math import VectorPeriodicTransform2D
       >>>
       >>> def f1(x, y):
       >>>     return Vector3D(x, y, 0)
       >>>
       >>> f2 = VectorPeriodicTransform2D(f1, 1., 1.)
       >>>
       >>> f2(1.5, 1.5)
       Vector3D(0.5, 0.5, 0)
       >>> f2(-0.3, -1.3)
       Vector3D(0.7, 0.7, 0)
       >>>
       >>> f3 = VectorPeriodicTransform2D(f1, 1., 0)
       >>>
       >>> f3(1.5, 1.5)
       Vector3D(0.5, 1.5, 0)
       >>> f3(-0.3, -1.3)
       Vector3D(0.7, -1.3, 0)
    """

    def __init__(self, object function2d, double period_x, double period_y):

        if not callable(function2d):
            raise TypeError("function2d is not callable.")

        self.function2d = autowrap_vectorfunction2d(function2d)

        if period_x < 0:
            raise ValueError("Argument period_x must be >= 0.")
        if period_y < 0:
            raise ValueError("Argument period_y must be >= 0.")

        self.period_x = period_x
        self.period_y = period_y

    cdef Vector3D evaluate(self, double x, double y):
        """Return the value of periodic function."""

        x = remainder(x, self.period_x)
        y = remainder(y, self.period_y)

        return self.function2d.evaluate(x, y)


cdef class VectorPeriodicTransform3D(VectorFunction3D):
    """
    Extends a periodic 3D vector function to an infinite 3D space.

    Set period_x/period_y/period_z to 0 if the function is not periodic along x/y/z axis.

    :param VectorFunction3D function3d: The periodic 3D vector function defined in the
                                        ([0, period_x), [0, period_y), [0, period_z)) intervals.
    :param double period_x: The period of the function along x-axis.
                            0 if not periodic.
    :param double period_y: The period of the function along y-axis.
                            0 if not periodic.
    :param double period_z: The period of the function along z-axis.
                            0 if not periodic.

    .. code-block:: pycon

       >>> from cherab.core.math import PeriodicTransform3D
       >>>
       >>> def f1(x, y, z):
       >>>     return Vector3D(x, y, z)
       >>>
       >>> f2 = VectorPeriodicTransform3D(f1, 1., 1., 1.)
       >>>
       >>> f2(1.5, 1.5, 1.5)
       Vector3D(0.5, 0.5, 0.5)
       >>> f2(-0.3, -1.3, -2.3)
       Vector3D(0.7, 0.7, 0.7)
       >>>
       >>> f3 = VectorPeriodicTransform3D(f1, 0, 1., 0)
       >>>
       >>> f3(1.5, 0.5, 1.5)
       Vector3D(1.5, 0.5, 1.5)
    """

    def __init__(self, object function3d, double period_x, double period_y, double period_z):

        if not callable(function3d):
            raise TypeError("function2d is not callable.")

        self.function3d = autowrap_vectorfunction3d(function3d)

        if period_x < 0:
            raise ValueError("Argument period_x must be >= 0.")
        if period_y < 0:
            raise ValueError("Argument period_y must be >= 0.")
        if period_z < 0:
            raise ValueError("Argument period_z must be >= 0.")

        self.period_x = period_x
        self.period_y = period_y
        self.period_z = period_z

    cdef Vector3D evaluate(self, double x, double y, double z):
        """Return the value of periodic function."""

        x = remainder(x, self.period_x)
        y = remainder(y, self.period_y)
        z = remainder(z, self.period_z)

        return self.function3d.evaluate(x, y, z)
