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

from libc.math cimport sqrt, atan2, M_PI

from raysect.core.math.function.float cimport autowrap_function1d, autowrap_function2d, autowrap_function3d
from raysect.core.math.function.vector3d cimport autowrap_function1d as autowrap_vectorfunction1d
from raysect.core.math.function.vector3d cimport autowrap_function2d as autowrap_vectorfunction2d
from raysect.core.math.function.vector3d cimport autowrap_function3d as autowrap_vectorfunction3d
from raysect.core cimport rotate_z
cimport cython


cdef class IsoMapper2D(Function2D):
    """
    Applies a 1D function to modify the value of a 2D scalar field.

    For a given 2D scalar field f(x,y) and 1D function g(x) this object
    returns g(f(x,y)).

    :param Function2D function2d: the 2D scalar field
    :param Function1D function1d: the 1D function

    .. code-block:: pycon

       >>> from raysect.core.math.function.float import Interpolator1DArray
       >>> from cherab.core.math import IsoMapper2D
       >>> from cherab.tools.equilibrium import example_equilibrium
       >>>
       >>> equilibrium = example_equilibrium()
       >>>
       >>> # extract the 2D psi function
       >>> psi_n = equilibrium.psi_normalised
       >>> # make a 1D psi profile
       >>> profile = Interpolator1DArray([0, 0.5, 0.9, 1.0], [2500, 2000, 1000, 0], 'cubic', 'none', 0)
       >>> # perform the flux function mapping
       >>> f = IsoMapper2D(psi_n, profile)
       >>>
       >>> f(2, 0)
       2499.97177
       >>> f(2.2, 0.5)
       1990.03783
    """

    def __init__(self, object function2d, object function1d):

        if not (callable(function2d) and callable(function1d)):
            raise TypeError("function1d or function2d is not callable.")

        self.function1d = autowrap_function1d(function1d)
        self.function2d = autowrap_function2d(function2d)

    cdef double evaluate(self, double x, double y) except? -1e999:
        """Return the mapped value at (x,y)."""

        return self.function1d.evaluate(self.function2d.evaluate(x, y))


cdef class IsoMapper3D(Function3D):
    """
    Applies a 1D function to modify the value of a 3D scalar field.

    For a given 3D scalar field f(x,y,z) and 1D function g(x) this object
    returns g(f(x,y,z)).

    :param Function3D function3d: the 3D scalar field
    :param Function1D function1d: the 1D function

    .. code-block:: pycon

       >>> from raysect.core.math.function.float import Interpolator1DArray
       >>> from cherab.core.math import IsoMapper2D, AxisymmetricMapper
       >>> from cherab.tools.equilibrium import example_equilibrium
       >>>
       >>> equilibrium = example_equilibrium()
       >>>
       >>> # extract the 3D psi function
       >>> psi_n = equilibrium.psi_normalised
       >>> psi_n_3d = AxisymmetricMapper(psi_n)
       >>> # make a 1D psi profile
       >>> profile = Interpolator1DArray([0, 0.5, 0.9, 1.0], [2500, 2000, 1000, 0], 'cubic', 'none', 0)
       >>> # perform the flux function mapping
       >>> f = IsoMapper3D(psi_n_3d, profile)
       >>>
       >>> f(2, 0, 0)
       2499.97177
       >>> f(0, 2, 0)
       2499.97177
    """

    def __init__(self, object function3d, object function1d):

        if not (callable(function3d) and callable(function1d)):
            raise TypeError("function1d or function3d is not callable.")

        self.function3d = autowrap_function3d(function3d)
        self.function1d = autowrap_function1d(function1d)

    cdef double evaluate(self, double x, double y, double z) except? -1e999:
        """Return the mapped value at (x,y,z)."""

        return self.function1d.evaluate(self.function3d.evaluate(x, y, z))


cdef class Swizzle2D(Function2D):
    """
    Inverts the argument order of the specified function.

    :param Function2D function2d: The 2D function you want to inverse the arguments.

    .. code-block:: pycon

       >>> from cherab.core.math import Swizzle2D
       >>>
       >>> def f1(r, z):
       >>>     return r**2 + z
       >>>
       >>> f2 = Swizzle2D(f1)
       >>>
       >>> f2(3, 0)
       3.0
    """

    def __init__(self, object function2d):

        if not callable(function2d):
            raise TypeError("function2d is not callable.")

        self.function2d = autowrap_function2d(function2d)

    cdef double evaluate(self, double x, double y) except? -1e999:
        """Return the value with inversed arguments."""

        return self.function2d.evaluate(y, x)


cdef class Swizzle3D(Function3D):
    """
    Rearranges the order of a 3D functions arguments.

    For instance, a 90 degree rotation of function coordinates can be performed
    by swapping arguments: xyz -> xzy

    Shape is a tuple of 3 integers from 0,1,2 imposing the order of
    arguments. 0, 1 and 2 correspond respectively to x, y and z where (
    x,y,z) are the initial arguments.
    For instance:
    shape = (0,2,1) transforms f(x,y,z) in f(x,z,y)
    shape = (1,0,1) transforms f(x,y,z) in f(y,x,y)

    :param Function3D function3d: the 3D function you want to reorder the arguments.
    :param tuple shape: a tuple of integers imposing the order of the arguments.

    .. code-block:: pycon

       >>> from cherab.core.math import Swizzle3D
       >>>
       >>> def f1(x, y, z):
       >>>     return x**3 + y**2 + z
       >>>
       >>> f2 = Swizzle3D(f1, (0, 2, 1))
       >>>
       >>> f2(3, 2, 1)
       30.0
    """

    def __init__(self, object function3d, shape):
        """


        """

        if not callable(function3d):
            raise TypeError("function3d is not callable.")

        for i in shape:
            if i not in [0, 1, 2]:
                raise ValueError("shape must contain integers among 0, "
                                 "1 and 2 only")

        if isinstance(shape, tuple) and len(shape) == 3:
            self.function3d = autowrap_function3d(function3d)
            for i in range(3):
                self.shape[i] = shape[i]
        else:
            raise TypeError("shape must be a tuple of length 3.")

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double evaluate(self, double x, double y, double z) except? -1e999:
        """Return the value of function3d at position (x,y,z) reorganized
        according to shape."""

        cdef:
            double d[3]
            int i

        for i in range(3):
            if self.shape[i] == 0:
                d[i] = x
            elif self.shape[i] == 1:
                d[i] = y
            elif self.shape[i] == 2:
                d[i] = z
            else:
                raise ValueError("Shape must contain integers among 0, 1 and 2 only")

        return self.function3d.evaluate(d[0], d[1], d[2])


cdef class AxisymmetricMapper(Function3D):
    """
    Performs an 360 degree rotation of a 2D function (defined on the xz plane)
    around the z-axis.

    Due to the nature of this mapping, only the positive region of the x range
    of the supplied function is mapped.
    
    :param Function2D function2d: The function to be mapped.

    .. code-block:: pycon

       >>> from numpy import sqrt
       >>> from cherab.core.math import AxisymmetricMapper
       >>>
       >>> def f1(r, z):
       >>>     return r
       >>>
       >>> f2 = AxisymmetricMapper(f1)
       >>>
       >>> f2(1, 0, 0)
       1.0
       >>> f2(1/sqrt(2), 1/sqrt(2), 0)
       0.99999999
    """

    def __init__(self, object function2d):

        if not callable(function2d):
            raise TypeError("Function2D is not callable.")

        self.function2d = autowrap_function2d(function2d)

    cdef double evaluate(self, double x, double y, double z) except? -1e999:
        """Return the value of function2d when it is y-axis symmetrically
        extended to the 3D space."""

        return self.function2d.evaluate(sqrt(x*x + y*y), z)


cdef class VectorAxisymmetricMapper(VectorFunction3D):
    """
    Performs an 360 degree rotation of a 2D vector function (defined on the xz plane)
    around the z-axis.

    Due to the nature of this mapping, only the positive region of the x range
    of the supplied function is mapped.

    :param VectorFunction2D vectorfunction2d: The vector function to be mapped.

    .. code-block:: pycon

       >>> from cherab.core.math import VectorAxisymmetricMapper
       >>>
       >>> def my_func(r, z):
       >>>     v = Vector3D(1, 0, 0)
       >>>     v.length = r
       >>>     return v
       >>>
       >>> f = VectorAxisymmetricMapper(my_func)
       >>>
       >>> f(1, 0, 0)
       Vector3D(1.0, 0.0, 0.0)
       >>> f(1/sqrt(2), 1/sqrt(2), 0)
       Vector3D(0.70710678, 0.70710678, 0.0)
    """

    def __init__(self, object vectorfunction2d):

        if not callable(vectorfunction2d):
            raise TypeError("Function3D is not callable.")

        self.function2d = autowrap_vectorfunction2d(vectorfunction2d)

    @cython.cdivision(True)
    cdef Vector3D evaluate(self, double x, double y, double z):
        """Return the value of function2d when it is y-axis symmetrically
        extended to the 3D space."""
        cdef double r, phi

        # convert to cylindrical coordinates
        phi = atan2(y, x) / M_PI * 180
        r, z = sqrt(x*x + y*y), z

        # perform axisymmetric rotation
        return self.function2d.evaluate(r, z).transform(rotate_z(phi))


cdef class CylindricalMapper(Function3D):
    """
    Converts Cartesian coordinates to cylindrical coordinates and calls a 3D function
    defined in cylindrical coordinates, f(r, :math:`\\phi`, z).

    The angular coordinate is given in radians.

    Positive angular coordinate is measured counterclockwise from the xz plane.
    
    :param Function3D function3d: The function to be mapped. Must be defined
                                  in the interval (:math:`-\\pi`, :math:`\\pi`]
                                  on the angular axis.

    .. code-block:: pycon

       >>> from math import sqrt, cos
       >>> from cherab.core.math import CylindricalMapper
       >>>
       >>> def my_func(r, phi, z):
       >>>     return r * cos(phi)
       >>>
       >>> f = CylindricalMapper(my_func)
       >>>
       >>> f(1, 0, 0)
       1.0
       >>> f(0.5 * sqrt(3), 0.5, 0)
       0.8660254037844385
    """

    def __init__(self, object function3d):

        if not callable(function3d):
            raise TypeError("Function3D is not callable.")

        self.function3d = autowrap_function3d(function3d)

    cdef double evaluate(self, double x, double y, double z) except? -1e999:
        """
        Converts to cylindrical coordinates and evaluates the function
        defined in cylindrical coordinates.
        """
        cdef double r, phi

        r = sqrt(x * x + y * y)
        phi = atan2(y, x)

        return self.function3d.evaluate(r, phi, z)


cdef class VectorCylindricalMapper(VectorFunction3D):
    """
    Converts Cartesian coordinates to cylindrical coordinates, calls
    a 3D vector function defined in cylindrical coordinates, f(r, :math:`\\phi`, z),
    then converts the returned 3D vector to Cartesian coordinates.

    The angular coordinate is given in radians.

    Positive angular coordinate is measured counterclockwise from the xz plane.
    
    :param VectorFunction3D function3d: The function to be mapped. Must be defined
                                        in the interval (:math:`-\\pi`, :math:`\\pi`]
                                        on the angular axis.

    .. code-block:: pycon

       >>> from math import sqrt, cos
       >>> from raysect.core.math import Vector3D
       >>> from cherab.core.math import VectorCylindricalMapper
       >>>
       >>> def my_vec_func(r, phi, z):
       >>>     v = Vector3D(0, 1, 0)
       >>>     v.length = r * abs(cos(phi))
       >>>     return v
       >>>
       >>> f = VectorCylindricalMapper(my_vec_func)
       >>>
       >>> f(1, 0, 0)
       Vector3D(0.0, 1.0, 0.0)
       >>> f(1/sqrt(2), 1/sqrt(2), 0)
       Vector3D(-0.5, 0.5, 0.0)
    """

    def __init__(self, object function3d):

        if not callable(function3d):
            raise TypeError("Function3D is not callable.")

        self.function3d = autowrap_vectorfunction3d(function3d)

    @cython.cdivision(True)
    cdef Vector3D evaluate(self, double x, double y, double z):
        """
        Converts to cylindrical coordinates, evaluates the vector function
        defined in cylindrical coordinates and rotates the resulting vector
        around z-axis.
        """
        cdef double r, phi

        r = sqrt(x * x + y * y)
        phi = atan2(y, x)

        return self.function3d.evaluate(r, phi, z).transform(rotate_z(phi / M_PI * 180))


cdef class PeriodicMapper1D(Function1D):
    """
    Maps a periodic 1D function into 1D space.

    :param Function1D function1d: The periodic 1D function to map defined
                                  in the [0, period) interval.
    :param double period: The period of the function.

    .. code-block:: pycon

       >>> from cherab.core.math import PeriodicMapper1D
       >>>
       >>> def f1(x):
       >>>     return x
       >>>
       >>> f2 = PeriodicMapper1D(f1, 1.)
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


cdef class PeriodicMapper2D(Function2D):
    """
    Maps a periodic 2D function into 2D space.

    Set period_x/period_y to 0 if the function is not periodic along x/y axis.

    :param Function2D function2d: The periodic 2D function to map defined
                                  in the ([0, period_x), [0, period_y)) intervals.
    :param double period_x: The period of the function along x-axis.
                            0 if not periodic.
    :param double period_y: The period of the function along y-axis.
                            0 if not periodic.

    .. code-block:: pycon

       >>> from cherab.core.math import PeriodicMapper2D
       >>>
       >>> def f1(x, y):
       >>>     return x * y
       >>>
       >>> f2 = PeriodicMapper2D(f1, 1., 1.)
       >>>
       >>> f2(1.5, 1.5)
       0.25
       >>> f2(-0.3, -1.3)
       0.49
       >>>
       >>> f3 = PeriodicMapper2D(f1, 1., 0)
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


cdef class PeriodicMapper3D(Function3D):
    """
    Maps a periodic 3D function into 3D space.

    Set period_x/period_y/period_z to 0 if the function is not periodic along x/y/z axis.

    :param Function3D function3d: The periodic 3D function to map defined in the
                                  ([0, period_x), [0, period_y), [0, period_z)) intervals.
    :param double period_x: The period of the function along x-axis.
                            0 if not periodic.
    :param double period_y: The period of the function along y-axis.
                            0 if not periodic.
    :param double period_z: The period of the function along z-axis.
                            0 if not periodic.

    .. code-block:: pycon

       >>> from cherab.core.math import PeriodicMapper3D
       >>>
       >>> def f1(x, y, z):
       >>>     return x * y * z
       >>>
       >>> f2 = PeriodicMapper3D(f1, 1., 1., 1.)
       >>>
       >>> f2(1.5, 1.5, 1.5)
       0.125
       >>> f2(-0.3, -1.3, -2.3)
       0.343
       >>>
       >>> f3 = PeriodicMapper3D(f1, 0, 1., 0)
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


cdef class VectorPeriodicMapper1D(VectorFunction1D):
    """
    Maps a periodic 1D vector function into 1D space.

    :param VectorFunction1D function1d: The periodic 1D vector function to map
                                        defined in the [0, period) interval.
    :param double period: The period of the function.

    .. code-block:: pycon

       >>> from raysect.core.math import Vector3D
       >>> from cherab.core.math import VectorPeriodicMapper1D
       >>>
       >>> def f1(x):
       >>>     return Vector3D(x, 0, 0)
       >>>
       >>> f2 = VectorPeriodicMapper1D(f1, 1.)
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


cdef class VectorPeriodicMapper2D(VectorFunction2D):
    """
    Maps a periodic 2D vector function into 2D space.

    Set period_x/period_y to 0 if the function is not periodic along x/y axis.

    :param VectorFunction2D function2d: The periodic 2D vector function to map defined in
                                        the ([0, period_x), [0, period_y)) intervals.
    :param double period_x: The period of the function along x-axis.
                            0 if not periodic.
    :param double period_y: The period of the function along y-axis.
                            0 if not periodic.

    .. code-block:: pycon

       >>> from cherab.core.math import VectorPeriodicMapper2D
       >>>
       >>> def f1(x, y):
       >>>     return Vector3D(x, y, 0)
       >>>
       >>> f2 = VectorPeriodicMapper2D(f1, 1., 1.)
       >>>
       >>> f2(1.5, 1.5)
       Vector3D(0.5, 0.5, 0)
       >>> f2(-0.3, -1.3)
       Vector3D(0.7, 0.7, 0)
       >>>
       >>> f3 = VectorPeriodicMapper2D(f1, 1., 0)
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


cdef class VectorPeriodicMapper3D(VectorFunction3D):
    """
    Maps a periodic 3D vector function into 3D space.

    Set period_x/period_y/period_z to 0 if the function is not periodic along x/y/z axis.

    :param VectorFunction3D function3d: The periodic 3D vector function to map defined in the
                                        ([0, period_x), [0, period_y), [0, period_z)) intervals.
    :param double period_x: The period of the function along x-axis.
                            0 if not periodic.
    :param double period_y: The period of the function along y-axis.
                            0 if not periodic.
    :param double period_z: The period of the function along z-axis.
                            0 if not periodic.

    .. code-block:: pycon

       >>> from cherab.core.math import PeriodicMapper3D
       >>>
       >>> def f1(x, y, z):
       >>>     return Vector3D(x, y, z)
       >>>
       >>> f2 = VectorPeriodicMapper3D(f1, 1., 1., 1.)
       >>>
       >>> f2(1.5, 1.5, 1.5)
       Vector3D(0.5, 0.5, 0.5)
       >>> f2(-0.3, -1.3, -2.3)
       Vector3D(0.7, 0.7, 0.7)
       >>>
       >>> f3 = VectorPeriodicMapper3D(f1, 0, 1., 0)
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
