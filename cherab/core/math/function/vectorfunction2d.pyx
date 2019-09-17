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

from raysect.core.math cimport new_vector3d, Vector3D
from raysect.core.math.function cimport autowrap_function2d


cdef class VectorFunction2D:
    """
    Cython optimised class for representing an arbitrary 2D vector function.

    Returns a Vector3D for a given 2D coordinate.

    Using __call__() in cython is slow. This class provides an overloadable
    cython cdef evaluate() method which has much less overhead than a python
    function call.

    For use in cython code only, this class cannot be extended via python.

    To create a new function object, inherit this class and implement the
    evaluate() method. The new function object can then be used with any code
    that accepts a function object.
    """

    cdef Vector3D evaluate(self, double x, double y):

        raise NotImplementedError("The evaluate() method has not been implemented.")

    def __call__(self, double x, double y):

        return self.evaluate(x, y)


cdef class PythonVectorFunction2D(VectorFunction2D):
    """
    Wraps a python callable object with a VectorFunction2D object.

    This class allows a python object to interact with cython code that requires
    a VectorFunction3D object. The python object must implement __call__()
    expecting three arguments and return a Vector object.

    This class is intended to be used to transparently wrap python objects that
    are passed via constructors or methods into cython optimised code. It is not
    intended that the users should need to directly interact with these wrapping
    objects. Constructors and methods expecting a VectorFunction3D object should
    be designed to accept a generic python object and then test that object to
    determine if it is an instance of VectorFunction3D. If the object is not a
    VectorFunction3D object it should be wrapped using this class for internal
    use.

    See also: autowrap_vectorfunction2d()

    :param object function: the 2d vector function object to wrap.

    .. code-block:: pycon

       >>> from raysect.core import Vector3D
       >>> from cherab.core.math import PythonVectorFunction2D
       >>>
       >>> def vectorfunction2d(r, z):
       >>>     return Vector3D(1, 0, 0)
       >>>
       >>> fv2 = PythonVectorFunction2D(vectorfunction2d)
       >>> fv2(1, 3.5)
       Vector3D(1.0, 0.0, 0.0)
    """

    def __init__(self, object function):

        self.function = function

    cdef Vector3D evaluate(self, double x, double y):

        return self.function(x, y)


cdef VectorFunction2D autowrap_vectorfunction2d(object function):
    """
    Automatically wraps the supplied python object in a PythonVectorFunction2D
    object.

    If this function is passed a valid VectorFunction2D object, then the
    VectorFunction3D object is simply returned without wrapping.

    This convenience function is provided to simplify the handling of
    VectorFunction2D and python callable objects in constructors, functions and
    setters.
    """

    if isinstance(function, VectorFunction2D):

        return <VectorFunction2D> function

    else:

        return PythonVectorFunction2D(function)


cdef class ScalarToVectorFunction2D(VectorFunction2D):
    """
    Combines three Function2D objects to produce a VectorFunction2D.

    The three Function2D objects correspond to the x, y and z components of the
    resulting vector object.

    :param Function2D x_function: the Vx(x, y) 2d function.
    :param Function2D y_function: the Vy(x, y) 2d function.
    :param Function2D z_function: the Vz(x, y) 2d function.

    .. code-block:: pycon

       >>> from cherab.core.math import Constant2D
       >>> from cherab.core.math.function import ScalarToVectorFunction2D
       >>>
       >>> vx = Constant2D(1)
       >>> vy = Constant2D(2)
       >>> vz = Constant2D(3)
       >>>
       >>> fv = ScalarToVectorFunction2D(vx, vy, vz)
       >>> fv(3.5, 6.2)
       Vector3D(1.0, 2.0, 3.0)
    """

    def __init__(self, object x_function, object y_function, object z_function):

        super().__init__()
        self._x = autowrap_function2d(x_function)
        self._y = autowrap_function2d(y_function)
        self._z = autowrap_function2d(z_function)

    cdef Vector3D evaluate(self, double x, double y):

        return new_vector3d(self._x.evaluate(x, y),
                            self._y.evaluate(x, y),
                            self._z.evaluate(x, y))

