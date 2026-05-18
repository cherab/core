# cython: language_level=3

# Copyright 2016-2025 Euratom
# Copyright 2016-2025 United Kingdom Atomic Energy Authority
# Copyright 2016-2025 Centro de Investigaciones Energéticas, Medioambientales y Tecnológicas
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

cimport libc.math as cmath
from cherab.core.math.function.float.function6d.base cimport Function6D
from cherab.core.math.function.float.function6d.autowrap cimport autowrap_function6d


cdef class Exp6D(Function6D):
    """
    A Function6D class that implements the exponential of the result of a Function6D object: exp(f())

    :param Function6D function: A Function6D object.
    """
    def __init__(self, object function):
        self._function = autowrap_function6d(function)

    cdef double evaluate(self, double x, double y, double z, double u, double w, double v) except? -1e999:
        return cmath.exp(self._function.evaluate(x, y, z, u, w, v))


cdef class Sin6D(Function6D):
    """
    A Function6D class that implements the sine of the result of a Function6D object: sin(f())

    :param Function6D function: A Function6D object.
    """
    def __init__(self, object function):
        self._function = autowrap_function6d(function)

    cdef double evaluate(self, double x, double y, double z, double u, double w, double v) except? -1e999:
        return cmath.sin(self._function.evaluate(x, y, z, u, w, v))


cdef class Cos6D(Function6D):
    """
    A Function6D class that implements the cosine of the result of a Function6D object: cos(f())

    :param Function6D function: A Function6D object.
    """
    def __init__(self, object function):
        self._function = autowrap_function6d(function)

    cdef double evaluate(self, double x, double y, double z, double u, double w, double v) except? -1e999:
        return cmath.cos(self._function.evaluate(x, y, z, u, w, v))


cdef class Tan6D(Function6D):
    """
    A Function6D class that implements the tangent of the result of a Function6D object: tan(f())

    :param Function6D function: A Function6D object.
    """
    def __init__(self, object function):
        self._function = autowrap_function6d(function)

    cdef double evaluate(self, double x, double y, double z, double u, double w, double v) except? -1e999:
        return cmath.tan(self._function.evaluate(x, y, z, u, w, v))


cdef class Asin6D(Function6D):
    """
    A Function6D class that implements the arcsine of the result of a Function6D object: asin(f())

    :param Function6D function: A Function6D object.
    """
    def __init__(self, object function):
        self._function = autowrap_function6d(function)

    cdef double evaluate(self, double x, double y, double z, double u, double w, double v) except? -1e999:
        cdef double val = self._function.evaluate(x, y, z, u, w, v)
        if -1.0 <= val <= 1.0:
            return cmath.asin(val)
        raise ValueError("The function returned a value outside of the arcsine domain of [-1, 1].")


cdef class Acos6D(Function6D):
    """
    A Function6D class that implements the arccosine of the result of a Function6D object: acos(f())

    :param Function6D function: A Function6D object.
    """
    def __init__(self, object function):
        self._function = autowrap_function6d(function)

    cdef double evaluate(self, double x, double y, double z, double u, double w, double v) except? -1e999:
        cdef double val = self._function.evaluate(x, y, z, u, w, v)
        if -1.0 <= val <= 1.0:
            return cmath.acos(val)
        raise ValueError("The function returned a value outside of the arccosine domain of [-1, 1].")


cdef class Atan6D(Function6D):
    """
    A Function6D class that implements the arctangent of the result of a Function6D object: atan(f())

    :param Function6D function: A Function6D object.
    """
    def __init__(self, object function):
        self._function = autowrap_function6d(function)

    cdef double evaluate(self, double x, double y, double z, double u, double w, double v) except? -1e999:
        return cmath.atan(self._function.evaluate(x, y, z, u, w, v))


cdef class Atan4Q6D(Function6D):
    """
    A Function6D class that implements the arctangent of the result of 2 Function6D objects: atan2(f1(), f2())

    This differs from Atan6D in that it takes separate functions for the
    numerator and denominator, in order to get the quadrant correct.

    :param Function6D numerator: A Function6D object representing the numerator
    :param Function6D denominator: A Function6D object representing the denominator
    """
    def __init__(self, object numerator, object denominator):
        self._numerator = autowrap_function6d(numerator)
        self._denominator = autowrap_function6d(denominator)

    cdef double evaluate(self, double x, double y, double z, double u, double w, double v) except? -1e999:
        return cmath.atan2(self._numerator.evaluate(x, y, z, u, w, v),
                          self._denominator.evaluate(x, y, z, u, w, v))


cdef class Sqrt6D(Function6D):
    """
    A Function6D class that implements the square root of the result of a Function6D object: sqrt(f())

    :param Function6D function: A Function6D object.
    """
    def __init__(self, object function):
        self._function = autowrap_function6d(function)

    cdef double evaluate(self, double x, double y, double z, double u, double w, double v) except? -1e999:
        cdef double f = self._function.evaluate(x, y, z, u, w, v)
        if f < 0: # complex values are not supported
            raise ValueError("Math domain error in sqrt({0}). Sqrt of a negative value is not supported.".format(f))
        return cmath.sqrt(f)


cdef class Erf6D(Function6D):
    """
    A Function6D class that implements the error function of the result of a Function6D object: erf(f())

    :param Function6D function: A Function6D object.
    """
    def __init__(self, object function):
        self._function = autowrap_function6d(function)

    cdef double evaluate(self, double x, double y, double z, double u, double w, double v) except? -1e999:
        return cmath.erf(self._function.evaluate(x, y, z, u, w, v))