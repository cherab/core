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

import numbers
from cpython.object cimport Py_LT, Py_EQ, Py_GT, Py_LE, Py_NE, Py_GE
cimport cython
from libc.math cimport floor
from .autowrap cimport autowrap_function6d


cdef class Function6D(FloatFunction):
    """
    Cython optimised class for representing an arbitrary 6D function returning a float.

    Using __call__() in cython is slow. This class provides an overloadable
    cython cdef evaluate() method which has much less overhead than a python
    function call.

    For use in cython code only, this class cannot be extended via python.

    To create a new function object, inherit this class and implement the
    evaluate() method. The new function object can then be used with any code
    that accepts a function object.
    """

    cdef double evaluate(self, double x, double y, double z, double u, double w, double v) except? -1e999:
        raise NotImplementedError("The evaluate() method has not been implemented.")

    def __call__(self, double x, double y, double z, double u, double w, double v):
        """ Evaluate the function f(x, y, z, u, w, v)

        :param float x: function parameter x
        :param float y: function parameter y
        :param float z: function parameter z
        :param float u: function parameter u
        :param float w: function parameter w
        :param float v: function parameter v
        :rtype: float
        """
        return self.evaluate(x, y, z, u, w, v)
    def __add__(self, object b):
        if is_callable(b):
            # a() + b()
            return AddFunction6D(self, b)
        elif isinstance(b, numbers.Real):
            # a() + B -> B + a()
            return AddScalar6D(<double> b, self)
        return NotImplemented

    def __radd__(self, object a):
        return self.__add__(a)

    def __sub__(self, object b):
        if is_callable(b):
            # a() - b()
            return SubtractFunction6D(self, b)
        elif isinstance(b, numbers.Real):
            # a() - B -> -B + a()
            return AddScalar6D(-(<double> b), self)
        return NotImplemented
    
    def __rsub__(self, object a):
        if is_callable(a):
            # a() - b()
            return SubtractFunction6D(a, self)
        elif isinstance(a, numbers.Real):
            # A - b()
            return SubtractScalar6D(<double> a, self)
        return NotImplemented

    def __mul__(self, object b):
        if is_callable(b):
            # a() * b()
            return MultiplyFunction6D(self, b)
        elif isinstance(b, numbers.Real):
            # a() * B -> B * a()
            return MultiplyScalar6D(<double> b, self)
        return NotImplemented
    
    def __rmul__(self, object a):
        return self.__mul__(a)

    @cython.cdivision(True)
    def __truediv__(self, object b):
        cdef double v
        if is_callable(b):
            # a() / b()
            return DivideFunction6D(self, b)
        elif isinstance(b, numbers.Real):
            # a() / B -> 1/B * a()
            v = <double> b
            if v == 0.0:
                raise ZeroDivisionError("Scalar used as the denominator of the division is zero valued.")
            return MultiplyScalar6D(1/v, self)
        return NotImplemented

    @cython.cdivision(True)
    def __rtruediv__(self, object a):
        if is_callable(a):
            # a() / b()
            return DivideFunction6D(a, self)
        elif isinstance(a, numbers.Real):
            # A / b()
            return DivideScalar6D(<double> a, self)
        return NotImplemented

    def __mod__(self, object b):
        cdef double v
        if is_callable(b):
            # a() % b()
            return ModuloFunction6D(self, b)
        elif isinstance(b, numbers.Real):
            # a() % B
            v = <double> b
            if v == 0.0:
                raise ZeroDivisionError("Scalar used as the divisor of the division is zero valued.")
            return ModuloFunctionScalar6D(self, v)
        return NotImplemented
    
    def __rmod__(self, object a):
        if is_callable(a):
            # a() % b()
            return ModuloFunction6D(a, self)
        elif isinstance(a, numbers.Real):
            # A % b()
            return ModuloScalarFunction6D(<double> a, self)
        return NotImplemented

    def __neg__(self):
        return MultiplyScalar6D(-1, self)

    def __pow__(self, object b, object c):
        if c is not None:
            # Optimised implementation of pow(a, b, c) not available: fall back
            # to general implementation
            return PowFunction6D(self, b) % c
        if is_callable(b):
            # a() ** b()
            return PowFunction6D(self, b)
        elif isinstance(b, numbers.Real):
            # a() ** b
            return PowFunctionScalar6D(self, <double> b)
        return NotImplemented
    
    def __rpow__(self, object a, object c):
        if c is not None:
            # Optimised implementation of pow(a, b, c) not available: fall back
            # to general implementation
            return PowFunction6D(a, self) % c
        if is_callable(a):
            # a() ** b()
            return PowFunction6D(a, self)
        elif isinstance(a, numbers.Real):
            # A ** b()
            return PowScalarFunction6D(<double> a, self)
        return NotImplemented

    def __abs__(self):
        return AbsFunction6D(self)

    def __richcmp__(self, object other, int op):
        if is_callable(other):
            if op == Py_EQ:
                return EqualsFunction6D(self, other)
            if op == Py_NE:
                return NotEqualsFunction6D(self, other)
            if op == Py_LT:
                return LessThanFunction6D(self, other)
            if op == Py_GT:
                return GreaterThanFunction6D(self, other)
            if op == Py_LE:
                return LessEqualsFunction6D(self, other)
            if op == Py_GE:
                return GreaterEqualsFunction6D(self, other)
        if isinstance(other, numbers.Real):
            if op == Py_EQ:
                return EqualsScalar6D(<double> other, self)
            if op == Py_NE:
                return NotEqualsScalar6D(<double> other, self)
            if op == Py_LT:
                # f() < K -> K > f
                return GreaterThanScalar6D(<double> other, self)
            if op == Py_GT:
                # f() > K -> K < f
                return LessThanScalar6D(<double> other, self)
            if op == Py_LE:
                # f() <= K -> K >= f
                return GreaterEqualsScalar6D(<double> other, self)
            if op == Py_GE:
                # f() >= K -> K <= f
                return LessEqualsScalar6D(<double> other, self)
        return NotImplemented


cdef class AddFunction6D(Function6D):
    """
    A Function6D class that implements the addition of the results of two Function6D objects: f1() + f2()

    This class is not intended to be used directly, but rather returned as the result of an __add__() call on a
    Function6D object.

    :param function1: A Function6D object.
    :param function2: A Function6D object.
    """

    def __init__(self, object function1, object function2):
        self._function1 = autowrap_function6d(function1)
        self._function2 = autowrap_function6d(function2)

    cdef double evaluate(self, double x, double y, double z, double u, double w, double v) except? -1e999:
        return self._function1.evaluate(x, y, z, u, w, v) + self._function2.evaluate(x, y, z, u, w, v)


cdef class SubtractFunction6D(Function6D):
    """
    A Function6D class that implements the subtraction of the results of two Function6D objects: f1() - f2()

    This class is not intended to be used directly, but rather returned as the result of a __sub__() call on a
    Function6D object.

    :param function1: A Function6D object.
    :param function2: A Function6D object.
    """

    def __init__(self, object function1, object function2):
        self._function1 = autowrap_function6d(function1)
        self._function2 = autowrap_function6d(function2)

    cdef double evaluate(self, double x, double y, double z, double u, double w, double v) except? -1e999:
        return self._function1.evaluate(x, y, z, u, w, v) - self._function2.evaluate(x, y, z, u, w, v)


cdef class MultiplyFunction6D(Function6D):
    """
    A Function6D class that implements the multiplication of the results of two Function6D objects: f1() * f2()

    This class is not intended to be used directly, but rather returned as the result of a __mul__() call on a
    Function6D object.

    :param function1: A Function6D object.
    :param function2: A Function6D object.
    """

    def __init__(self, function1, function2):
        self._function1 = autowrap_function6d(function1)
        self._function2 = autowrap_function6d(function2)

    cdef double evaluate(self, double x, double y, double z, double u, double w, double v) except? -1e999:
        return self._function1.evaluate(x, y, z, u, w, v) * self._function2.evaluate(x, y, z, u, w, v)


cdef class DivideFunction6D(Function6D):
    """
    A Function6D class that implements the division of the results of two Function6D objects: f1() / f2()

    This class is not intended to be used directly, but rather returned as the result of a __truediv__() call on a
    Function6D object.

    :param function1: A Function6D object.
    :param function2: A Function6D object.
    """

    def __init__(self, function1, function2):
        self._function1 = autowrap_function6d(function1)
        self._function2 = autowrap_function6d(function2)

    @cython.cdivision(True)
    cdef double evaluate(self, double x, double y, double z, double u, double w, double v) except? -1e999:
        cdef double denominator = self._function2.evaluate(x, y, z, u, w, v)
        if denominator == 0.0:
            raise ZeroDivisionError("Function used as the denominator of the division returned a zero value.")
        return self._function1.evaluate(x, y, z, u, w, v) / denominator


cdef class ModuloFunction6D(Function6D):
    """
    A Function6D class that implements the modulo of the results of two Function6D objects: f1() % f2()

    This class is not intended to be used directly, but rather returned as the result of a __mod__() call on a
    Function6D object.

    :param object function1: A Function6D object or Python callable.
    :param object function2: A Function6D object or Python callable.
    """
    def __init__(self, function1, function2):
        self._function1 = autowrap_function6d(function1)
        self._function2 = autowrap_function6d(function2)

    @cython.cdivision(True)
    cdef double evaluate(self, double x, double y, double z, double u, double w, double v) except? -1e999:
        cdef double divisor = self._function2.evaluate(x, y, z, u, w, v)
        if divisor == 0.0:
            raise ZeroDivisionError("Function used as the divisor of the modulo returned a zero value.")
        return self._function1.evaluate(x, y, z, u, w, v) % divisor


cdef class PowFunction6D(Function6D):
    """
    A Function6D class that implements the pow() operator on two Function6D objects.

    This class is not intended to be used directly, but rather returned as the result of a __pow__() call on a
    Function6D object.

    :param object function1: A Function6D object or Python callable.
    :param object function2: A Function6D object or Python callable.
    """
    def __init__(self, function1, function2):
        self._function1 = autowrap_function6d(function1)
        self._function2 = autowrap_function6d(function2)

    cdef double evaluate(self, double x, double y, double z, double u, double w, double v) except? -1e999:
        cdef double base, exponent
        base = self._function1.evaluate(x, y, z, u, w, v)
        exponent = self._function2.evaluate(x, y, z, u, w, v)
        if base < 0 and floor(exponent) != exponent:  # Would return a complex value rather than double
            raise ValueError("Negative base and non-integral exponent is not supported")
        if base == 0 and exponent < 0:
            raise ZeroDivisionError("0.0 cannot be raised to a negative power")
        return base ** exponent


cdef class AbsFunction6D(Function6D):
    """
    A Function6D class that implements the absolute value of the result of a Function6D object: abs(f()).

    This class is not intended to be used directly, but rather returned as the
    result of an __abs__() call on a Function6D object.

    :param object function: A Function6D object or Python callable.
    """
    def __init__(self, object function):
        self._function = autowrap_function6d(function)

    cdef double evaluate(self, double x, double y, double z, double u, double w, double v) except? -1e999:
        return abs(self._function.evaluate(x, y, z, u, w, v))


cdef class EqualsFunction6D(Function6D):
    """
    A Function6D class that tests the equality of the results of two Function6D objects: f1() == f2()

    This class is not intended to be used directly, but rather returned as the result of an __eq__() call on a
    Function6D object.

    :param object function1: A Function6D object or Python callable.
    :param object function2: A Function6D object or Python callable.
    """
    def __init__(self, object function1, object function2):
        self._function1 = autowrap_function6d(function1)
        self._function2 = autowrap_function6d(function2)

    cdef double evaluate(self, double x, double y, double z, double u, double w, double v) except? -1e999:
        return self._function1.evaluate(x, y, z, u, w, v) == self._function2.evaluate(x, y, z, u, w, v)


cdef class NotEqualsFunction6D(Function6D):
    """
    A Function6D class that tests the inequality of the results of two Function6D objects: f1() != f2()

    This class is not intended to be used directly, but rather returned as the result of an __ne__() call on a
    Function6D object.

    :param object function1: A Function6D object or Python callable.
    :param object function2: A Function6D object or Python callable.
    """
    def __init__(self, object function1, object function2):
        self._function1 = autowrap_function6d(function1)
        self._function2 = autowrap_function6d(function2)

    cdef double evaluate(self, double x, double y, double z, double u, double w, double v) except? -1e999:
        return self._function1.evaluate(x, y, z, u, w, v) != self._function2.evaluate(x, y, z, u, w, v)


cdef class LessThanFunction6D(Function6D):
    """
    A Function6D class that implements < of the results of two Function6D objects: f1() < f2()

    This class is not intended to be used directly, but rather returned as the result of an __lt__() call on a
    Function6D object.

    :param object function1: A Function6D object or Python callable.
    :param object function2: A Function6D object or Python callable.
    """
    def __init__(self, object function1, object function2):
        self._function1 = autowrap_function6d(function1)
        self._function2 = autowrap_function6d(function2)

    cdef double evaluate(self, double x, double y, double z, double u, double w, double v) except? -1e999:
        return self._function1.evaluate(x, y, z, u, w, v) < self._function2.evaluate(x, y, z, u, w, v)


cdef class GreaterThanFunction6D(Function6D):
    """
    A Function6D class that implements > of the results of two Function6D objects: f1() > f2()

    This class is not intended to be used directly, but rather returned as the result of a __gt__() call on a
    Function6D object.

    :param object function1: A Function6D object or Python callable.
    :param object function2: A Function6D object or Python callable.
    """
    def __init__(self, object function1, object function2):
        self._function1 = autowrap_function6d(function1)
        self._function2 = autowrap_function6d(function2)

    cdef double evaluate(self, double x, double y, double z, double u, double w, double v) except? -1e999:
        return self._function1.evaluate(x, y, z, u, w, v) > self._function2.evaluate(x, y, z, u, w, v)


cdef class LessEqualsFunction6D(Function6D):
    """
    A Function6D class that implements <= of the results of two Function6D objects: f1() <= f2()

    This class is not intended to be used directly, but rather returned as the result of an __le__() call on a
    Function6D object.

    :param object function1: A Function6D object or Python callable.
    :param object function2: A Function6D object or Python callable.
    """
    def __init__(self, object function1, object function2):
        self._function1 = autowrap_function6d(function1)
        self._function2 = autowrap_function6d(function2)

    cdef double evaluate(self, double x, double y, double z, double u, double w, double v) except? -1e999:
        return self._function1.evaluate(x, y, z, u, w, v) <= self._function2.evaluate(x, y, z, u, w, v)


cdef class GreaterEqualsFunction6D(Function6D):
    """
    A Function6D class that implements >= of the results of two Function6D objects: f1() >= f2()

    This class is not intended to be used directly, but rather returned as the result of an __ge__() call on a
    Function6D object.

    :param object function1: A Function6D object or Python callable.
    :param object function2: A Function6D object or Python callable.
    """
    def __init__(self, object function1, object function2):
        self._function1 = autowrap_function6d(function1)
        self._function2 = autowrap_function6d(function2)

    cdef double evaluate(self, double x, double y, double z, double u, double w, double v) except? -1e999:
        return self._function1.evaluate(x, y, z, u, w, v) >= self._function2.evaluate(x, y, z, u, w, v)


cdef class AddScalar6D(Function6D):
    """
    A Function6D class that implements the addition of scalar and the result of a Function6D object: K + f()

    This class is not intended to be used directly, but rather returned as the result of an __add__() call on a
    Function6D object.

    :param value: A double value.
    :param function: A Function6D object or Python callable.
    """

    def __init__(self, double value, object function):
        self._value = value
        self._function = autowrap_function6d(function)

    cdef double evaluate(self, double x, double y, double z, double u, double w, double v) except? -1e999:
        return self._value + self._function.evaluate(x, y, z, u, w, v)


cdef class SubtractScalar6D(Function6D):
    """
    A Function6D class that implements the subtraction of scalar and the result of a Function6D object: K - f()

    This class is not intended to be used directly, but rather returned as the result of an __sub__() call on a
    Function6D object.

    :param value: A double value.
    :param function: A Function6D object or Python callable.
    """

    def __init__(self, double value, object function):
        self._value = value
        self._function = autowrap_function6d(function)

    cdef double evaluate(self, double x, double y, double z, double u, double w, double v) except? -1e999:
        return self._value - self._function.evaluate(x, y, z, u, w, v)


cdef class MultiplyScalar6D(Function6D):
    """
    A Function6D class that implements the multiplication of scalar and the result of a Function6D object: K * f()

    This class is not intended to be used directly, but rather returned as the result of an __mul__() call on a
    Function6D object.

    :param value: A double value.
    :param function: A Function6D object or Python callable.
    """

    def __init__(self, double value, object function):
        self._value = value
        self._function = autowrap_function6d(function)

    cdef double evaluate(self, double x, double y, double z, double u, double w, double v) except? -1e999:
        return self._value * self._function.evaluate(x, y, z, u, w, v)


cdef class DivideScalar6D(Function6D):
    """
    A Function6D class that implements the subtraction of scalar and the result of a Function6D object: K / f()

    This class is not intended to be used directly, but rather returned as the result of an __div__() call on a
    Function6D object.

    :param value: A double value.
    :param function: A Function6D object or Python callable.
    """

    def __init__(self, double value, object function):
        self._value = value
        self._function = autowrap_function6d(function)

    @cython.cdivision(True)
    cdef double evaluate(self, double x, double y, double z, double u, double w, double v) except? -1e999:
        cdef double denominator = self._function.evaluate(x, y, z, u, w, v)
        if denominator == 0.0:
            raise ZeroDivisionError("Function used as the denominator of the division returned a zero value.")
        return self._value / denominator


cdef class ModuloScalarFunction6D(Function6D):
    """
    A Function6D class that implements the modulo of scalar and the result of a Function6D object: K % f()

    This class is not intended to be used directly, but rather returned as the result of a __mod__() call on a
    Function6D object.

    :param float value: A double value.
    :param object function: A Function6D object or Python callable.
    """
    def __init__(self, double value, object function):
        self._value = value
        self._function = autowrap_function6d(function)

    @cython.cdivision(True)
    cdef double evaluate(self, double x, double y, double z, double u, double w, double v) except? -1e999:
        cdef double divisor = self._function.evaluate(x, y, z, u, w, v)
        if divisor == 0.0:
            raise ZeroDivisionError("Function used as the divisor of the modulo returned a zero value.")
        return self._value % divisor


cdef class ModuloFunctionScalar6D(Function6D):
    """
    A Function6D class that implements the modulo of the result of a Function6D object and a scalar: f() % K

    This class is not intended to be used directly, but rather returned as the result of a __mod__() call on a
    Function6D object.

    :param object function: A Function6D object or Python callable.
    :param float value: A double value.
    """
    def __init__(self, object function, double value):
        if value == 0:
            raise ValueError("Divisor cannot be zero")
        self._value = value
        self._function = autowrap_function6d(function)

    @cython.cdivision(True)
    cdef double evaluate(self, double x, double y, double z, double u, double w, double v) except? -1e999:
        return self._function.evaluate(x, y, z, u, w, v) % self._value


cdef class PowScalarFunction6D(Function6D):
    """
    A Function6D class that implements the pow of scalar and the result of a Function6D object: K ** f()

    This class is not intended to be used directly, but rather returned as the result of an __pow__() call on a
    Function6D object.

    :param float value: A double value.
    :param object function: A Function6D object or Python callable.
    """
    def __init__(self, double value, object function):
        self._value = value
        self._function = autowrap_function6d(function)

    cdef double evaluate(self, double x, double y, double z, double u, double w, double v) except? -1e999:
        cdef double exponent = self._function.evaluate(x, y, z, u, w, v)
        if self._value < 0 and floor(exponent) != exponent:
            raise ValueError("Negative base and non-integral exponent is not supported")
        if self._value == 0 and exponent < 0:
            raise ZeroDivisionError("0.0 cannot be raised to a negative power")
        return self._value ** exponent


cdef class PowFunctionScalar6D(Function6D):
    """
    A Function6D class that implements the pow of the result of a Function6D object and a scalar: f() ** K

    This class is not intended to be used directly, but rather returned as the result of an __pow__() call on a
    Function6D object.

    :param object function: A Function6D object or Python callable.
    :param float value: A double value.
    """
    def __init__(self, object function, double value):
        self._value = value
        self._function = autowrap_function6d(function)

    cdef double evaluate(self, double x, double y, double z, double u, double w, double v) except? -1e999:
        cdef double base = self._function.evaluate(x, y, z, u, w, v)
        if base < 0 and floor(self._value) != self._value:
            raise ValueError("Negative base and non-integral exponent is not supported")
        if base == 0 and self._value < 0:
            raise ZeroDivisionError("0.0 cannot be raised to a negative power")
        return base ** self._value


cdef class EqualsScalar6D(Function6D):
    """
    A Function6D class that tests the equality of a scalar and the result of a Function6D object: K == f2()

    This class is not intended to be used directly, but rather returned as the result of an __eq__() call on a
    Function6D object.

    :param value: A double value.
    :param object function: A Function6D object or Python callable.
    """
    def __init__(self, double value, object function):
        self._value = value
        self._function = autowrap_function6d(function)

    cdef double evaluate(self, double x, double y, double z, double u, double w, double v) except? -1e999:
        return self._value == self._function.evaluate(x, y, z, u, w, v)


cdef class NotEqualsScalar6D(Function6D):
    """
    A Function6D class that tests the inequality of a scalar and the result of a Function6D object: K != f2()

    This class is not intended to be used directly, but rather returned as the result of an __ne__() call on a
    Function6D object.

    :param value: A double value.
    :param object function: A Function6D object or Python callable.
    """
    def __init__(self, double value, object function):
        self._value = value
        self._function = autowrap_function6d(function)

    cdef double evaluate(self, double x, double y, double z, double u, double w, double v) except? -1e999:
        return self._value != self._function.evaluate(x, y, z, u, w, v)


cdef class LessThanScalar6D(Function6D):
    """
    A Function6D class that implements < of a scalar and the result of a Function6D object: K < f2()

    This class is not intended to be used directly, but rather returned as the result of an __lt__() call on a
    Function6D object.

    :param value: A double value.
    :param object function: A Function6D object or Python callable.
    """
    def __init__(self, double value, object function):
        self._value = value
        self._function = autowrap_function6d(function)

    cdef double evaluate(self, double x, double y, double z, double u, double w, double v) except? -1e999:
        return self._value < self._function.evaluate(x, y, z, u, w, v)


cdef class GreaterThanScalar6D(Function6D):
    """
    A Function6D class that implements > of a scalar and the result of a Function6D object: K > f2()

    This class is not intended to be used directly, but rather returned as the result of a __gt__() call on a
    Function6D object.

    :param value: A double value.
    :param object function: A Function6D object or Python callable.
    """
    def __init__(self, double value, object function):
        self._value = value
        self._function = autowrap_function6d(function)

    cdef double evaluate(self, double x, double y, double z, double u, double w, double v) except? -1e999:
        return self._value > self._function.evaluate(x, y, z, u, w, v)


cdef class LessEqualsScalar6D(Function6D):
    """
    A Function6D class that implements <= of a scalar and the result of a Function6D object: K <= f2()

    This class is not intended to be used directly, but rather returned as the result of an __le__() call on a
    Function6D object.

    :param value: A double value.
    :param object function: A Function6D object or Python callable.
    """
    def __init__(self, double value, object function):
        self._value = value
        self._function = autowrap_function6d(function)

    cdef double evaluate(self, double x, double y, double z, double u, double w, double v) except? -1e999:
        return self._value <= self._function.evaluate(x, y, z, u, w, v)


cdef class GreaterEqualsScalar6D(Function6D):
    """
    A Function6D class that implements >= of a scalar and the result of a Function6D object: K >= f2()

    This class is not intended to be used directly, but rather returned as the result of an __ge__() call on a
    Function6D object.

    :param value: A double value.
    :param object function: A Function6D object or Python callable.
    """
    def __init__(self, double value, object function):
        self._value = value
        self._function = autowrap_function6d(function)

    cdef double evaluate(self, double x, double y, double z, double u, double w, double v) except? -1e999:
        return self._value >= self._function.evaluate(x, y, z, u, w, v)
