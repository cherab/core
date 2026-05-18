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

from raysect.core.math.function.base cimport Function
from raysect.core.math.function.float.base cimport FloatFunction


cdef class Function6D(FloatFunction):
    cdef double evaluate(self, double x, double y, double z, double u, double w, double v) except? -1e999


cdef class AddFunction6D(Function6D):
    cdef Function6D _function1, _function2


cdef class SubtractFunction6D(Function6D):
    cdef Function6D _function1, _function2


cdef class MultiplyFunction6D(Function6D):
    cdef Function6D _function1, _function2


cdef class DivideFunction6D(Function6D):
    cdef Function6D _function1, _function2


cdef class ModuloFunction6D(Function6D):
    cdef Function6D _function1, _function2


cdef class PowFunction6D(Function6D):
    cdef Function6D _function1, _function2


cdef class AbsFunction6D(Function6D):
    cdef Function6D _function


cdef class EqualsFunction6D(Function6D):
    cdef Function6D _function1, _function2


cdef class NotEqualsFunction6D(Function6D):
    cdef Function6D _function1, _function2


cdef class LessThanFunction6D(Function6D):
    cdef Function6D _function1, _function2


cdef class GreaterThanFunction6D(Function6D):
    cdef Function6D _function1, _function2


cdef class LessEqualsFunction6D(Function6D):
    cdef Function6D _function1, _function2


cdef class GreaterEqualsFunction6D(Function6D):
    cdef Function6D _function1, _function2


cdef class AddScalar6D(Function6D):
    cdef double _value
    cdef Function6D _function


cdef class SubtractScalar6D(Function6D):
    cdef double _value
    cdef Function6D _function


cdef class MultiplyScalar6D(Function6D):
    cdef double _value
    cdef Function6D _function


cdef class DivideScalar6D(Function6D):
    cdef double _value
    cdef Function6D _function


cdef class ModuloScalarFunction6D(Function6D):
    cdef double _value
    cdef Function6D _function


cdef class ModuloFunctionScalar6D(Function6D):
    cdef double _value
    cdef Function6D _function


cdef class PowScalarFunction6D(Function6D):
    cdef double _value
    cdef Function6D _function


cdef class PowFunctionScalar6D(Function6D):
    cdef double _value
    cdef Function6D _function


cdef class EqualsScalar6D(Function6D):
    cdef double _value
    cdef Function6D _function


cdef class NotEqualsScalar6D(Function6D):
    cdef double _value
    cdef Function6D _function


cdef class LessThanScalar6D(Function6D):
    cdef double _value
    cdef Function6D _function


cdef class GreaterThanScalar6D(Function6D):
    cdef double _value
    cdef Function6D _function


cdef class LessEqualsScalar6D(Function6D):
    cdef double _value
    cdef Function6D _function


cdef class GreaterEqualsScalar6D(Function6D):
    cdef double _value
    cdef Function6D _function


cdef inline bint is_callable(object f):
    """
    Tests if an object is a python callable or a Function6D object.

    :param object f: Object to test.
    :return: True if callable, False otherwise.
    """
    if isinstance(f, Function6D):
        return True

    # other function classes are incompatible
    if isinstance(f, Function):
        return False

    return callable(f)