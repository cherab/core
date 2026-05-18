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

from cherab.core.math.function.float.function6d.base cimport Function6D


cdef class Constant6D(Function6D):
    """
    Wraps a scalar constant with a Function6D object.

    This class allows a numeric Python scalar, such as a float or an integer, to
    interact with cython code that requires a Function6D object. The scalar must
    be convertible to double. The value of the scalar constant will be returned
    independent of the arguments the function is called with.

    This class is intended to be used to transparently wrap python objects that
    are passed via constructors or methods into cython optimised code. It is not
    intended that the users should need to directly interact with these wrapping
    objects. Constructors and methods expecting a Function6D object should be
    designed to accept a generic python object and then test that object to
    determine if it is an instance of Function6D. If the object is not a
    Function6D object it should be wrapped using this class for internal use.

    See also: autowrap_function6d()

    :param float value: the constant value to return when called
    """
    def __init__(self, double value):
        self._value = value

    cdef double evaluate(self, double x, double y, double z, double u, double w, double v) except? -1e999:
        return self._value
