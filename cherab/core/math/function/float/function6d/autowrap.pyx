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
from cherab.core.math.function.float.function6d.base cimport Function6D
from cherab.core.math.function.float.function6d.constant cimport Constant6D
from raysect.core.math.function.base cimport Function


cdef class PythonFunction6D(Function6D):
    """
    Wraps a python callable object with a Function6D object.

    This class allows a python object to interact with cython code that requires
    a Function6D object. The python object must implement __call__() expecting
    six arguments.

    This class is intended to be used to transparently wrap python objects that
    are passed via constructors or methods into cython optimised code. It is not
    intended that the users should need to directly interact with these wrapping
    objects. Constructors and methods expecting a Function6D object should be
    designed to accept a generic python object and then test that object to
    determine if it is an instance of Function6D. If the object is not a
    Function6D object it should be wrapped using this class for internal use.

    See also: autowrap_function6d()
    """

    def __init__(self, object function):
        self.function = function

    cdef double evaluate(self, double x, double y, double z, double u, double w, double v) except? -1e999:
        return self.function(x, y, z, u, w, v)


cdef Function6D autowrap_function6d(object obj):
    """
    Automatically wraps the supplied python object in a PythonFunction6D or Constant6D object.

    If this function is passed a valid Function6D object, then the Function6D
    object is simply returned without wrapping.

    If this function is passed a numerical scalar (int or float), a Constant6D
    object is returned.

    This convenience function is provided to simplify the handling of Function6D
    and python callable objects in constructors, functions and setters.
    """

    if isinstance(obj, Function6D):
        return <Function6D> obj
    elif isinstance(obj, Function):
        raise TypeError('A Function6D object is required.')
    elif isinstance(obj, numbers.Real):
        return Constant6D(obj)
    else:
        return PythonFunction6D(obj)


def _autowrap_function6d(obj):
    """Expose cython function for testing."""
    return autowrap_function6d(obj)


cdef inline bint is_callable(object f):
    """
    Tests if an object is a python callable or a Function6D object.
    """
    print(f"Checking if callable:", f)
    print(f"isinstance(Function6D):", isinstance(f, Function6D))
    print(f"isinstance(Function):", isinstance(f, Function))
    print(f"callable():", callable(f))
    
    if isinstance(f, Function6D):
        return True

    # other function classes are incompatible
    if isinstance(f, Function):
        return False

    return callable(f)
