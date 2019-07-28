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

cdef class Constant1D(Function1D):
    """Constant 1D real function.

    Inherits from Function1D, implements `__call__(x)`.

    .. code-block:: pycon

       >>> from cherab.core.math import Constant1D
       >>>
       >>> f = Constant1D(5.3)
       >>>
       >>> f(0.3)
       5.3
       >>> f(-1.3e6)
       5.3
    """

    def __init__(self, double value):
        self.value = value

    cdef double evaluate(self, double x) except? -1e999:
        return self.value


cdef class Constant2D(Function2D):
    """Constant 2D real function.

    Inherits from Function2D, implements `__call__(x, y)`.

    .. code-block:: pycon

       >>> from cherab.core.math import Constant2D
       >>>
       >>> f = Constant2D(-7.9)
       >>>
       >>> f(3.7, -8.3)
       -7.9
       >>> f(3e999, -5)
       -7.9
    """

    def __init__(self, double value):
        self.value = value

    cdef double evaluate(self, double x, double y) except? -1e999:
        return self.value


cdef class Constant3D(Function3D):
    """Constant 3D real function.

    Inherits from Function3D, implements `__call__(x, y, z)`.

    .. code-block:: pycon

       >>> from cherab.core.math import Constant3D
       >>>
       >>> f = Constant3D(0)
       >>>
       >>> f(1,2,3)
       0.0
       >>> f(-3, 100, -7e999)
       0.0
    """

    def __init__(self, double value):
        self.value = value

    cdef double evaluate(self, double x, double y, double z) except? -1e999:
        return self.value


cdef class ConstantVector2D(VectorFunction2D):
    """Constant 2D real vector function.

    Inherits from Function2D, implements `__call__(x, y)`.

    .. code-block:: pycon

       >>> from raysect.core.math import Vector3D
       >>> from cherab.core.math import ConstantVector2D
       >>>
       >>> f2 = ConstantVector2D(Vector3D(0.5, 0.5, 0.5))
       >>>
       >>> f2(1, 1)
       Vector3D(0.5, 0.5, 0.5)
       >>> f2(-1, 7e6)
       Vector3D(0.5, 0.5, 0.5)
    """

    def __init__(self, Vector3D value not None):
        self.value = value

    cdef Vector3D evaluate(self, double x, double y):
        return self.value


cdef class ConstantVector3D(VectorFunction3D):
    """Constant 3D real vector function.

    Inherits from Function3D, implements `__call__(x, y, z)`.

    .. code-block:: pycon

       >>> from raysect.core.math import Vector3D
       >>> from cherab.core.math import ConstantVector3D
       >>>
       >>> f3 = ConstantVector3D(Vector3D(0.5, 0.5, 0.5))
       >>>
       >>> f3(1, 1, 6)
       Vector3D(0.5, 0.5, 0.5)
       >>> f3(-1, 7e6, -1e999)
       Vector3D(0.5, 0.5, 0.5)
    """

    def __init__(self, Vector3D value not None):
        self.value = value

    cdef Vector3D evaluate(self, double x, double y, double z):
        return self.value
