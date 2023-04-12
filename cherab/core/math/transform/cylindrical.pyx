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

from libc.math cimport sqrt, atan2, M_PI

from raysect.core.math cimport Vector3D
from raysect.core.math.function.float cimport autowrap_function3d
from raysect.core.math.function.vector3d cimport autowrap_function3d as autowrap_vectorfunction3d
from raysect.core cimport rotate_z
cimport cython

cdef class CylindricalTransform(Function3D):
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
       >>> from cherab.core.math import CylindricalTransform
       >>>
       >>> def my_func(r, phi, z):
       >>>     return r * cos(phi)
       >>>
       >>> f = CylindricalTransform(my_func)
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


cdef class VectorCylindricalTransform(VectorFunction3D):
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
       >>> from cherab.core.math import VectorCylindricalTransform
       >>>
       >>> def my_vec_func(r, phi, z):
       >>>     v = Vector3D(0, 1, 0)
       >>>     v.length = r * abs(cos(phi))
       >>>     return v
       >>>
       >>> f = VectorCylindricalTransform(my_vec_func)
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
