# Copyright 2014-2017 United Kingdom Atomic Energy Authority
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

from raysect.core.math.vector cimport Vector3D
from raysect.core.math.function cimport Function2D


cdef class VectorFunction2D:

    cdef Vector3D evaluate(self, double x, double y)


cdef class PythonVectorFunction2D(VectorFunction2D):

    cdef public object function


cdef inline VectorFunction2D autowrap_vectorfunction2d(object function)


cdef class ScalarToVectorFunction2D(VectorFunction2D):

    cdef Function2D _x, _y, _z
