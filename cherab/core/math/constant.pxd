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

from raysect.optical cimport Vector3D
from cherab.core.math.function cimport Function1D, Function2D, Function3D, VectorFunction2D, VectorFunction3D


cdef class Constant1D(Function1D):
    cdef double value


cdef class Constant2D(Function2D):
    cdef double value


cdef class Constant3D(Function3D):
    cdef double value


cdef class ConstantVector2D(VectorFunction2D):
    cdef Vector3D value


cdef class ConstantVector3D(VectorFunction3D):
    cdef Vector3D value
