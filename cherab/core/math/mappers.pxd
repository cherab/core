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

from cherab.core.math.function cimport Function1D, Function2D, Function3D, VectorFunction2D, VectorFunction3D
from raysect.core cimport Vector3D


cdef class IsoMapper2D(Function2D):

    cdef:
        readonly Function1D function1d
        readonly Function2D function2d

    cdef double evaluate(self, double x, double y) except? -1e999


cdef class IsoMapper3D(Function3D):

    cdef:
        readonly Function3D function3d
        readonly Function1D function1d

    cdef double evaluate(self, double x, double y, double z) except? -1e999


cdef class Swizzle2D(Function2D):

    cdef readonly Function2D function2d

    cdef double evaluate(self, double x, double y) except? -1e999


cdef class Swizzle3D(Function3D):

    cdef:
        readonly Function3D function3d
        int shape[3]

    cdef double evaluate(self, double x, double y, double z) except? -1e999


cdef class AxisymmetricMapper(Function3D):

    cdef readonly Function2D function2d

    cdef double evaluate(self, double x, double y, double z) except? -1e999


cdef class VectorAxisymmetricMapper(VectorFunction3D):

    cdef readonly VectorFunction2D function2d

    cdef Vector3D evaluate(self, double x, double y, double z)