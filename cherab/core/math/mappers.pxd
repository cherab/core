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

from libc.math cimport fmod
from raysect.core.math.function.float cimport Function1D, Function2D, Function3D
from raysect.core.math.function.vector3d cimport Function1D as VectorFunction1D
from raysect.core.math.function.vector3d cimport Function2D as VectorFunction2D
from raysect.core.math.function.vector3d cimport Function3D as VectorFunction3D
from raysect.core cimport Vector3D


cdef class IsoMapper2D(Function2D):

    cdef:
        readonly Function1D function1d
        readonly Function2D function2d


cdef class IsoMapper3D(Function3D):

    cdef:
        readonly Function3D function3d
        readonly Function1D function1d


cdef class Swizzle2D(Function2D):

    cdef readonly Function2D function2d


cdef class Swizzle3D(Function3D):

    cdef:
        readonly Function3D function3d
        int shape[3]


cdef class AxisymmetricMapper(Function3D):

    cdef readonly Function2D function2d


cdef class VectorAxisymmetricMapper(VectorFunction3D):

    cdef readonly VectorFunction2D function2d


cdef class CylindricalMapper(Function3D):

    cdef readonly Function3D function3d


cdef class VectorCylindricalMapper(VectorFunction3D):

    cdef readonly VectorFunction3D function3d


cdef inline double remainder(double x1, double x2) nogil:
    if x2 == 0:
        return x1
    x1 = fmod(x1, x2)
    return x1 + x2 if (x1 < 0) else x1


cdef class PeriodicMapper1D(Function1D):

    cdef:
        readonly Function1D function1d
        readonly double period


cdef class PeriodicMapper2D(Function2D):

    cdef:
        readonly Function2D function2d
        double period_x, period_y


cdef class PeriodicMapper3D(Function3D):

    cdef:
        readonly Function3D function3d
        readonly double period_x, period_y, period_z


cdef class VectorPeriodicMapper1D(VectorFunction1D):

    cdef:
        readonly VectorFunction1D function1d
        readonly double period


cdef class VectorPeriodicMapper2D(VectorFunction2D):

    cdef:
        readonly VectorFunction2D function2d
        readonly double period_x, period_y


cdef class VectorPeriodicMapper3D(VectorFunction3D):

    cdef:
        readonly VectorFunction3D function3d
        readonly double period_x, period_y, period_z
