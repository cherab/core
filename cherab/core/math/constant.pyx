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
    """Constant 1D real function. Do not propagate NaN values."""

    def __init__(self, double value):
        self.value = value

    cdef double evaluate(self, double x) except? -1e999:
        return self.value


cdef class Constant2D(Function2D):
    """Constant 2D real function. Do not propagate NaN values."""

    def __init__(self, double value):
        self.value = value

    cdef double evaluate(self, double x, double y) except? -1e999:
        return self.value


cdef class Constant3D(Function3D):
    """Constant 3D real function. Do not propagate NaN values."""

    def __init__(self, double value):
        self.value = value

    cdef double evaluate(self, double x, double y, double z) except? -1e999:
        return self.value


cdef class ConstantVector2D(VectorFunction2D):
    """Constant 2D real vector function. Do not propagate NaN values."""

    def __init__(self, Vector3D value not None):
        self.value = value

    cdef Vector3D evaluate(self, double x, double y):
        return self.value


cdef class ConstantVector3D(VectorFunction3D):
    """Constant 3D real vector function. Do not propagate NaN values."""

    def __init__(self, Vector3D value not None):
        self.value = value

    cdef Vector3D evaluate(self, double x, double y, double z):
        return self.value
