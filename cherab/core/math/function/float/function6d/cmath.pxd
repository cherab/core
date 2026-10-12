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


cdef class Exp6D(Function6D):
    cdef Function6D _function


cdef class Sin6D(Function6D):
    cdef Function6D _function


cdef class Cos6D(Function6D):
    cdef Function6D _function


cdef class Tan6D(Function6D):
    cdef Function6D _function


cdef class Asin6D(Function6D):
    cdef Function6D _function


cdef class Acos6D(Function6D):
    cdef Function6D _function


cdef class Atan6D(Function6D):
    cdef Function6D _function


cdef class Atan4Q6D(Function6D):
    cdef Function6D _numerator, _denominator


cdef class Sqrt6D(Function6D):
    cdef Function6D _function


cdef class Erf6D(Function6D):
    cdef Function6D _function
