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

from cherab.core.math.function cimport Function1D, Function2D, Function3D


cdef class ClampOutput1D(Function1D):

    cdef:
        Function1D _f
        double _min, _max


cdef class ClampOutput2D(Function2D):

    cdef:
        Function2D _f
        double _min, _max


cdef class ClampOutput3D(Function3D):

    cdef:
        Function3D _f
        double _min, _max


cdef class ClampInput1D(Function1D):

    cdef:
        Function1D _f
        double _xmin, _xmax


cdef class ClampInput2D(Function2D):

    cdef:
        Function2D _f
        double _xmin, _xmax, _ymin, _ymax


cdef class ClampInput3D(Function3D):

    cdef:
        Function3D _f
        double _xmin, _xmax, _ymin, _ymax, _zmin, _zmax
