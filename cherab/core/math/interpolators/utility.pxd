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

cimport cython

cdef inline int find_index(double[::1] x_view, int size, double v, double padding=*)

cdef inline double[::1] derivatives_array(double v, int deriv)

cdef int factorial(int n)

@cython.cdivision(True)
cdef inline double lerp(double x0, double x1, double y0, double y1, double x):
    return ((y1 - y0) / (x1 - x0)) * (x - x0) + y0

cdef int factorial(int n)
