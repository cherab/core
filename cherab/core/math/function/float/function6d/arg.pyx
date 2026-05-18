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


cdef class Arg6D(Function6D):
    """
    Returns one of the arguments the function is passed, unmodified

    This is used to pass coordinates through to other functions in the
    function framework which expect a Function6D object.

    Valid options for argument are "x", "y", "z", "u", "w", or "v".

    >>> argx = Arg6D("x")
    >>> argx(2, 3, 5, 7, 11, 13)
    2.0
    >>> argy = Arg6D("y")
    >>> argy(2, 3, 5, 7, 11, 13)
    3.0
    >>> argz = Arg6D("z")
    >>> argz(2, 3, 5, 7, 11, 13)
    5.0
    >>> argu = Arg6D("u")
    >>> argu(2, 3, 5, 7, 11, 13)
    7.0
    >>> argw = Arg6D("w")
    >>> argw(2, 3, 5, 7, 11, 13)
    11.0
    >>> argv = Arg6D("v")
    >>> argv(2, 3, 5, 7, 11, 13)
    13.0

    :param str argument: either "x", "y", "z", "u", "w", or "v", the argument to return
    """
    def __init__(self, object argument):
        if argument == "x":
            self._argument = X
        elif argument == "y":
            self._argument = Y
        elif argument == "z":
            self._argument = Z
        elif argument == "u":
            self._argument = U
        elif argument == "w":
            self._argument = W
        elif argument == "v":
            self._argument = V
        else:
            raise ValueError("The argument to Arg6D must be either 'x', 'y', 'z', 'u', 'w' or 'v'")

    cdef double evaluate(self, double x, double y, double z, double u, double w, double v) except? -1e999:
        if self._argument == X:
            return x
        elif self._argument == Y:
            return y
        elif self._argument == Z:
            return z
        elif self._argument == U:
            return u
        elif self._argument == W:
            return w
        else:  # V
            return v
