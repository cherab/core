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

from raysect.core.math.function.float cimport Function1D, autowrap_function2d


cdef class Integrator2D:
    """
    Compute a definite integral of a two-dimensional function.

    :ivar Function2D integrand: A 2D function to integrate.
    """

    @property
    def integrand(self):
        """
        A 2D function to integrate.

        :rtype: int
        """
        return self.function

    @integrand.setter
    def integrand(self, object func not None):

        self.function = autowrap_function2d(func)

    cdef double evaluate(self, double x_lower, double x_upper, Function1D y_lower, Function1D y_upper) except? -1e999:

        raise NotImplementedError("The evaluate() method has not been implemented.")

    def __call__(self, double x_lower, double x_upper, Function1D y_lower, Function1D y_upper):
        """
        Integrates a two-dimensional function over a finite interval.

        :param double x_lower: Lower limit of integration in the x dimension.
        :param double x_upper: Upper limit of integration in the x dimension.
        :param Function1D y_lower: Lower limit of integration in the y dimension.
        :param Function1D y_upper: Upper limit of integration in the y dimension.

        :returns: Definite integral of a two-dimensional function.
        """

        return self.evaluate(x_lower, x_upper, y_lower, y_upper)
