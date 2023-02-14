# Copyright 2016-2021 Euratom
# Copyright 2016-2021 United Kingdom Atomic Energy Authority
# Copyright 2016-2021 Centro de Investigaciones Energéticas, Medioambientales y Tecnológicas
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

from raysect.core.math.function.float cimport Function3D
from raysect.core.math.cython.utility cimport find_index

from libc.math cimport sqrt, exp, pi


cdef class ConstantAxisymmetricGaussian3D(Function3D):
    """
    A function with a 2D Gaussian in the x-y plane and equal standard deviations in x and y directions.

    .. math::
    F(x, y, z) = \\frac{1}{2 * \\pi \\sigma^2} exp\\left(-\\frac{x^2 + y^2}{2 * \\sigma^2}\\right)

    The function value has a Gaussian shape in the x-y plane with the standard deviations in
    x and y direction being equal. The integral over an x-y plane is equal to 1 
    and the mean values in x and y directions are equal to 0.

    :param float stddev: The standard deviation in both the x and y directions.
    """

    def __init__(self, stddev):

        super().__init__()

        self.stddev = stddev

    @property
    def stddev(self):

        return self._stddev

    @stddev.setter
    def stddev(self, value):
        if value <= 0:
            raise ValueError("Value has to be larger than 0.")

        self._stddev = value
        self._kr = -1 / (2 * value ** 2)
        self._normalisation = 1 / (2 * pi * value ** 2)

    cdef double evaluate(self, double x, double y, double z) except? -1e999:
        cdef:
            double r2
        r2 = x ** 2 + y ** 2
        return self._normalisation * exp(r2 * self._kr)


cdef class ConstantBivariateGaussian3D(Function3D):
    """
    A function with a 2D Gaussian in the x-y plane.

    .. math::
    F(x, y, z) = \\frac{1}{2 * \\pi \\sigma_x \\sigma_y} exp\\left(-\\frac{x^2 + y^2}{2 * \\sigma_x \\sigma_y}\\right)

    The function value has a Gaussian shape in the x-y plane. The integral over an x-y plane is equal to 1
    and the mean values in x and y directions are equal to 0.
    The correlation between the standard deviations in x and y directions is equal to 0.

    :param float stddev_x: The standard deviation in the x directions.
    :param float stddev_y: The standard deviation in the y directions.
    """

    def __init__(self, stddev_x, stddev_y):

        super().__init__()
        self._init_params()

        self.stddev_x = stddev_x
        self.stddev_y = stddev_y

    def _init_params(self):
        self._stddev_x = 1
        self._stddev_y = 1

    @property
    def stddev_x(self):
        return self._stddev_x

    @stddev_x.setter
    def stddev_x(self, value):
        if value <= 0:
            raise ValueError("Value has to be larger than 0.")

        self._stddev_x = value

        self._cache_constants()

    @property
    def stddev_y(self):
        return self._stddev_y

    @stddev_y.setter
    def stddev_y(self, value):
        if value <= 0:
            raise ValueError("Value has to be larger than 0.")

        self._stddev_y = value

        self._cache_constants()

    def _cache_constants(self):
        self._kx = -1 / (2 * self._stddev_x ** 2)
        self._ky = -1 / (2 * self._stddev_y ** 2)
        self._normalisation = 1 / (2 * pi * self._stddev_x * self._stddev_y)

    cdef double evaluate(self, double x, double y, double z) except? -1e999:
        return self._normalisation * exp(x ** 2 * self._kx +
                                                   y ** 2 * self._ky)


cdef class TrivariateGaussian3D(Function3D):
    """
    A function with a 3D Gaussian shape.

    .. math::
    F(x, y, z) = \\frac{1}{\\sqrt{2 \\pi^3} \\sigma_x \\sigma_y \\sigma_z} exp\\left(-\\frac{x^2}{2 \\sigma_x^2} -\\frac{y^2}{2 \\sigma_y^2} - \\frac{(z - \\mu_z)^2}{2 \\sigma_z^2}\\right)

    The integral over the whole 3D space is equal to 1.The correlation between the standard deviations in x and y directions is equal to 0. The mean value in the
    x and y directions are equal to 0.

    :param float mean_z: Mean value in the z direction.
    :param float stddev_x: The standard deviation in the x directions.
    :param float stddev_y: The standard deviation in the y directions.
    :param float stddev_z: The standard deviation in the z directions.
    """

    def __init__(self, mean_z, stddev_x, stddev_y, stddev_z):

        super().__init__()
        self._init_params()

        self.stddev_x = stddev_x
        self.stddev_y = stddev_y
        self.stddev_z = stddev_z
        self.mean_z = mean_z

    def _init_params(self):
        self._mean_z = 1
        self._stddev_x = 1
        self._stddev_y = 1
        self._stddev_z = 1

    @property
    def stddev_x(self):
        return self._stddev_x

    @stddev_x.setter
    def stddev_x(self, double value):
        if value <= 0:
            raise ValueError("Value has to be larger than 0.")

        self._stddev_x = value

        self._cache_constants()

    @property
    def stddev_y(self):
        return self._stddev_y

    @stddev_y.setter
    def stddev_y(self, double value):
        if value <= 0:
            raise ValueError("Value has to be larger than 0.")

        self._stddev_y = value

        self._cache_constants()

    @property
    def stddev_z(self):
        return self._stddev_z

    @stddev_z.setter
    def stddev_z(self, double value):
        if value <= 0:
            raise ValueError("Value has to be larger than 0.")

        self._stddev_z = value

        self._cache_constants()

    @property
    def mean_z(self):
        return self._mean_z

    @mean_z.setter
    def mean_z(self, double value):
        self._mean_z = value

    def _cache_constants(self):
        self._kx = -1 / (2 * self._stddev_x ** 2)
        self._ky = -1 / (2 * self._stddev_y ** 2)
        self._kz = -1 / (2 * self._stddev_z ** 2)
        self._normalisation = 1 / (sqrt((2 * pi) ** 3) * self._stddev_x * self._stddev_y * self._stddev_z)

    cdef double evaluate(self, double x, double y, double z) except? -1e999:
        return self._normalisation * exp(x ** 2 * self._kx +
                                         y ** 2 * self._ky +
                                        (z - self._mean_z) ** 2 * self._kz)


cdef class GaussianBeamModel(Function3D):
    """
    A Gaussian beam function (https://en.wikipedia.org/wiki/Gaussian_beam)

    .. math::
      F(x, y, z) = \\frac{1}{2 \\pi \\sigma^2_z} exp\\left( -\\frac{x^2 + y^2}{2 \\sigma_z(z)^2 }\\right)

    where the standard deviation in the z direction

    .. math::
      \\sigma_z(z) = \\sigma_0 \\sqrt{1 + \\left(\\frac{z - z_0}{z_R}\\right)^2}

    is a function of position and the

    .. math::
      z_R = \\frac{\\pi \\omega_0^2 n}{\\lambda_l}

    is the Rayleigh range.
    """

    def __init__(self, double wavelength, double waist_z, double stddev_waist):

        # preset default values
        self._wavelength = 1e3
        self._waist_z = 0
        self._stddev_waist = 1e-3

        self.wavelength = wavelength
        self.waist_z = waist_z
        self.stddev_waist = stddev_waist

    @property
    def wavelength(self):
        return self._wavelength

    @wavelength.setter
    def wavelength(self, double value):
        if value <= 0:
            raise ValueError("Value has to be larger than 0, but {0} passed.".format(value))

        self._wavelength = value
        self._cache_constants()

    @property
    def waist_z(self):
        return self._waist_z

    @waist_z.setter
    def waist_z(self, double value):
        self._waist_z = value

    @property
    def stddev_waist(self):
        return self._stddev_waist

    @stddev_waist.setter
    def stddev_waist(self, double value):
        if value <= 0:
            raise ValueError("Value has to be larger than 0, but {0} passed.".format(value))

        self._stddev_waist = value
        self._stddev_waist2 = self._stddev_waist ** 2
        self._cache_constants()

    def _cache_constants(self):

        n = 1  # refractive index of vacuum
        self._rayleigh_range = 2 * pi * n * self._stddev_waist2 / self._wavelength / 1e-9

    @cython.cdivision(True)
    cdef double evaluate(self, double x, double y, double z) except? -1e999:

        cdef:
            double r2, stddev_z2, z_prime

        # shift to correct gaussiam beam model coords, it works with waist at z=0
        z_prime = z - self._waist_z

        r2 = x ** 2 + y ** 2
        stddev_z2 = self._stddev_waist2 * (1 + ((z_prime) / self._rayleigh_range) ** 2)

        return 1 / (2 * pi * stddev_z2) * exp(r2 / (-2 * stddev_z2))
