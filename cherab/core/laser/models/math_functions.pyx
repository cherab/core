from raysect.core.math.function.float cimport Function3D
from raysect.optical cimport Spectrum
from raysect.core.math.cython.utility cimport find_index

from libc.math cimport sqrt, exp, pi


cdef class ConstantAxisymmetricGaussian3D(Function3D):

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
        self._recip_negative_2stddev2 = -1 / (2 * value ** 2)
        self._recip_2pistddev2 = 1 / (2 * pi * value ** 2)

    cdef double evaluate(self, double x, double y, double z) except? -1e999:
        cdef:
            double r2
        r2 = x ** 2 + y ** 2
        return self._recip_2pistddev2 * exp(r2 * self._recip_negative_2stddev2)


cdef class ConstantBivariateGaussian3D(Function3D):

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
        self._negative_recip_2stddevz2 = -1 / (2 * self._stddev_z ** 2)
        self._normalisationstddevz = 1 / (sqrt((2 * pi) ** 3) * self._stddev_x * self._stddev_y * self._stddev_z)

    cdef double evaluate(self, double x, double y, double z) except? -1e999:
        return self._normalisationstddevz * exp(x ** 2 * self._kx +
                                                          y ** 2 * self._ky +
                                                          (z - self._mean_z) ** 2 * self._negative_recip_2stddevz2)


cdef class GaussianBeamModel(Function3D):

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
        if not value > 0:
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
        if not value > 0:
            raise ValueError("Value has to be larger than 0, but {0} passed.".format(value))

        self._stddev_waist = value
        self._stddev_waist2 = self._stddev_waist ** 2
        self._cache_constants()

    def _cache_constants(self):

        n = 1  # refractive index of vacuum
        self._rayleigh_range = 2 * pi * n * self._stddev_waist2 / self._wavelength / 1e-9

    cdef double evaluate(self, double x, double y, double z) except? -1e999:

        cdef:
            double r2, stddev_z2, z_prime

        # shift to correct gaussiam beam model coords, it works with waist at z=0
        z_prime = z - self._waist_z

        r2 = x ** 2 + y ** 2
        stddev_z2 = self._stddev_waist2 * (1 + ((z_prime) / self._rayleigh_range) ** 2)

        return 1 / (2 * pi * stddev_z2) * exp(r2 / (-2 * stddev_z2))
