from raysect.core.math.function cimport Function3D
from raysect.optical cimport Spectrum
from raysect.core.math.cython.utility cimport find_index

from libc.math cimport sqrt, exp, pi


cdef class AxisymmetricGaussian3D(Function3D):

    def __init__(self, sigma = None):

        super().__init__()

        self.sigma = sigma

    @property
    def sigma(self):

        return self._sigma

    @sigma.setter
    def sigma(self, value):

        self._sigma = value
        self._recip_negative_2sigma2 = -1 / (2 * value ** 2)
        self._recip_sqrt_2pisigma2 = 1 / sqrt(2 * pi * value ** 2)

    cdef double evaluate(self, double x, double y, double z) except? -1e999:
        cdef:
            double r2

        r2 = x ** 2 + y ** 2
        return self._recip_sqrt_2pisigma2 * exp(r2 * self._recip_negative_2sigma2)


cdef class GaussianBeamModel(Function3D):

    def __init__(self, double wavelength, double z_focus, double focus_width, double m2=1):

        self.wavelength = wavelength
        self.z_focus = z_focus
        self.focus_width = focus_width
        self.m2 = m2

    @property
    def wavelength (self):
        return self._wavelength

    @wavelength.setter
    def wavelength(self, double value):
        if not value > 0:
            raise ValueError("Value has to be larger than 0, but {0} passed.".format(value))
        
        self._wavelength = value
        self._cache_constants()

    @property
    def z_focus (self):
        return self._z_focus

    @z_focus.setter
    def z_focus(self, double value):
        self._z_focus = value
        self._cache_constants()
    
    @property
    def focus_width(self):
        return self._focus_width

    @focus_width.setter
    def focus_width(self, double value):
        if not value > 0:
            raise ValueError("Value has to be larger than 0, but {0} passed.".format(value))
    
        self._focus_width = value
    
    @property
    def m2(self):
        return self._m2

    @m2.setter
    def m2(self, double value):
        if not value > 0:
            raise ValueError("Value has to be larger than 0, but {0} passed.".format(value))
        
        self._m2 = value
        self._cache_constants(self)

    def _cache_constants(self):
        self._wavelengthm_div_pi = (self._wavelength * sqrt(self._m2)) / pi
        self._focus_width2m = self._focus_width ** 2 * sqrt(self._m2)

    cdef double evaluate(self, double x, double y, double z) except? -1e999:

        cdef:
            double r2, w2_z

        r2 = x ** 2 + y ** 2
        w2_z = self._focus_width2m + self._wavelengthm_div_pi * (z - self._focus_z) 

        return 1 - exp(-2 * r2 / w2_z)