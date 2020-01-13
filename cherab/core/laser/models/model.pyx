from raysect.core.math.function.function3d import Constant3D
from raysect.optical cimport Spectrum, Vector3D

from cherab.core.math cimport ConstantVector3D
from cherab.core cimport Plasma
from cherab.core.laser.node cimport Laser
from cherab.core.laser.models.model_base cimport LaserModel
from cherab.core.laser.models.math_functions cimport AxisymmetricGaussian3D
from libc.math cimport M_PI, sqrt, exp

cdef class UniformPowerDensity(LaserModel):
    def __init__(self, Laser laser = None, power = 1,  Vector3D polarization = Vector3D(0, 1, 0)):
        super().__init__(laser)
    
        self.set_polarization(polarization)
        self.set_pointing_function(ConstantVector3D(Vector3D(0, 0, 1)))
        self.power = power

    def set_polarization(self, Vector3D value):
        value = value.normalise()
        self.set_polarization_function(ConstantVector3D(value))

    @property
    def power(self):
        return self._power

    @power.setter
    def power(self, double value):
        if not value >0:
            raise ValueError("Value has to be larger than 0.")
        
        self._power = value
        self.set_power_density_function(Constant3D(value))

cdef class GaussianBeamAxisymmetric(LaserModel):
    def __init__(self, Laser laser = None, Vector3D polarization = Vector3D(0, 1, 0), power=1,
                 laser_sigma = 0.01, waist_radius=0.001, beam_quality_factor = 1, focus_z = 0.0, central_wavelength = 1064):

        super().__init__(laser)

        #laser sigma dependent constants
        #set laser constants
        self.polarization = polarization
        self.waist_radius = waist_radius
        self.beam_quality_factor = beam_quality_factor
        self.focus_z = focus_z

    cpdef double get_power_axis(self, double z, double wavelength):
        return self._power_const / self.get_beam_width2(z, wavelength) ** 2

    cpdef Vector3D get_polarization(self, double x, double y, double z):
        """
        Returns vector denoting the laser polarisation.
        
        The point is specified in the laser beam space.
        
        :param x: x coordinate in meters.
        :param y: y coordinate in meters. 
        :param z: z coordinate in meters.
        :return: power density in Wm^-3. 
        """

        return self._polarization_vector

    cpdef double get_beam_width2(self, double z, double wavelength):
        return self._waist2 + wavelength ** 2 * self._waist_const * (z - self._focus_z) ** 2

    @property
    def beam_quality_factor(self):
        return self._beam_quality_factor
    @beam_quality_factor.setter
    def beam_quality_factor(self, value):

        if not value > 0:
            raise ValueError("Laser power has to be larger than 0.")

        self._beam_quality_factor = value
        self._change_waist_const()

    @property
    def polarization(self):
        return self._polarization_vector

    @polarization.setter
    def polarization(self, Vector3D value):

        if value.length == 0:
            raise ValueError("Polarisation can't be a zero length vector.")

        self._polarization_vector = value.normalise()

    @property
    def waist_radius(self):
        return self._waist_radius

    @waist_radius.setter
    def waist_radius(self, double value):

        if not value > 0:
            raise ValueError("Value has to be larger than 0.")

        self._waist_radius = value
        self._waist2 = value ** 2
        self._change_waist_const()

    @property
    def focus_z(self):
        return self._focus_z

    @focus_z.setter
    def focus_z(self, double value):
        self._focus_z = value

    def _change_waist_const(self):
        self._waist_const = (1e-9 * self._beam_quality_factor / (M_PI * self._waist_radius)) ** 2

cdef class AxisymmetricGaussian(LaserModel):
    def __init__(self, Laser laser = None, laser_sigma = 0.01, Vector3D polarization = Vector3D(0, 1, 0)):

        super().__init__(laser)

        self.set_polarization(polarization)
        self.set_pointing_function(ConstantVector3D(Vector3D(0, 0, 1)))
        self.laser_sigma = laser_sigma

        #set laser constants
        self.set_polarization(polarization)

    def set_polarization(self, Vector3D value):
        value = value.normalise()
        self.set_polarization_function(ConstantVector3D(value))

    @property
    def laser_sigma(self):
        return self._laser_sigma

    @laser_sigma.setter
    def laser_sigma(self, value):
        if value <= 0:
            raise ValueError("Standard deviation of the laser power has to be larger than 0.")

        self._laser_sigma = value
        function = AxisymmetricGaussian3D(value)
        self.set_power_density_function(function)

