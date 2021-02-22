from raysect.core.math.function.float import Constant3D
from raysect.core.math.function.float import MultiplyScalar3D
from raysect.optical cimport Spectrum, Vector3D

from raysect.core.math.function.vector3d cimport Constant3D as ConstantVector3D
from cherab.core.laser.node cimport Laser
from cherab.core.laser.models.profile_base cimport LaserProfile
from cherab.core.laser.models.math_functions cimport ConstantAxisymmetricGaussian3D, ConstantBivariateGaussian3D, TrivariateGaussian3D, GaussianBeamModel 

from cherab.core.utility.constants cimport SPEED_OF_LIGHT

from libc.math cimport M_PI, sqrt, exp


cdef class UniformEnergyDensity(LaserProfile):

    def __init__(self, power_density=1,  Vector3D polarization=Vector3D(0, 1, 0)):
        super().__init__()

        self.set_polarization(polarization)
        self.set_pointing_function(ConstantVector3D(Vector3D(0, 0, 1)))
        self.power_density = power_density

    def set_polarization(self, Vector3D value):
        value = value.normalise()
        self.set_polarization_function(ConstantVector3D(value))

    @property
    def power_density(self):
        return self._power_density

    @power_density.setter
    def power_density(self, value):
        if not value > 0:
            raise ValueError("Laser power density has to be larger than 0.")

        self._power_density = value
        funct = Constant3D(value)
        self.set_power_density_function(funct)


cdef class ConstantBivariateGaussian(LaserProfile):
    def __init__(self, double pulse_energy=1, pulse_length=1, double stddev_x=0.01, double stddev_y=0.01,
                 Vector3D polarization=Vector3D(0, 1, 0)):

        super().__init__()
        # set initial values
        self._pulse_energy = 1
        self._pulse_length = 1
        self._stddev_x = 0.1
        self._stddev_y = 0.1

        self.set_polarization(polarization)
        self.set_pointing_function(ConstantVector3D(Vector3D(0, 0, 1)))

        self.stddev_x = stddev_x
        self.stddev_y = stddev_y

        self.pulse_energy = pulse_energy
        self.pulse_length = pulse_length

        # set laser constants
        self.set_polarization(polarization)

    def set_polarization(self, Vector3D value):
        value = value.normalise()
        self.set_polarization_function(ConstantVector3D(value))

    @property
    def pulse_energy(self):
        return self._pulse_energy

    @pulse_energy.setter
    def pulse_energy(self, double value):
        if value <= 0:
            raise ValueError("Value has to be larger than 0.")

        self._pulse_energy = value
        self._function_changed()

    @property
    def pulse_length(self):
        return self._pulse_length

    @pulse_length.setter
    def pulse_length(self, double value):
        if value <= 0:
            raise ValueError("Value has to be larger than 0.")

        self._pulse_length = value
        self._function_changed()

    @property
    def stddev_x(self):
        return self._stddev_x

    @stddev_x.setter
    def stddev_x(self, value):
        if value <= 0:
            raise ValueError("Standard deviation of the laser power has to be larger than 0.")

        self._stddev_x = value
        self._function_changed()

    @property
    def stddev_y(self):
        return self._stddev_y

    @stddev_y.setter
    def stddev_y(self, value):
        if value <= 0:
            raise ValueError("Standard deviation of the laser power has to be larger than 0.")

        self._stddev_y = value
        self._function_changed()

    def _function_changed(self):
        """
        Energy density should be returned in units [J/m ** 3]. Energy shape in xy
        plane is defined by normal distribution (integral over xy plane for
        constant z is 1). The units of such distribution are [m ** -2].
        In the z axis direction (direction of laser propagation),
        the laser_energy is spread along the z axis using the velocity
        of light SPEED_OF_LIGHT and the temporal duration of the pulse:
        length = SPEED_OF_LIGTH * pulse_length. Combining the normal distribution with the normalisation
         pulse_energy / length gives the units [J / m ** 3].
        """
        self._distribution = ConstantBivariateGaussian3D(self._stddev_x, self._stddev_y)

        length = SPEED_OF_LIGHT * self._pulse_length  # convert from temporal to spatial length of pulse
        normalisation = self._pulse_energy / length   # normalisation to have correct spatial energy density [J / m**3]

        function = MultiplyScalar3D(normalisation, self._distribution)
        self.set_power_density_function(function)


cdef class TrivariateGaussian(LaserProfile):
    def __init__(self, double pulse_energy=1, double pulse_length=1, double mean_z=0,
                 double stddev_x=0.01, double stddev_y=0.01,
                 Vector3D polarization=Vector3D(0, 1, 0)):

        super().__init__()
        # set initial values
        self._pulse_energy = 1
        self._pulse_length = 1
        self._stddev_x = 0.1
        self._stddev_y = 0.1
        self._stddev_z = 1
        self._mean_z = mean_z

        self.stddev_x = stddev_x
        self.stddev_y = stddev_y
        self.mean_z = mean_z

        self.pulse_energy = pulse_energy
        self.pulse_length = pulse_length

        self.set_polarization(polarization)
        self.set_pointing_function(ConstantVector3D(Vector3D(0, 0, 1)))

    def set_polarization(self, Vector3D value):
        value = value.normalise()
        self.set_polarization_function(ConstantVector3D(value))

    @property
    def pulse_energy(self):
        return self._pulse_energy

    @pulse_energy.setter
    def pulse_energy(self, double value):
        if value <= 0:
            raise ValueError("Value has to be larger than 0.")

        self._pulse_energy = value
        self._function_changed()

    @property
    def pulse_length(self):
        return self._pulse_length

    @pulse_length.setter
    def pulse_length(self, double value):
        if value <= 0:
            raise ValueError("Value has to be larger than 0.")

        self._pulse_length = value
        self._stddev_z = self._pulse_length * SPEED_OF_LIGHT
        self._function_changed()

    @property
    def stddev_x(self):
        return self._stddev_x

    @stddev_x.setter
    def stddev_x(self, value):
        if value <= 0:
            raise ValueError("Standard deviation of the laser power has to be larger than 0.")

        self._stddev_x = value
        self._function_changed()

    @property
    def stddev_y(self):
        return self._stddev_y

    @stddev_y.setter
    def stddev_y(self, value):
        if value <= 0:
            raise ValueError("Standard deviation of the laser power has to be larger than 0.")

        self._stddev_y = value
        self._function_changed()

    @property
    def mean_z(self):
        return self._mean_z

    @mean_z.setter
    def mean_z(self, double value):
        self._mean_z = value
        self._function_changed()

    def _function_changed(self):
        """
        Energy density should be returned in units [J/m ** 3]. Energy shape in xy
        plane is defined by normal distribution (integral over xy plane for
        constant z is 1). The units of such distribution are [m ** -2].
        In the z axis direction (direction of laser propagation),
        the laser_energy is spread along the z axis using the velocity
        of light SPEED_OF_LIGHT and the temporal duration of the pulse:
        length = SPEED_OF_LIGTH * pulse_length. Combining the normal distribution with the normalisation
         pulse_energy / length gives the units [J / m ** 3].
        """

        self._distribution = TrivariateGaussian3D(self._mean_z, self._stddev_x, self._stddev_y,
                                                  self._stddev_z)

        one_stddev = 0.682689492137  # ratio of energy in one standart deviation

        # pulse length is given by a standart deviation, which contains only one_stddev part of the energy
        normalisation =  self._pulse_energy / self._pulse_length

        function = MultiplyScalar3D(normalisation, self._distribution)
        self.set_power_density_function(function)


cdef class GaussianBeamAxisymmetric(LaserProfile):

    def __init__(self, double pulse_energy=1, double pulse_length=1,
                 double waist_z=0, double stddev_waist=0.01,
                 double laser_wavelength=1e3, Vector3D polarization=Vector3D(0, 1, 0)):

        super().__init__()
        # set initial values
        self._pulse_energy = 1
        self._pulse_length = 1
        self._stddev_waist = 0.1
        self._waist_z = waist_z
        self._laser_wavelength = 1e3

        self.set_polarization(polarization)
        self.set_pointing_function(ConstantVector3D(Vector3D(0, 0, 1)))

        self.stddev_waist = stddev_waist
        self.waist_z = waist_z

        self.pulse_energy = pulse_energy
        self.pulse_length = pulse_length
        self.laser_wavelength = laser_wavelength

    def set_polarization(self, Vector3D value):
        value = value.normalise()
        self.set_polarization_function(ConstantVector3D(value))

    @property
    def pulse_energy(self):
        return self._pulse_energy

    @pulse_energy.setter
    def pulse_energy(self, double value):
        if value <= 0:
            raise ValueError("Value has to be larger than 0.")

        self._pulse_energy = value
        self._function_changed()

    @property
    def pulse_length(self):
        return self._pulse_length

    @pulse_length.setter
    def pulse_length(self, double value):
        if value <= 0:
            raise ValueError("Value has to be larger than 0.")

        self._pulse_length = value
        self._function_changed()

    @property
    def waist_z(self):
        return self._waist_z

    @waist_z.setter
    def waist_z(self, double value):
        self._waist_z = value
        self._function_changed()

    @property
    def stddev_waist(self):
        return self._stddev_waist

    @stddev_waist.setter
    def stddev_waist(self, double value):
        self._stddev_waist = value
        self._function_changed()

    @property
    def laser_wavelength(self):
        return self._laser_wavelength

    @laser_wavelength.setter
    def laser_wavelength(self, double value):
        self._laser_wavelength = value
        self._function_changed()

    def _function_changed(self):
        """
        Energy density should be returned in units [J/m ** 3]. Energy shape in xy
        plane is defined by normal distribution (integral over xy plane for
        constant z is 1). The units of such distribution are [m ** -2].
        In the z axis direction (direction of laser propagation),
        the laser_energy is spread along the z axis using the velocity
        of light SPEED_OF_LIGHT and the temporal duration of the pulse:
        length = SPEED_OF_LIGTH * pulse_length. Combining the normal distribution with the normalisation
         pulse_energy / length gives the units [J / m ** 3].
        """

        self._distribution = GaussianBeamModel(self.laser_wavelength, self._waist_z, self.stddev_waist)
        # calculate volumetric power dentiy
        length = SPEED_OF_LIGHT * self._pulse_length  # convert from temporal to spatial length of pulse
        normalisation = self._pulse_energy / length  # normalisation to have correct spatial energy density [J / m**3]

        function = MultiplyScalar3D(normalisation, self._distribution)
        self.set_power_density_function(function)
