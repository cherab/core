from raysect.core.math.function.float import Constant3D
from raysect.core.math.function.vector3d cimport Constant3D as ConstantVector3D
from raysect.primitive import Cylinder
from raysect.optical cimport Spectrum, Vector3D, translate

from cherab.core.laser.node cimport Laser
from cherab.core.laser.models.profile_base cimport LaserProfile
from cherab.core.laser.models.math_functions cimport ConstantAxisymmetricGaussian3D, ConstantBivariateGaussian3D, TrivariateGaussian3D, GaussianBeamModel 

from cherab.core.utility.constants cimport SPEED_OF_LIGHT

from libc.math cimport M_PI, sqrt, exp


cdef class UniformEnergyDensity(LaserProfile):

    def __init__(self, double energy_density=1., double laser_radius=0.05, double laser_length=1.,  Vector3D polarization=Vector3D(0, 1, 0)):
        super().__init__()

        self.set_polarization(polarization)
        self.set_pointing_function(ConstantVector3D(Vector3D(0, 0, 1)))
        self.energy_density = energy_density

        self._laser_radius =  0.05
        self._laser_length = 1.

        self.laser_radius = laser_radius
        self.laser_length = laser_length

    def set_polarization(self, Vector3D value):
        value = value.normalise()
        self.set_polarization_function(ConstantVector3D(value))

    @property
    def laser_length(self):
        return self._laser_length

    @laser_length.setter
    def laser_length(self, value):

        if value <= 0:
            raise ValueError("Laser length has to be larger than 0.")

        self._laser_length = value
        self.notifier.notify()

    @property
    def laser_radius(self):
        return self._laser_radius

    @laser_radius.setter
    def laser_radius(self, value):

        if value <= 0:
            raise ValueError("Laser radius has to be larger than 0.")

        self._laser_radius = value
        self.notifier.notify()

    @property
    def energy_density(self):
        return self._energy_density

    @energy_density.setter
    def energy_density(self, value):
        if not value > 0:
            raise ValueError("Laser power density has to be larger than 0.")

        self._energy_density = value
        funct = Constant3D(value)
        self.set_energy_density_function(funct)

    cpdef list generate_geometry(self):

        return generate_segmented_cylinder(self.laser_radius, self.laser_length)
    

cdef class ConstantBivariateGaussian(LaserProfile):
    def __init__(self, double pulse_energy=1, double pulse_length=1, double laser_radius=0.05, double laser_length=1.,
                 double stddev_x=0.01, double stddev_y=0.01, Vector3D polarization=Vector3D(0, 1, 0)):

        super().__init__()
        # set initial values
        self._pulse_energy = 1
        self._pulse_length = 1
        self._stddev_x = 0.1
        self._stddev_y = 0.1

        self._laser_radius = 0.05
        self._laser_length = 1

        self.laser_radius = laser_radius
        self.laser_length = laser_length

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
    def laser_length(self):
        return self._laser_length

    @laser_length.setter
    def laser_length(self, value):

        if value <= 0:
            raise ValueError("Laser length has to be larger than 0.")

        self._laser_length = value
        self.notifier.notify()

    @property
    def laser_radius(self):
        return self._laser_radius

    @laser_radius.setter
    def laser_radius(self, value):

        if value <= 0:
            raise ValueError("Laser radius has to be larger than 0.")

        self._laser_radius = value
        self.notifier.notify()


    @property
    def pulse_energy(self):
        return self._pulse_energy

    @pulse_energy.setter
    def pulse_energy(self, double value):
        if value <= 0:
            raise ValueError("Value has to be larger than 0.")

        self._pulse_energy = value
        self.notifier.notify()

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

    def _geometry_changed(self):

        if self._laser is not None:
            self._laser._configure_geometry()

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

        function = normalisation * self._distribution
        self.set_energy_density_function(function)

    cpdef list generate_geometry(self):

        return generate_segmented_cylinder(self.laser_radius, self.laser_length)

cdef class TrivariateGaussian(LaserProfile):
    def __init__(self, double pulse_energy=1, double pulse_length=1, double mean_z=0,
                 double laser_length=1., double laser_radius=0.05,
                 double stddev_x=0.01, double stddev_y=0.01,
                 Vector3D polarization=Vector3D(0, 1, 0)):

        super().__init__()
        # set initial values
        self._pulse_energy = 1
        self._pulse_length = 1
        self._stddev_x = 0.1
        self._stddev_y = 0.1
        self._stddev_z = 1
        self._laser_radius = 0.05
        self._laser_length = 1
        self._mean_z = mean_z


        self.laser_radius = laser_radius
        self.laser_length = laser_length
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
    def laser_length(self):
        return self._laser_length

    @laser_length.setter
    def laser_length(self, value):

        if value <= 0:
            raise ValueError("Laser length has to be larger than 0.")

        self._laser_length = value
        self.notifier.notify()

    @property
    def laser_radius(self):
        return self._laser_radius

    @laser_radius.setter
    def laser_radius(self, value):

        if value <= 0:
            raise ValueError("Laser radius has to be larger than 0.")

        self._laser_radius = value
        self.notifier.notify()


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

        normalisation =  self._pulse_energy 

        function = normalisation * self._distribution
        self.set_energy_density_function(function)

    cpdef list generate_geometry(self):

        return generate_segmented_cylinder(self.laser_radius, self.laser_length)

cdef class GaussianBeamAxisymmetric(LaserProfile):

    def __init__(self, double pulse_energy=1, double pulse_length=1,
                 double laser_length=1., double laser_radius=0.05,
                 double waist_z=0, double stddev_waist=0.01,
                 double laser_wavelength=1e3, Vector3D polarization=Vector3D(0, 1, 0)):

        super().__init__()
        # set initial values
        self._pulse_energy = 1
        self._pulse_length = 1
        self._stddev_waist = 0.1
        self._waist_z = waist_z
        self._laser_wavelength = 1e3
        self._laser_radius = 0.05
        self._laser_length = 1

        self.laser_length = laser_length
        self.laser_radius = laser_radius

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
    def laser_length(self):
        return self._laser_length

    @laser_length.setter
    def laser_length(self, value):

        if value <= 0:
            raise ValueError("Laser length has to be larger than 0.")

        self._laser_length = value
        self.notifier.notify()

    @property
    def laser_radius(self):
        return self._laser_radius

    @laser_radius.setter
    def laser_radius(self, value):

        if value <= 0:
            raise ValueError("Laser radius has to be larger than 0.")

        self._laser_radius = value
        self.notifier.notify()

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

        function = normalisation * self._distribution
        self.set_energy_density_function(function)

    cpdef list generate_geometry(self):

        return generate_segmented_cylinder(self.laser_radius, self.laser_length)

def generate_segmented_cylinder(radius, length):

    n_segments = int(length // (2 * radius))  # number of segments
    geometry = []

    #length of segment is either length / n_segments if length > radius or length i f length < radius
    if n_segments > 1:
        segment_length = length / n_segments
        for i in range(n_segments):
            segment = Cylinder(name="Laser segment {0:d}".format(i), radius=radius, height=segment_length,
                                transform=translate(0, 0, i * segment_length))

            geometry.append(segment)
    elif 0 <= n_segments < 2:
            segment = Cylinder(name="Laser segment {0:d}".format(0), radius=radius, height=length)

            geometry.append(segment)
    else:
        raise ValueError("Incorrect number of segments calculated.")
    
    return geometry