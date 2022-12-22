from raysect.core.math.function.float import Constant3D
from raysect.core.math.function.vector3d cimport Constant3D as ConstantVector3D
from raysect.primitive import Cylinder
from raysect.optical cimport Spectrum, Vector3D, translate

from cherab.core.laser cimport Laser, LaserProfile
from cherab.core.model.laser.math_functions cimport ConstantAxisymmetricGaussian3D, ConstantBivariateGaussian3D, TrivariateGaussian3D, GaussianBeamModel 

from cherab.core.utility.constants cimport SPEED_OF_LIGHT

from libc.math cimport M_PI, sqrt, exp


cdef class UniformEnergyDensity(LaserProfile):
    """
    LaserProfile with a constant volumetric energy density.

    Returns a laser with a cylindrical shape within which the laser volumentric energy density is constant.
    The laser starts at z=0 and extends in the positive z direction.

    .. note:
        The methods get_pointing, get_polarization and get_energy_density are not limited to the inside
        of the laser cylinder.  If called alone for position (x, y, z) outisde the laser cylinder,
        they will still return non-zero values.
      
    In the following example, a laser of length of 2 m (extending from z=0 to z=2 m) with a radius of 3 cm
    and volumetric energy density of 5 J*m^-3 and polarisation in the y direction is created:

    .. code-block:: pycon
       
       >>> from raysect.core import Vector3D
       >>> from cherab.core.model.laser import UniformEnergyDensity
       
       >>> energy = 5 # energy density in J
       >>> radius = 3e-2 # laser radius in m
       >>> length = 2 # laser length in m
       >>> polarisation = Vector3D(0, 1, 0) # polarisation direction
       
           # create the laser profile
       >>> laser_profile = UniformEnergyDensity(energy, radius, length, polarisation)

    :param float energy_density: The volumetric energy density of the laser light.
    :param float laser_length: The length of the laser cylinder.
    :param float laser_radius: The radius of the laser cylinder.
    :param Vector3D polarization: The direction of the laser polarization:

    :ivar float energy_density: The volumetric energy density of the laser light.
    :ivar float laser_radius: The radius of the laser cylinder.
    :ivar float laser_length: The length of the laser cylinder.
    """

    def __init__(self, double energy_density=1., double laser_length=1., double laser_radius=0.05,  Vector3D polarization=Vector3D(0, 1, 0)):
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
        if value <= 0:
            raise ValueError("Laser power density has to be larger than 0.")

        self._energy_density = value
        funct = Constant3D(value)
        self.set_energy_density_function(funct)

    cpdef list generate_geometry(self):

        return generate_segmented_cylinder(self.laser_radius, self.laser_length)
    

cdef class ConstantBivariateGaussian(LaserProfile):
    """
    LaserProfile with a Gaussian-shaped volumetric energy density distribution in the xy plane
    and constant pulse intensity.

    Returns a laser with a cylindrical shape and the propagation of the laser light in the positive z direction.

    The model imitates a laser beam with a uniform power output within a single pulse. This results
    in the distribution of the energy density along the propagation direction of the laser (z-axis) to be also
    uniform. The integral value of laser energy Exy in an x-y plane is given by
    
    .. math:: 
         E_{xy} = \\frac{E_p}{(c * \\tau)},

    where Ep is the energy of the laser pulse, tau is the temporal pulse length and c is the speed of light in vacuum.
    In an x-y plane, the volumetric energy density follows a bivariate Gaussian with a zero correlation:

    .. math::
         E(x, y) = \\frac{E_{xy}}{2 \\pi \\sigma_x \\sigma_y} exp\\left(-\\frac{x^2 + y^2}{2 \\sigma_x \\sigma_y}\\right).

    The sigma_x and sigma_y are standard deviations in x and y directions, respectively.

    .. note::
        The height of the cylinder, forming the laser beam, is given by the laser_length and is independent from the 
        temporal length of the laser pulse given by pulse_length. This gives the possibility to independently control
        the size of the laser primitive and the value of the volumetric energy density.
      
        The methods get_pointing, get_polarization and get_energy_density are not limited to the inside
        of the laser cylinder.  If called for position (x, y, z) outisde the laser cylinder, they can still
        return non-zero values.
    

    The following example shows how to create a laser with sigma_x= 1 cm and sigma_y=2 cm, which makes the laser
    profile in x-y plane to be elliptical. The pulse energy is 5 J and the laser temporal pulse length is 10 ns:

    .. code-block:: pycon
       
       >>> from raysect.core import Vector3D
       >>> from cherab.core.model.laser import ConstantBivariateGaussian
       
       >>> radius = 3e-2 # laser radius in m
       >>> length = 2 # laser length in m
       >>> polarisation = Vector3D(0, 1, 0) # polarisation direction
       >>> pulse_energy = 5 # energy in a laser pulse in J
       >>> pulse_length = 1e-8 # pulse length in s
       >>> width_x = 1e-2 # standard deviation in x direction in m
       >>> width_y = 2e-2 # standard deviation in y direction in m
       
           # create the laser profile
       >>> laser_profile = ConstantBivariateGaussian(pulse_energy, pulse_length, radius, length, width_x, width_y, polarisation)

    :param float pulse_energy: The energy of the laser in Joules delivered in a single laser pulse.
    :param float pulse_length: The temporal length of the laser pulse in seconds.
    :param float laser_length: The length of the laser cylinder.
    :param float laser_radius: The radius of the laser cylinder.
    :param float stddev_x: The standard deviation of the bivariate Gaussian distribution of the volumetric energy
      density distribution of the laser light in the x axis in meters.
    :param float stddev_y: The standard deviation of the bivariate Gaussian distribution of the volumetric energy
      density distribution of the laser light in the y axis in meters.
    :param Vector3D polarization: The direction of the laser polarization:

    :ivar float pulse_energy: The energy of the laser in Joules delivered in a single laser pulse.
    :ivar float pulse_length: The temporal length of the laser pulse in seconds.
    :ivar float laser_radius: The radius of the laser cylinder.
    :ivar float laser_length: The length of the laser cylinder.
    :ivar float stddev_x: The standard deviation of the bivariate Gaussian distribution of the volumetric energy
      density distribution of the laser light in the x axis in meters.
    :ivar float stddev_y: The standard deviation of the bivariate Gaussian distribution of the volumetric energy
      density distribution of the laser light in the y axis in meters.
    """
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
    """
    LaserProfile with a trivariate Gaussian-shaped volumetric energy density.

    Returns a laser with a cylindrical shape and the propagation of the laser light in the positive z direction.
    This model imitates a laser beam with a Gaussian distribution of power output within a single pulse frozen in time:

    .. math::
         E(x, y, z) = \\frac{E_p}{\\sqrt{2 \\pi^3} \\sigma_x \\sigma_y \\sigma_z} exp\\left(-\\frac{x^2}{2 \\sigma_x^2} -\\frac{y^2}{2 \\sigma_y^2} -\\frac{(z - \\mu_z)^2}{2 \\sigma_z^2}\\right).


    The sigma_x and sigma_y are standard deviations in x and y directions, respectively, and E_p is the energy deliverd by laser in a
    single laser pulse. The mu_z is the mean of the distribution in the z direction and controls th position of the laser pulse along the z direction.
    The standard deviation in z direction sigma_z is calculated from the pulse length tau_p, which is the 
    standard deviation of the Gaussian distributed ouput power of the laser within a single pulse:

    .. math::
         \\sigma_z = \\tau_p c.

    The c stands for the speed of light in vacuum.

    .. note::
        The height of the cylinder, forming the laser beam, is given by the laser_length and is independent from the 
        temporal length of the laser pulse given by pulse_length. This gives the possibility to independently control
        the size of the laser primitive and the value of the volumetric energy density.
      
        The methods get_pointing, get_polarization and get_energy_density are not limited to the inside
        of the laser cylinder.  If called alone for position (x, y, z) outisde the laser cylinder, they can still
        return non-zero values.
    

    The following example shows how to create a laser with sigma_x = 1 cm and sigma_y = 2 cm, which makes the laser
    profile in an x-y plane to be elliptical. The pulse energy is 5 J and the laser temporal pulse length is 10 ns.
    The position of the laser pulse maximum mean_z is set to 0.5:

    .. code-block:: pycon
       
       >>> from raysect.core import Vector3D
       >>> from cherab.core.model.laser import ConstantBivariateGaussian
       
       >>> radius = 3e-2 # laser radius in m
       >>> length = 2 # laser length in m
       >>> polarisation = Vector3D(0, 1, 0) # polarisation direction
       >>> pulse_energy = 5 # energy in a laser pulse in J
       >>> pulse_length = 1e-8 # pulse length in s
       >>> pulse_z = 0.5 # position of the pulse mean
       >>> width_x = 1e-2 # standard deviation in x direction in m
       >>> width_y = 2e-2 # standard deviation in y direction in m
       
           # create the laser profile
       >>> laser_profile = ConstantBivariateGaussian(pulse_energy, pulse_length, pulse_z, radius, length, width_x, width_y, polarisation)


    :param float pulse_energy: The energy of the laser in Joules delivered in a single laser pulse.
    :param float pulse_length: The standard deviation of the laser pulse length in the temporal domain.
    :param float mean_z: Position of the mean value of the laser pulse in the z direction. Can be used to control the
      position of the laser pulse along the laser propagation.
    :param float laser_length: The length of the laser cylinder.
    :param float laser_radius: The radius of the laser cylinder.
    :param float stddev_x: The standard deviation of the bivariate Gaussian distribution of the volumetric energy
      density distribution of the laser light in the x axis in meters.
    :param float stddev_y: The standard deviation of the bivariate Gaussian distribution of the volumetric energy
      density distribution of the laser light in the y axis in meters.
    :param Vector3D polarization: The direction of the laser polarization.

    :ivar float pulse_energy: The energy of the laser in Joules delivered in a single laser pulse.
    :ivar float pulse_length: The standard deviation of the laser pulse length in the temporal domain.
    :ivar float mean_z: Position of the mean value of the laser pulse in the z direction.
     Can be used to control the position of the laser pulse along the laser propagation.
    :ivar float laser_radius: The radius of the laser cylinder.
    :ivar float laser_length: The length of the laser cylinder.
    :ivar float stddev_x: The standard deviation of the bivariate Gaussian distribution of the volumetric energy
      density distribution of the laser light in the x axis in meters.
    :ivar float stddev_y: The standard deviation of the bivariate Gaussian distribution of the volumetric energy
      density distribution of the laser light in the y axis in meters.
    """
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
        Energy density should be returned in units [J/m ** 3]. The integral value of the _distribution
        is 1, thus multiplying _distribution by _pulse_energy gives correct values.
        """

        self._distribution = TrivariateGaussian3D(self._mean_z, self._stddev_x, self._stddev_y,
                                                  self._stddev_z)

        normalisation =  self._pulse_energy 

        function = normalisation * self._distribution
        self.set_energy_density_function(function)

    cpdef list generate_geometry(self):

        return generate_segmented_cylinder(self.laser_radius, self.laser_length)

cdef class GaussianBeamAxisymmetric(LaserProfile):
    """
    LaserProfile with volumetric energy density following the Gaussian beam model.

    Returns a laser with a cylindrical shape and the propagation of the laser light in the positive z direction. This model implements
    the axisymmetrical Gaussian beam model. It immitates a focused axis symmetrical laser beam with a uniform power ouput in a laser pulse.
    The volumetric energy density is given by

    .. math::
         E(x, y, z) = \\frac{E_{xy}}{2 \\pi \\sigma^2(z)} exp\\left( -\\frac{x^2 + y^2}{2 \\sigma^2(z) }\\right) \\\\

    where the sigma is the standard deviation of the Gaussian shape in the xy plane and is given by

    .. math::
         sigma(z) = \\sigma_0 \\sqrt{1 + \\left(\\frac{z - z_0}{z_R}\\right)^2}.

    The z_0 is the position of the beam focus and z_R is the Rayleigh length

    .. math::
         z_R = \\frac{\\pi \\omega_0^2 n}{\\lambda_l}
    
    where the omega_0 is the standard deviation in the xy plane in the focal point (beam waist) and lambda_l is the central wavelength of
    the laser. The E_xy stand for the laser energy in an xy plane and is calculated as:
    
    .. math::
         E_{xy} = \\frac{E_p}{(c * \\tau)},

    where the E_p is the energy in a single laser pulse and tau is the temporal pulse length.

    .. note::      
        For more information about the Gaussian beam model see https://en.wikipedia.org/wiki/Gaussian_beam

        The methods get_pointing, get_polarization and get_energy_density are not limited to the inside
        of the laser cylinder.  If called alone for position (x, y, z) outisde the laser cylinder, they can still
        return non-zero values.

    The following example shows how to create a laser with pulse energy 5J, pulse length 10 ns and with the laser cylinder primitive
    being 2m long with 5 cm in diameter. The the standard deviation of the beam in the focal point (waist) is 5mm and the position of the
    waist is z=50 cm. The laser wavelength is 1060 nm.

    .. code-block:: pycon
       
       >>> from raysect.core import Vector3D
       >>> from cherab.core.model.laser import GaussianBeamAxisymmetric
       
       >>> radius = 5e-2 # laser radius in m
       >>> length = 2 # laser length in m
       >>> polarisation = Vector3D(0, 1, 0) # polarisation direction
       >>> pulse_energy = 5 # energy in a laser pulse in J
       >>> pulse_length = 1e-8 # pulse length in s
       >>> waist_width = 5e-3 # standard deviation in the waist
       >>> waist_z = 0.5 # position of the pulse mean
       >>> width_x = 1e-2 # standard deviation in x direction in m
       >>> width_y = 2e-2 # standard deviation in y direction in m
       >>> laser_wlen = 1060 # laser wavelength in nm
       
           # create the laser profile
       >>> laser_profile = GaussianBeamAxisymmetric(pulse_energy, pulse_length, length, radius, waist_z, waist_width, laser_wlen)

    :param float pulse_energy: The energy of the laser in Joules delivered in a single laser pulse.
    :param float pulse_length: The temporal length of the laser pulse in seconds.
    :param float laser_length: The length of the laser cylinder in meters.
    :param float laser_radius: The radius of the laser cylinder in meters.
    :param float waist_z: Position of the laser waist along the z axis in m.
    :param float stddev_waist: The standard deviation of the laser width in the focal point (waist) in m.
    :param float laser_wavelength: The central wavelength of the laser light in nanometers.
    :param Vector3D polarization: The direction of the laser polarization.

    :ivar float pulse_energy: The energy of the laser in Joules delivered in a single laser pulse.
    :ivar float pulse_length: The temporal length of the laser pulse in seconds.
    :ivar float laser_length: The length of the laser cylinder in meters.
    :ivar float laser_radius: The radius of the laser cylinder in meters.
    :ivar float waist_z: Position of the laser waist along the z axis in m.
    :ivar float stddev_waist: The standard deviation of the laser width in the focal point (waist) in m.
    :ivar float laser_wavelength: The central wavelength of the laser light in nanometers.
    :ivar Vector3D polarization: The direction of the laser polarization.
    """

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

        self._distribution = GaussianBeamModel(self._laser_wavelength, self._waist_z, self._stddev_waist)
        # calculate volumetric power dentiy
        length = SPEED_OF_LIGHT * self._pulse_length  # convert from temporal to spatial length of pulse
        normalisation = self._pulse_energy / length  # normalisation to have correct spatial energy density [J / m**3]

        function = normalisation * self._distribution
        self.set_energy_density_function(function)

    cpdef list generate_geometry(self):

        return generate_segmented_cylinder(self.laser_radius, self.laser_length)


def generate_segmented_cylinder(radius, length):
    """
    Generates a segmented cylindrical laser geometry

    Approximates a long cylinder with a cylindrical segments to optimize
    targetted and importance sampling. The height of a cylinder segments is roughly
    2 * cylinder radius.

    :return: List of cylinders
    """

    n_segments = int(length // (2 * radius))  # number of segments
    geometry = []

    #length of segment is either length / n_segments if length > radius or length if length < radius
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