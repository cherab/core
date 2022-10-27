from raysect.core cimport Vector3D, new_vector3d
from raysect.primitive import Cylinder, Cone, Intersect

from raysect.core cimport translate, rotate_x

from cherab.core.beam.material cimport BeamMaterial
from cherab.core.beam.distribution cimport BeamDistribution
from cherab.core.atomic cimport AtomicData
from cherab.core.utility import Notifier, EvAmuToMS

from libc.math cimport tan, M_PI

cdef double DEGREES_TO_RADIANS = M_PI / 180


cdef class ThinBeam(BeamDistribution):

    def __init__(self):

        super().__init__()

        # beam properties
        self.BEAM_AXIS = Vector3D(0.0, 0.0, 1.0)
        self._energy = 0.0                         # eV/amu
        self._power = 0.0                          # total beam power, W
        self._speed = 0.0                          # speed of the particles (m/s)
        self._temperature = 0.0                    # Broadening of the beam (eV)
        self._element = element = None             # beam species, an Element object
        self._divergence_x = 0.0                   # beam divergence x (degrees)
        self._divergence_y = 0.0                   # beam divergence y (degrees)
        self._length = 1.0                         # m
        self._sigma = 0.1                          # m (gaussian beam width at origin)
        self._z_outofbounds = False

    cpdef Vector3D bulk_velocity(self, double x, double y, double z):
        """
        Evaluates the species' bulk velocity at the specified 3D coordinate.

        :param float x: position in meters
        :param float y: position in meters
        :param float z: position in meters
        :return: velocity vector in m/s
        :rtype: Vector3D
        """
        # if behind the beam just return the beam axis (for want of a better value)
        if z <= 0:
            return self.BEAM_AXIS

        # calculate direction from divergence
        cdef double dx = tan(DEGREES_TO_RADIANS * self._divergence_x)
        cdef double dy = tan(DEGREES_TO_RADIANS * self._divergence_y)

        return new_vector3d(dx, dy, 1.0).normalise().mul(self._speed) 

    cpdef double effective_temperature(self, double x, double y, double z) except? -1e999:
        """
        Evaluates the species' effective temperature at the specified 3D coordinate.

        :param float x: position in meters
        :param float y: position in meters
        :param float z: position in meters
        :return: temperature in eV
        :rtype: float
        """

        return self._temperature

    cpdef double density(self, double x, double y, double z) except? -1e999:
        """
        Evaluates the species' density at the specified 3D coordinate.

        :param float x: position in meters
        :param float y: position in meters
        :param float z: position in meters
        :return: density in m^-3
        :rtype: float
        """

        if self._attenuator is None:
            raise ValueError('The beam must have an attenuator model to provide density values.')

        # todo: make z > length return 0 as a non-default toggle. It should throw an error by default to warn users they are requesting data outside the domain.
        if z < 0 or z > self._length:
            if self._z_outofbounds:
                return 0
            else:
                raise ValueError("z coordinate in beam space out of bounds: z e (0, length)")

        return self._attenuator.density(x, y, z)

    @property
    def z_outofbounds(self):
        return self._z_outofbounds
    
    @z_outofbounds.setter
    def z_outofbounds(self, bint value):
        self._z_outofbounds = value

    @property
    def temperature(self):
        return self._temperature

    @temperature.setter
    def temperature(self, double value):
  
        if value < 0:
            raise ValueError('Temperature cannot be less than zero.')

        self._temperature = value

    @property
    def energy(self):
        return self._energy

    @energy.setter
    def energy(self, double value):
        if value < 0:
            raise ValueError('Beam energy cannot be less than zero.')
        self._energy = value
        self.notifier.notify()
        self._speed = EvAmuToMS.to(value)

    cpdef double get_energy(self):
        return self._energy

    cpdef double get_speed(self):
        return self._speed

    @property
    def power(self):
        return self._power

    @power.setter
    def power(self, double value):
        if value < 0:
            raise ValueError('Beam power cannot be less than zero.')
        self._power = value
        self.notifier.notify()

    cpdef double get_power(self):
        return self._power

    @property
    def divergence_x(self):
        return self._divergence_x

    @divergence_x.setter
    def divergence_x(self, double value):
        if value < 0:
            raise ValueError('Beam x divergence cannot be less than zero.')
        self._divergence_x = value
        self.notifier.notify()

    cpdef double get_divergence_x(self):
        return self._divergence_x

    @property
    def divergence_y(self):
        return self._divergence_y

    @divergence_y.setter
    def divergence_y(self, double value):
        if value < 0:
            raise ValueError('Beam y divergence cannot be less than zero.')
        self._divergence_y = value
        self.notifier.notify()

    cpdef double get_divergence_y(self):
        return self._divergence_y

    @property
    def length(self):
        return self._length

    @length.setter
    def length(self, double value):
        if value <= 0:
            raise ValueError('Beam length must be greater than zero.')
        self._length = value
        self.notifier.notify()

    cpdef double get_length(self):
        return self._length

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, double value):
        if value <= 0:
            raise ValueError('Beam sigma (width) must be greater than zero.')
        self._sigma = value
        self.notifier.notify()

    cpdef double get_sigma(self):
        return self._sigma

    @property
    def atomic_data(self):
        return self._atomic_data

    @atomic_data.setter
    def atomic_data(self, AtomicData value not None):
        self._atomic_data = value
        self._configure_attenuator()

    @property
    def plasma(self):
        return self._plasma

    @plasma.setter
    def plasma(self, Plasma value not None):
        
        self._plasma = value
        self._configure_attenuator()

    cpdef Plasma get_plasma(self):
        return self._plasma

    @property
    def attenuator(self):
        return self._attenuator
    
    @attenuator.setter
    def attenuator(self, BeamAttenuator value not None):

        # disconnect from previous attenuator's notifications
        if self._attenuator:
            self._attenuator.notifier.remove(self._modified)

        self._attenuator = value
        self._configure_attenuator()

        # connect to new attenuator's notifications
        self._attenuator.notifier.add(self._modified)

        # attenuator supplies beam density, notify dependents there is a data change
        self.notifier.notify()

    cpdef list get_geometry(self):
        
        # check necessary data is available
        if not self._plasma:
            raise ValueError('The beam must have a reference to a plasma object to be used with an emission model.')

        if not self._attenuator:
            raise ValueError('The beam must have an attenuator model to be used with an emission model.')

        if not self._atomic_data:
            raise ValueError('The beam must have an atomic data source to be used with an emission model.')

        return [self._generate_geometry()]

    def _generate_geometry(self):
        """
        Generate the bounding geometry for the beam model.

        Where possible the beam is bound by a cone as this offers the tightest
        fitting bounding volume. To avoid numerical issues caused by creating
        extremely long cones in low divergence cases, the geometry is switched
        to a cylinder where the difference in volume between the cone and a
        cylinder is less than 10%.

        :return: Beam geometry Primitive.
        """

        # number of beam sigma the bounding volume lies from the beam axis
        num_sigma = self._attenuator.clamp_sigma

        # return Cylinder(NUM_SIGMA * self.sigma, height=self.length)

        # no divergence, use a cylinder
        if self._divergence_x == 0 and self._divergence_y == 0:
            return Cylinder(num_sigma * self.sigma, height=self.length)

        # rate of change of beam radius with z (using largest divergence)
        drdz = tan(DEGREES_TO_RADIANS * max(self._divergence_x, self._divergence_y))

        # radii of bounds at the beam origin (z=0) and the beam end (z=length)
        radius_start = num_sigma * self.sigma
        radius_end = radius_start + self.length * num_sigma * drdz

        # distance of the cone apex to the beam origin
        distance_apex = radius_start / (num_sigma * drdz)
        cone_height = self.length + distance_apex

        # calculate volumes
        cylinder_volume = self.length * M_PI * radius_end**2
        cone_volume = M_PI * (cone_height * radius_end**2 - distance_apex * radius_start**2) / 3
        volume_ratio = cone_volume / cylinder_volume

        # if the volume difference is <10%, generate a cylinder
        if volume_ratio > 0.9:
            return Cylinder(num_sigma * self.sigma, height=self.length)

        # cone has to be rotated by 180 deg and shifted by beam length in the +z direction
        cone_transform = translate(0, 0, self.length) * rotate_x(180)

        # create cone and cut off -z protrusion
        return Intersect(
            Cone(radius_end, cone_height, transform=cone_transform),
            Cylinder(radius_end * 1.01, self.length * 1.01)
        )

    def _configure_attenuator(self):

        # there must be an attenuator present to configure
        if not self._attenuator:
            return
        # there must be plasma present to configure
        if not self._plasma:
            return
        # there must be atomic_data present to configure
        if not self._atomic_data:
            return

        # setup attenuator
        self._attenuator.distribution = self
        self._attenuator.plasma = self._plasma
        self._attenuator.atomic_data = self._atomic_data

    def _beam_changed(self):
        """
        Reaction to _beam changes
        """
        self._plasma = self._beam.plasma
        self._atomic_data = self._beam.atomic_data
        self._configure_attenuator()