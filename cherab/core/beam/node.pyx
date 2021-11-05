# cython: language_level=3

# Copyright 2016-2018 Euratom
# Copyright 2016-2018 United Kingdom Atomic Energy Authority
# Copyright 2016-2018 Centro de Investigaciones Energéticas, Medioambientales y Tecnológicas
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


from raysect.primitive import Cylinder, Cone, Intersect

from raysect.core cimport translate, rotate_x

from raysect.optical cimport World, AffineMatrix3D, Primitive, Ray, new_vector3d
from raysect.optical.material.emitter.inhomogeneous cimport NumericalIntegrator

from cherab.core.beam.material cimport BeamMaterial
from cherab.core.beam.model cimport BeamModel
from cherab.core.atomic cimport AtomicData, Element
from cherab.core.utility import Notifier
from libc.math cimport tan, M_PI


cdef double DEGREES_TO_RADIANS = M_PI / 180


cdef class ModelManager:

    def __init__(self):
        self._models = []
        self.notifier = Notifier()

    def __iter__(self):
        return iter(self._models)

    cpdef object set(self, object models):

        # copy models and test it is an iterable
        models = list(models)

        # check contents of list are beam models
        for model in models:
            if not isinstance(model, BeamModel):
                raise TypeError('The model list must consist of only BeamModel objects.')

        self._models = models
        self.notifier.notify()

    cpdef object add(self, BeamModel model):

        if not model:
            raise ValueError('Model must not be None type.')

        self._models.append(model)
        self.notifier.notify()

    cpdef object clear(self):
        self._models = []
        self.notifier.notify()


# todo: beam sigma defines the width, is this really a good way to specify the width? beam.width = fwhm?
cdef class Beam(Node):
    """
    A scene-graph object representing a Gaussian mono-energetic beam.

    The Cherab beam object holds all the properties and state of a mono-energetic
    particle beam to which beam attenuation and emission models may be attached.
    The Beam object is defined in terms of its power, energy, geometric properties
    and the plasma it interacts with.

    The Beam object is a Raysect scene-graph node and lives in it's own
    coordinate space. This coordinate space is defined relative to it's parent
    scene-graph object by an AffineTransform. The beam parameters are defined
    in the Beam object coordinate space. Models using the beam object must
    convert any spatial coordinates into beam space before requesting values
    from the Beam object. The Beam axis is defined to lie along the positive
    z-axis with its origin at the origin of the local coordinate system.

    While a Beam object can be used to simply hold and sample beam properties,
    it can also be used as an emitter in Raysect scenes by attaching
    emission models. The Beam's bounding geometry is automatically defined from
    the Beam's initial width and divergence. The length of the Beam geometry
    needs to be set by the user.

    Beam emission models may be attached to the beam
    object by either setting the full list of models or adding to the list of
    models. See the Beam's ModelManager for more information. The beam emission models
    must be derived from the BeamModel base class.

    Any change to the beam object properties and models
    will result in a automatic notification being sent to objects that register
    with the Beam objects' Notifier. All Cherab models and associated scene-graph
    objects automatically handle the notifications internally to clear
    cached data. If you need to keep track of beam changes in your own classes,
    a callback can be registered with the beam Notifier which will be called in
    the event of a change to the Beam object. See the Notifier documentation.

    .. warning::
       In the current implementation of the Beam class, the Beam can only be associated
       with a single plasma instance. If your scene has overlapping plasmas the
       beam attenuation will only be calculated for the plasma instance to which
       this beam is attached.

    :param Node parent: The parent node in the Raysect scene-graph.
      See the Raysect documentation for more guidance.
    :param AffineMatrix3D transform: The transform defining the spatial position
      and orientation of this beam. See the Raysect documentation if you need
      guidance on how to use AffineMatrix3D transforms.
    :param str name: The name for this beam object.

    :ivar AtomicData atomic_data: The atomic data provider class for this beam.
      All beam emission and attenuation rates will be calculated from the same provider.
    :ivar BeamAttenuator attenuator: The method used for calculating the attenuation
      of this beam into the plasma. Defaults to a SingleRayAttenuator().
    :ivar float divergence_x: The beam profile divergence in the x dimension in beam
      coordinates (degrees).
    :ivar float divergence_y: The beam profile divergence in the y dimension in beam
      coordinates (degrees).
    :ivar Element element: The element of which this beam is composed.
    :ivar float energy: The beam energy in eV/amu.
    :ivar VolumeIntegrator integrator: The configurable method for doing
      volumetric integration through the beam along a Ray's path. Defaults to
      a numerical integrator with 1mm step size, NumericalIntegrator(step=0.001).
    :ivar float length: The approximate length of this beam from source to extinction
      in the plasma. This is used for setting the bounding geometry over which calculations
      will occur. Units of m.
    :ivar ModelManager models: The manager class that sets and provides access to the
      emission models for this beam.
    :ivar Plasma plasma: The plasma instance with which this beam interacts.
    :ivar float power: The total beam power in W.
    :ivar float sigma: The Gaussian beam width at the origin in m.
    :ivar float temperature: The broadening of the beam (eV).

    .. code-block:: pycon

       >>> # This example shows how to initialise and populate a basic beam
       >>>
       >>> from raysect.core.math import Vector3D, translate, rotate
       >>> from raysect.optical import World
       >>>
       >>> from cherab.core.atomic import carbon, deuterium, Line
       >>> from cherab.core.model import BeamCXLine
       >>> from cherab.openadas import OpenADAS
       >>>
       >>>
       >>> world = World()
       >>>
       >>> beam = Beam(parent=world, transform=translate(1.0, 0.0, 0) * rotate(90, 0, 0))
       >>> beam.plasma = plasma  # put your plasma object here
       >>> beam.atomic_data = OpenADAS()
       >>> beam.energy = 60000
       >>> beam.power = 1e4
       >>> beam.element = deuterium
       >>> beam.sigma = 0.025
       >>> beam.divergence_x = 0.5
       >>> beam.divergence_y = 0.5
       >>> beam.length = 3.0
       >>> beam.models = [BeamCXLine(Line(carbon, 5, (8, 7)))]
       >>> beam.integrator.step = 0.001
       >>> beam.integrator.min_samples = 5
    """

    def __init__(self, object parent=None, AffineMatrix3D transform=None, str name=None):

        super().__init__(parent, transform, name)

        # change reporting and tracking
        self.notifier = Notifier()

        # beam properties
        self.BEAM_AXIS = Vector3D(0.0, 0.0, 1.0)
        self._energy = 0.0                         # eV/amu
        self._power = 0.0                          # total beam power, W
        self._temperature = 0.0                    # Broadening of the beam (eV)
        self._element = element = None             # beam species, an Element object
        self._divergence_x = 0.0                   # beam divergence x (degrees)
        self._divergence_y = 0.0                   # beam divergence y (degrees)
        self._length = 1.0                         # m
        self._sigma = 0.1                          # m (gaussian beam width at origin)

        # external data dependencies
        self._plasma = None
        self._atomic_data = None

        # setup emission model handler and trigger geometry rebuilding if the models change
        self._models = ModelManager()
        self._models.notifier.add(self._configure_geometry)

        # beam attenuation model
        self._attenuator = None

        # beam geometry
        self._geometry = None

        # emission model integrator
        self._integrator = NumericalIntegrator(step=0.001)

    cpdef double density(self, double x, double y, double z) except? -1e999:
        """
        Returns the bean density at the specified position in beam coordinates.
        
        Note: this function is only defined over the domain 0 < z < beam_length.
        Outside of this range the density is clamped to zero.
        
        :param x: x coordinate in meters.
        :param y: y coordinate in meters.
        :param z: z coordinate in meters.
        :return: Beam density in m^-3
        """

        if self._attenuator is None:
            raise ValueError('The beam must have an attenuator model to provide density values.')

        # todo: make z > length return 0 as a non-default toggle. It should throw an error by default to warn users they are requesting data outside the domain.
        if z < 0 or z > self._length:
            return 0

        return self._attenuator.density(x, y, z)

    cpdef Vector3D direction(self, double x, double y, double z):
        """
        Calculates the beam direction vector at a point in space.
        
        Note the values of the beam outside of the beam envelope should be
        treated with caution.
        
        :param x: x coordinate in meters.
        :param y: y coordinate in meters.
        :param z: z coordinate in meters. 
        :return: Direction vector.
        """

        # if behind the beam just return the beam axis (for want of a better value)
        if z <= 0:
            return self.BEAM_AXIS

        # calculate direction from divergence
        cdef double dx = tan(DEGREES_TO_RADIANS * self._divergence_x)
        cdef double dy = tan(DEGREES_TO_RADIANS * self._divergence_y)
        return new_vector3d(dx, dy, 1.0).normalise()

    @property
    def energy(self):
        return self._energy

    @energy.setter
    def energy(self, double value):
        if value < 0:
            raise ValueError('Beam energy cannot be less than zero.')
        self._energy = value
        self.notifier.notify()

    cdef double get_energy(self):
        return self._energy

    @property
    def power(self):
        return self._power

    @power.setter
    def power(self, double value):
        if value < 0:
            raise ValueError('Beam power cannot be less than zero.')
        self._power = value
        self.notifier.notify()

    cdef double get_power(self):
        return self._power

    @property
    def temperature(self):
        return self._temperature

    @temperature.setter
    def temperature(self, double value):
        if value < 0:
            raise ValueError('Beam temperature cannot be less than zero.')
        self._temperature = value
        self.notifier.notify()

    cdef double get_temperature(self):
        return self._temperature

    @property
    def element(self):
        return self._element

    @element.setter
    def element(self, Element value not None):
        self._element = value
        self.notifier.notify()

    cdef Element get_element(self):
        return self._element

    @property
    def divergence_x(self):
        return self._divergence_x

    @divergence_x.setter
    def divergence_x(self, double value):
        if value < 0:
            raise ValueError('Beam x divergence cannot be less than zero.')
        self._divergence_x = value
        self.notifier.notify()

    cdef double get_divergence_x(self):
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

    cdef double get_divergence_y(self):
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

    cdef double get_length(self):
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

    cdef double get_sigma(self):
        return self._sigma

    @property
    def atomic_data(self):
        return self._atomic_data

    @atomic_data.setter
    def atomic_data(self, AtomicData value not None):
        self._atomic_data = value
        self._configure_geometry()
        self._configure_attenuator()

    @property
    def plasma(self):
        return self._plasma

    @plasma.setter
    def plasma(self, Plasma value not None):
        self._plasma = value
        self._configure_geometry()
        self._configure_attenuator()

    cdef Plasma get_plasma(self):
        return self._plasma

    @property
    def attenuator(self):
        return self._attenuator
    
    @attenuator.setter
    def attenuator(self, BeamAttenuator value not None):

        # check necessary data is available
        if not self._plasma:
            raise ValueError('The beam must have a reference to a plasma object to be used with an attenuator.')

        if not self._atomic_data:
            raise ValueError('The beam must have an atomic data source to be used with an emission model.')

        # disconnect from previous attenuator's notifications
        if self._attenuator:
            self._attenuator.notifier.remove(self._modified)

        self._attenuator = value
        self._configure_attenuator()

        # connect to new attenuator's notifications
        self._attenuator.notifier.add(self._modified)

        # attenuator supplies beam density, notify dependents there is a data change
        self.notifier.notify()

    @property
    def models(self):
        return self._models

    @models.setter
    def models(self, object values):

        # check necessary data is available
        if not self._plasma:
            raise ValueError('The beam must have a reference to a plasma object to be used with an emission model.')

        if not self._attenuator:
            raise ValueError('The beam must have an attenuator model to be used with an emission model.')

        if not self._atomic_data:
            raise ValueError('The beam must have an atomic data source to be used with an emission model.')

        # setting the emission models causes ModelManager to notify the Beam object to configure geometry
        # so no need to explicitly rebuild here
        self._models.set(values)

    @property
    def integrator(self):
        return self._integrator

    @integrator.setter
    def integrator(self, VolumeIntegrator value):
        self._integrator = value
        self._configure_geometry()

    def _configure_geometry(self):

        # detach existing geometry
        # take a copy of self.children as it will be modified when unparenting
        children = self.children.copy()
        for child in children:
            child.parent = None

        # no further work if there are no emission models
        if not list(self._models):
            return

        # check necessary data is available
        if not self._plasma:
            raise ValueError('The beam must have a reference to a plasma object to be used with an emission model.')

        if not self._attenuator:
            raise ValueError('The beam must have an attenuator model to be used with an emission model.')

        if not self._atomic_data:
            raise ValueError('The beam must have an atomic data source to be used with an emission model.')

        # build geometry to fit beam
        self._geometry = self._generate_geometry()

        # attach geometry to the beam
        self._geometry.parent = self
        self._geometry.name = 'Beam Geometry'

        # add plasma material
        self._geometry.material = BeamMaterial(self, self._plasma, self._atomic_data, list(self._models), self.integrator)

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

        # check necessary data is available
        if not self._plasma:
            raise ValueError('The beam must have a reference to a plasma object to be used with an attenuator.')

        if not self._atomic_data:
            raise ValueError('The beam must have an atomic data source to be used with an emission model.')

        # setup attenuator
        self._attenuator.beam = self
        self._attenuator.plasma = self._plasma
        self._attenuator.atomic_data = self._atomic_data

    cdef int _modified(self) except -1:
        """
        Called when a scene-graph change occurs that modifies this Node's root
        transforms. This will occur if the Node's transform is modified, a
        parent node transform is modified or if the Node's section of scene-
        graph is re-parented.
        """

        # beams section of the scene-graph has been modified, alert dependents
        self.notifier.notify()
