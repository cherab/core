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


from raysect.primitive import Cylinder

from raysect.optical cimport World, AffineMatrix3D, Primitive, Ray, new_vector3d
from raysect.optical.material cimport Material
from raysect.optical.material.emitter.inhomogeneous cimport NumericalIntegrator

from cherab.core.beam.model cimport BeamModel
from cherab.core.beam.material cimport BeamMaterial
from cherab.core.atomic cimport AtomicData, Element
from cherab.core.utility import Notifier
from libc.math cimport tan, M_PI


cdef double DEGREES_TO_RADIANS = (M_PI / 180)


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
    Represents a mono-energetic beam of particles with a Gaussian profile.

    :param parent:
    :param transform:
    :param name:
    :return:
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
        Returns the bean density at the specified coordinates.
        
        :param x: x coordinate in meters.
        :param y: y coordinate in meters.
        :param z: z coordinate in meters.
        :return: Beam density in m^-3
        """

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

        # build plasma material
        self._geometry.material = BeamMaterial(self, self._plasma, self._atomic_data, list(self._models), self.integrator)

    def _generate_geometry(self):

        # todo: switch this for a Cone primitive
        # the beam bounding envelope is a cylinder aligned with the beam axis, sharing the same coordinate space
        # the cylinder radius is set to 5 sigma around the widest section of the gaussian beam
        radius = 5.0 * (self.sigma + self.length * tan(DEGREES_TO_RADIANS * max(self._divergence_x, self._divergence_y)))
        return Cylinder(radius=radius, height=self.length)

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
