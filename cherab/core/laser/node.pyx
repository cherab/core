from raysect.primitive import Cylinder

from raysect.optical cimport World, AffineMatrix3D, Primitive, Ray, new_vector3d
from raysect.optical.material cimport Material
from raysect.optical.material.emitter.inhomogeneous cimport NumericalIntegrator

from cherab.core.laser.model cimport LaserModel
from cherab.core.laser.material cimport LaserMaterial
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
            if not isinstance(model, LaserModel):
                raise TypeError('The model list must consist of only LaserModel objects.')

        self._models = models
        self.notifier.notify()

    cpdef object add(self, LaserModel model):

        if not model:
            raise ValueError('Model must not be None type.')

        self._models.append(model)
        self.notifier.notify()

    cpdef object clear(self):
        self._models = []
        self.notifier.notify()

cdef class Laser(Node):

    def __init__(self, object parent=None, AffineMatrix3D transform=None, str name=None):

        super().__init__(parent, transform, name)

        # change reporting and tracking
        self.notifier = Notifier()

        # laser beam properties
        self.BEAM_AXIS = Vector3D(0.0, 0.0, 1.0)
        self._length = 1.0                         # [m]
        self._radius = 0.1                         # [m]

        # external data dependencies
        self._plasma = None

        # setup emission model handler and trigger geometry rebuilding if the models change
        self._laser_models = ModelManager()
        self._laser_models.notifier.add(self._configure_geometry)

        self._integrator = NumericalIntegrator(step=0.001)

    @property
    def length(self):
        return self._length

    @length.setter
    def length(self, value):

        if value <= 0:
            raise ValueError("Laser length has to be larger than 0.")

        self._length = value
        self._configure_geometry()

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, value):

        if value <= 0:
            raise ValueError("Laser radius has to be larger than 0.")

        self._radius = value

    @property
    def plasma(self):
        return self._plasma

    @plasma.setter
    def plasma(self, Plasma value not None):
        self._plasma = value
        self._configure_geometry()

    cdef Plasma get_plasma(self):
        return self._plasma

    def _configure_geometry(self):

        # detach existing geometry
        # take a copy of self.children as it will be modified when unparenting
        children = self.children.copy()
        for child in children:
            child.parent = None

        # check necessary data is available
        if not self._plasma:
            raise ValueError('The laser beam must have a reference to a plasma object to be used with a scattering model.')

        # build geometry to fit beam
        self._geometry = self._generate_geometry()

        # attach geometry to the beam
        self._geometry.parent = self
        self._geometry.name = 'Laser Beam Geometry'

        # build plasma material
        self._geometry.material = LaserMaterial(self, self._plasma, list(self._laser_models), self._integrator)

    @property
    def laser_models(self):
        return self._laser_models

    @laser_models.setter
    def laser_models(self, object values):

        # check necessary data is available
        if not self._plasma:
            raise ValueError('The beam must have a reference to a plasma object to be used with an emission model.')

        # setting the emission models causes ModelManager to notify the Beam object to configure geometry
        # so no need to explicitly rebuild here
        self._laser_models.set(values)

    def _generate_geometry(self):

        # the beam bounding envelope is a cylinder aligned with the beam axis, sharing the same coordinate space
        # the cylinder radius is width, the cylinder length is length
        return Cylinder(radius=self.radius, height=self.length)

    @property
    def integrator(self):
        return self._integrator

    @integrator.setter
    def integrator(self, VolumeIntegrator value):
        self._integrator = value
        self._configure_geometry()