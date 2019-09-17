from raysect.primitive import Cylinder

from raysect.optical cimport World, AffineMatrix3D, Primitive, Ray, new_vector3d
from raysect.optical.material cimport Material
from raysect.optical.material.emitter.inhomogeneous cimport NumericalIntegrator
from raysect.core import translate

from cherab.core.laser.model cimport LaserModel
from cherab.core.laser.material cimport LaserMaterial
from cherab.core.laser.scattering cimport ScatteringModel
from cherab.core.atomic cimport AtomicData, Element
from cherab.core.utility import Notifier
from libc.math cimport tan, M_PI


cdef double DEGREES_TO_RADIANS = (M_PI / 180)


cdef class ModelManager:

    def __init__(self, type):
        self._models = []
        self.notifier = Notifier()
        self._object_type = type

    def __iter__(self):
        return iter(self._models)

    cpdef object set(self, object models):

        # copy models and test it is an iterable
        models = list(models)

        # check contents of list are beam models
        for model in models:
            if not isinstance(model, self._object_type):
                raise TypeError("Model has to be of type {0} but {1} passed.".format(self._object_type, type(model)))

        self._models = models
        self.notifier.notify()

    cpdef object add(self, object model):

        if not model:
            raise ValueError('Model must not be None type.')

        if not isinstance(model, self._object_type):
            raise TypeError("Model has to be of type {0} but {1} passed.".format(self._object_type, type(model)))

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

        self._laser_model = None
        self._scattering_models = ModelManager(ScatteringModel)

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
        self._configure_scattering_models()

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
        #self._geometry.transform = translate(0, 0, 0)

        # build plasma material
        self._geometry.material = LaserMaterial(self, self._integrator)

    @property
    def laser_model(self):
        return self._laser_model

    @laser_model.setter
    def laser_model(self, object value):

        # setting the emission models causes ModelManager to notify the Beam object to configure geometry
        # so no need to explicitly rebuild here

        if not isinstance(value, LaserModel):
            raise TypeError("Value has to be of type LaserModel but {0} passed.".format(type(value)))

        self._laser_model = value
        self._configure_scattering_models()

    @property
    def scattering_models(self):
        return self._scattering_models

    @scattering_models.setter
    def scattering_models(self, value):

        # check necessary data is available
        if not self._plasma:
            raise ValueError('The laser must have a reference to a plasma object to be used with a scattering model.')

        if not self._laser_model:
            raise ValueError('The laser must have a reference to have laser models specified to be used with a scattering model.')

        self._scattering_models.set(value)
        self._configure_geometry()
        self._configure_scattering_models()

    def _configure_scattering_models(self):

        for scattering_model in self._scattering_models:
            scattering_model.plasma = self._plasma
            scattering_model.laser_model = self._laser_model


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