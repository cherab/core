from raysect.primitive cimport Cylinder
from raysect.optical cimport World, AffineMatrix3D, Primitive, Ray
from raysect.optical.material.emitter.inhomogeneous cimport NumericalIntegrator
from raysect.core cimport translate, Material

from cherab.core.laser.material cimport LaserMaterial
from cherab.core.laser.model cimport LaserModel
from cherab.core.laser.models.laserspectrum_base import LaserSpectrum
from cherab.core.utility import Notifier
from libc.math cimport M_PI

from math import ceil

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

        # check contents of list are laser models
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

    def __init__(self, object parent=None, AffineMatrix3D transform=None,
                 double importance=1., str name=None):

        super().__init__(parent, transform, name)

        # set init values of the laser
        self._set_init_values()

        # change reporting and tracking
        self.notifier = Notifier()

        #setup model manager
        self._models = ModelManager()
        
        # set material integrator
        self._integrator = NumericalIntegrator(step=1e-3)

        self._importance = importance
        self._configure_geometry()

    def _set_init_values(self):
        """
        Sets initial values of the laser shape to avoid errors.
        """
        self._importance = 0.
        self._geometry = []

    @property
    def plasma(self):
        return self._plasma

    @plasma.setter
    def plasma(self, Plasma value not None):

        #unregister from old plasma notifier
        if self._plasma is not None:
            self._plasma.notifier.remove(self._plasma_changed)

        self._plasma = value
        #self._plasma.notifier.add(self._plasma_changed)
        self._configure_geometry()

    @property
    def importance(self):
        return self._importance

    @importance.setter
    def importance(self, double value):
        
        self._importance = value
        self._configure_geometry()

    def _configure_geometry(self):
        """
        The beam bounding envelope is a cylinder aligned with the beam axis.
        The overall length of the cylinder is self.length and radius self.radius.
        The cylinder is divided into cylindical segments to optimize bounding shapes
        The length of the cylidrical segments is equal to 2 * self.radius, except the last segment.
        Length of the last segment can be different to match self.length value.
        """

        # no further work if there are no emission models
        if not list(self._models):
            return
        
        #no further work if there is no laser profile connected
        if self._laser_profile is None:
            return

        # clear geometry to remove segments
        for i in self._geometry:
            i.parent = None
        self._geometry = []

        #get geometry from laser_profile
        primitives = self._laser_profile.generate_geometry()

        #assign material and parent
        for i in primitives:
            i.parent = self
            i.material = LaserMaterial(self, i, list(self._models), self._integrator)

        self._geometry = primitives

    @property
    def laser_spectrum(self):
        return self._laser_spectrum

    @laser_spectrum.setter
    def laser_spectrum(self, LaserSpectrum value):
        self._laser_spectrum = value
        self._configure_geometry()

    @property
    def laser_profile(self):
        return self._laser_profile

    @laser_profile.setter
    def laser_profile(self, object value):

        self._laser_profile = value
        self._laser_profile.laser = self

        self._configure_geometry()

    @property
    def models(self):
        return list(self._models)

    @models.setter
    def models(self, value):
        # check necessary data is available
        if not self._plasma:
            raise ValueError('The laser must have a reference to a plasma object before specifying the scattering model.')

        if not self._laser_profile:
            raise ValueError('The laser must have a reference to a laser model object before specifying the scattering model.')

        if not self._laser_spectrum:
            raise ValueError('The laser must have a reference to a laser spectrum object before specifying scattering model.')

        self._models.set(value)
        self._configure_geometry()

    @property
    def integrator(self):
        return self._integrator

    @integrator.setter
    def integrator(self, VolumeIntegrator value):
        self._integrator = value
        self._configure_geometry()

    def get_geometry(self):
        return self._geometry

    def _plasma_changed(self):
        """React to change of plasma and propagate the information."""
        self._configure_geometry()

    def _modified(self):
        self._configure_geometry()
