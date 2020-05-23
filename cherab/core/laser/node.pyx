from raysect.primitive cimport Cylinder
from raysect.optical cimport World, AffineMatrix3D, Primitive, Ray
from raysect.optical.material.emitter.inhomogeneous cimport NumericalIntegrator
from raysect.core cimport translate, Material

from cherab.core.laser.material cimport LaserMaterial
from cherab.core.laser.scattering cimport LaserEmissionModel
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
            if not isinstance(model, LaserEmissionModel):
                raise TypeError('The model list must consist of only LaserEmissionModel objects.')

        self._models = models
        self.notifier.notify()

    cpdef object add(self, LaserEmissionModel model):

        if not model:
            raise ValueError('Model must not be None type.')

        self._models.append(model)
        self.notifier.notify()

    cpdef object clear(self):
        self._models = []
        self.notifier.notify()

cdef class Laser(Node):

    def __init__(self, double length=1, double radius=0.05, object parent=None, AffineMatrix3D transform=None,
                 double importance=0., str name=None):

        super().__init__(parent, transform, name)

        # set init values of the laser
        self._set_init_values()

        # change reporting and tracking
        self.notifier = Notifier()
        self.notifier.add(self._configure_geometry)

        #setup model manager
        self._models = ModelManager()
        
        # set material integrator
        self._integrator = NumericalIntegrator(step=1e-3)

        # laser beam properties
        self.length = length                         # [m]
        self.radius = radius                         # [m]

        self._importance = importance
        self._configure_geometry()

    def _set_init_values(self):
        """
        Sets initial values of the laser shape to avoid errors.
        """
        self._length = 1
        self._radius = 0.5
        self._importance = 0.
        self._geometry = []

    @property
    def length(self):
        return self._length

    @length.setter
    def length(self, value):

        if value <= 0:
            raise ValueError("Laser length has to be larger than 0.")

        self._length = value
        self.notifier.notify()

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, value):

        if value <= 0:
            raise ValueError("Laser radius has to be larger than 0.")

        self._radius = value
        self.notifier.notify()

    @property
    def plasma(self):
        return self._plasma

    @plasma.setter
    def plasma(self, Plasma value not None):

        #unregister from old plasma notifier
        if self._plasma is not None:
            self._plasma.notifier.remove(self._plasma_changed)

        self._plasma = value
        self._plasma.notifier.add(self._plasma_changed)
        self.notifier.notify()

    @property
    def importance(self):
        return self._importance

    @importance.setter
    def importance(self, double value):
        
        self._importance = value
        self.notifier.notify()

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

        # clear geometry to remove segments
        for i in self._geometry:
            i.parent = None
        self._geometry[:] = []

        # length of first n-1 segments is 2 * radius
        radius = self.radius  # radius of segments
        n_segments = int(self.length // (2 * radius))  # number of segments

        if n_segments > 1:
            segment_length = self.length / n_segments
            for i in range(n_segments):
                segment = Cylinder(name="Laser segment {0:d}".format(i), radius=radius, height=segment_length,
                                    transform=translate(0, 0, i * segment_length), parent=self)
                segment.material = LaserMaterial(self, segment, list(self._models), self._integrator)

                self._geometry.append(segment)
        elif 0 <= n_segments < 2:
                segment = Cylinder(name="Laser segment {0:d}".format(0), radius=radius, height=self.length,
                                    parent=self)
                segment.material = LaserMaterial(self, segment, list(self._models), self._integrator)

                self._geometry.append(segment)
        else:
            raise ValueError("Incorrect number of segments calculated.")

    @property
    def laser_spectrum(self):
        return self._laser_spectrum

    @laser_spectrum.setter
    def laser_spectrum(self, LaserSpectrum value):
        self._laser_spectrum = value
        self.notifier.notify()

    @property
    def laser_model(self):
        return self._laser_model

    @laser_model.setter
    def laser_model(self, object value):

        self._laser_model = value

        self.notifier.notify()

    @property
    def models(self):
        return list(self._models)

    @models.setter
    def models(self, value):
        # check necessary data is available
        if not self._plasma:
            raise ValueError('The laser must have a reference to a plasma object before specifying the scattering model.')

        if not self._laser_model:
            raise ValueError('The laser must have a reference to a laser model object before specifying the scattering model.')

        if not self._laser_spectrum:
            raise ValueError('The laser must have a reference to a laser spectrum object before specifying scattering model.')

        self._models.set(value)
        self.notifier.notify()

    @property
    def integrator(self):
        return self._integrator

    @integrator.setter
    def integrator(self, VolumeIntegrator value):
        self._integrator = value
        self.notifier.notify()

    def get_geometry(self):
        return self._geometry

    def _plasma_changed(self):
        """React to change of plasma and propagate the information."""
        self.notifier.notify()

    def _modified(self):
        self.notifier.notify()
