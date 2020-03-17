from raysect.primitive cimport Cylinder

from raysect.optical cimport World, AffineMatrix3D, Primitive, Ray
from raysect.optical.material.emitter.inhomogeneous cimport NumericalIntegrator
from raysect.core cimport translate

from cherab.core.laser.material cimport LaserMaterial
from cherab.core.laser.scattering cimport ScatteringModel
from cherab.core.laser.models.laserspectrum_base import LaserSpectrum
from cherab.core.utility import Notifier
from libc.math cimport M_PI

from math import ceil

cdef double DEGREES_TO_RADIANS = (M_PI / 180)

cdef class Laser(Node):

    def __init__(self, object parent=None, AffineMatrix3D transform=None,
                 str name=None):

        super().__init__(parent, transform, name)

        # change reporting and tracking
        self.notifier = Notifier()

        # set material integrator
        self._integrator = NumericalIntegrator(step=1e-3)

        # laser beam properties
        self.BEAM_AXIS = Vector3D(0.0, 0.0, 1.0)
        self._length = 1.0                         # [m]
        self._radius = 0.1                         # [m]
        self._geometry = []
        self._configure_geometry()

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
        self._configure_geometry()

    @property
    def plasma(self):
        return self._plasma

    @plasma.setter
    def plasma(self, Plasma value not None):
        # check necessary data is available
        if not self._scattering_model:
            raise ValueError('The laser must have a reference to a scattering model object before specifying plasma.')

        if not self._laser_model:
            raise ValueError('The laser must have a reference to a laser model object before specifying plasma.')

        if not self._laser_spectrum:
            raise ValueError('The laser must have a reference to a laser spectrum object before specifying plasma.')

        self._plasma = value

        self.notifier.notify()

    def set_importance(self, value):
        for i in self._geometry:
            i.material.importance = value

    cdef Plasma get_plasma(self):
        return self._plasma

    def _configure_geometry(self):
        """
        The beam bounding envelope is a cylinder aligned with the beam axis.
        The overall length of the cylinder is self.length and radius self.radius.
        The cylinder is divided into cylindical segments to optimize bounding shapes
        The length of the cylidrical segments is equal to 2 * self.radius, except the last segment.
        Length of the last segment can be different to match self.length value.
        """
        # disconnect segments
        for i in self._geometry:
            i.parent = None

        # length of first n-1 segments is 2 * radius
        radius = self.radius  # radius of segments
        segment_length = 2 * radius  # length of segments
        n_segments = int(self.length // segment_length)  # number of segments
        self._geometry = []
        if n_segments > 1:
            for i in range(n_segments - 1):
                cylinder = Cylinder(name="Laser segment {0:d}".format(i), radius=radius, height=segment_length,
                                    transform=translate(0, 0, i * segment_length), parent=self,
                                    material=LaserMaterial(self, self._integrator))
                cylinder.material.laser = self

                self._geometry.append(cylinder)

        # length of the last segment is laser_length - n * 2 radius to avoid infinitesimal reminder segments
        segment_length_last = self.length - (n_segments - 1) * segment_length
        cylinder = Cylinder(name="Laser segment {0:d}".format(n_segments - 1), radius=radius, height=segment_length_last,
                            transform=translate(0, 0, (n_segments - 1) * segment_length), parent=self,
                            material=LaserMaterial(self, self._integrator))

        self._geometry.append(cylinder)

        self.notifier.notify()

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
    def scattering_model(self):
        return self._scattering_model

    @scattering_model.setter
    def scattering_model(self, ScatteringModel value):

        self._scattering_model = value

        self.notifier.notify()

    @property
    def integrator(self):
        return self._integrator

    @integrator.setter
    def integrator(self, VolumeIntegrator value):
        self._integrator = value
        self._configure_geometry()

    def get_geometry(self):
        return self._geometry
