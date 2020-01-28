from raysect.primitive cimport Cylinder

from raysect.optical cimport World, AffineMatrix3D, Primitive, Ray
from raysect.optical.material.emitter.inhomogeneous cimport NumericalIntegrator
from raysect.core cimport translate

from cherab.core.laser.material cimport LaserMaterial
from cherab.core.laser.scattering cimport ScatteringModel
from cherab.core.laser.models.laserspectrum_base import LaserSpectrum
from cherab.core.utility import Notifier
from libc.math cimport M_PI


cdef double DEGREES_TO_RADIANS = (M_PI / 180)

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
        self._configure_geometry()

    @property
    def plasma(self):
        return self._plasma

    @plasma.setter
    def plasma(self, Plasma value not None):
        self._plasma = value
        self._configure_geometry()

        self._plasma_changed()

    def set_importance(self, value):
        self._geometry.material.importance = value

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
        # self._geometry.transform = translate(0, 0, 0)

        # build plasma material
        self._geometry.material = LaserMaterial(self, self._integrator)

    @property
    def laser_spectrum(self):
        return self._laser_spectrum

    @laser_spectrum.setter
    def laser_spectrum(self, LaserSpectrum value):
        self._laser_spectrum = value
        self._laser_spectrum_changed()

    @property
    def laser_model(self):
        return self._laser_model

    @laser_model.setter
    def laser_model(self, object value):

        self._laser_model = value

        self._laser_model_changed()

    @property
    def scattering_model(self):
        return self._scattering_model

    @scattering_model.setter
    def scattering_model(self, ScatteringModel value):

        # check necessary data is available
        if not self._plasma:
            raise ValueError('The laser must have a reference to a plasma object to be used with a scattering model.')

        if not self._laser_model:
            raise ValueError('The laser must have a reference to have laser models specified to be used with a scattering model.')

        if not self._laser_spectrum:
            raise ValueError('The laser must have a reference to have laser spectrum specified to be used with a scattering model.')

        self._scattering_model = value
        self._configure_geometry()

        self._scattering_changed()

    def _laser_model_changed(self):
        if self._scattering_model is not None:
            self._scattering_model.laser_model = self._laser_model

    def _laser_spectrum_changed(self):
        if self._scattering_model is not None:
            self._scattering_model.set_laser_spectrum(self._laser_spectrum)

    def _scattering_changed(self):
        if self._plasma is not None:
            self._scattering_model.plasma = self._plasma
        if self._laser_model is not None:
            self._scattering_model.laser_model = self._laser_model
        if self._laser_spectrum is not None:
            self._scattering_model.set_laser_spectrum(self._laser_spectrum)

    def _plasma_changed(self):
        self._configure_geometry()
        if self._scattering_model is not None:
            self._scattering_model.plasma = self._plasma

    def _generate_geometry(self):
        # the beam bounding envelope is a cylinder aligned with the beam axis,
        # sharing the same coordinate space the cylinder radius is width,
        # the cylinder length is length
        return Cylinder(radius=self.radius, height=self.length,
                        transform=translate(0, 0, 0))

    @property
    def integrator(self):
        return self._integrator

    @integrator.setter
    def integrator(self, VolumeIntegrator value):
        self._integrator = value
        self._configure_geometry()
