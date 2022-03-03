from raysect.primitive cimport Cylinder
from raysect.optical cimport World, AffineMatrix3D, Primitive, Ray
from raysect.optical.material.emitter.inhomogeneous cimport NumericalIntegrator
from raysect.core cimport translate, Material

from cherab.core.laser.material cimport LaserMaterial
from cherab.core.laser.model cimport LaserModel
from cherab.core.laser.profile import LaserProfile
from cherab.core.laser.laserspectrum import LaserSpectrum
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
    """
    A scene-graph object representing a laser of laser light.

    The Cherab laser object holds basic information about the laser and connects
    the components which are needed for the laser description. With specified
    emission models it can contribute to observed radiation.

    The Laser object is a Raysect scene-graph node and lives in it's own
    coordinate space. This coordinate space is defined relative to it's parent
    scene-graph object by an AffineTransform. The beam parameters are defined
    in the Laser object coordinate space. Models using the beam object must
    convert any spatial coordinates into beam space before requesting values
    from the Laser object.

    The main physical properties of the laser are defined by the three
    attributes laser_spectrum, laser_profile and models. The laser_spectrum
    has to be an instance of LaserSpectrum and defines the spectral properties
    of the laser light. The laser_profile has to be an instance of LaserProfile
    and it holds all the space related definitions as volumetric distribution
    of laser light energy polarisation direction. In the models a list of LaserModels
    can be stored, which calculate the contribution of the laser ligth to the observed
    radiation. The models can cover various applications as for example
    Thomson scattering. Please see the documentation of individual classes
    for more detail.

    The shape of the laser (e.g. cylinder) and its parameters (e.g. radius)
    is controled by the LaserProfile.

    The plasma reference has to be specified to attach the any models.

    :param Node parent: The parent node in the Raysect scene-graph.
      See the Raysect documentation for more guidance.
    :param AffineMatrix3D transform: The transform defining the spatial position
      and orientation of this laser. See the Raysect documentation if you need
      guidance on how to use AffineMatrix3D transforms.
    :param str name: The name for this laser object.
    :ivar Plasma plasma: The plasma instance with which this laser interacts.
    :ivar float importance: The importance sampling factor.
    :ivar LaserSpectrum laser_spectrum: The LaserSpectrum instance with which this laser interacts.
    :ivar LaserProfile laser_profile: The LaserProfile instance with which this laser interacts.
    :ivar ModelManager models: The manager class that sets and provides access to the
      emission models for this laser.
    :ivar VolumeIntegrator integrator: The configurable method for doing
      volumetric integration through the laser along a Ray's path. Defaults to
      a numerical integrator with 1mm step size, NumericalIntegrator(step=0.001).
    """

    def __init__(self, object parent=None, AffineMatrix3D transform=None, str name=None):

        super().__init__(parent, transform, name)

        self._set_init_values()

        self.notifier = Notifier()

        self._models = ModelManager()
        
        self._integrator = NumericalIntegrator(step=1e-3)

        self._importance = 1.
        self._configure()

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
        self._configure()

    @property
    def importance(self):
        return self._importance

    @importance.setter
    def importance(self, double value):
        
        self._importance = value
        self._configure()

    def _configure(self):
        """
        Reconfigure the laser primitives and update references after changes.
        """

        # no further work if there are no emission models
        if not list(self._models):
            return
        
        #no further work if there is no laser profile connected
        if self._laser_profile is None:
            return

        # update the plasma reference in models
        for model in list(self._models):
            model.plasma = self._plasma

        # clear geometry parents to remove segments
        for i in self._geometry:
            i.parent = None
        self._geometry = []

        # rebuild geometry
        primitives = self._laser_profile.generate_geometry()

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
        self._configure()

    @property
    def laser_profile(self):
        return self._laser_profile

    @laser_profile.setter
    def laser_profile(self, LaserProfile value):

        self._laser_profile = value
        self._laser_profile.laser = self

        self._configure()

    @property
    def models(self):
        return list(self._models)

    @models.setter
    def models(self, value):
        
        # check necessary data is available
        if not all([self._plasma, self._laser_profile, self._laser_spectrum]):
            raise ValueError("The plasma, laser_profile and laser_spectrum must be set before before specifying any models.")

        self._models.set(value)
        self._configure()

    @property
    def integrator(self):
        return self._integrator

    @integrator.setter
    def integrator(self, VolumeIntegrator value):
        self._integrator = value
        self._configure()

    def get_geometry(self):
        return self._geometry

    def _plasma_changed(self):
        """React to change of plasma and propagate the information."""
        self._configure()

    def _modified(self):
        self._configure()
