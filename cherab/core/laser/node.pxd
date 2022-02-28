from raysect.optical cimport Point3D, Vector3D, Node, Spectrum, Primitive
from raysect.optical.material.emitter.inhomogeneous cimport VolumeIntegrator
from raysect.primitive cimport Cylinder

from cherab.core.plasma cimport Plasma
from cherab.core.laser.profile cimport LaserProfile
from cherab.core.laser.laserspectrum cimport LaserSpectrum
from cherab.core.laser.model cimport LaserModel


cdef class ModelManager:

    cdef:
        list _models
        readonly object notifier

    cpdef object set(self, object models)

    cpdef object add(self, LaserModel model)

    cpdef object clear(self)


cdef class Laser(Node):

    cdef:
        readonly object notifier
        double _importance
        Plasma _plasma
        ModelManager  _models
        LaserProfile _laser_profile
        LaserSpectrum _laser_spectrum
        list _geometry
        VolumeIntegrator _integrator
        

    cdef object __weakref__