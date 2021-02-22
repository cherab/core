from raysect.optical cimport Point3D, Vector3D, Node, Spectrum, Primitive
from raysect.optical.material.emitter.inhomogeneous cimport VolumeIntegrator
from raysect.primitive cimport Cylinder

from cherab.core.plasma cimport Plasma
from cherab.core.laser.models.profile_base cimport LaserProfile
from cherab.core.laser.models.laserspectrum_base cimport LaserSpectrum
from cherab.core.laser.scattering cimport LaserEmissionModel
cdef class ModelManager:

    cdef:
        list _models
        readonly object notifier

    cpdef object set(self, object models)

    cpdef object add(self, LaserEmissionModel model)

    cpdef object clear(self)

cdef class Laser(Node):

    cdef:
        readonly object notifier
        double _length, _radius, _importance
        Plasma _plasma
        ModelManager  _models
        LaserProfile _laser_profile
        LaserSpectrum _laser_spectrum
        list _geometry
        VolumeIntegrator _integrator
        

    cdef object __weakref__