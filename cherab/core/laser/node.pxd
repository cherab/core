from raysect.optical cimport Point3D, Vector3D, Node, Spectrum, Primitive
from raysect.optical.material.emitter.inhomogeneous cimport VolumeIntegrator

from cherab.core.plasma cimport Plasma
from cherab.core.laser.model cimport LaserModel
from cherab.core.laser.scattering cimport ScatteringModel

cdef class ModelManager:

    cdef:
        list _models
        readonly object notifier
        object _object_type

    cpdef object set(self, object models)

    cpdef object add(self, object model)

    cpdef object clear(self)


cdef class Laser(Node):

    cdef:
        object notifier
        Vector3D BEAM_AXIS
        double _length, _radius
        Plasma _plasma
        ModelManager  _scattering_models
        LaserModel _laser_model
        Primitive _geometry
        VolumeIntegrator _integrator


    cdef object __weakref__

    cdef Plasma get_plasma(self)