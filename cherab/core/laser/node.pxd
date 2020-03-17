from raysect.optical cimport Point3D, Vector3D, Node, Spectrum, Primitive
from raysect.optical.material.emitter.inhomogeneous cimport VolumeIntegrator

from cherab.core.plasma cimport Plasma
from cherab.core.laser.models.model_base cimport LaserModel
from cherab.core.laser.models.laserspectrum_base cimport LaserSpectrum
from cherab.core.laser.scattering cimport ScatteringModel

cdef class Laser(Node):

    cdef:
        object notifier
        Vector3D BEAM_AXIS
        double _length, _radius
        Plasma _plasma
        ScatteringModel  _scattering_model
        LaserModel _laser_model
        LaserSpectrum _laser_spectrum
        list _geometry
        VolumeIntegrator _integrator

    cdef object __weakref__

    cdef Plasma get_plasma(self)