from raysect.optical.material.emitter cimport InhomogeneousVolumeEmitter
from raysect.core.math cimport AffineMatrix3D
from cherab.core.plasma cimport Plasma
from cherab.core.laser.node cimport Laser
from cherab.core.laser.scattering cimport ScatteringModel

cdef class LaserMaterial(InhomogeneousVolumeEmitter):

    cdef:
        Laser _laser
        Plasma _plasma
        ScatteringModel _scattering_model
        AffineMatrix3D _laser_to_plasma


    cdef object __weakref__
