from raysect.core.scenegraph._nodebase cimport _NodeBase
from raysect.core.math cimport AffineMatrix3D
from raysect.optical.material.emitter cimport InhomogeneousVolumeEmitter

from cherab.core.laser.node cimport Laser

cdef class LaserMaterial(InhomogeneousVolumeEmitter):

    cdef:
        AffineMatrix3D _laser_to_plasma, _laser_segment_to_laser_node
        list _models

    cdef object __weakref__
