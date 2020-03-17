from raysect.core.math.function cimport Function3D
from raysect.optical cimport Spectrum, Point3D, Vector3D

from cherab.core.math.function cimport VectorFunction3D
from cherab.core.laser.node cimport Laser

cdef class LaserModel:

    cdef:
        object __weakref__
        VectorFunction3D _polarization3d, _pointing3d
        Function3D _power_density3d

    cpdef Vector3D get_pointing(self, double x, double y, double z)

    cpdef Vector3D get_polarization(self, double x, double y, double z)

    cpdef double get_power_density(self, double x, double y, double z)
