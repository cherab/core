from raysect.core.math.function.float cimport Function3D
from raysect.core.math.function.vector3d cimport Function3D as VectorFunction3D

from raysect.optical cimport Spectrum, Point3D, Vector3D

from cherab.core.laser.node cimport Laser

cdef class LaserProfile:

    cdef:
        VectorFunction3D _polarization3d, _pointing3d
        Function3D _energy_density3d
        Laser _laser
        readonly object notifier

    cpdef Vector3D get_pointing(self, double x, double y, double z)

    cpdef Vector3D get_polarization(self, double x, double y, double z)

    cpdef double get_energy_density(self, double x, double y, double z)

    cpdef list generate_geometry(self)