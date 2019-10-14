from raysect.optical cimport Spectrum, Point3D, Vector3D

from cherab.core.laser.scattering cimport ScatteringModel

cdef class ModelManager:

    cdef:
        list _models
        readonly object notifier

    cpdef object set(self, object models)

    cpdef object add(self, ScatteringModel model)

    cpdef object clear(self)

cdef class LaserModel:

    cdef:
        _laser_spectrum

    cdef object __weakref__

    cpdef Vector3D get_pointing(self, double x, double y, double z)

    cpdef Vector3D get_polarization(self, double x, double y, double z)

    cpdef double get_power_density(self, double x, double y, double z, double wavelength)

    cpdef Spectrum get_power_density_spectrum(self, double x, double y, double z)

    cpdef double get_laser_spectrum(self)

