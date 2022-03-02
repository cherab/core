from raysect.optical cimport Vector3D, Point3D
from raysect.optical.spectrum cimport Spectrum

from cherab.core cimport Plasma
from cherab.core.laser.profile cimport LaserProfile
from cherab.core.laser.laserspectrum cimport LaserSpectrum


cdef class LaserModel:
    cdef:
        Plasma _plasma
        LaserSpectrum _laser_spectrum
        LaserProfile _laser_profile


    cpdef Spectrum emission(self, Point3D point_plasma, Vector3D observation_plasma, Point3D point_laser,
                            Vector3D observation_laser, Spectrum spectrum)

    cdef object __weakref__
