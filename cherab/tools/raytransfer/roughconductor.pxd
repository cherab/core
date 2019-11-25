from raysect.optical cimport SpectralFunction
from raysect.optical.material cimport RoughConductor
from raysect.optical cimport Vector3D, Spectrum


cdef class RToptimisedRoughConductor(RoughConductor):

    cdef double _d(self, Vector3D s_half)

    cdef double _g(self, Vector3D s_incoming, Vector3D s_outgoing)

    cdef double _g1(self, Vector3D v)

    cdef double _fresnel_conductor(self, double ci, double n, double k) nogil
