from raysect.core cimport Vector3D
from raysect.optical cimport Primitive
from raysect.optical.material.emitter.inhomogeneous cimport NumericalIntegrator

from cherab.core.beam.distribution cimport BeamDistribution
from cherab.core.atomic cimport AtomicData, Element
from cherab.core.plasma cimport Plasma
from cherab.core.beam.model cimport BeamAttenuator

cdef class ThinBeam(BeamDistribution):

    cdef:
        readonly Vector3D BEAM_AXIS
        double _energy, _power, _temperature, _speed
        double _divergence_x, _divergence_y
        bint _z_outofbounds
        double _length, _sigma
        Plasma _plasma
        AtomicData _atomic_data
        BeamAttenuator _attenuator
        Primitive _geometry
    
    cpdef double get_energy(self)

    cpdef double get_speed(self)

    cpdef double get_power(self)

    cpdef double get_divergence_x(self)

    cpdef double get_divergence_y(self)

    cpdef double get_length(self)

    cpdef double get_sigma(self)

    cpdef Plasma get_plasma(self)