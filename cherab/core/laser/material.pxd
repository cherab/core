from raysect.optical.material.emitter cimport InhomogeneousVolumeEmitter
from raysect.core.math cimport AffineMatrix3D

from cherab.core.plasma cimport Plasma
from cherab.core.laser.node cimport Laser
from cherab.core.laser.models.model_base cimport LaserModel
from cherab.core.laser.scattering cimport ScatteringModel
from cherab.core.laser.models.laserspectrum_base cimport LaserSpectrum

cdef class LaserMaterial(InhomogeneousVolumeEmitter):

    cdef:
        Laser _laser
        Plasma _plasma
        ScatteringModel _scattering_model
        AffineMatrix3D _laser_to_plasma
        double[::1] _laser_wavelength_mv, _laser_spectrum_power_mv
        double _laser_delta_wavelength
        LaserSpectrum _laser_spectrum
        LaserModel _laser_model
        int _laser_bins

    cdef object __weakref__
