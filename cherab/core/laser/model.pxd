from raysect.optical cimport Spectrum, Point3D, Vector3D

from cherab.core.plasma.node cimport Plasma
from cherab.core.laser.node cimport Laser
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
        Spectrum _laser_spectrum

    cdef object __weakref__

    cpdef Vector3D get_pointing(self, x, y, z)

    cpdef Vector3D get_polarization(self, x, y, z)

    cpdef float get_power_density(self, x, y, z)

cdef class UniformCylinderLaser_tester(LaserModel):

    cdef:
        float _power_density, _spectral_sigma, _central_wavelength, _wlen_min, _wlen_max, _nbins
        Vector3D _polarization_vector


cdef class GaussianBeamAxisymmetric(LaserModel):

    cdef:
        double _laser_power, _spectral_mu, _spectral_sigma, _spectrum_min_wavelength, _spectrum_max_wavelength
        double _laser_sigma, _recip_laser_sigma2, _const_width, _waist_radius, _waist_const, _waist2, _power_const
        double _m2, _z_focus # laser beam quality
        int _spectrum_nbins
        Vector3D _polarization_vector

    cpdef double get_beam_width2(self, z)

    cpdef double get_power_axis(self, z)


cdef class GaussianAxisymmetricalConstant(LaserModel):

    cdef:
        double _laser_power, _spectral_mu, _spectral_sigma, _spectrum_min_wavelength, _spectrum_max_wavelength, _laser_sigma, _recip_laser_sigma2, _const_width
        int _spectrum_nbins
        Vector3D _polarization_vector