from raysect.optical cimport Spectrum, Point3D, Vector3D

from cherab.core.laser.models.model_base cimport LaserModel

cdef class UniformPowerDensity(LaserModel):

    cdef:
        float _power_density, _spectral_sigma, _central_wavelength, _wlen_min, _wlen_max, _nbins
        Vector3D _polarization_vector


cdef class GaussianBeamAxisymmetric(LaserModel):

    cdef:
        double _laser_power, _const_width, _waist_radius, _waist_const, _waist2, _power_const, _focus_z
        double _m2 # laser beam quality
        Vector3D _polarization_vector

    cpdef double get_beam_width2(self, double z, double wavelength)

    cpdef double get_power_axis(self, double z, double wavelength)


cdef class GaussianAxisymmetricalConstant(LaserModel):

    cdef:
        double _laser_power, _spectral_mu, _spectral_sigma, _spectrum_min_wavelength, _spectrum_max_wavelength, _laser_sigma, _recip_laser_sigma2, _const_width
        int _spectrum_nbins
        Vector3D _polarization_vector
