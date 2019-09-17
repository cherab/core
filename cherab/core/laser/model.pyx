from cherab.core.utility import Notifier
from cherab.core.laser.node cimport Laser
from raysect.optical cimport SpectralFunction, Spectrum, InterpolatedSF, Point3D, Vector3D
from cherab.core.laser.scattering cimport ScatteringModel
from cherab.core.utility.constants cimport RECIP_2_PI

from libc.math cimport exp, sqrt, M_PI as pi, INFINITY as INF

import numpy as np

cdef class ModelManager:

    def __init__(self):
        self._models = []
        self.notifier = Notifier()

    def __iter__(self):
        return iter(self._models)

    cpdef object set(self, object models):

        # copy models and test it is an iterable
        models = list(models)

        # check contents of list are beam models
        for model in models:
            if not isinstance(model, ScatteringModel):
                raise TypeError('The model list must consist of only LaserModel objects.')

        self._models = models
        self.notifier.notify()

    cpdef object add(self, ScatteringModel model):

        if not model:
            raise ValueError('Model must not be None type.')

        self._models.append(model)
        self.notifier.notify()

    cpdef object clear(self):
        self._models = []
        self.notifier.notify()

cdef class LaserModel:

    def __init__(self):

        self._laser_spectrum = None

    @property
    def laser_spectrum(self):
        return self._laser_spectrum

    @laser_spectrum.setter
    def laser_spectrum(self, value):

        if not isinstance(value, Spectrum):
            raise TypeError("Laser spectrum has to be of type Spectrum, but {0} passed".format(type(value)))

        self._laser_spectrum = value

    cpdef Vector3D get_pointing(self, x, y, z):
        """
        Returns the pointing vector of the light at the specified point.
        
        The point is specified in the laser beam space.
        
        :param x: x coordinate in meters.
        :param y: y coordinate in meters. 
        :param z: z coordinate in meters.
        :return: Intensity in m^-3. 
        """

        raise NotImplementedError('Virtual method must be implemented in a sub-class.')

    cpdef Vector3D get_polarization(self, x, y, z):
        """
        Returns vector denoting the laser polarisation.
        
        The point is specified in the laser beam space.
        
        :param x: x coordinate in meters.
        :param y: y coordinate in meters. 
        :param z: z coordinate in meters.
        :return: power density in Wm^-3. 
        """

        raise NotImplementedError('Virtual method must be implemented in a sub-class.')

    cpdef float get_power_density(self, x, y, z):
        """
        Returns the power density of the light at the specified point.
        
        The point is specified in the laser beam space.
        
        :param x: x coordinate in meters.
        :param y: y coordinate in meters. 
        :param z: z coordinate in meters.
        :return: power density in Wm^-3. 
        """

        raise NotImplementedError('Virtual method must be implemented in a sub-class.')

    def _change(self):
        """
        Called if the plasma, beam or the atomic data source properties change.

        If the model caches calculation data that would be invalidated if its
        source data changes then this method may be overridden to clear the
        cache.
        """

        pass

cdef class UniformCylinderLaser_tester(LaserModel):

    def __init__(self, power_density=0, Vector3D polarization = Vector3D(0, 1, 0), central_wavelength = 1060, spectral_sigma = 0.01, wlen_min = 1059.8,
                 wlen_max=1060.2, nbins=100):

        super().__init__()

        self._power_density = power_density
        self._central_wavelength = central_wavelength
        self._spectral_sigma = spectral_sigma
        self._wlen_min = wlen_min
        self._wlen_max = wlen_max
        self._nbins = nbins
        self._polarization_vector = Vector3D(0, 1, 0)

        self._gaussian_spectrum()


    cpdef Vector3D get_pointing(self, x, y, z):
        """
        Returns the pointing vector of the light at the specified point.
        
        The point is specified in the laser beam space.
        
        :param x: x coordinate in meters.
        :param y: y coordinate in meters. 
        :param z: z coordinate in meters.
        :return: Intensity in m^-3. 
        """

        return Vector3D(0, 0, 1)

    cpdef float get_power_density(self, x, y, z):
        """
        Returns the power density of the light at the specified point.
        
        The point is specified in the laser beam space.
        
        :param x: x coordinate in meters.
        :param y: y coordinate in meters. 
        :param z: z coordinate in meters.
        :return: power density in Wm^-3. 
        """

        return self._power_density

    cpdef Vector3D get_polarization(self, x, y, z):
        """
        Returns vector denoting the laser polarisation.
        
        The point is specified in the laser beam space.
        
        :param x: x coordinate in meters.
        :param y: y coordinate in meters. 
        :param z: z coordinate in meters.
        :return: power density in Wm^-3. 
        """

        return self._polarization_vector

    @property
    def polarization(self):
        return self._polarization_vector

    @polarization.setter
    def polarization(self, Vector3D vector):

        if vector.length < 0:
            raise ValueError("Vector of 0 length is not allowed.")

        self._polarization_vector = vector.normalise()

    def _gaussian_spectrum(self):
        spectrum = Spectrum(self._wlen_min, self._wlen_max, self._nbins)
        wavelengths = spectrum.wavelengths
        samples = np.exp(-1/2 * np.power(wavelengths - self._central_wavelength, 2)/self._spectral_sigma**2) / spectrum.delta_wavelength

        interpolated = InterpolatedSF(wavelengths, samples, normalise=True)
        for index, value in enumerate(wavelengths):
            spectrum.samples_mv[index] = interpolated(value)

        self._laser_spectrum = spectrum


cdef class GaussianBeamAxisymmetric(LaserModel):

    def __init__(self, Laser laser = None, Vector3D polarization = Vector3D(0, 1, 0), power=1,
                 central_wavelength = 1060, spectral_sigma = 0.01, spectrum_wlen_min = 1059.8,
                 spectrum_wlen_max=1060.2, spectrum_nbins=100, laser_sigma = 0.01, waist_radius=0.001,
                 m2 = 1, z_focus=0):

        super().__init__()

        self._set_defaults()
        #laser sigma dependent constants
        self.spectrum_min_wavelength = spectrum_wlen_min
        self.spectrum_max_wavelength = spectrum_wlen_max
        #set laser constants
        self.laser_power = power
        self.spectral_mu = central_wavelength
        self.spectral_sigma = spectral_sigma
        self.spectrum_nbins = spectrum_nbins
        self.polarization = polarization.normalise()
        self.waist_radius = waist_radius
        self.m2 = m2
        self.z_focus = z_focus

    def _set_defaults(self):
        self._waist_radius = 0.1
        self._waist2 = 0.001
        self._spectral_mu = 1060
        self._spectrum_min_wavelength = 0
        self._spectrum_max_wavelength = INF
        self._spectral_sigma = 0.05
        self._spectrum_nbins = 1
        self._m2 = 1

    cpdef Vector3D get_pointing(self, x, y, z):
        """
        Returns the pointing vector of the light at the specified point.
        
        The point is specified in the laser beam space.
        
        :param x: x coordinate in meters.
        :param y: y coordinate in meters. 
        :param z: z coordinate in meters.
        :return: Intensity in m^-3. 
        """

        return Vector3D(0, 0, 1)

    cpdef float get_power_density(self, x, y, z):
        """
        Returns the power density of the light at the specified point.
        
        The point is specified in the laser beam space.
        
        :param x: x coordinate in meters.
        :param y: y coordinate in meters. 
        :param z: z coordinate in meters.
        :return: power density in Wm^-3. 
        """

        cdef:
            double power, width, r, power_axis

        r2 = x ** 2 + y ** 2

        width2 = self.get_beam_width2(z)
        power_axis = self._power_const / width2
        power = power_axis * exp(-2 * r2 / width2)
        return power

    cpdef double get_power_axis(self, z):
        return self._power_const / self.get_beam_width(z) ** 2

    cpdef Vector3D get_polarization(self, x, y, z):
        """
        Returns vector denoting the laser polarisation.
        
        The point is specified in the laser beam space.
        
        :param x: x coordinate in meters.
        :param y: y coordinate in meters. 
        :param z: z coordinate in meters.
        :return: power density in Wm^-3. 
        """

        return self._polarization_vector

    cpdef double get_beam_width2(self, z):
        return self._waist2 + self._waist_const * (z - self._z_focus) ** 2

    @property
    def z_focus(self):
        return self._z_focus

    @z_focus.setter
    def z_focus(self,value):
        self._z_focus = value

    @property
    def m2 (self):
        return self._m2
    @m2.setter
    def m2(self, value):

        if not value > 0:
            raise ValueError("Laser power has to be larger than 0.")

        self._m2 = value
        self._change_waist_const()

    @property
    def laser_power(self):
        return self._laser_power

    @laser_power.setter
    def laser_power(self, value):
        if not value > 0:
            raise ValueError("Laser power has to be larger than 0.")

        self._laser_power = value
        self._power_const = 2 * value / pi

    @property
    def polarization(self):
        return self._polarization_vector

    @polarization.setter
    def polarization(self, Vector3D value):

        if value.length == 0:
            raise ValueError("Polarisation can't be a zero length vector.")

        self._polarization_vector = value.normalise()

    @property
    def waist_radius(self):
        return self._waist_radius

    @waist_radius.setter
    def waist_radius(self,double value):

        if not value > 0:
            raise ValueError("Value has to be larger than 0.")

        self._waist_radius = value
        self._waist2 = value ** 2
        self._change_waist_const()


    def _change_waist_const(self):
        self._waist_const = (self._spectral_mu * 1e-9 * self._m2 / (pi * self._waist_radius)) ** 2


    @property
    def spectral_sigma(self):
        return self._spectral_sigma

    @spectral_sigma.setter
    def spectral_sigma(self, value):
        if value <= 0:
            raise ValueError("Spectral sigma has to be larger than 0.")

        self._spectral_sigma = value
        self._create_laser_spectrum()

    @property
    def spectral_mu(self):
        return self._spectral_mu

    @spectral_mu.setter
    def spectral_mu(self, value):
        if value <=0:
            raise ValueError("Central wavelength of the laser has to be larger than 0.")
        self._spectral_mu = value
        self._change_waist_const()
        self._create_laser_spectrum()

    @property
    def spectrum_nbins(self):
        return self._spectrum_nbins

    @spectrum_nbins.setter
    def spectrum_nbins(self, value):
        #verify correct value of value
        if not isinstance(value, int):
            try:
                value = int(value)
                print("Converting number of bins to integer.")
            except TypeError:
                raise TypeError("Value has to be convertable to integer.")

            if value <= 0:
                raise ValueError("Number of bins has to be larger than 0.")

        self._spectrum_nbins = value
        self._create_laser_spectrum()

    @property
    def spectrum_min_wavelength(self):
        return self._spectrum_min_wavelength


    @spectrum_min_wavelength.setter
    def spectrum_min_wavelength(self, value):

        if value >= self._spectrum_max_wavelength and self._spectrum_max_wavelength is not None:
            raise ValueError("Minimum spectrum wavelength should be smaller than maximum laser-spectrum wavelength.")

        self._spectrum_min_wavelength = value

    @property
    def spectrum_max_wavelength(self):
        return self._spectrum_max_wavelength


    @spectrum_max_wavelength.setter
    def spectrum_max_wavelength(self, value):

        if value <= self._spectrum_min_wavelength and self._spectrum_min_wavelength is not None:
            raise ValueError("Maximum spectrum wavelength should be larger than minimum laser-spectrum wavelength.")

        self._spectrum_max_wavelength = value

    def _create_laser_spectrum(self):
        if self._spectrum_max_wavelength is INF:
            raise UserWarning("Max laser wavelength is infinite, can't create laser spectrum.")
        if  self._spectrum_min_wavelength == 0:
            raise UserWarning("Min laser wavelength is o, can't create laser spectrum.")

        self._laser_spectrum = Spectrum(self._spectrum_min_wavelength, self._spectrum_max_wavelength, self._spectrum_nbins)
        wavelengths = self._laser_spectrum.wavelengths
        samples = np.exp(-1/2 * np.power(wavelengths - self._spectral_mu, 2)/self._spectral_sigma**2) / self._laser_spectrum.delta_wavelength

        interpolated = InterpolatedSF(wavelengths, samples, normalise=True)
        for index, value in enumerate(wavelengths):
            self._laser_spectrum.samples_mv[index] = interpolated(value)

cdef class GaussianAxisymmetricalConstant(LaserModel):

    def __init__(self, Laser laser=None, Plasma plasma=None, power=0, central_wavelength = 1060,
                 spectral_sigma = 0.01, spectrum_wlen_min = 1059.8, spectrum_wlen_max=1060.2, spectrum_nbins=100,
                 laser_sigma = 0.01, Vector3D polarisation_vector = Vector3D(0, 1, 0)):

        super().__init__()



        #laser sigma dependent constants
        self._const_width = 0
        self._recip_laser_sigma2 = 0
        self.spectrum_min_wavelength = spectrum_wlen_min
        self.spectrum_max_wavelength = spectrum_wlen_max

        #set laser constants
        self._laser_power = power
        self._spectral_mu = central_wavelength
        self._spectral_sigma = spectral_sigma
        self.spectrum_min_wavelength = spectrum_wlen_min
        self.spectrum_max_wavelength = spectrum_wlen_max
        self.spectrum_nbins = spectrum_nbins
        self.laser_sigma = laser_sigma
        self._polarization_vector = polarisation_vector.normalise()
        self._create_laser_spectrum()


    cpdef Vector3D get_pointing(self, x, y, z):
        """
        Returns the pointing vector of the light at the specified point.
        
        The point is specified in the laser beam space.
        
        :param x: x coordinate in meters.
        :param y: y coordinate in meters. 
        :param z: z coordinate in meters.
        :return: Intensity in m^-3. 
        """

        return Vector3D(0, 0, 1)

    cpdef float get_power_density(self, x, y, z):
        """
        Returns the power density of the light at the specified point.
        
        The point is specified in the laser beam space.
        
        :param x: x coordinate in meters.
        :param y: y coordinate in meters. 
        :param z: z coordinate in meters.
        :return: power density in Wm^-3. 
        """

        cdef:
            double r2, norm_intensity

        r2 = x ** 2 + y ** 2

        norm_intensity = self._const_width * exp(-0.5 * r2 * self._recip_laser_sigma2)

        return self._laser_power * norm_intensity

    cpdef Vector3D get_polarization(self, x, y, z):
        """
        Returns vector denoting the laser polarisation.
        
        The point is specified in the laser beam space.
        
        :param x: x coordinate in meters.
        :param y: y coordinate in meters. 
        :param z: z coordinate in meters.
        :return: power density in Wm^-3. 
        """
        return self._polarization_vector

    @property
    def polarization(self):
        return self._polarization_vector

    @polarization.setter
    def polarization(self, Vector3D vector):

        if vector.length < 0:
            raise ValueError("Vector of 0 length is not allowed.")

        self._polarization_vector = vector.normalise()

    @property
    def spectral_sigma(self):
        return self._spectral_sigma

    @spectral_sigma.setter
    def spectral_sigma(self, value):
        if value <= 0:
            raise ValueError("Spectral sigma has to be larger than 0.")

        self._spectral_sigma = value

        self._create_laser_spectrum()

    @property
    def spectral_mu(self):
        return self._spectral_mu

    @spectral_mu.setter
    def spectral_mu(self, value):
        if value <=0:
            raise ValueError("Central wavelength of the laser has to be larger than 0.")

        self._spectral_mu = value

        self._create_laser_spectrum()

    @property
    def spectrum_nbins(self):
        return self._spectrum_nbins

    @spectrum_nbins.setter
    def spectrum_nbins(self, value):
        #verify correct value of value
        if not isinstance(value, int):
            try:
                value = int(value)
                print("Converting number of bins to integer.")
            except TypeError:
                raise TypeError("Value has to be convertable to integer.")

            if value <= 0:
                raise ValueError("Number of bins has to be larger than 0.")

        self._spectrum_nbins = value
        self._create_laser_spectrum()

    @property
    def spectrum_min_wavelength(self):
        return self._spectrum_min_wavelength


    @spectrum_min_wavelength.setter
    def spectrum_min_wavelength(self, value):

        if value >= self._spectrum_max_wavelength:
            raise ValueError("Minimum spectrum wavelength should be smaller than maximum laser-spectrum wavelength.")

        self._spectrum_min_wavelength = value

    @property
    def spectrum_max_wavelength(self):
        return self._spectrum_max_wavelength


    @spectrum_max_wavelength.setter
    def spectrum_max_wavelength(self, value):

        if value <= self._spectrum_min_wavelength:
            raise ValueError("Maximum spectrum wavelength should be larger than minimum laser-spectrum wavelength.")

        self._spectrum_max_wavelength = value

    def _create_laser_spectrum(self):

        self._laser_spectrum = Spectrum(self._spectrum_min_wavelength, self._spectrum_max_wavelength, self._spectrum_nbins)
        wavelengths = self._laser_spectrum.wavelengths
        print("test")
        samples = np.exp(-1/2 * np.power(wavelengths - self._spectral_mu, 2)/self._spectral_sigma**2) / self._laser_spectrum.delta_wavelength

        interpolated = InterpolatedSF(wavelengths, samples, normalise=True)
        for index, value in enumerate(wavelengths):
            self._laser_spectrum.samples_mv[index] = interpolated(value)

        self._laser_spectrum = self._laser_spectrum


    @property
    def laser_sigma(self):
        return self._laser_sigma

    @laser_sigma.setter
    def laser_sigma(self, value):
        if value <= 0:
            raise ValueError("Standard deviation of the laser power has to be larger than 0.")

        self._laser_sigma = value

        self._const_width = 1 / (self._laser_sigma * sqrt(2 * pi))
        self._recip_laser_sigma2 = 1 / self._laser_sigma ** 2

    @property
    def laser_power(self):
        return self._laser_power

    @laser_power.setter
    def laser_power(self, value):
        if value <=0:
            raise ValueError("Laser power has to be larger than 0.")

        self._laser_power = value

