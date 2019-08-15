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

    cpdef Vector3D pointing(self, x, y, z):
        """
        Returns the pointing vector of the light at the specified point.
        
        The point is specified in the laser beam space.
        
        :param x: x coordinate in meters.
        :param y: y coordinate in meters. 
        :param z: z coordinate in meters.
        :return: Intensity in m^-3. 
        """

        raise NotImplementedError('Virtual method must be implemented in a sub-class.')

    cpdef Vector3D polarization(self, x, y, z):
        """
        Returns vector denoting the laser polarisation.
        
        The point is specified in the laser beam space.
        
        :param x: x coordinate in meters.
        :param y: y coordinate in meters. 
        :param z: z coordinate in meters.
        :return: power density in Wm^-3. 
        """

        raise NotImplementedError('Virtual method must be implemented in a sub-class.')

    cpdef float power_density(self, x, y, z):
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

    def __init__(self, power_density=0, central_wavelength = 1060, spectral_sigma = 0.01, wlen_min = 1059.8,
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


    cpdef Vector3D pointing(self, x, y, z):
        """
        Returns the pointing vector of the light at the specified point.
        
        The point is specified in the laser beam space.
        
        :param x: x coordinate in meters.
        :param y: y coordinate in meters. 
        :param z: z coordinate in meters.
        :return: Intensity in m^-3. 
        """

        return Vector3D(0, 0, 1)

    cpdef float power_density(self, x, y, z):
        """
        Returns the power density of the light at the specified point.
        
        The point is specified in the laser beam space.
        
        :param x: x coordinate in meters.
        :param y: y coordinate in meters. 
        :param z: z coordinate in meters.
        :return: power density in Wm^-3. 
        """

        return self._power_density

    cpdef Vector3D polarization(self, x, y, z):
        """
        Returns vector denoting the laser polarisation.
        
        The point is specified in the laser beam space.
        
        :param x: x coordinate in meters.
        :param y: y coordinate in meters. 
        :param z: z coordinate in meters.
        :return: power density in Wm^-3. 
        """

        return self._polarization_vector

    def set_polarisation(self, Vector3D vector):

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


cdef class GaussianAxisymmetricalConstant(LaserModel):

    def __init__(self, Laser laser=None, Plasma plasma=None, power=0, central_wavelength = 1060,
                 spectral_sigma = 0.01, spectrum_wlen_min = 1059.8, spectrum_wlen_max=1060.2, spectrum_nbins=100,
                 laser_sigma = 0.01):

        super().__init__()

        #laser sigma dependent constants
        self._const_width = 0
        self._recip_laser_sigma2 = 0
        self._spectrum_min_wavelength = 1
        self._spectrum_max_wavelength = INF

        #set laser constants
        self._laser_power = power
        self._spectral_mu = central_wavelength
        self._spectral_sigma = spectral_sigma
        self.spectrum_min_wavelength = spectrum_wlen_min
        self.spectrum_max_wavelength = spectrum_wlen_max
        self.spectrum_nbins = spectrum_nbins
        self.laser_sigma = laser_sigma
        self._polarization_vector = Vector3D(0, 1, 0)
        self._create_laser_spectrum()


    cpdef Vector3D pointing(self, x, y, z):
        """
        Returns the pointing vector of the light at the specified point.
        
        The point is specified in the laser beam space.
        
        :param x: x coordinate in meters.
        :param y: y coordinate in meters. 
        :param z: z coordinate in meters.
        :return: Intensity in m^-3. 
        """

        return Vector3D(0, 0, 1)

    cpdef float power_density(self, x, y, z):
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

    cpdef Vector3D polarization(self, x, y, z):
        """
        Returns vector denoting the laser polarisation.
        
        The point is specified in the laser beam space.
        
        :param x: x coordinate in meters.
        :param y: y coordinate in meters. 
        :param z: z coordinate in meters.
        :return: power density in Wm^-3. 
        """

        return self._polarization_vector

    def set_polarisation(self, Vector3D vector):

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

