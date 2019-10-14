
from raysect.optical cimport SpectralFunction, Spectrum, InterpolatedSF, Point3D, Vector3D

from cherab.core.utility import Notifier
from cherab.core.laser.scattering cimport ScatteringModel

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

    cpdef Vector3D get_pointing(self, double x, double y, double z):
        """
        Returns the pointing vector of the light at the specified point.
        
        The point is specified in the laser beam space.
        
        :param x: x coordinate in meters.
        :param y: y coordinate in meters. 
        :param z: z coordinate in meters.
        :return: Intensity in m^-3. 
        """

        raise NotImplementedError('Virtual method must be implemented in a sub-class.')

    cpdef Vector3D get_polarization(self, double x, double y, double z):
        """
        Returns vector denoting the laser polarisation.
        
        The point is specified in the laser beam space.
        
        :param x: x coordinate in meters.
        :param y: y coordinate in meters. 
        :param z: z coordinate in meters.
        :return: power density in Wm^-3. 
        """

        raise NotImplementedError('Virtual method must be implemented in a sub-class.')

    cpdef double get_power_density(self, double x, double y, double z, double wavelength):
        """
        Returns the volumetric power density of the laser light at the specified point.
        The return value is a sum for all laser wavelengths.
        
        The point is specified in the laser beam space.
        
        :param x: x coordinate in meters in the laser frame. 
        :param y: y coordinate in meters in the laser frame. 
        :param z: z coordinate in meters in the laser frame.
        :return: power density in W*m^-3. 
        """

        raise NotImplementedError('Virtual method must be implemented in a sub-class.')

    cpdef Spectrum get_power_density_spectrum(self, double x, double y, double z):
        """
        Returns Spectrum of the the volumetric spectral power density of the laser light at the specified point.
        
        The point is specified in the laser beam space.
        
        :param x: x coordinate in meters in the laser frame. 
        :param y: y coordinate in meters in the laser frame. 
        :param z: z coordinate in meters in the laser frame.
        :return: spectral power density in W * m^-3 * nm^-1. 
        """

        raise NotImplementedError('Virtual method must be implemented in a sub-class.')

    cpdef double get_laser_spectrum(self):
        """
        Retruns ratio of the power enclosed in the specified wavelength range
        
        :param minimum_wavelength: Lower boundary of the wavelength band. 
        :param maximum_wavelength: Upper boundary of the wavelength band. 
        :return: 
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


