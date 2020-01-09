
from raysect.optical cimport SpectralFunction, Spectrum, InterpolatedSF, Point3D, Vector3D
from cherab.core.laser.node cimport Laser

cdef class LaserModel:

    def __init__(self, Laser laser = None):
        if laser:
            self.laser = laser

    @property
    def laser(self):
        return self._laser

    @laser.setter
    def laser(self, Laser value):
        self._laser = value

    def set_polarization_function(self, VectorFunction3D function):
        self._polarization = function

    def set_pointing_function(self, VectorFunction3D function):
        self._pointing = function

    def set_power_density_function(self, Function3D function):
        self._power_density = function

    cpdef Vector3D get_pointing(self, double x, double y, double z):
        """
        Returns the pointing vector of the light at the specified point.
        
        The point is specified in the laser beam space.
        
        :param x: x coordinate in meters.
        :param y: y coordinate in meters. 
        :param z: z coordinate in meters.
        :return: Intensity in m^-3. 
        """

        return self._pointing.evaluate(x, y, z)

    cpdef Vector3D get_polarization(self, double x, double y, double z):
        """
        Returns vector denoting the laser polarisation.
        
        The point is specified in the laser beam space.
        
        :param x: x coordinate in meters.
        :param y: y coordinate in meters. 
        :param z: z coordinate in meters.
        :return: power density in Wm^-3. 
        """

        return self._polarization(x, y, z)

    cpdef double get_power_density(self, double x, double y, double z):
        """
        Returns the volumetric power density of the laser light at the specified point.
        The return value is a sum for all laser wavelengths.
        
        The point is specified in the laser beam space.
        
        :param x: x coordinate in meters in the laser frame. 
        :param y: y coordinate in meters in the laser frame. 
        :param z: z coordinate in meters in the laser frame.
        :return: power density in W*m^-3. 
        """

        return self._power_density.evaluate(x, y, z)


    def _change(self):
        """
        Called if the plasma, beam or the atomic data source properties change.

        If the model caches calculation data that would be invalidated if its
        source data changes then this method may be overridden to clear the
        cache.
        """

        pass


