from raysect.core.math.function.float cimport Function3D
from raysect.core.math.function.vector3d cimport Function3D as VectorFunction3D

from raysect.optical cimport SpectralFunction, Spectrum, InterpolatedSF, Point3D, Vector3D

from cherab.core.laser.node cimport Laser
from cherab.core.utility import Notifier

cdef class LaserProfile:

    def __init__(self):

        self.notifier = Notifier()

    def set_polarization_function(self, VectorFunction3D function):
        self._polarization3d = function

    def set_pointing_function(self, VectorFunction3D function):
        self._pointing3d = function

    def set_energy_density_function(self, Function3D function):
        self._energy_density3d = function
    
    cpdef Vector3D get_pointing(self, double x, double y, double z):
        """
        Returns the pointing vector of the light at the specified point.

        The point is specified in the laser beam space.

        :param x: x coordinate in meters.
        :param y: y coordinate in meters.
        :param z: z coordinate in meters.
        :return: Intensity in m^-3.
        """

        return self._pointing3d.evaluate(x, y, z)

    cpdef Vector3D get_polarization(self, double x, double y, double z):
        """
        Returns vector denoting the laser polarisation.

        The point is specified in the laser beam space.

        :param x: x coordinate in meters.
        :param y: y coordinate in meters.
        :param z: z coordinate in meters.
        :return: power density in Wm^-3.
        """

        return self._polarization3d(x, y, z)

    cpdef double get_energy_density(self, double x, double y, double z):
        """
        Returns the volumetric power density of the laser light at the specified point.
        The return value is a sum for all laser wavelengths.

        The point is specified in the laser beam space.

        :param x: x coordinate in meters in the laser frame.
        :param y: y coordinate in meters in the laser frame.
        :param z: z coordinate in meters in the laser frame.
        :return: power density in W*m^-3.
        """

        return self._energy_density3d.evaluate(x, y, z)

    @property
    def laser(self):
        return self._laser
    
    @laser.setter
    def laser(self, value):
        if not isinstance(value, Laser):
            raise TypeError("Value has to instance of Laser class.")
        
        if self._laser is not None:
            self.notifier.remove(self._laser._configure_geometry)
        
        self._laser = value
        self.notifier.add(self._laser._configure_geometry)
        self._change()
        
        self.notifier.notify()

    cpdef list generate_geometry(self):
        """returns list of raysect primitives composing the laser geometry"""

        raise NotImplementedError("Virtual function density not defined.")
    
    def _change(self):
        """
        Called if the laser properties change.

        If the model caches calculation data that would be invalidated if its
        source data changes then this method may be overridden to clear the
        cache.
        """

        pass
