# Copyright 2016-2021 Euratom
# Copyright 2016-2021 United Kingdom Atomic Energy Authority
# Copyright 2016-2021 Centro de Investigaciones Energéticas, Medioambientales y Tecnológicas
#
# Licensed under the EUPL, Version 1.1 or – as soon they will be approved by the
# European Commission - subsequent versions of the EUPL (the "Licence");
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at:
#
# https://joinup.ec.europa.eu/software/page/eupl5
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the Licence is distributed on an "AS IS" basis, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied.
#
# See the Licence for the specific language governing permissions and limitations
# under the Licence.


from raysect.core.math.function.float cimport Function3D
from raysect.core.math.function.vector3d cimport Function3D as VectorFunction3D

from raysect.optical cimport SpectralFunction, Spectrum, InterpolatedSF, Point3D, Vector3D

from cherab.core.laser.node cimport Laser
from cherab.core.utility import Notifier


cdef class LaserProfile:
    """
    LaserProfile base class.

    This is an abstract class and cannot be used for observing.
    
    Provides information about spatial properties of the laser beam:
    direction of the laser propagation (direction 
    of the Poynting vector), polarisation of the ligth as the direction
    of the electric component vector and volumetric energy density of 
    the laser light.

    All the laser properties are evaluated in the frame of reference of
    the laser.

    .. warning::
        When combining a LaserProfile with a LaserSpectrum for a laser,
        a special care has to be given to obtain the correct power
        of the scattered spectrum. Scattering models can multiply
        both the spectral power density given by the LaserProfile and
        the volumetric energy density given by the LaserProfile.
        Combination of incompatible cases may yield incorrect
        values of scattered power.

    :ivar Laser laser: The Laser scenegraph node the LaserProfile
      is connected to.
    """

    def __init__(self):

        self.notifier = Notifier()

    def set_polarization_function(self, VectorFunction3D function):
        """
        Assigns the 3D vector function describing the polarisation vector.

        The polarisation is given as the direction of the electric
        component of the electromagnetic wave.

        The function is specified in the laser space.

        :param VectorFunction3D function: A 3D vector function describing
          the polarisation vector.
        """
        self._polarization3d = function

    def set_pointing_function(self, VectorFunction3D function):
        """
        Assings the 3D vector function describing the direction of the laser propagation.

        The direction of the laser light propagation is the direction
        of the Poynting vector.

        :param VectorFunction3D function: A 3D vector function describing
          the laser light propagation direction 
        """
        self._pointing3d = function

    def set_energy_density_function(self, Function3D function):
        """
        Assigns the 3D scalar function describing the laser energy distribution.

        The laser power distribution is the value of the volumetric
        energy density of the laser light.
        """
        self._energy_density3d = function
    
    cpdef Vector3D get_pointing(self, double x, double y, double z):
        """
        Returns the laser light propagation direction.

        At the point (x, y, z) in the laser space.

        :param x: x coordinate in meters.
        :param y: y coordinate in meters.
        :param z: z coordinate in meters.
        :return: Intensity in m^-3.
        """

        return self._pointing3d.evaluate(x, y, z)

    cpdef Vector3D get_polarization(self, double x, double y, double z):
        """
        Returns a vector denoting the laser polarisation.

        The polarisation direction is the direction of the electric
        component of the electromagnetic wave for the point (x, y, z)
        in the laser space.

        :param x: x coordinate in meters.
        :param y: y coordinate in meters.
        :param z: z coordinate in meters.
        :return: power density in Wm^-3.
        """

        return self._polarization3d(x, y, z)

    cpdef double get_energy_density(self, double x, double y, double z):
        """
        Returns the volumetric energy density of the laser light in W*m^-3.
        
        At the point (x, y, z) in the laser space.

        :param x: x coordinate in meters in the laser frame.
        :param y: y coordinate in meters in the laser frame.
        :param z: z coordinate in meters in the laser frame.
        :return: power density in W*m^-3.
        """

        return self._energy_density3d.evaluate(x, y, z)

    cpdef list generate_geometry(self):
        """
        returns list of raysect primitives composing the laser geometry
        
        This method is called from the Laser instance to which the instance
        of Profile is attached to. The Laser instance will be assigned as
        the parent to the returned primitives in the Laser._configure method.
        The Laser._configure method does not change any transforms. This is
        why the returned primitives have to have their transforms already
        initialised in the frame of the laser, when returned.
        """

        raise NotImplementedError("Virtual function density not defined.")
    
    def _change(self):
        """
        Called if the laser properties change.

        If the model caches calculation data that would be invalidated if its
        source data changes then this method may be overridden to clear the
        cache.
        """

        pass
