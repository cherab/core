# Copyright 2016-2018 Euratom
# Copyright 2016-2018 United Kingdom Atomic Energy Authority
# Copyright 2016-2018 Centro de Investigaciones Energéticas, Medioambientales y Tecnológicas
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

# cython: language_level=3

from cherab.core.utility import Notifier

from libc.math cimport exp, M_PI
from raysect.optical cimport Vector3D
cimport cython

from cherab.core.math cimport autowrap_function3d, autowrap_vectorfunction3d
from cherab.core.utility.constants cimport ELEMENTARY_CHARGE


# must be immutable, once created cannot be modified as changes are not tracked.
cdef class DistributionFunction:
    """
    The distribution function base class.

    All plasma Species objects are defined through distribution functions. This base
    class defines the core interface which is used by the rest of the framework.
    This base class cannot be used on its own, it raises a NotImplementedError() on all
    method calls. Users should either use a simpler distribution instance
    (such as the Maxwellian distribution) or implement their own.
    """

    def __init__(self):
        self.notifier = Notifier()

    def __call__(self, double x, double y, double z, double vx, double vy, double vz):
        """
        Evaluates the phase space density at the specified point in 6D phase space.

        Wraps the cython evaluate() function for fast evaluation.

        :param float x: position in meters
        :param float y: position in meters
        :param float z: position in meters
        :param float vx: velocity in meters per second
        :param float vy: velocity in meters per second
        :param float vz: velocity in meters per second
        :return: phase space density in s^3/m^6
        """
        return self.evaluate(x, y, z, vx, vy, vz)

    cdef double evaluate(self, double x, double y, double z, double vx, double vy, double vz) except? -1e999:
        """
        Evaluates the phase space density at the specified point in 6D phase space.
        
        :param float x: position in meters
        :param float y: position in meters
        :param float z: position in meters
        :param float vx: velocity in meters per second
        :param float vy: velocity in meters per second
        :param float vz: velocity in meters per second
        :return: phase space density in s^3/m^6
        """
        raise NotImplementedError()

    cpdef Vector3D bulk_velocity(self, double x, double y, double z):
        """
        Evaluates the species' bulk velocity at the specified 3D coordinate.

        :param float x: position in meters
        :param float y: position in meters
        :param float z: position in meters
        :return: velocity vector in m/s
        :rtype: Vector3D
        """
        raise NotImplementedError()

    cpdef double effective_temperature(self, double x, double y, double z) except? -1e999:
        """
        Evaluates the species' effective temperature at the specified 3D coordinate.

        :param float x: position in meters
        :param float y: position in meters
        :param float z: position in meters
        :return: temperature in eV
        :rtype: float
        """
        raise NotImplementedError()

    cpdef double density(self, double x, double y, double z) except? -1e999:
        """
        Evaluates the species' density at the specified 3D coordinate.

        :param float x: position in meters
        :param float y: position in meters
        :param float z: position in meters
        :return: density in m^-3
        :rtype: float
        """
        raise NotImplementedError()


cdef class ZeroDistribution(DistributionFunction):
    """
    A zero distribution function.

    All distribution properties are zero.
    """

    def __init__(self):
        super().__init__()

    def __call__(self, double x, double y, double z, double vx, double vy, double vz):
        """
        Evaluates the phase space density at the specified point in 6D phase space.

        Wraps the cython evaluate() function for fast evaluation.

        :param float x: position in meters
        :param float y: position in meters
        :param float z: position in meters
        :param float vx: velocity in meters per second
        :param float vy: velocity in meters per second
        :param float vz: velocity in meters per second
        :return: phase space density 0 s^3/m^6
        """
        return self.evaluate(x, y, z, vx, vy, vz)

    cdef double evaluate(self, double x, double y, double z, double vx, double vy, double vz) except? -1e999:
        """
        Evaluates the phase space density at the specified point in 6D phase space.
        
        :param float x: position in meters
        :param float y: position in meters
        :param float z: position in meters
        :param float vx: velocity in meters per second
        :param float vy: velocity in meters per second
        :param float vz: velocity in meters per second
        :return: phase space density 0 s^3/m^6
        """
        return 0.0

    cpdef Vector3D bulk_velocity(self, double x, double y, double z):
        """
        Evaluates the species' bulk velocity at the specified 3D coordinate.

        :param float x: position in meters
        :param float y: position in meters
        :param float z: position in meters
        :return: velocity vector (0, 0, 0) in m/s
        :rtype: Vector3D
        """
        return Vector3D(0, 0, 0)

    cpdef double effective_temperature(self, double x, double y, double z) except? -1e999:
        """
        Returns 0 species' effective temperature at the specified 3D coordinate.

        :param float x: position in meters
        :param float y: position in meters
        :param float z: position in meters
        :return: temperature 0 eV
        :rtype: float
        """
        return 0.0

    cpdef double density(self, double x, double y, double z) except? -1e999:
        """
        Returns 0 species' density.

        :param float x: position in meters
        :param float y: position in meters
        :param float z: position in meters
        :return: density 0 m^-3
        :rtype: float
        """
        return 0.0


cdef class Maxwellian(DistributionFunction):
    """
    A Maxwellian distribution function.

    This class implements a Maxwell-Boltzmann distribution, the statistical distribution
    describing a system of particles that have reached thermodynamic equilibrium. The
    user supplies 3D functions that provide the mean density, temperature and velocity
    respectively.

    :param Function3D density: 3D function defining the density in cubic meters.
    :param Function3D temperature: 3D function defining the temperature in eV.
    :param VectorFunction3D velocity: 3D vector function defining the bulk velocity in meters per second.
    :param double atomic_mass: Atomic mass of the species in kg.

    .. code-block:: pycon

       >>> from scipy.constants import atomic_mass
       >>> from raysect.core.math import Vector3D
       >>> from cherab.core import Maxwellian
       >>> from cherab.core.atomic import deuterium
       >>>
       >>> # Setup distribution for a slab of plasma in thermodynamic equilibrium
       >>> d0_density = 1E17
       >>> d0_temperature = 1
       >>> bulk_velocity = Vector3D(0, 0, 0)
       >>> d0_distribution = Maxwellian(d0_density, d0_temperature, bulk_velocity, deuterium.atomic_weight * atomic_mass)
    """

    def __init__(self, object density, object temperature, object velocity, double atomic_mass):

        super().__init__()
        self._density = autowrap_function3d(density)
        self._temperature = autowrap_function3d(temperature)
        self._velocity = autowrap_vectorfunction3d(velocity)
        self._atomic_mass = atomic_mass

    @cython.cdivision(True)
    cdef double evaluate(self, double x, double y, double z, double vx, double vy, double vz) except? -1e999:
        """
        Evaluates the phase space density at the specified point in 6D phase space.

        :param x: position in meters
        :param y: position in meters
        :param z: position in meters
        :param vx: velocity in meters per second
        :param vy: velocity in meters per second
        :param vz: velocity in meters per second
        :return: phase space density in s^3/m^6
        """

        cdef:
            double k1, k2, ux, uy, uz
            Vector3D bulk_velocity

        k1 = self._atomic_mass / (2 * ELEMENTARY_CHARGE * self._temperature.evaluate(x, y, z))
        k2 = (k1 / M_PI) ** 1.5

        bulk_velocity = self._velocity.evaluate(x, y, z)
        ux = vx - bulk_velocity.x
        uy = vy - bulk_velocity.y
        uz = vz - bulk_velocity.z

        return self._density.evaluate(x, y, z) * exp(-k1 * (ux*ux + uy*uy + uz*uz)) * k2

    cpdef Vector3D bulk_velocity(self, double x, double y, double z):
        """
        Evaluates the species' bulk velocity at the specified 3D coordinate.

        :param x: position in meters
        :param y: position in meters
        :param z: position in meters
        :return: velocity vector in m/s
        
        .. code-block:: pycon

           >>> d0_distribution.bulk_velocity(1, 0, 0)
           Vector3D(0.0, 0.0, 0.0)
        """

        return self._velocity.evaluate(x, y, z)

    cpdef double effective_temperature(self, double x, double y, double z) except? -1e999:
        """

        :param x: position in meters
        :param y: position in meters
        :param z: position in meters
        :return: temperature in eV
        
        .. code-block:: pycon
        
           >>> d0_distribution.effective_temperature(1, 0, 0)
           1.0
        """

        return self._temperature.evaluate(x, y, z)

    cpdef double density(self, double x, double y, double z) except? -1e999:
        """

        :param x: position in meters
        :param y: position in meters
        :param z: position in meters
        :return: density in m^-3

        .. code-block:: pycon

           >>> d0_distribution.density(1, 0, 0)
           1e+17
        """

        return self._density.evaluate(x, y, z)


