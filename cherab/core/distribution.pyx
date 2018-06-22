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

    def __init__(self):
        self.notifier = Notifier()

    cdef double evaluate(self, double x, double y, double z, double vx, double vy, double vz) except? -1e999:
        """

        :param x: position in meters
        :param y: position in meters
        :param z: position in meters
        :param vx: velocity in meters per second
        :param vy: velocity in meters per second
        :param vz: velocity in meters per second
        :return: phase space density in s^3/m^6
        """

        raise NotImplementedError()

    def __call__(self, double x, double y, double z, double vx, double vy, double vz):
        return self.evaluate(x, y, z, vx, vy, vz)

    cpdef Vector3D bulk_velocity(self, double x, double y, double z):
        """

        :param x: position in meters
        :param y: position in meters
        :param z: position in meters
        :return: velocity vector in m/s
        """

        raise NotImplementedError()

    cpdef double effective_temperature(self, double x, double y, double z) except? -1e999:
        """

        :param x: position in meters
        :param y: position in meters
        :param z: position in meters
        :return: temperature in eV
        """

        raise NotImplementedError()

    cpdef double density(self, double x, double y, double z) except? -1e999:
        """

        :param x: position in meters
        :param y: position in meters
        :param z: position in meters
        :return: density in m^-3
        """

        raise NotImplementedError()


cdef class Maxwellian(DistributionFunction):

    def __init__(self, object density, object temperature, object velocity, double atomic_mass):
        """
        :param Function3D density: 3D function defining the density in cubic meters.
        :param Function3D temperature: 3D function defining the temperature in eV.
        :param VectorFunction3D velocity: 3D vector function defining the bulk velocity in meters per second.
        :param double atomic_mass: Atomic mass of the species in kg.
        """

        super().__init__()
        self._density = autowrap_function3d(density)
        self._temperature = autowrap_function3d(temperature)
        self._velocity = autowrap_vectorfunction3d(velocity)
        self._atomic_mass = atomic_mass

    @cython.cdivision(True)
    cdef double evaluate(self, double x, double y, double z, double vx, double vy, double vz) except? -1e999:
        """

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

        :param x: position in meters
        :param y: position in meters
        :param z: position in meters
        :return: velocity vector in m/s
        """

        return self._velocity.evaluate(x, y, z)

    cpdef double effective_temperature(self, double x, double y, double z) except? -1e999:
        """

        :param x: position in meters
        :param y: position in meters
        :param z: position in meters
        :return: temperature in eV
        """

        return self._temperature.evaluate(x, y, z)

    cpdef double density(self, double x, double y, double z) except? -1e999:
        """

        :param x: position in meters
        :param y: position in meters
        :param z: position in meters
        :return: density in m^-3
        """

        return self._density.evaluate(x, y, z)


