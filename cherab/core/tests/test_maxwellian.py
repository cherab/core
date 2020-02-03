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

import unittest

import numpy as np

from cherab.core.distribution import Maxwellian
from raysect.core import Vector3D

ATOMIC_MASS = 1.66053906660e-27
ELEMENTARY_CHARGE = 1.602176634e-19


class TestMaxwellian(unittest.TestCase):

    def setUp(self):
        self.x = np.linspace(-10, 10, 5)  # m
        self.y = np.linspace(-10, 10, 5)  # m
        self.z = np.linspace(-10, 10, 5)  # m
        self.vx = np.linspace(-10e5, 10e5, 5)  # m/s
        self.vy = np.linspace(-10e5, 10e5, 5)  # m/s
        self.vz = np.linspace(-10e5, 10e5, 5)  # m/s

    def tearDown(self):
        pass

    def test_bulk_velocity(self):
        density = lambda x, y, z: 6e19 * (1 + 0.1 * np.sin(x) * np.sin(y) * np.sin(z))  # m^-3
        temperature = lambda x, y, z: 3e3 * (1 + 0.1 * np.sin(x + 1) * np.sin(y + 1) * np.sin(z + 1))  # eV
        velocity = lambda x, y, z: 1.6e5 * (1 + 0.1 * np.sin(x + 2) * np.sin(y + 2) * np.sin(z + 2)) * Vector3D(1, 2, 3).normalise()  # m/s
        mass = 4 * ATOMIC_MASS  # kg
        maxwellian = Maxwellian(density, temperature, velocity, mass)

        for x in self.x:
            for y in self.y:
                for z in self.z:
                    self.assertAlmostEqual(maxwellian.bulk_velocity(x, y, z).x, velocity(x, y, z).x, delta=1e-10,
                                           msg='bulk_velocity method gives a wrong value at ({}, {}, {}).'.format(x, y, z))
                    self.assertAlmostEqual(maxwellian.bulk_velocity(x, y, z).y, velocity(x, y, z).y, delta=1e-10,
                                           msg='bulk_velocity method gives a wrong value at ({}, {}, {}).'.format(x, y, z))
                    self.assertAlmostEqual(maxwellian.bulk_velocity(x, y, z).z, velocity(x, y, z).z, delta=1e-10,
                                           msg='bulk_velocity method gives a wrong value at ({}, {}, {}).'.format(x, y, z))

    def test_effective_temperature(self):
        density = lambda x, y, z: 6e19 * (1 + 0.1 * np.sin(x) * np.sin(y) * np.sin(z))  # m^-3
        temperature = lambda x, y, z: 3e3 * (1 + 0.1 * np.sin(x + 1) * np.sin(y + 1) * np.sin(z + 1))  # eV
        velocity = lambda x, y, z: 1.6e5 * (1 + 0.1 * np.sin(x + 2) * np.sin(y + 2) * np.sin(z + 2)) * Vector3D(1, 2, 3).normalise()  # m/s
        mass = 4 * ATOMIC_MASS  # kg
        maxwellian = Maxwellian(density, temperature, velocity, mass)

        for x in self.x:
            for y in self.y:
                for z in self.z:
                    self.assertAlmostEqual(maxwellian.effective_temperature(x, y, z), temperature(x, y, z), delta=1e-10,
                                           msg='effective_temperature method gives a wrong value at ({}, {}, {}).'.format(x, y, z))

    def test_density(self):
        density = lambda x, y, z: 6e19 * (1 + 0.1 * np.sin(x) * np.sin(y) * np.sin(z))  # m^-3
        temperature = lambda x, y, z: 3e3 * (1 + 0.1 * np.sin(x + 1) * np.sin(y + 1) * np.sin(z + 1))  # eV
        velocity = lambda x, y, z: 1.6e5 * (1 + 0.1 * np.sin(x + 2) * np.sin(y + 2) * np.sin(z + 2)) * Vector3D(1, 2, 3).normalise()  # m/s
        mass = 4 * ATOMIC_MASS  # kg
        maxwellian = Maxwellian(density, temperature, velocity, mass)

        for x in self.x:
            for y in self.y:
                for z in self.z:
                    self.assertAlmostEqual(maxwellian.density(x, y, z), density(x, y, z), delta=1e-10,
                                           msg='density method gives a wrong value at ({}, {}, {}).'.format(x, y, z))

    def test_value(self):
        density = lambda x, y, z: 6e19 * (1 + 0.1 * np.sin(x) * np.sin(y) * np.sin(z))  # m^-3
        temperature = lambda x, y, z: 3e3 * (1 + 0.1 * np.sin(x + 1) * np.sin(y + 1) * np.sin(z + 1))  # eV
        velocity = lambda x, y, z: 1.6e5 * (1 + 0.1 * np.sin(x + 2) * np.sin(y + 2) * np.sin(z + 2)) * Vector3D(1, 2, 3).normalise()  # m/s
        mass = 4 * ATOMIC_MASS  # kg
        maxwellian = Maxwellian(density, temperature, velocity, mass)

        sigma = lambda x, y, z: np.sqrt(temperature(x, y, z) * ELEMENTARY_CHARGE / mass)  # m/s
        phase_space_density = lambda x, y, z, vx, vy, vz: density(x, y, z) / (np.sqrt(2 * np.pi) * sigma(x, y, z)) ** 3 \
                                                          * np.exp(-(Vector3D(vx, vy, vz) - velocity(x, y, z)).length ** 2 / (2 * sigma(x, y, z) ** 2))  # s^3/m^6

        # testing only half the values to avoid huge execution time
        for x in self.x[::2]:
            for y in self.y[::2]:
                for z in self.z[::2]:
                    for vx in self.vx[::2]:
                        for vy in self.vy[::2]:
                            for vz in self.vz[::2]:
                                self.assertAlmostEqual(maxwellian(x, y, z, vx, vy, vz), phase_space_density(x, y, z, vx, vy, vz), delta=1e-10,
                                                       msg='call method gives a wrong phase space density at ({}, {}, {}, {}, {}, {}).'.format(x, y, z, vx, vy, vz))


if __name__ == '__main__':
    unittest.main()