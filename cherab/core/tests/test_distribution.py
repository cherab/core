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

from raysect.core import Vector3D

from cherab.core.distribution import ZeroDistribution, GenericDistribution, Maxwellian
from cherab.core.utility.constants import ATOMIC_MASS, ELEMENTARY_CHARGE


# Note: DistributionFunction is a cdef class (abstract base class) that cannot be
# directly instantiated or subclassed from Python. The abstract methods raise
# NotImplementedError, which is tested implicitly through the concrete implementations
# (ZeroDistribution, Maxwellian, GenericDistribution) that inherit from it.


class TestZeroDistribution(unittest.TestCase):
    """
    Test cases for the ZeroDistribution class.
    
    ZeroDistribution should return zero for all distribution properties.
    """

    def setUp(self):
        self.distribution = ZeroDistribution()
        self.x = np.linspace(-10, 10, 5)  # m
        self.y = np.linspace(-10, 10, 5)  # m
        self.z = np.linspace(-10, 10, 5)  # m
        self.vx = np.linspace(-10e5, 10e5, 5)  # m/s
        self.vy = np.linspace(-10e5, 10e5, 5)  # m/s
        self.vz = np.linspace(-10e5, 10e5, 5)  # m/s

    def tearDown(self):
        pass

    def test_call_returns_zero(self):
        """Test that __call__() returns zero for all inputs."""
        for x in self.x[::2]:  # Test subset to avoid long execution time
            for y in self.y[::2]:
                for z in self.z[::2]:
                    for vx in self.vx[::2]:
                        for vy in self.vy[::2]:
                            for vz in self.vz[::2]:
                                result = self.distribution(x, y, z, vx, vy, vz)
                                self.assertEqual(result, 0.0,
                                                 msg='__call__() should return 0.0 at ({}, {}, {}, {}, {}, {}).'.format(
                                                     x, y, z, vx, vy, vz))

    def test_bulk_velocity_returns_zero_vector(self):
        """Test that bulk_velocity() returns zero vector for all positions."""
        for x in self.x:
            for y in self.y:
                for z in self.z:
                    velocity = self.distribution.bulk_velocity(x, y, z)
                    self.assertAlmostEqual(velocity.x, 0.0, delta=1e-10,
                                           msg='bulk_velocity().x should be 0.0 at ({}, {}, {}).'.format(x, y, z))
                    self.assertAlmostEqual(velocity.y, 0.0, delta=1e-10,
                                           msg='bulk_velocity().y should be 0.0 at ({}, {}, {}).'.format(x, y, z))
                    self.assertAlmostEqual(velocity.z, 0.0, delta=1e-10,
                                           msg='bulk_velocity().z should be 0.0 at ({}, {}, {}).'.format(x, y, z))

    def test_effective_temperature_returns_zero(self):
        """Test that effective_temperature() returns zero for all positions."""
        for x in self.x:
            for y in self.y:
                for z in self.z:
                    temperature = self.distribution.effective_temperature(x, y, z)
                    self.assertEqual(temperature, 0.0,
                                     msg='effective_temperature() should return 0.0 at ({}, {}, {}).'.format(x, y, z))

    def test_density_returns_zero(self):
        """Test that density() returns zero for all positions."""
        for x in self.x:
            for y in self.y:
                for z in self.z:
                    density = self.distribution.density(x, y, z)
                    self.assertEqual(density, 0.0,
                                     msg='density() should return 0.0 at ({}, {}, {}).'.format(x, y, z))


class TestGenericDistribution(unittest.TestCase):
    """
    Test cases for the GenericDistribution class.
    
    GenericDistribution allows users to provide custom 6D phase space density,
    density, temperature, and velocity functions.
    """

    def setUp(self):
        self.x = np.linspace(-5, 5, 3)  # m
        self.y = np.linspace(-5, 5, 3)  # m
        self.z = np.linspace(-5, 5, 3)  # m
        self.vx = np.linspace(-5e5, 5e5, 3)  # m/s
        self.vy = np.linspace(-5e5, 5e5, 3)  # m/s
        self.vz = np.linspace(-5e5, 5e5, 3)  # m/s

        # Define atomic mass for Gaussian distribution (using deuterium mass)
        self.atomic_mass = 2 * ATOMIC_MASS  # kg

        # Define shared density and temperature functions for 3D Gaussian distribution
        self.density = lambda x, y, z: 1e20 * (1 + 0.1 * np.sin(x) * np.sin(y) * np.sin(z))  # m^-3
        self.temperature = lambda x, y, z: 1e3 * (1 + 0.1 * np.sin(x + 1) * np.sin(y + 1) * np.sin(z + 1))  # eV
        self.velocity = lambda x, y, z: Vector3D(1e5 * x, 2e5 * y, 3e5 * z)  # m/s
        
        
        # Define 3D Gaussian phase space density function using density and temperature
        # This implements a Maxwellian distribution: f = n * (m/(2*pi*e*T))^(3/2) * exp(-m*v^2/(2*e*T))
        def phase_space_density_gaussian(x, y, z, vx, vy, vz):
            n = self.density(x, y, z)
            T = self.temperature(x, y, z)
            m = self.atomic_mass
            
            # Thermal velocity spread squared
            sigma_sq = T * ELEMENTARY_CHARGE / m  # (m/s)^2
            
            # Velocity magnitude squared (assuming zero bulk velocity for simplicity)
            v_sq = vx**2 + vy**2 + vz**2
            
            # Normalization factor
            norm = (m / (2 * np.pi * ELEMENTARY_CHARGE * T)) ** 1.5
            
            # Gaussian distribution
            return n * norm * np.exp(-v_sq / (2 * sigma_sq))
        
        self.phase_space_density = phase_space_density_gaussian

    def test_bulk_velocity(self):
        """Test that bulk_velocity() returns the correct velocity vector."""
        # Define velocity function
        
        distribution = GenericDistribution(self.phase_space_density, self.density, self.temperature, self.velocity)
        
        for x in self.x:
            for y in self.y:
                for z in self.z:
                    result = distribution.bulk_velocity(x, y, z)
                    expected = self.velocity(x, y, z)
                    self.assertAlmostEqual(result.x, expected.x, delta=1e-10,
                                           msg='bulk_velocity().x is wrong at ({}, {}, {}).'.format(x, y, z))
                    self.assertAlmostEqual(result.y, expected.y, delta=1e-10,
                                           msg='bulk_velocity().y is wrong at ({}, {}, {}).'.format(x, y, z))
                    self.assertAlmostEqual(result.z, expected.z, delta=1e-10,
                                           msg='bulk_velocity().z is wrong at ({}, {}, {}).'.format(x, y, z))

    def test_effective_temperature(self):
        """Test that effective_temperature() returns the correct temperature."""
        
        distribution = GenericDistribution(self.phase_space_density, self.density, self.temperature, self.velocity)
        
        for x in self.x:
            for y in self.y:
                for z in self.z:
                    result = distribution.effective_temperature(x, y, z)
                    expected = self.temperature(x, y, z)
                    self.assertAlmostEqual(result, expected, delta=1e-10,
                                           msg='effective_temperature() is wrong at ({}, {}, {}).'.format(x, y, z))

    def test_density(self):
        """Test that density() returns the correct density."""
        
        distribution = GenericDistribution(self.phase_space_density, self.density, self.temperature, self.velocity)
        
        for x in self.x:
            for y in self.y:
                for z in self.z:
                    result = distribution.density(x, y, z)
                    expected = self.density(x, y, z)
                    self.assertAlmostEqual(result, expected, delta=1e-10,
                                           msg='density() is wrong at ({}, {}, {}).'.format(x, y, z))

    def test_call(self):
        """Test that __call__() returns the correct phase space density using 3D Gaussian."""
        
        distribution = GenericDistribution(self.phase_space_density, self.density, self.temperature, self.velocity)
        
        # Test subset to avoid long execution time
        for x in self.x:
            for y in self.y:
                for z in self.z[::2]:
                    for vx in self.vx[::2]:
                        for vy in self.vy[::2]:
                            for vz in self.vz[::2]:
                                result = distribution(x, y, z, vx, vy, vz)
                                expected = self.phase_space_density(x, y, z, vx, vy, vz)
                                self.assertAlmostEqual(result, expected, delta=1e-10,
                                                       msg='__call__() is wrong at ({}, {}, {}, {}, {}, {}).'.format(
                                                           x, y, z, vx, vy, vz))

    def test_float_inputs(self):
        """Test GenericDistribution with float inputs instead of lambda functions."""
        # Pass floats directly - they should be converted to constant functions
        density_float = 1e20  # m^-3
        temperature_float = 1e3  # eV
        velocity_constant = Vector3D(1e5, 2e5, 3e5)  # m/s (constant vector function)
        phase_space_density_float = 1e17  # s^3/m^6
        
        distribution = GenericDistribution(phase_space_density_float, density_float, temperature_float, velocity_constant)
        
        # Test that all methods work with constant float inputs
        for x in self.x:
            for y in self.y:
                for z in self.z:
                    # Density should be constant
                    self.assertAlmostEqual(distribution.density(x, y, z), density_float, delta=1e-10,
                                           msg='density() should return constant value at ({}, {}, {}).'.format(x, y, z))
                    # Temperature should be constant
                    self.assertAlmostEqual(distribution.effective_temperature(x, y, z), temperature_float, delta=1e-10,
                                           msg='effective_temperature() should return constant value at ({}, {}, {}).'.format(x, y, z))
                    # Velocity should be constant
                    vel = distribution.bulk_velocity(x, y, z)
                    self.assertAlmostEqual(vel.x, velocity_constant.x, delta=1e-10,
                                           msg='bulk_velocity().x should return constant value at ({}, {}, {}).'.format(x, y, z))
                    self.assertAlmostEqual(vel.y, velocity_constant.y, delta=1e-10,
                                           msg='bulk_velocity().y should return constant value at ({}, {}, {}).'.format(x, y, z))
                    self.assertAlmostEqual(vel.z, velocity_constant.z, delta=1e-10,
                                           msg='bulk_velocity().z should return constant value at ({}, {}, {}).'.format(x, y, z))
        
        # Test phase space density is constant
        for x in self.x[::2]:
            for y in self.y[::2]:
                for z in self.z[::2]:
                    for vx in self.vx[::2]:
                        for vy in self.vy[::2]:
                            for vz in self.vz[::2]:
                                result = distribution(x, y, z, vx, vy, vz)
                                self.assertAlmostEqual(result, phase_space_density_float, delta=1e-10,
                                                       msg='__call__() should return constant value at ({}, {}, {}, {}, {}, {}).'.format(
                                                           x, y, z, vx, vy, vz))


class TestMaxwellian(unittest.TestCase):
    """
    Test cases for the Maxwellian class.
    
    Maxwellian implements a Maxwell-Boltzmann distribution function.
    """

    def setUp(self):
        self.x = np.linspace(-10, 10, 5)  # m
        self.y = np.linspace(-10, 10, 5)  # m
        self.z = np.linspace(-10, 10, 5)  # m
        self.vx = np.linspace(-10e5, 10e5, 5)  # m/s
        self.vy = np.linspace(-10e5, 10e5, 5)  # m/s
        self.vz = np.linspace(-10e5, 10e5, 5)  # m/s
        
        # Define shared density, temperature, velocity, and mass for all tests
        self.density = lambda x, y, z: 6e19 * (1 + 0.1 * np.sin(x) * np.sin(y) * np.sin(z))  # m^-3
        self.temperature = lambda x, y, z: 3e3 * (1 + 0.1 * np.sin(x + 1) * np.sin(y + 1) * np.sin(z + 1))  # eV
        self.velocity = lambda x, y, z: 1.6e5 * (1 + 0.1 * np.sin(x + 2) * np.sin(y + 2) * np.sin(z + 2)) * Vector3D(1, 2, 3).normalise()  # m/s
        self.mass = 4 * ATOMIC_MASS  # kg
        
        # Define sigma and phase_space_density for test_value
        self.sigma = lambda x, y, z: np.sqrt(self.temperature(x, y, z) * ELEMENTARY_CHARGE / self.mass)  # m/s
        self.phase_space_density = lambda x, y, z, vx, vy, vz: self.density(x, y, z) / (np.sqrt(2 * np.pi) * self.sigma(x, y, z)) ** 3 \
                                                          * np.exp(-(Vector3D(vx, vy, vz) - self.velocity(x, y, z)).length ** 2 / (2 * self.sigma(x, y, z) ** 2))  # s^3/m^6

    def tearDown(self):
        pass

    def test_bulk_velocity(self):
        """Test that bulk_velocity() returns the correct velocity vector."""
        maxwellian = Maxwellian(self.density, self.temperature, self.velocity, self.mass)

        for x in self.x:
            for y in self.y:
                for z in self.z:
                    self.assertAlmostEqual(maxwellian.bulk_velocity(x, y, z).x, self.velocity(x, y, z).x, delta=1e-10,
                                           msg='bulk_velocity method gives a wrong value at ({}, {}, {}).'.format(x, y, z))
                    self.assertAlmostEqual(maxwellian.bulk_velocity(x, y, z).y, self.velocity(x, y, z).y, delta=1e-10,
                                           msg='bulk_velocity method gives a wrong value at ({}, {}, {}).'.format(x, y, z))
                    self.assertAlmostEqual(maxwellian.bulk_velocity(x, y, z).z, self.velocity(x, y, z).z, delta=1e-10,
                                           msg='bulk_velocity method gives a wrong value at ({}, {}, {}).'.format(x, y, z))

    def test_effective_temperature(self):
        """Test that effective_temperature() returns the correct temperature."""
        maxwellian = Maxwellian(self.density, self.temperature, self.velocity, self.mass)

        for x in self.x:
            for y in self.y:
                for z in self.z:
                    self.assertAlmostEqual(maxwellian.effective_temperature(x, y, z), self.temperature(x, y, z), delta=1e-10,
                                           msg='effective_temperature method gives a wrong value at ({}, {}, {}).'.format(x, y, z))

    def test_density(self):
        """Test that density() returns the correct density."""
        maxwellian = Maxwellian(self.density, self.temperature, self.velocity, self.mass)

        for x in self.x:
            for y in self.y:
                for z in self.z:
                    self.assertAlmostEqual(maxwellian.density(x, y, z), self.density(x, y, z), delta=1e-10,
                                           msg='density method gives a wrong value at ({}, {}, {}).'.format(x, y, z))

    def test_value(self):
        """Test that __call__() returns the correct phase space density."""
        maxwellian = Maxwellian(self.density, self.temperature, self.velocity, self.mass)

        # testing only half the values to avoid huge execution time
        for x in self.x[::2]:
            for y in self.y[::2]:
                for z in self.z[::2]:
                    for vx in self.vx[::2]:
                        for vy in self.vy[::2]:
                            for vz in self.vz[::2]:
                                self.assertAlmostEqual(maxwellian(x, y, z, vx, vy, vz), self.phase_space_density(x, y, z, vx, vy, vz), delta=1e-10,
                                                       msg='call method gives a wrong phase space density at ({}, {}, {}, {}, {}, {}).'.format(x, y, z, vx, vy, vz))