# Copyright 2016-2022 Euratom
# Copyright 2016-2022 United Kingdom Atomic Energy Authority
# Copyright 2016-2022 Centro de Investigaciones Energéticas, Medioambientales y Tecnológicas
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

from raysect.core.math import Vector3D
from cherab.core.math import transform
import numpy as np
import unittest


class TestCylindricalTransform(unittest.TestCase):
    """Cylindrical transform tests."""

    def setUp(self):
        """Initialisation with functions to map."""

        def f3d(r, phi, z):
            return r * np.cos(phi) + z
        self.function3d = f3d

        def vecf3d(r, phi, z):
            return Vector3D(np.sin(phi), r * z, np.cos(phi))
        self.vectorfunction3d = vecf3d

    def test_cylindrical_transform(self):
        """Cylindrical transform."""
        cyl_func = transform.CylindricalTransform(self.function3d)
        self.assertAlmostEqual(cyl_func(1., 1., 0.5),
                               self.function3d(np.sqrt(2.), 0.25 * np.pi, 0.5),
                               places=10)

    def test_cylindrical_transform_invalid_arg(self):
        """An error must be raised if the given argument is not callable."""
        self.assertRaises(TypeError, transform.CylindricalTransform, "blah")

    def test_vector_cylindrical_transform(self):
        """Cylindrical transform."""
        cyl_func = transform.VectorCylindricalTransform(self.vectorfunction3d)
        vec1 = cyl_func(1., 1., 1.)
        vec2 = Vector3D(-0.5, 1.5, 1 / np.sqrt(2))
        np.testing.assert_almost_equal([vec1.x, vec1.y, vec1.z], [vec2.x, vec2.y, vec2.z], decimal=10)

    def test_vector_cylindrical_transform_invalid_arg(self):
        """An error must be raised if the given argument is not callable."""
        self.assertRaises(TypeError, transform.VectorCylindricalTransform, "blah")


class TestPeriodicTransform(unittest.TestCase):
    """Periodic transform tests."""

    def setUp(self):
        """Initialisation with functions to map."""

        def f1d(x):
            return x * np.cos(x - 3)
        self.function1d = f1d

        def f2d(x, y):
            return x * np.sin(y)
        self.function2d = f2d

        def f3d(x, y, z):
            return x * x * np.exp(y) - 2 * z * y
        self.function3d = f3d

        def vecf1d(x):
            return Vector3D(x, x**2, x**3)
        self.vectorfunction1d = vecf1d

        def vecf2d(x, y):
            return Vector3D(x, y, x * y)
        self.vectorfunction2d = vecf2d

        def vecf3d(x, y, z):
            return Vector3D(x + y + z, (x + y) * z, x * y * z)
        self.vectorfunction3d = vecf3d

    def test_periodic_transform_1d(self):
        """1D periodic transform"""
        period_func = transform.PeriodicTransform1D(self.function1d, np.pi)
        self.assertAlmostEqual(period_func(1.4 * np.pi),
                               self.function1d(0.4 * np.pi),
                               places=10)
        self.assertAlmostEqual(period_func(-0.4 * np.pi),
                               self.function1d(0.6 * np.pi),
                               places=10)

    def test_periodic_transform_1d_invalid_arg(self):
        """1D periodic transform. Invalid arguments."""
        # 1st argument is not callable
        self.assertRaises(TypeError, transform.PeriodicTransform1D, "blah", np.pi)
        # period is not a number
        self.assertRaises(TypeError, transform.PeriodicTransform1D, self.function1d, "blah")
        # period is negative
        self.assertRaises(ValueError, transform.PeriodicTransform1D, self.function1d, -1)

    def test_periodic_transform_2d(self):
        """2D periodic transform"""
        period_func = transform.PeriodicTransform2D(self.function2d, 1, np.pi)
        self.assertAlmostEqual(period_func(-0.4, 1.4 * np.pi),
                               self.function2d(0.6, 0.4 * np.pi),
                               places=10)
        # Periodic only along x
        period_func = transform.PeriodicTransform2D(self.function2d, 1., 0)
        self.assertAlmostEqual(period_func(-0.4, 1.4 * np.pi),
                               self.function2d(0.6, 1.4 * np.pi),
                               places=10)
        # Periodic only along y
        period_func = transform.PeriodicTransform2D(self.function2d, 0, np.pi)
        self.assertAlmostEqual(period_func(-0.4, 1.4 * np.pi),
                               self.function2d(-0.4, 0.4 * np.pi),
                               places=10)

    def test_periodic_transform_2d_invalid_arg(self):
        """2D periodic transform. Invalid arguments."""
        # 1st argument is not callable
        self.assertRaises(TypeError, transform.PeriodicTransform2D, "blah", np.pi, np.pi)
        # period is not a number
        self.assertRaises(TypeError, transform.PeriodicTransform2D, self.function2d, "blah", np.pi)
        self.assertRaises(TypeError, transform.PeriodicTransform2D, self.function2d, np.pi, "blah")
        # period is negative
        self.assertRaises(ValueError, transform.PeriodicTransform2D, self.function2d, -1, np.pi)
        self.assertRaises(ValueError, transform.PeriodicTransform2D, self.function2d, np.pi, -1)

    def test_periodic_transform_3d(self):
        """3D periodic transform"""
        period_func = transform.PeriodicTransform3D(self.function3d, 1, 1, 1)
        self.assertAlmostEqual(period_func(-0.4, 1.4, 2.1),
                               self.function3d(0.6, 0.4, 0.1),
                               places=10)
        # Periodic only along y and z
        period_func = transform.PeriodicTransform3D(self.function3d, 0, 1, 1)
        self.assertAlmostEqual(period_func(-0.4, 1.4, 2.1),
                               self.function3d(-0.4, 0.4, 0.1),
                               places=10)
        # Periodic only along x and z
        period_func = transform.PeriodicTransform3D(self.function3d, 1, 0, 1)
        self.assertAlmostEqual(period_func(-0.4, 1.4, 2.1),
                               self.function3d(0.6, 1.4, 0.1),
                               places=10)
        # Periodic only along x and y
        period_func = transform.PeriodicTransform3D(self.function3d, 1, 1, 0)
        self.assertAlmostEqual(period_func(-0.4, 1.4, 2.1),
                               self.function3d(0.6, 0.4, 2.1),
                               places=10)

    def test_periodic_transform_3d_invalid_arg(self):
        """3D periodic transform. Invalid arguments."""
        # 1st argument is not callable
        self.assertRaises(TypeError, transform.PeriodicTransform3D, "blah", np.pi, np.pi, np.pi)
        # period is not a number
        self.assertRaises(TypeError, transform.PeriodicTransform3D, self.function3d, "blah", np.pi, np.pi)
        self.assertRaises(TypeError, transform.PeriodicTransform3D, self.function3d, np.pi, "blah", np.pi)
        self.assertRaises(TypeError, transform.PeriodicTransform3D, self.function3d, np.pi, np.pi, "blah")
        # period is negative
        self.assertRaises(ValueError, transform.PeriodicTransform3D, self.function3d, -1, np.pi, np.pi)
        self.assertRaises(ValueError, transform.PeriodicTransform3D, self.function3d, np.pi, -1, np.pi)
        self.assertRaises(ValueError, transform.PeriodicTransform3D, self.function3d, np.pi, np.pi, -1)

    def test_vector_periodic_transform_1d(self):
        """1D vector periodic transform"""
        period_func = transform.VectorPeriodicTransform1D(self.vectorfunction1d, 1)
        vec1 = period_func(1.4)
        vec2 = Vector3D(0.4, 0.16, 0.064)
        np.testing.assert_almost_equal([vec1.x, vec1.y, vec1.z], [vec2.x, vec2.y, vec2.z], decimal=10)

    def test_vector_periodic_transform_1d_invalid_arg(self):
        """1D vector periodic transform. Invalid arguments."""
        # 1st argument is not callable
        self.assertRaises(TypeError, transform.VectorPeriodicTransform1D, "blah", 1.)
        # period is not a number
        self.assertRaises(TypeError, transform.VectorPeriodicTransform1D, self.vectorfunction1d, "blah")
        # period is negative
        self.assertRaises(ValueError, transform.VectorPeriodicTransform1D, self.vectorfunction1d, -1)

    def test_vector_periodic_transform_2d(self):
        """2D vector periodic transform"""
        period_func = transform.VectorPeriodicTransform2D(self.vectorfunction2d, 1, 1)
        vec1 = period_func(-0.4, 1.6)
        vec2 = Vector3D(0.6, 0.6, 0.36)
        np.testing.assert_almost_equal([vec1.x, vec1.y, vec1.z], [vec2.x, vec2.y, vec2.z], decimal=10)

        # Periodic only along x
        period_func = transform.VectorPeriodicTransform2D(self.vectorfunction2d, 1, 0)
        vec1 = period_func(-0.4, 1.6)
        vec2 = Vector3D(0.6, 1.6, 0.96)
        np.testing.assert_almost_equal([vec1.x, vec1.y, vec1.z], [vec2.x, vec2.y, vec2.z], decimal=10)

        # Periodic only along y
        period_func = transform.VectorPeriodicTransform2D(self.vectorfunction2d, 0, 1)
        vec1 = period_func(-0.4, 1.6)
        vec2 = Vector3D(-0.4, 0.6, -0.24)
        np.testing.assert_almost_equal([vec1.x, vec1.y, vec1.z], [vec2.x, vec2.y, vec2.z], decimal=10)

    def test_vector_periodic_transform_2d_invalid_arg(self):
        """2D vector periodic transform. Invalid arguments."""
        # 1st argument is not callable
        self.assertRaises(TypeError, transform.VectorPeriodicTransform2D, "blah", 1, 1)
        # period is not a number
        self.assertRaises(TypeError, transform.VectorPeriodicTransform2D, self.vectorfunction2d, "blah", 1)
        self.assertRaises(TypeError, transform.VectorPeriodicTransform2D, self.vectorfunction2d, 1, "blah")
        # period is negative
        self.assertRaises(ValueError, transform.VectorPeriodicTransform2D, self.vectorfunction2d, -1, 1)
        self.assertRaises(ValueError, transform.VectorPeriodicTransform2D, self.vectorfunction2d, 1, -1)

    def test_vector_periodic_transform_3d(self):
        """3D vector periodic transform"""
        period_func = transform.VectorPeriodicTransform3D(self.vectorfunction3d, 1, 1, 1)
        vec1 = period_func(-0.4, 1.6, 1.2)
        vec2 = Vector3D(1.4, 0.24, 0.072)
        np.testing.assert_almost_equal([vec1.x, vec1.y, vec1.z], [vec2.x, vec2.y, vec2.z], decimal=10)

        # Periodic along y and z
        period_func = transform.VectorPeriodicTransform3D(self.vectorfunction3d, 0, 1, 1)
        vec1 = period_func(-0.4, 1.6, 1.2)
        vec2 = Vector3D(0.4, 0.04, -0.048)
        np.testing.assert_almost_equal([vec1.x, vec1.y, vec1.z], [vec2.x, vec2.y, vec2.z], decimal=10)

        # Periodic along x and z
        period_func = transform.VectorPeriodicTransform3D(self.vectorfunction3d, 1, 0, 1)
        vec1 = period_func(-0.4, 1.6, 1.2)
        vec2 = Vector3D(2.4, 0.44, 0.192)
        np.testing.assert_almost_equal([vec1.x, vec1.y, vec1.z], [vec2.x, vec2.y, vec2.z], decimal=10)

        # Periodic along x and y
        period_func = transform.VectorPeriodicTransform3D(self.vectorfunction3d, 1, 1, 0)
        vec1 = period_func(-0.4, 1.6, 1.2)
        vec2 = Vector3D(2.4, 1.44, 0.432)
        np.testing.assert_almost_equal([vec1.x, vec1.y, vec1.z], [vec2.x, vec2.y, vec2.z], decimal=10)

    def test_vector_periodic_transform_3d_invalid_arg(self):
        """3D vector periodic transform. Invalid arguments."""
        # 1st argument is not callable
        self.assertRaises(TypeError, transform.VectorPeriodicTransform3D, "blah", 1, 1, 1)
        # period is not a number
        self.assertRaises(TypeError, transform.VectorPeriodicTransform3D, self.vectorfunction3d, "blah", 1, 1)
        self.assertRaises(TypeError, transform.VectorPeriodicTransform3D, self.vectorfunction3d, 1, "blah", 1)
        self.assertRaises(TypeError, transform.VectorPeriodicTransform3D, self.vectorfunction3d, 1, 1, "blah")
        # period is negative
        self.assertRaises(ValueError, transform.VectorPeriodicTransform3D, self.vectorfunction3d, -1, 1, 1)
        self.assertRaises(ValueError, transform.VectorPeriodicTransform3D, self.vectorfunction3d, 1, -1, 1)
        self.assertRaises(ValueError, transform.VectorPeriodicTransform3D, self.vectorfunction3d, 1, 1, -1)


if __name__ == '__main__':
    unittest.main()
