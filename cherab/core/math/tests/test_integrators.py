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

from raysect.core.math.function.float import Exp1D, Arg1D, Exp2D, Arg2D, Constant1D
from cherab.core.math.integrators import GaussianQuadrature, GaussianQuadrature2D
from math import sqrt, pi
from scipy.special import erf
import unittest


class TestGaussianQuadrature(unittest.TestCase):
    """Gaussian quadrature integrator tests."""

    def test_properties(self):
        """Test property assignment."""
        min_order = 3
        max_order = 30
        reltol = 1.0e-6
        quadrature = GaussianQuadrature(
            integrand=Arg1D,
            relative_tolerance=reltol,
            max_order=max_order,
            min_order=min_order,
        )

        self.assertEqual(quadrature.relative_tolerance, reltol)
        self.assertEqual(quadrature.max_order, max_order)
        self.assertEqual(quadrature.min_order, min_order)
        self.assertEqual(quadrature.integrand, Arg1D)

        min_order = 0
        max_order = 2  # < min_order
        reltol = -1

        with self.assertRaises(ValueError):
            quadrature.max_order = max_order

        with self.assertRaises(ValueError):
            quadrature.min_order = min_order

        with self.assertRaises(ValueError):
            quadrature.relative_tolerance = reltol

        min_order = 1
        max_order = 20
        reltol = 1.0e-5

        quadrature.relative_tolerance = reltol
        quadrature.min_order = min_order
        quadrature.max_order = max_order
        quadrature.integrand = Exp1D

        self.assertEqual(quadrature.relative_tolerance, reltol)
        self.assertEqual(quadrature.min_order, min_order)
        self.assertEqual(quadrature.max_order, max_order)
        self.assertEqual(quadrature.integrand, Exp1D)

    def test_integrate(self):
        """Test integration."""
        quadrature = GaussianQuadrature(relative_tolerance=1.0e-8)
        a = -0.5
        b = 3.0
        quadrature.integrand = (2 / sqrt(pi)) * Exp1D(-Arg1D() * Arg1D())
        exact_integral = erf(b) - erf(a)

        self.assertAlmostEqual(quadrature(a, b), exact_integral, places=8)


class TestGaussianQuadrature2D(unittest.TestCase):
    """Gaussian quadrature 2D integrator tests."""

    def test_properties(self):
        """Test property assignment."""
        x_min_order = 3
        y_min_order = 4
        x_max_order = 30
        y_max_order = 40
        reltol = 1.0e-6
        integrand = Exp2D(Arg2D("x") + Arg2D("y"))

        quadrature = GaussianQuadrature2D(
            integrand=integrand,
            relative_tolerance=reltol,
            x_max_order=x_max_order,
            x_min_order=x_min_order,
            y_max_order=y_max_order,
            y_min_order=y_min_order,
        )

        self.assertEqual(quadrature.relative_tolerance, reltol)
        self.assertEqual(quadrature.x_max_order, x_max_order)
        self.assertEqual(quadrature.y_max_order, y_max_order)
        self.assertEqual(quadrature.x_min_order, x_min_order)
        self.assertEqual(quadrature.y_min_order, y_min_order)
        self.assertEqual(quadrature.integrand, integrand)

        x_min_order = 50  # > x_max_order
        x_max_order = 2  # < x_min_order
        y_min_order = 50  # > y_max_order
        y_max_order = 1  # < y_min_order
        reltol = -1

        with self.assertRaises(ValueError):
            quadrature.x_max_order = x_max_order

        with self.assertRaises(ValueError):
            quadrature.x_min_order = x_min_order

        with self.assertRaises(ValueError):
            quadrature.y_max_order = y_max_order

        with self.assertRaises(ValueError):
            quadrature.y_min_order = y_min_order

        with self.assertRaises(ValueError):
            quadrature.relative_tolerance = reltol

        x_min_order = 0
        y_min_order = 0

        with self.assertRaises(ValueError):
            quadrature.x_min_order = x_min_order

        with self.assertRaises(ValueError):
            quadrature.y_min_order = y_min_order

        x_min_order = 1
        x_max_order = 20
        y_min_order = 2
        y_max_order = 30
        reltol = 1.0e-5

        quadrature.relative_tolerance = reltol
        quadrature.x_min_order = x_min_order
        quadrature.x_max_order = x_max_order
        quadrature.y_min_order = y_min_order
        quadrature.y_max_order = y_max_order
        quadrature.integrand = Arg2D("x")

        self.assertEqual(quadrature.relative_tolerance, reltol)
        self.assertEqual(quadrature.x_min_order, x_min_order)
        self.assertEqual(quadrature.x_max_order, x_max_order)
        self.assertEqual(quadrature.y_min_order, y_min_order)
        self.assertEqual(quadrature.y_max_order, y_max_order)
        self.assertEqual(quadrature.integrand, Arg2D("x"))

    def test_integrate(self):
        """Test 2D integration."""
        quadrature = GaussianQuadrature2D(relative_tolerance=1.0e-8)

        # Integration limits
        a_x, b_x = -2.0, 2.0
        a_y, b_y = -3.0, 3.0
        
        # Bivariate Normal distribution with std_dev=1, mean=0 and no correlation
        quadrature.integrand = (
            1 / (2 * pi) * Exp2D(-0.5 * (Arg2D("x") ** 2 + Arg2D("y") ** 2))
        )

        # Exact integral of the bivariate normal distribution
        exact_integral = (
            1 / 4.0
            * (erf(b_x / sqrt(2)) - erf(a_x / sqrt(2)))
            * (erf(b_y / sqrt(2)) - erf(a_y / sqrt(2)))
        )

        self.assertAlmostEqual(
            quadrature(a_x, b_x, Constant1D(a_y), Constant1D(b_y)),
            exact_integral,
            places=8,
        )


if __name__ == "__main__":
    unittest.main()
