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

from raysect.core.math.function.float import Exp1D, Arg1D
from cherab.core.math.integrators import GaussianQuadrature
from math import sqrt, pi
from scipy.special import erf
import unittest


class TestGaussianQuadrature(unittest.TestCase):
    """Gaussian quadrature integrator tests."""

    def test_properties(self):
        """Test property assignment."""
        min_order = 3
        max_order = 30
        reltol = 1.e-6
        quadrature = GaussianQuadrature(integrand=Arg1D, relative_tolerance=reltol, max_order=max_order, min_order=min_order)

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
        reltol = 1.e-5

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
        quadrature = GaussianQuadrature(relative_tolerance=1.e-8)
        a = -0.5
        b = 3.
        quadrature.integrand = (2 / sqrt(pi)) * Exp1D(- Arg1D() * Arg1D())
        exact_integral = erf(b) - erf(a)

        self.assertAlmostEqual(quadrature(a, b), exact_integral, places=8)


if __name__ == '__main__':
    unittest.main()
