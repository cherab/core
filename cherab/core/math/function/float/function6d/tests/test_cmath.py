# cython: language_level=3

# Copyright 2016-2025 Euratom
# Copyright 2016-2025 United Kingdom Atomic Energy Authority
# Copyright 2016-2025 Centro de Investigaciones Energéticas, Medioambientales y Tecnológicas
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

"""
Unit tests for the cmath wrapper classes.
"""

import math
import unittest
import itertools
import cherab.core.math.function.float.function6d.cmath as cmath6d
from cherab.core.math.function.float.function6d.autowrap import PythonFunction6D

# TODO: expand tests to cover the cython interface
class TestCmath6D(unittest.TestCase):

    def setUp(self):
        self.f1 = PythonFunction6D(lambda x, y, z, u, w, v: x / 10 + y + z + u/2 + w/3 + v/4)
        self.f2 = PythonFunction6D(lambda x, y, z, u, w, v: x * x + y * y - z * z + u * u + w * w - v * v)

    def test_exp(self):
        testvals = [-10.0, -7, -0.001, 0.0, 0.00003, 10, 23.4]
        for (x, y, z, u, w, v) in itertools.product(testvals, repeat=6):
            function = cmath6d.Exp6D(self.f1)
            expected = math.exp(self.f1(x, y, z, u, w, v))
            self.assertEqual(function(x, y, z, u, w, v), expected, "Exp6D call did not match reference value.")

    def test_sin(self):
        testvals = [-10.0, -7, -0.001, 0.0, 0.00003, 10, 23.4]
        for (x, y, z, u, w, v) in itertools.product(testvals, repeat=6):
            function = cmath6d.Sin6D(self.f1)
            expected = math.sin(self.f1(x, y, z, u, w, v))
            self.assertEqual(function(x, y, z, u, w, v), expected, "Sin6D call did not match reference value.")

    def test_cos(self):
        testvals = [-10.0, -7, -0.001, 0.0, 0.00003, 10, 23.4]
        for (x, y, z, u, w, v) in itertools.product(testvals, repeat=6):
            function = cmath6d.Cos6D(self.f1)
            expected = math.cos(self.f1(x, y, z, u, w, v))
            self.assertEqual(function(x, y, z, u, w, v), expected, "Cos6D call did not match reference value.")

    def test_tan(self):
        testvals = [-10.0, -7, -0.001, 0.0, 0.00003, 10, 23.4]
        for (x, y, z, u, w, v) in itertools.product(testvals, repeat=6):
            function = cmath6d.Tan6D(self.f1)
            expected = math.tan(self.f1(x, y, z, u, w, v))
            self.assertEqual(function(x, y, z, u, w, v), expected, "Tan6D call did not match reference value.")

    def test_asin(self):
        v = [-10, -6, -2, -0.001, 0, 0.001, 2, 6, 10]
        function = cmath6d.Asin6D(self.f1)
        for x in v:
            expected = math.asin(self.f1(x, 0, 0, 0, 0, 0))
            self.assertEqual(function(x, 0, 0, 0, 0, 0), expected, "Asin3D call did not match reference value.")

        with self.assertRaises(ValueError, msg="Asin3D did not raise a ValueError with value outside domain."):
            function(100, 0, 0, 0, 0, 0)

    def test_acos(self):
        v = [-10, -6, -2, -0.001, 0, 0.001, 2, 6, 10]
        function = cmath6d.Acos6D(self.f1)
        for x in v:
            expected = math.acos(self.f1(x, 0, 0, 0, 0, 0))
            self.assertEqual(function(x, 0, 0, 0, 0, 0), expected, "Acos6D call did not match reference value.")

        with self.assertRaises(ValueError, msg="Acos3D did not raise a ValueError with value outside domain."):
            function(100, 0, 0, 0, 0, 0)

    def test_atan(self):
        testvals = [-10.0, -7, -0.001, 0.0, 0.00003, 10, 23.4]
        for (x, y, z, u, w, v) in itertools.product(testvals, repeat=6):
            function = cmath6d.Atan6D(self.f1)
            expected = math.atan(self.f1(x, y, z, u, w, v))
            self.assertEqual(function(x, y, z, u, w, v), expected, "Atan6D call did not match reference value.")

    def test_atan2(self):
        testvals = [-10.0, -7, -0.001, 0.0, 0.00003, 10, 23.4]
        for (x, y, z, u, w, v) in itertools.product(testvals, repeat=6):
            function = cmath6d.Atan4Q6D(self.f1, self.f2)
            expected = math.atan2(self.f1(x, y, z, u, w, v), self.f2(x, y, z, u, w, v))
            self.assertEqual(function(x, y, z, u, w, v), expected, "Atan4Q6D call did not match reference value.")

    def test_erf(self):
        testvals = [-1e5, -7, -0.001, 0.0, 0.00003, 10, 23.4, 1e5]
        function = cmath6d.Erf6D(self.f1)
        for (x, y, z, u, w, v) in itertools.product(testvals, repeat=6):
            expected = math.erf(self.f1(x, y, z, u, w, v))
            self.assertAlmostEqual(function(x, y, z, u, w, v), expected, 10, "Erf6D call did not match reference value.")

    def test_sqrt(self):
        testvals = [0.0, 0.00003, 10, 23.4, 1e5]
        function = cmath6d.Sqrt6D(self.f1)
        for (x, y, z, u, w, v) in itertools.product(testvals, repeat=6):
            expected = math.sqrt(self.f1(x, y, z, u, w, v))
            self.assertEqual(function(x, y, z, u, w, v), expected, "Sqrt6D call did not match reference value.")

        with self.assertRaises(ValueError, msg="Sqrt6D did not raise a ValueError with value outside domain."):
            function(-0.1, -0.1, -0.1, -0.1, -0.1, -0.1)