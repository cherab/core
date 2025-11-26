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
Unit tests for the Arg6D class.
"""

import unittest
import itertools
from cherab.core.math.function.float.function6d.arg import Arg6D

# TODO: expand tests to cover the cython interface
class TestArg6D(unittest.TestCase):

    def test_arg(self):
        testvals = [-1e10, -7, -0.001, 0.0, 0.00003, 10, 2.3e49]
        for (x, y, z, u, w, v) in itertools.product(testvals, repeat=6):
            argx = Arg6D("x")
            argy = Arg6D("y")
            argz = Arg6D("z")
            argu = Arg6D("u")
            argw = Arg6D("w")
            argv = Arg6D("v")
            self.assertEqual(argx(x, y, z, u, w, v), x, "Arg6D('x') call did not match reference value.")
            self.assertEqual(argy(x, y, z, u, w, v), y, "Arg6D('y') call did not match reference value.")
            self.assertEqual(argz(x, y, z, u, w, v), z, "Arg6D('z') call did not match reference value.")
            self.assertEqual(argu(x, y, z, u, w, v), u, "Arg6D('u') call did not match reference value.")
            self.assertEqual(argw(x, y, z, u, w, v), w, "Arg6D('w') call did not match reference value.")
            self.assertEqual(argv(x, y, z, u, w, v), v, "Arg6D('v') call did not match reference value.")

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError, msg="Arg6D did not raise ValueError with incorrect string."):
            Arg6D("q")
