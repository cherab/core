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
Unit tests for the Constant6D class.
"""

import unittest
from cherab.core.math.function.float.function6d.constant import Constant6D

# TODO: expand tests to cover the cython interface
class TestConstant6D(unittest.TestCase):

    def test_constant(self):
        testvals = [-1e10, -7, -0.001, 0.0, 0.00003, 10, 2.3e49]
        for x in testvals:
            constant = Constant6D(x)
            # Test with a single set of values since it's a constant function
            self.assertEqual(constant(500, 1.5, -3.14, 2.7, 1.8, 3.6), x, "Constant6D call did not match reference value.")
