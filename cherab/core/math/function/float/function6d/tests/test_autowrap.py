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
Unit tests for the autowrap_6d function
"""

import unittest
from cherab.core.math.function.float.function6d.autowrap import _autowrap_function6d, PythonFunction6D
from cherab.core.math.function.float.function6d.constant import Constant6D

class TestAutowrap6D(unittest.TestCase):

    def test_constant(self):
        function = _autowrap_function6d(5.0)
        self.assertIsInstance(function, Constant6D, "Autowrapped scalar float is not a Constant6D.")

    def test_python_function(self):
        function = _autowrap_function6d(lambda x, y, z, u, w, v: 10*x + 5*y + 2*z + u + 3*w + 4*v)
        self.assertIsInstance(function, PythonFunction6D, "Autowrapped function is not a PythonFunction6D.")
