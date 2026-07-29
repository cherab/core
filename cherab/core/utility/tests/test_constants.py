# Copyright 2026 Oak Ridge National Laboratory
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

import math
import unittest
from cherab.core.utility import constants


class TestConstants(unittest.TestCase):
    def setUp(self):
        self._expected_constants = dict(
            # sourced c standard maths library
            # CPython wraps libc's math so uses the same constants as cython's
            # cimport of libc.math.
            RECIP_2_PI=1 / (2 * math.pi),
            RECIP_4_PI=1 / (4 * math.pi),
            DEGREES_TO_RADIANS=math.pi / 180,
            RADIANS_TO_DEGREES=180 / math.pi,

            # sourced from NIST, CODATA 2018=https://physics.nist.gov/cuu/Constants/Table/allascii.txt
            ATOMIC_MASS=1.66053906660e-27,
            ELEMENTARY_CHARGE=1.602176634e-19,
            SPEED_OF_LIGHT=299792458.0,
            PLANCK_CONSTANT=6.62607015e-34,
            HC_EV_NM=1239.8419738620933,  # (Planck constant in eV s) x (speed of light in nm/s)
            ELECTRON_CLASSICAL_RADIUS=2.8179403262e-15,
            ELECTRON_REST_MASS=9.1093837015e-31,
            RYDBERG_CONSTANT_EV=13.605693122994,
            VACUUM_PERMITTIVITY=8.8541878128e-12,
            BOHR_MAGNETON=5.78838180123e-5,  # in eV/T
        )

    def test_all_exported(self):
        """
        Test Cython constants exported as Python floats.
        """
        for name, value in self._expected_constants.items():
            self.assertEqual(value, getattr(constants, name))

    def test_exported_literal(self):
        """
        Test a constant accessed by literal name.
        """
        self.assertEqual(self._expected_constants['ATOMIC_MASS'], constants.ATOMIC_MASS)

    def test_exported_names(self):
        """
        Test all exported names are as expected by this test class.
        """
        self.assertEqual(sorted(self._expected_constants.keys()), sorted(dir(constants)))

    def test_readonly(self):
        """
        Test that constants can't be modified or removed and new constants can't be added.
        """
        with self.assertRaises(AttributeError):
            constants.ATOMIC_MASS = 1.66e-27

    def test_nonew(self):
        """
        Test that attempting to assign a new constant from Python errors.
        """
        with self.assertRaises(AttributeError):
            constants.TAU = math.tau

    def test_nodel(self):
        """
        Test that attempting to delete and constant from Python errors.
        """
        with self.assertRaises(AttributeError):
            del constants.RYDBERG_CONSTANT_EV
