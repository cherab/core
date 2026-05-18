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
Unit tests for the Function6D class.
"""

import math
import unittest
import itertools
from cherab.core.math.function.float.function6d.autowrap import PythonFunction6D

# TODO: expand tests to cover the cython interface
class TestFunction6D(unittest.TestCase):

    def setUp(self):
        self.ref1 = lambda x, y, z, u, w, v: 10 * x + 5 * y + 2 * z + u + 3 * w + 4 * v
        self.ref2 = lambda x, y, z, u, w, v: abs(x + y + z + u + w + v)

        self.f1 = PythonFunction6D(self.ref1)
        self.f2 = PythonFunction6D(self.ref2)

    def test_call(self):
        testvals = [-1e10, -7, -0.001, 0.0, 0.00003, 10, 2.3e49]
        for (x, y, z, u, w, v) in itertools.product(testvals, repeat=6):
            self.assertEqual(self.f1(x, y, z, u, w, v), self.ref1(x, y, z, u, w, v), 
                          "Function6D call did not match reference function value.")

    def test_negate(self):
        testvals = [-1e10, -7, -0.001, 0.0, 0.00003, 10, 2.3e49]
        r = -self.f1
        for (x, y, z, u, w, v) in itertools.product(testvals, repeat=6):
            self.assertEqual(r(x, y, z, u, w, v), -self.ref1(x, y, z, u, w, v), 
                          "Function6D negate did not match reference function value.")

    def test_add_scalar(self):
        testvals = [-1e10, -7, -0.001, 0.0, 0.00003, 10, 2.3e49]
        r1 = 8 + self.f1
        r2 = self.f1 + 65
        for (x, y, z, u, w, v) in itertools.product(testvals, repeat=6):
            self.assertEqual(r1(x, y, z, u, w, v), 8 + self.ref1(x, y, z, u, w, v), 
                          "Function6D add scalar (K + f()) did not match reference function value.")
            self.assertEqual(r2(x, y, z, u, w, v), self.ref1(x, y, z, u, w, v) + 65, 
                          "Function6D add scalar (f() + K) did not match reference function value.")

    def test_sub_scalar(self):
        testvals = [-1e10, -7, -0.001, 0.0, 0.00003, 10, 2.3e49]
        r1 = 8 - self.f1
        r2 = self.f1 - 65
        for (x, y, z, u, w, v) in itertools.product(testvals, repeat=6):
            self.assertEqual(r1(x, y, z, u, w, v), 8 - self.ref1(x, y, z, u, w, v), 
                          "Function6D subtract scalar (K - f()) did not match reference function value.")
            self.assertEqual(r2(x, y, z, u, w, v), self.ref1(x, y, z, u, w, v) - 65, 
                          "Function6D subtract scalar (f() - K) did not match reference function value.")

    def test_mul_scalar(self):
        testvals = [-1e10, -7, -0.001, 0.0, 0.00003, 10, 2.3e49]
        r1 = 5 * self.f1
        r2 = self.f1 * -7.8
        for (x, y, z, u, w, v) in itertools.product(testvals, repeat=6):
            self.assertEqual(r1(x, y, z, u, w, v), 5 * self.ref1(x, y, z, u, w, v), 
                          "Function6D multiply scalar (K * f()) did not match reference function value.")
            self.assertEqual(r2(x, y, z, u, w, v), self.ref1(x, y, z, u, w, v) * -7.8, 
                          "Function6D multiply scalar (f() * K) did not match reference function value.")

    def test_div_scalar(self):
        testvals = [-1e10, -7, -0.001, 0.000031, 10.3, 2.3e49]
        r1 = 5.451 / self.f1
        r2 = self.f1 / -7.8
        for (x, y, z, u, w, v) in itertools.product(testvals, repeat=6):
            self.assertEqual(r1(x, y, z, u, w, v), 5.451 / self.ref1(x, y, z, u, w, v), 
                          "Function6D divide scalar (K / f()) did not match reference function value.")
            self.assertAlmostEqual(r2(x, y, z, u, w, v), self.ref1(x, y, z, u, w, v) / -7.8, 
                                 delta=abs(r2(x, y, z, u, w, v)) * 1e-12, 
                                 msg="Function6D divide scalar (f() / K) did not match reference function value.")

        r = 5 / self.f1
        with self.assertRaises(ZeroDivisionError, msg="ZeroDivisionError not raised when function returns zero."):
            r(0, 0, 0, 0, 0, 0)

    def test_mod_function6d_scalar(self):
        # Note that Function6D objects work with doubles, so the floating modulo
        # operator is used rather than the integer one. For accurate testing we
        # therefore need to use the math.fmod operator rather than % in Python.
        testvals = [-10, -7, -0.001, 0.00003, 10, 12.3]
        r1 = 5 % self.f1
        r2 = self.f1 % -7.8
        for (x, y, z, u, w, v) in itertools.product(testvals, repeat=6):
            if self.ref1(x, y, z, u, w, v) == 0:
                with self.assertRaises(ZeroDivisionError, msg="ZeroDivisionError not raised when function returns 0."):
                    r1(x, y, z, u, w, v)
            else:
                self.assertAlmostEqual(r1(x, y, z, u, w, v), math.fmod(5, self.ref1(x, y, z, u, w, v)), 15, "Function6D modulo scalar (K % f()) did not match reference function value.")
                self.assertAlmostEqual(r2(x, y, z, u, w, v), math.fmod(self.ref1(x, y, z, u, w, v), -7.8), 15, "Function6D modulo scalar (f() % K) did not match reference function value.")
        with self.assertRaises(ZeroDivisionError, msg="ZeroDivisionError not raised when function returns 0."):
            r1(0, 0, 0, 0, 0, 0)
        with self.assertRaises(ZeroDivisionError, msg="ZeroDivisionError not raised when modulo scalar is 0."):
            self.f1 % 0

    def test_pow_function6d_scalar(self):
        testvals = [-10, -7, -0.001, 0.00003, 10, 12.3]
        r1 = 5. ** self.f1
        r2 = self.f1 ** -7.8
        r3 = (-5.) ** self.f1
        for (x, y, z, u, w, v) in itertools.product(testvals, repeat=6):
            self.assertAlmostEqual(r1(x, y, z, u, w, v), 5. ** self.ref1(x, y, z, u, w, v), 15, "Function6D power scalar (K ** f()) did not match reference function value.")
            if self.ref1(x, y, z, u, w, v) < 0:
                with self.assertRaises(ValueError, msg="ValueError not raised when base is negative and exponent non-integral."):
                    r2(x, y, z, u, w, v)
            elif not float(self.ref1(x, y, z, u, w, v)).is_integer():
                with self.assertRaises(ValueError, msg="ValueError not raised when base is negative and exponent non-integral."):
                    r3(x, y, z, u, w, v)
            else:
                if self.ref1(x, y, z, u, w, v) == 0:
                    with self.assertRaises(ZeroDivisionError, msg="ZeroDivisionError not raised when base is 0 and exponent negative."):
                        r2(x, y, z, u, w, v)
                else:
                    self.assertAlmostEqual(r2(x, y, z, u, w, v), self.ref1(x, y, z, u, w, v) ** -7.8, 15, "Function6D power scalar (f() ** K) did not match reference function value.")
        with self.assertRaises(ZeroDivisionError, msg="ZeroDivisionError not raised when base is 0 and exponent negative."):
            r2(0, 0, 0, 0, 0, 0)
        with self.assertRaises(ZeroDivisionError, msg="ZeroDivisionError not raised when base is zero and exponent negative."):
            r4 = 0 ** self.f1
            r4(-1, 0, 0, 0, 0, 0)

    def test_richcmp_scalar(self):
        testvals = [-1e10, -7, -0.001, 0.0, 0.00003, 10, 2.3e49]
        for (x, y, z, u, w, v) in itertools.product(testvals, repeat=6):
            ref_value = self.ref1(x, y, z, u, w, v)
            higher_value = ref_value + abs(ref_value) + 1
            lower_value = ref_value - abs(ref_value) - 1
            self.assertEqual(
                (self.f1 == ref_value)(x, y, z, u, w, v), 1.0,
                msg="Function6D equals scalar (f() == K) did not return true when it should."
            )
            self.assertEqual(
                (ref_value == self.f1)(x, y, z, u, w, v), 1.0,
                msg="Scalar equals Function6D (K == f()) did not return true when it should."
            )
            self.assertEqual(
                (self.f1 == higher_value)(x, y, z, u, w, v), 0.0,
                msg="Function6D equals scalar (f() == K) did not return false when it should."
            )
            self.assertEqual(
                (higher_value == self.f1)(x, y, z, u, w, v), 0.0,
                msg="Scalar equals Function6D (K == f()) did not return false when it should."
            )
            self.assertEqual(
                (self.f1 != higher_value)(x, y, z, u, w, v), 1.0,
                msg="Function6D not equals scalar (f() != K) did not return true when it should."
            )
            self.assertEqual(
                (higher_value != self.f1)(x, y, z, u, w, v), 1.0,
                msg="Scalar not equals Function6D (K != f()) did not return true when it should."
            )
            self.assertEqual(
                (self.f1 != ref_value)(x, y, z, u, w, v), 0.0,
                msg="Function6D not equals scalar (f() != K) did not return false when it should."
            )
            self.assertEqual(
                (ref_value != self.f1)(x, y, z, u, w, v), 0.0,
                msg="Scalar not equals Function6D (K != f()) did not return false when it should."
            )
            self.assertEqual(
                (self.f1 < higher_value)(x, y, z, u, w, v), 1.0,
                msg="Function6D less than scalar (f() < K) did not return true when it should."
            )
            self.assertEqual(
                (lower_value < self.f1)(x, y, z, u, w, v), 1.0,
                msg="Scalar less than Function6D (K < f()) did not return true when it should."
            )
            self.assertEqual(
                (self.f1 < lower_value)(x, y, z, u, w, v), 0.0,
                msg="Function6D less than scalar (f() < K) did not return false when it should."
            )
            self.assertEqual(
                (higher_value < self.f1)(x, y, z, u, w, v), 0.0,
                msg="Scalar less than Function6D (K < f()) did not return false when it should."
            )
            self.assertEqual(
                (self.f1 > lower_value)(x, y, z, u, w, v), 1.0,
                msg="Function6D greater than scalar (f() > K) did not return true when it should."
            )
            self.assertEqual(
                (higher_value > self.f1)(x, y, z, u, w, v), 1.0,
                msg="Scalar greater than Function6D (K > f()) did not return true when it should."
            )
            self.assertEqual(
                (self.f1 > higher_value)(x, y, z, u, w, v), 0.0,
                msg="Function6D greater than scalar (f() > K) did not return false when it should."
            )
            self.assertEqual(
                (lower_value > self.f1)(x, y, z, u, w, v), 0.0,
                msg="Scalar greater than Function6D (K > f()) did not return false when it should."
            )
            self.assertEqual(
                (self.f1 <= higher_value)(x, y, z, u, w, v), 1.0,
                msg="Function6D less equals scalar (f() <= K) did not return true when it should."
            )
            self.assertEqual(
                (lower_value <= self.f1)(x, y, z, u, w, v), 1.0,
                msg="Scalar less equals Function6D (K <= f()) did not return true when it should."
            )
            self.assertEqual(
                (self.f1 <= ref_value)(x, y, z, u, w, v), 1.0,
                msg="Function6D less equals scalar (f() <= K) did not return true when it should."
            )
            self.assertEqual(
                (ref_value <= self.f1)(x, y, z, u, w, v), 1.0,
                msg="Scalar less equals Function6D (K <= f()) did not return true when it should."
            )
            self.assertEqual(
                (self.f1 <= lower_value)(x, y, z, u, w, v), 0.0,
                msg="Function6D less equals scalar (f() <= K) did not return false when it should."
            )
            self.assertEqual(
                (higher_value <= self.f1)(x, y, z, u, w, v), 0.0,
                msg="Scalar less equals Function6D (K <= f()) did not return false when it should."
            )
            self.assertEqual(
                (self.f1 >= lower_value)(x, y, z, u, w, v), 1.0,
                msg="Function6D greater equals scalar (f() >= K) did not return true when it should."
            )
            self.assertEqual(
                (higher_value >= self.f1)(x, y, z, u, w, v), 1.0,
                msg="Scalar greater equals Function6D (K >= f()) did not return true when it should."
            )
            self.assertEqual(
                (self.f1 >= ref_value)(x, y, z, u, w, v), 1.0,
                msg="Function6D greater equals scalar (f() >= K) did not return true when it should."
            )
            self.assertEqual(
                (ref_value >= self.f1)(x, y, z, u, w, v), 1.0,
                msg="Scalar greater equals Function6D (K >= f()) did not return true when it should."
            )
            self.assertEqual(
                (self.f1 >= higher_value)(x, y, z, u, w, v), 0.0,
                msg="Function6D greater equals scalar (f() >= K) did not return false when it should."
            )
            self.assertEqual(
                (lower_value >= self.f1)(x, y, z, u, w, v), 0.0,
                msg="Scalar greater equals Function6D (K >= f()) did not return false when it should."
            )

    def test_add_function6d(self):
        testvals = [-1e10, -7, -0.001, 0.0, 0.00003, 10, 2.3e49]
        r1 = self.f1 + self.f2
        r2 = self.ref1 + self.f2
        r3 = self.f1 + self.ref2
        for (x, y, z, u, w, v) in itertools.product(testvals, repeat=6):
            self.assertEqual(r1(x, y, z, u, w, v), self.ref1(x, y, z, u, w, v) + self.ref2(x, y, z, u, w, v), "Function6D add function (f1() + f2()) did not match reference function value.")
            self.assertEqual(r2(x, y, z, u, w, v), self.ref1(x, y, z, u, w, v) + self.ref2(x, y, z, u, w, v), "Function6D add function (p1() + f2()) did not match reference function value.")
            self.assertEqual(r3(x, y, z, u, w, v), self.ref1(x, y, z, u, w, v) + self.ref2(x, y, z, u, w, v), "Function6D add function (f1() + p2()) did not match reference function value.")

    def test_sub_function6d(self):
        testvals = [-1e10, -7, -0.001, 0.0, 0.00003, 10, 2.3e49]
        r1 = self.f1 - self.f2
        r2 = self.ref1 - self.f2
        r3 = self.f1 - self.ref2
        for (x, y, z, u, w, v) in itertools.product(testvals, repeat=6):
            self.assertEqual(r1(x, y, z, u, w, v), self.ref1(x, y, z, u, w, v) - self.ref2(x, y, z, u, w, v), "Function6D subtract function (f1() - f2()) did not match reference function value.")
            self.assertEqual(r2(x, y, z, u, w, v), self.ref1(x, y, z, u, w, v) - self.ref2(x, y, z, u, w, v), "Function6D subtract function (p1() - f2()) did not match reference function value.")
            self.assertEqual(r3(x, y, z, u, w, v), self.ref1(x, y, z, u, w, v) - self.ref2(x, y, z, u, w, v), "Function6D subtract function (f1() - p2()) did not match reference function value.")

    def test_mul_function6d(self):
        testvals = [-1e10, -7, -0.001, 0.0, 0.00003, 10, 2.3e49]
        r1 = self.f1 * self.f2
        r2 = self.ref1 * self.f2
        r3 = self.f1 * self.ref2
        for (x, y, z, u, w, v) in itertools.product(testvals, repeat=6):
            self.assertEqual(r1(x, y, z, u, w, v), self.ref1(x, y, z, u, w, v) * self.ref2(x, y, z, u, w, v), "Function6D multiply function (f1() * f2()) did not match reference function value.")
            self.assertEqual(r2(x, y, z, u, w, v), self.ref1(x, y, z, u, w, v) * self.ref2(x, y, z, u, w, v), "Function6D multiply function (p1() * f2()) did not match reference function value.")
            self.assertEqual(r3(x, y, z, u, w, v), self.ref1(x, y, z, u, w, v) * self.ref2(x, y, z, u, w, v), "Function6D multiply function (f1() * p2()) did not match reference function value.")

    def test_div_function6d(self):
        testvals = [-1e10, -7, -0.001, 0.00003, 10, 2.3e49]
        r1 = self.f1 / self.f2
        r2 = self.ref1 / self.f2
        r3 = self.f1 / self.ref2
        for (x, y, z, u, w, v) in itertools.product(testvals, repeat=6):
            self.assertAlmostEqual(r1(x, y, z, u, w, v), self.ref1(x, y, z, u, w, v) / self.ref2(x, y, z, u, w, v), delta=abs(r1(x, y, z, u, w, v)) * 1e-12, msg="Function6D divide function (f1() / f2()) did not match reference function value.")
            self.assertAlmostEqual(r2(x, y, z, u, w, v), self.ref1(x, y, z, u, w, v) / self.ref2(x, y, z, u, w, v), delta=abs(r2(x, y, z, u, w, v)) * 1e-12, msg="Function6D divide function (p1() / f2()) did not match reference function value.")
            self.assertAlmostEqual(r3(x, y, z, u, w, v), self.ref1(x, y, z, u, w, v) / self.ref2(x, y, z, u, w, v), delta=abs(r3(x, y, z, u, w, v)) * 1e-12, msg="Function6D divide function (f1() / p2()) did not match reference function value.")

        with self.assertRaises(ZeroDivisionError, msg="ZeroDivisionError not raised when function returns zero."):
            r1(0, 0, 0, 0, 0, 0)

    def test_mod_function6d(self):
        testvals = [-1e10, -7, -0.001, 0.00003, 10, 2.3e49]
        r1 = self.f1 % self.f2
        r2 = self.ref1 % self.f2
        r3 = self.f1 % self.ref2
        for (x, y, z, u, w, v) in itertools.product(testvals, repeat=6):
            self.assertAlmostEqual(r1(x, y, z, u, w, v), math.fmod(self.ref1(x, y, z, u, w, v), self.ref2(x, y, z, u, w, v)), delta=abs(r1(x, y, z, u, w, v)) * 1e-12, msg="Function6D modulo function (f1() % f2()) did not match reference function value.")
            self.assertAlmostEqual(r2(x, y, z, u, w, v), math.fmod(self.ref1(x, y, z, u, w, v), self.ref2(x, y, z, u, w, v)), delta=abs(r2(x, y, z, u, w, v)) * 1e-12, msg="Function6D modulo function (p1() % f2()) did not match reference function value.")
            self.assertAlmostEqual(r3(x, y, z, u, w, v), math.fmod(self.ref1(x, y, z, u, w, v), self.ref2(x, y, z, u, w, v)), delta=abs(r3(x, y, z, u, w, v)) * 1e-12, msg="Function6D modulo function (f1() % p2()) did not match reference function value.")

        with self.assertRaises(ZeroDivisionError, msg="ZeroDivisionError not raised when function returns zero."):
            r1(0, 0, 0, 0, 0, 0)

    def test_pow_function6d_function6d(self):
        testvals = [-3.0, -0.7, -0.001, 0.00003, 2]
        r1 = self.f1 ** self.f2
        r2 = self.ref1 ** self.f2
        r3 = self.f1 ** self.ref2
        for (x, y, z, u, w, v) in itertools.product(testvals, repeat=6):
            if self.ref1(x, y, z, u, w, v) < 0 and not float(self.ref2(x, y, z, u, w, v)).is_integer():
                with self.assertRaises(ValueError, msg="ValueError not raised when base is negative and exponent non-integral (1/3)."):
                    r1(x, y, z, u, w, v)
                with self.assertRaises(ValueError, msg="ValueError not raised when base is negative and exponent non-integral (2/3)."):
                    r2(x, y, z, u, w, v)
                with self.assertRaises(ValueError, msg="ValueError not raised when base is negative and exponent non-integral (3/3)."):
                    r3(x, y, z, u, w, v)
            else:
                self.assertAlmostEqual(r1(x, y, z, u, w, v), self.ref1(x, y, z, u, w, v) ** self.ref2(x, y, z, u, w, v), 15, "Function6D power function (f1() ** f2()) did not match reference function value.")
                self.assertAlmostEqual(r2(x, y, z, u, w, v), self.ref1(x, y, z, u, w, v) ** self.ref2(x, y, z, u, w, v), 15, "Function6D power function (p1() ** f2()) did not match reference function value.")
                self.assertAlmostEqual(r3(x, y, z, u, w, v), self.ref1(x, y, z, u, w, v) ** self.ref2(x, y, z, u, w, v), 15, "Function6D power function (f1() ** p2()) did not match reference function value.")

        with self.assertRaises(ZeroDivisionError, msg="ZeroDivisionError not raised when f1() == 0 and f2() is negative."):
            r4 = PythonFunction6D(lambda x, y, z, u, w, v: 0) ** self.f1
            r4(-1, 0, 0, 0, 0, 0)

    def test_pow_3_arguments(self):
        testvals = [-10, -7, -0.001, 0.00003, 0.8]
        r1 = pow(self.f1, 5, 3)
        r2 = pow(5, self.f1, 3)
        r3 = pow(5, self.f1, self.f2)
        r4 = pow(self.f2, self.f1, self.f2)
        r5 = pow(self.f2, self.ref1, self.ref2)
        r6 = pow(self.ref2, self.f1, self.f2)
        # Can't use 3 argument pow() if all arguments aren't integers, so
        # use fmod(a, b) % c instead
        for (x, y, z, u, w, v) in itertools.product(testvals, repeat=6):
            self.assertEqual(r1(x, y, z, u, w, v), math.fmod(self.ref1(x, y, z, u, w, v) ** 5, 3), "Function6D 3 argument pow(f1(), A, B) did not match reference value.")
            self.assertEqual(r2(x, y, z, u, w, v), math.fmod(5 ** self.ref1(x, y, z, u, w, v), 3), "Function6D 3 argument pow(A, f1(), B) did not match reference value.")
            self.assertEqual(r3(x, y, z, u, w, v), math.fmod(5 ** self.ref1(x, y, z, u, w, v), self.ref2(x, y, z, u, w, v)), "Function6D 3 argument pow(A, f1(), f2()) did not match reference value.")
            self.assertEqual(r4(x, y, z, u, w, v), math.fmod(self.ref2(x, y, z, u, w, v) ** self.ref1(x, y, z, u, w, v), self.ref2(x, y, z, u, w, v)), "Function6D 3 argument pow(f2(), f1(), f2()) did not match reference value.")
            self.assertEqual(r5(x, y, z, u, w, v), math.fmod(self.ref2(x, y, z, u, w, v) ** self.ref1(x, y, z, u, w, v), self.ref2(x, y, z, u, w, v)), "Function6D 3 argument pow(f2(), p1(), p2()) did not match reference value.")
            self.assertEqual(r6(x, y, z, u, w, v), math.fmod(self.ref2(x, y, z, u, w, v) ** self.ref1(x, y, z, u, w, v), self.ref2(x, y, z, u, w, v)), "Function6D 3 argument pow(p2(), f1(), f2()) did not match reference value.")

    def test_abs(self):
        testvals = [-1e10, -7, -0.001, 0.0, 0.0003, 10, 2.3e49]
        for (x, y, z, u, w, v) in itertools.product(testvals, repeat=6):
            self.assertEqual(abs(self.f1)(x, y, z, u, w, v), abs(self.ref1(x, y, z, u, w, v)),
                             msg="abs(Function6D) did not match reference value")

    def test_richcmp_function_callable(self):
        testvals = [-1e10, -7, -0.001, 0.0, 0.00003, 10, 2.3e49]
        for (x, y, z, u, w, v) in itertools.product(testvals, repeat=6):
            ref_value = self.ref1
            higher_value = lambda x, y, z, u, w, v: self.ref1(x, y, z, u, w, v) + abs(self.ref1(x, y, z, u, w, v)) + 1
            lower_value = lambda x, y, z, u, w, v: self.ref1(x, y, z, u, w, v) - abs(self.ref1(x, y, z, u, w, v)) - 1
            self.assertEqual(
                (self.f1 == ref_value)(x, y, z, u, w, v), 1.0,
                msg="Function6D equals callable (f1() == f2()) did not return true when it should."
            )
            self.assertEqual(
                (self.f1 == higher_value)(x, y, z, u, w, v), 0.0,
                msg="Function6D equals callable (f1() == f2()) did not return false when it should."
            )
            self.assertEqual(
                (self.f1 != higher_value)(x, y, z, u, w, v), 1.0,
                msg="Function6D not equals callable (f1() != f2()) did not return true when it should."
            )
            self.assertEqual(
                (self.f1 != ref_value)(x, y, z, u, w, v), 0.0,
                msg="Function6D not equals callable (f1() != f2()) did not return false when it should."
            )
            self.assertEqual(
                (self.f1 < higher_value)(x, y, z, u, w, v), 1.0,
                msg="Function6D less than callable (f1() < f2()) did not return true when it should."
            )
            self.assertEqual(
                (self.f1 < lower_value)(x, y, z, u, w, v), 0.0,
                msg="Function6D less than callable (f1() < f2()) did not return false when it should."
            )
            self.assertEqual(
                (self.f1 > lower_value)(x, y, z, u, w, v), 1.0,
                msg="Function6D greater than callable (f1() > f2()) did not return true when it should."
            )
            self.assertEqual(
                (self.f1 > higher_value)(x, y, z, u, w, v), 0.0,
                msg="Function6D greater than callable (f1() > f2()) did not return false when it should."
            )
            self.assertEqual(
                (self.f1 <= higher_value)(x, y, z, u, w, v), 1.0,
                msg="Function6D less equals callable (f1() <= f2()) did not return true when it should."
            )
            self.assertEqual(
                (self.f1 <= ref_value)(x, y, z, u, w, v), 1.0,
                msg="Function6D less equals callable (f1() <= f2()) did not return true when it should."
            )
            self.assertEqual(
                (self.f1 <= lower_value)(x, y, z, u, w, v), 0.0,
                msg="Function6D less equals callable (f1() <= f2()) did not return false when it should."
            )
            self.assertEqual(
                (self.f1 >= lower_value)(x, y, z, u, w, v), 1.0,
                msg="Function6D equals callable (f1() >= f2()) did not return true when it should."
            )
            self.assertEqual(
                (self.f1 >= ref_value)(x, y, z, u, w, v), 1.0,
                msg="Function6D greater equals callable (f1() >= f2()) did not return true when it should."
            )
            self.assertEqual(
                (self.f1 >= higher_value)(x, y, z, u, w, v), 0.0,
                msg="Function6D equals callable (f1() >= f2()) did not return false when it should."
            )

    def test_richcmp_callable_function(self):
        testvals = [-1e10, -7, -0.001, 0.0, 0.00003, 10, 2.3e49]
        for (x, y, z, u, w, v) in itertools.product(testvals, repeat=6):
            ref_value = self.ref1
            higher_value = lambda x, y, z, u, w, v: self.ref1(x, y, z, u, w, v) + abs(self.ref1(x, y, z, u, w, v)) + 1
            lower_value = lambda x, y, z, u, w, v: self.ref1(x, y, z, u, w, v) - abs(self.ref1(x, y, z, u, w, v)) - 1
            self.assertEqual(
                (ref_value == self.f1)(x, y, z, u, w, v), 1.0,
                msg="Callable equals Function6D (f1() == f2()) did not return true when it should."
            )
            self.assertEqual(
                (higher_value == self.f1)(x, y, z, u, w, v), 0.0,
                msg="Callable equals Function6D (f1() == f2()) did not return false when it should."
            )
            self.assertEqual(
                (higher_value != self.f1)(x, y, z, u, w, v), 1.0,
                msg="Callable not equals Function6D (f1() != f2()) did not return true when it should."
            )
            self.assertEqual(
                (ref_value != self.f1)(x, y, z, u, w, v), 0.0,
                msg="Callable not equals Function6D (f1() != f2()) did not return false when it should."
            )
            self.assertEqual(
                (lower_value < self.f1)(x, y, z, u, w, v), 1.0,
                msg="Callable less than Function6D (f1() < f2()) did not return true when it should."
            )
            self.assertEqual(
                (higher_value < self.f1)(x, y, z, u, w, v), 0.0,
                msg="Callable less than Function6D (f1() < f2()) did not return false when it should."
            )
            self.assertEqual(
                (higher_value > self.f1)(x, y, z, u, w, v), 1.0,
                msg="Callable greater than Function6D (f1() > f2()) did not return true when it should."
            )
            self.assertEqual(
                (lower_value > self.f1)(x, y, z, u, w, v), 0.0,
                msg="Callable greater than Function6D (f1() > f2()) did not return false when it should."
            )
            self.assertEqual(
                (lower_value <= self.f1)(x, y, z, u, w, v), 1.0,
                msg="Callable less equals Function6D (f1() <= f2()) did not return true when it should."
            )
            self.assertEqual(
                (ref_value <= self.f1)(x, y, z, u, w, v), 1.0,
                msg="Callable less equals Function6D (f1() <= f2()) did not return true when it should."
            )
            self.assertEqual(
                (higher_value <= self.f1)(x, y, z, u, w, v), 0.0,
                msg="Callable less equals Function6D (f1() <= f2()) did not return false when it should."
            )
            self.assertEqual(
                (higher_value >= self.f1)(x, y, z, u, w, v), 1.0,
                msg="Callable equals Function6D (f1() >= f2()) did not return true when it should."
            )
            self.assertEqual(
                (ref_value >= self.f1)(x, y, z, u, w, v), 1.0,
                msg="Callable greater equals Function6D (f1() >= f2()) did not return true when it should."
            )
            self.assertEqual(
                (lower_value >= self.f1)(x, y, z, u, w, v), 0.0,
                msg="Callable equals Function6D (f1() >= f2()) did not return false when it should."
            )

    def test_richcmp_function_function(self):
        testvals = [-1e10, -7, -0.001, 0.0, 0.00003, 10, 2.3e49]
        for (x, y, z, u, w, v) in itertools.product(testvals, repeat=6):
            ref_value = self.f1
            higher_value = self.f1 + abs(self.f1) + 1
            lower_value = self.f1 - abs(self.f1) - 1
            self.assertEqual(
                (self.f1 == ref_value)(x, y, z, u, w, v), 1.0,
                msg="Function6D equals Function6D (f1() == f2()) did not return true when it should."
            )
            self.assertEqual(
                (self.f1 == higher_value)(x, y, z, u, w, v), 0.0,
                msg="Function6D equals Function6D (f1() == f2()) did not return false when it should."
            )
            self.assertEqual(
                (self.f1 != higher_value)(x, y, z, u, w, v), 1.0,
                msg="Function6D not equals Function6D (f1() != f2()) did not return true when it should."
            )
            self.assertEqual(
                (self.f1 != ref_value)(x, y, z, u, w, v), 0.0,
                msg="Function6D not equals Function6D (f1() != f2()) did not return false when it should."
            )
            self.assertEqual(
                (self.f1 < higher_value)(x, y, z, u, w, v), 1.0,
                msg="Function6D less than Function6D (f1() < f2()) did not return true when it should."
            )
            self.assertEqual(
                (self.f1 < lower_value)(x, y, z, u, w, v), 0.0,
                msg="Function6D less than Function6D (f1() < f2()) did not return false when it should."
            )
            self.assertEqual(
                (self.f1 > lower_value)(x, y, z, u, w, v), 1.0,
                msg="Function6D greater than Function6D (f1() > f2()) did not return true when it should."
            )
            self.assertEqual(
                (self.f1 > higher_value)(x, y, z, u, w, v), 0.0,
                msg="Function6D greater than Function6D (f1() > f2()) did not return false when it should."
            )
            self.assertEqual(
                (self.f1 <= higher_value)(x, y, z, u, w, v), 1.0,
                msg="Function6D less equals Function6D (f1() <= f2()) did not return true when it should."
            )
            self.assertEqual(
                (self.f1 <= ref_value)(x, y, z, u, w, v), 1.0,
                msg="Function6D less equals Function6D (f1() <= f2()) did not return true when it should."
            )
            self.assertEqual(
                (self.f1 <= lower_value)(x, y, z, u, w, v), 0.0,
                msg="Function6D less equals Function6D (f1() <= f2()) did not return false when it should."
            )
            self.assertEqual(
                (self.f1 >= lower_value)(x, y, z, u, w, v), 1.0,
                msg="Function6D equals Function6D (f1() >= f2()) did not return true when it should."
            )
            self.assertEqual(
                (self.f1 >= ref_value)(x, y, z, u, w, v), 1.0,
                msg="Function6D greater equals Function6D (f1() >= f2()) did not return true when it should."
            )
            self.assertEqual(
                (self.f1 >= higher_value)(x, y, z, u, w, v), 0.0,
                msg="Function6D equals Function6D (f1() >= f2()) did not return false when it should."
            )
