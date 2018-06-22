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
from numpy import empty
from cherab.core.math.samplers import sample1d, sample2d, sample3d


def fn1d(x):
    """
    Python 1D test function.
    """

    return x * x


def fn2d(x, y):
    """
    Python 2D test function.
    """

    return x * x + 0.5 * y


def fn3d(x, y, z):
    """
    Python 3D test function.
    """

    return x * x + 0.5 * y - z


class TestSampler1D(unittest.TestCase):

    def test_sample1d_invalid_range_type(self):

        # invalid type
        with self.assertRaises(TypeError, msg="Type error was not raised when a string was (invalidly) supplied for the range."):

            sample1d(fn1d, "blah")

    def test_sample1d_invalid_range_length(self):

        # tuple too short
        with self.assertRaises(ValueError, msg="Passing a range tuple with too few values did not raise a ValueError."):

            sample1d(fn1d, (1.0, 2.0))

    def test_sample1d_invalid_range_minmax(self):

        # min range > max range
        with self.assertRaises(ValueError, msg="A ValueError was not raised when the min range was larger than the max range."):

            sample1d(fn1d, (10, 8, 100))

    def test_sample1d_invalid_range_samples(self):

        # number of samples < 1
        with self.assertRaises(ValueError, msg="A ValueError was not raised when the number of samples was < 1."):

            sample1d(fn1d, (10, 8, 0))

    def test_sample1d_invalid_function_called(self):

        # invalid function type
        with self.assertRaises(TypeError, msg="Type error was not raised when a string was (invalidly) supplied for the function."):

            sample1d("blah", (1, 2, 100))

    def test_sample1d_sample(self):

        rx = [1.0, 1.5, 2.0]
        rs = [1.0, 2.25, 4.0]

        tx, ts = sample1d(fn1d, (1.0, 2.0, 3))

        for i in range(3):

            self.assertEqual(tx[i], rx[i], "X points [{}] is incorrect.".format(i))
            self.assertEqual(ts[i], rs[i], "Sample point [{}] is incorrect.".format(i))


class TestSampler2D(unittest.TestCase):

    def test_sample2d_invalid_range_type(self):

        # invalid type for x
        with self.assertRaises(TypeError, msg="Type error was not raised when a string was (invalidly) supplied for the x range."):

            sample2d(fn2d, "blah", (1, 2, 3))

        # invalid type for y
        with self.assertRaises(TypeError, msg="Type error was not raised when a string was (invalidly) supplied for the y range."):

            sample2d(fn2d, (1, 2, 3), "blah")

    def test_sample2d_invalid_range_length(self):

        # tuple too short for x
        with self.assertRaises(ValueError, msg="Passing a range tuple for x with too few values did not raise a ValueError."):

            sample2d(fn2d, (1, 2), (1, 2, 3))

        # tuple too short for y
        with self.assertRaises(ValueError, msg="Passing a range tuple for y with too few values did not raise a ValueError."):

            sample2d(fn2d, (1, 2, 3), (1, 2))

    def test_sample2d_invalid_range_minmax(self):

        # min range > max range for x
        with self.assertRaises(ValueError, msg="A ValueError was not raised when the min x range was larger than the max x range."):

            sample2d(fn2d, (10, 8, 100), (1, 2, 3))

        # min range > max range for y
        with self.assertRaises(ValueError, msg="A ValueError was not raised when the min y range was larger than the max y range."):

            sample2d(fn2d, (1, 2, 3), (10, 8, 100))

    def test_sample2d_invalid_range_samples(self):

        # number of samples < 1 for x
        with self.assertRaises(ValueError, msg="A ValueError was not raised when the number of x samples was < 1."):

            sample2d(fn2d, (10, 8, 0), (1, 2, 3))

        # number of samples < 1 for y
        with self.assertRaises(ValueError, msg="A ValueError was not raised when the number of y samples was < 1."):

            sample2d(fn2d, (1, 2, 3), (10, 8, 0))

    def test_sample2d_invalid_function_called(self):

        # invalid function type
        with self.assertRaises(TypeError, msg="Type error was not raised when a string was (invalidly) supplied for the function."):

            sample2d("blah", (1, 2, 3), (1, 2, 3))

    def test_sample2d_sample(self):

        rx = [1.0, 1.5, 2.0]
        ry = [2.0, 2.5, 3.0]
        rs = empty((3, 3))

        for i in range(3):

            for j in range(3):

                rs[i][j] = rx[i] * rx[i] + 0.5 * ry[j]

        tx, ty, ts = sample2d(fn2d, (1.0, 2.0, 3), (2.0, 3.0, 3))

        for i in range(3):

            self.assertEqual(tx[i], rx[i], "X point [{}] is incorrect.".format(i))
            self.assertEqual(ty[i], ry[i], "Y point [{}] is incorrect.".format(i))

            for j in range(3):

                self.assertEqual(ts[i][j], rs[i][j], "Sample point [{}, {}] is incorrect.".format(i, j))


class TestSampler3D(unittest.TestCase):

    def test_sample3d_invalid_range_type(self):

        # invalid type for x
        with self.assertRaises(TypeError, msg="Type error was not raised when a string was (invalidly) supplied for the x range."):

            sample3d(fn3d, "blah", (1, 2, 3), (1, 2, 3))

        # invalid type for y
        with self.assertRaises(TypeError, msg="Type error was not raised when a string was (invalidly) supplied for the y range."):

            sample3d(fn3d, (1, 2, 3), "blah", (1, 2, 3))

        # invalid type for z
        with self.assertRaises(TypeError, msg="Type error was not raised when a string was (invalidly) supplied for the z range."):

            sample3d(fn3d, (1, 2, 3), (1, 2, 3), "blah")

    def test_sample3d_invalid_range_length(self):

        # tuple too short for x
        with self.assertRaises(ValueError, msg="Passing a range tuple for x with too few values did not raise a ValueError."):

            sample3d(fn3d, (1, 2), (1, 2, 3), (1, 2, 3))

        # tuple too short for y
        with self.assertRaises(ValueError, msg="Passing a range tuple for y with too few values did not raise a ValueError."):

            sample3d(fn3d, (1, 2, 3), (1, 2), (1, 2, 3))

        # tuple too short for z
        with self.assertRaises(ValueError, msg="Passing a range tuple for z with too few values did not raise a ValueError."):

            sample3d(fn3d, (1, 2, 3), (1, 2, 3), (1, 2))

    def test_sample3d_invalid_range_minmax(self):

        # min range > max range for x
        with self.assertRaises(ValueError, msg="A ValueError was not raised when the min x range was larger than the max x range."):

            sample3d(fn3d, (10, 8, 100), (1, 2, 3), (1, 2, 3))

        # min range > max range for y
        with self.assertRaises(ValueError, msg="A ValueError was not raised when the min y range was larger than the max y range."):

            sample3d(fn3d, (1, 2, 3), (10, 8, 100), (1, 2, 3))

        # min range > max range for z
        with self.assertRaises(ValueError, msg="A ValueError was not raised when the min z range was larger than the max z range."):

            sample3d(fn3d, (1, 2, 3), (1, 2, 3), (10, 8, 100))

    def test_sample3d_invalid_range_samples(self):

        # number of samples < 1 for x
        with self.assertRaises(ValueError, msg="A ValueError was not raised when the number of x samples was < 1."):

            sample3d(fn3d, (10, 8, 0), (1, 2, 3), (1, 2, 3))

        # number of samples < 1 for y
        with self.assertRaises(ValueError, msg="A ValueError was not raised when the number of y samples was < 1."):

            sample3d(fn3d, (1, 2, 3), (10, 8, 0), (1, 2, 3))

        # number of samples < 1 for z
        with self.assertRaises(ValueError, msg="A ValueError was not raised when the number of z samples was < 1."):

            sample3d(fn3d, (1, 2, 3), (1, 2, 3), (10, 8, 0))

    def test_sample3d_invalid_function_called(self):

        # invalid function type
        with self.assertRaises(TypeError, msg="Type error was not raised when a string was (invalidly) supplied for the function."):

            sample3d("blah", (1, 2, 3), (1, 2, 3), (1, 2, 3))

    def test_sample3d_sample(self):

        rx = [1.0, 1.5, 2.0]
        ry = [2.0, 2.5, 3.0]
        rz = [3.0, 3.5, 4.0]
        rs = empty((3, 3, 3))

        for i in range(3):

            for j in range(3):

                for k in range(3):

                    rs[i][j][k] = rx[i] * rx[i] + 0.5 * ry[j] - rz[k]

        tx, ty, tz, ts = sample3d(fn3d, (1.0, 2.0, 3), (2.0, 3.0, 3), (3.0, 4.0, 3))

        for i in range(3):

            self.assertEqual(tx[i], rx[i], "X point [{}] is incorrect.".format(i))
            self.assertEqual(ty[i], ry[i], "Y point [{}] is incorrect.".format(i))
            self.assertEqual(tz[i], rz[i], "Z point [{}] is incorrect.".format(i))

            for j in range(3):

                for k in range(3):

                    self.assertEqual(ts[i][j][k], rs[i][j][k], "Sample point [{}, {}, {}] is incorrect.".format(i, j, k))