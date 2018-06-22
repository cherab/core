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

from cherab.core.math import mappers
import numpy as np
import unittest

# todo: add tests for VectorAxisymmetricMapper

class TestMappers(unittest.TestCase):
    """Mappers tests."""
    
    def setUp(self):
        """Initialisation with functions to map."""

        def f1d(x): return x*np.cos(x-3)
        self.function1d = f1d
        def f2d(x, y): return x*np.sin(y)
        self.function2d = f2d
        def f3d(x, y, z): return x*x*np.exp(y)-2*z*y
        self.function3d = f3d


    def test_iso_mapper_2d(self):
        """Composition of a 1D and a 2D function."""
        compound_function = mappers.IsoMapper2D(self.function2d, self.function1d)
        self.assertAlmostEqual(compound_function(2., 1.),
                               self.function1d(self.function2d(2., 1.)),
                               places=10)

    def test_iso_mapper_2d_invalid_arg1(self):
        """An error must be raised if the first argument is not a callable."""
        self.assertRaises(TypeError, mappers.IsoMapper2D, "blah", self.function1d)

    def test_iso_mapper_2d_invalid_arg2(self):
        """An error must be raised if the second argument is not a callable."""
        self.assertRaises(TypeError, mappers.IsoMapper2D, self.function2d, "blah")

    def test_iso_mapper_3d(self):
        """Composition of a 1D and a 3D function."""
        compound_function = mappers.IsoMapper3D(self.function3d, self.function1d)
        self.assertAlmostEqual(compound_function(0.75, 3.7, 12),
                               self.function1d(self.function3d(0.75, 3.7, 12)),
                               places=10)

    def test_iso_mapper_3d_invalid_arg1(self):
        """An error must be raised if the first argument is not a callable."""
        self.assertRaises(TypeError, mappers.IsoMapper2D, "blah", self.function1d)

    def test_iso_mapper_3d_invalid_arg2(self):
        """An error must be raised if the second argument is not a callable."""
        self.assertRaises(TypeError, mappers.IsoMapper2D, self.function3d, "blah")

    def test_swizzle_2d(self):
        """Swap of a 2D function arguments."""
        swizzled_function = mappers.Swizzle2D(self.function2d)
        self.assertAlmostEqual(swizzled_function(3.6, -7.1),
                               self.function2d(-7.1, 3.6),
                               places=10)

    def test_swizzle_2d_invalid_arg(self):
        """An error must be raised if the given argument is not callable."""
        self.assertRaises(TypeError, mappers.Swizzle2D, "blah")

    def test_swizzle_3d_012(self):
        """Swap of a 3D function arguments: identity."""
        x, y, z = 4.2, 11, -0.3
        swizzled_function = mappers.Swizzle3D(self.function3d, (0, 1, 2))
        self.assertAlmostEqual(swizzled_function(x, y, z),
                               self.function3d(x, y, z),
                               places=10)

    def test_swizzle_3d_120(self):
        """Swap of a 3D function arguments: x,y,z -> y,z,x."""
        x, y, z = 4.2, 11, -0.3
        swizzled_function = mappers.Swizzle3D(self.function3d, (1, 2, 0))
        self.assertAlmostEqual(swizzled_function(x, y, z),
                               self.function3d(y, z, x),
                               places=10)

    def test_swizzle_3d_201(self):
        """Swap of a 3D function arguments: x,y,z -> z,x,y."""
        x, y, z = 4.2, 11, -0.3
        swizzled_function = mappers.Swizzle3D(self.function3d, (2, 0, 1))
        self.assertAlmostEqual(swizzled_function(x, y, z),
                               self.function3d(z, x, y),
                               places=10)

    def test_swizzle_3d_101(self):
        """Swap of a 3D function arguments: x,y,z -> y,x,y."""
        x, y, z = 4.2, 11, -0.3
        swizzled_function = mappers.Swizzle3D(self.function3d, (1, 0, 1))
        self.assertAlmostEqual(swizzled_function(x, y, z),
                               self.function3d(y, x, y),
                               places=10)

    def test_swizzle_3d_0120(self):
        """Swap of a 3D function arguments: argument error raising."""
        self.assertRaises(TypeError, mappers.Swizzle3D,
                          self.function3d, (0,1,2,0))

    def test_swizzle_3d_01a(self):
        """Swap of a 3D function arguments: argument error raising."""
        self.assertRaises(ValueError, mappers.Swizzle3D,
                          self.function3d, (0,1,'a'))

    def test_swizzle_3d_invalid_arg(self):
        """An error must be raised if the first argument is not callable."""
        self.assertRaises(TypeError, mappers.Swizzle3D, "blah", (0, 1, 2))

    def test_axisymmetric_mapper(self):
        """Axisymmetric mapper."""
        sym_func = mappers.AxisymmetricMapper(self.function2d)
        x, z, theta = 4.72, -2.8, 3.1
        self.assertAlmostEqual(sym_func(x*np.cos(theta), x*np.sin(theta), z),
                               self.function2d(x, z),
                               places=10)

    def test_axisymmetric_mapper_0(self):
        """Axisymmetric mapper. Test at the origin."""
        sym_func = mappers.AxisymmetricMapper(self.function2d)
        self.assertAlmostEqual(sym_func(0, 0, 0),
                               self.function2d(0, 0),
                               places=10)

    def test_axisymmetric_mapper_invalid_arg(self):
        """An error must be raised if the given argument is not callable."""
        self.assertRaises(TypeError, mappers.AxisymmetricMapper, "blah")

if __name__ == '__main__':
    unittest.main()