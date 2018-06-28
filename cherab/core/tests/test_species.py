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

from cherab.core import Species
from cherab.core.distribution import Maxwellian
from scipy.constants import atomic_mass

from cherab.core.atomic import elements
from raysect.core import Vector3D


class TestSpecies(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def get_maxwellian(self):
        return Maxwellian(lambda x, y, z: 1e20, lambda x, y, z: 1e3, lambda x, y, z: Vector3D(1e5, 0, 0), 4 * atomic_mass)

    def test_negative_ionisation(self):
        distribution = self.get_maxwellian()

        with self.assertRaises(ValueError, msg='The ionisation of a species must not be negative!'):
            Species(elements.carbon, -2, distribution)

    def test_too_big_ionisation(self):
        distribution = self.get_maxwellian()

        with self.assertRaises(ValueError, msg='The ionisation of a species must not be superior than the atomic number of the element!'):
            Species(elements.carbon, 7, distribution)

    def test_no_element_setter(self):
        species = Species(elements.carbon, 3, self.get_maxwellian())

        with self.assertRaises(AttributeError, msg='It must not be possible to change the element of a species!'):
            species.element = elements.fluorine

    def test_no_ionisation_setter(self):
        species = Species(elements.carbon, 3, self.get_maxwellian())

        with self.assertRaises(AttributeError, msg='It must not be possible to change the ionisation of a species!'):
            species.ionisation = 5

    def test_no_distribution_setter(self):
        species = Species(elements.carbon, 3, self.get_maxwellian())

        with self.assertRaises(AttributeError, msg='It must not be possible to change the distribution of a species!'):
            species.distribution = Maxwellian(lambda x, y, z: 2e20, lambda x, y, z: 3e3, lambda x, y, z: Vector3D(1.6e5, 0, 0), 4.1 * atomic_mass)

if __name__ == '__main__':
    unittest.main()