# Copyright 2016-2023 Euratom
# Copyright 2016-2023 United Kingdom Atomic Energy Authority
# Copyright 2016-2023 Centro de Investigaciones Energéticas, Medioambientales y Tecnológicas
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

import numpy as np

from raysect.core import Point3D, Vector3D
from raysect.optical import Ray, World

from cherab.core.atomic import AtomicData, LineRadiationPower, ContinuumPower, CXRadiationPower
from cherab.core.atomic import deuterium, hydrogen, nitrogen
from cherab.tools.plasmas.slab import build_constant_slab_plasma
from cherab.core.model import TotalRadiatedPower

from cherab.core.utility import EvAmuToMS, EvToJ


class ConstantLineRadiationPower(LineRadiationPower):
    """
    Constant line radiation power coefficient.
    """

    def __init__(self, value):
        self.value = value

    def evaluate(self, density, temperature):

        return self.value


class ConstantContinuumPower(ContinuumPower):
    """
    Constant continuum power coefficient.
    """

    def __init__(self, value):
        self.value = value

    def evaluate(self, density, temperature):

        return self.value


class ConstantCXRadiationPower(CXRadiationPower):
    """
    Constant charge exchange radiation power coefficient.
    """

    def __init__(self, value):
        self.value = value

    def evaluate(self, density, temperature):

        return self.value


class MockAtomicData(AtomicData):
    """Fake atomic data for test purpose."""

    def line_radiated_power_rate(self, element, charge):

        return ConstantLineRadiationPower(1.e-32)

    def continuum_radiated_power_rate(self, element, charge):

        return ConstantContinuumPower(1.e-33)

    def cx_radiated_power_rate(self, element, charge):

        return ConstantCXRadiationPower(1.e-31)


class TestTotalRadiatedPower(unittest.TestCase):

    def setUp(self):

        self.world = World()

        self.atomic_data = MockAtomicData()

        plasma_species = [(deuterium, 0, 1.e18, 500., Vector3D(0, 0, 0)),
                          (hydrogen, 0, 1.e18, 500., Vector3D(0, 0, 0)),
                          (nitrogen, 6, 5.e18, 1100., Vector3D(0, 0, 0)),
                          (nitrogen, 7, 1.e19, 1100., Vector3D(0, 0, 0))]
        self.plasma = build_constant_slab_plasma(length=1.2, width=1.2, height=1.2,
                                                 electron_density=1e19, electron_temperature=1000.,
                                                 plasma_species=plasma_species)
        self.plasma.parent = self.world
        self.plasma.atomic_data = self.atomic_data

    def test_total_radiated_power(self):

        self.plasma.models = [TotalRadiatedPower(nitrogen, 6)]

        # observing
        origin = Point3D(1.5, 0, 0)
        direction = Vector3D(-1, 0, 0)
        ray = Ray(origin=origin, direction=direction,
                  min_wavelength=500., max_wavelength=550., bins=2)
        radiated_power = ray.trace(self.world).total()

        # validating
        ne = self.plasma.electron_distribution.density(0.5, 0.5, 0.5)
        n_n6 = self.plasma.composition[(nitrogen, 6)].distribution.density(0.5, 0.5, 0.5)
        n_n7 = self.plasma.composition[(nitrogen, 7)].distribution.density(0.5, 0.5, 0.5)
        n_h0 = self.plasma.composition[(hydrogen, 0)].distribution.density(0.5, 0.5, 0.5)
        n_d0 = self.plasma.composition[(deuterium, 0)].distribution.density(0.5, 0.5, 0.5)

        integration_length = 1.2

        plt_rate = self.atomic_data.line_radiated_power_rate(nitrogen, 6).value
        plt_radiance = 0.25 / np.pi * plt_rate * ne * n_n6 * integration_length

        prb_rate = self.atomic_data.continuum_radiated_power_rate(nitrogen, 7).value
        prb_radiance = 0.25 / np.pi * prb_rate * ne * n_n7 * integration_length

        prc_rate = self.atomic_data.cx_radiated_power_rate(nitrogen, 7).value
        prc_radiance = 0.25 / np.pi * prc_rate * (n_h0 + n_d0) * n_n7 * integration_length

        test_radiated_power = plt_radiance + prb_radiance + prc_radiance

        self.assertAlmostEqual(radiated_power / test_radiated_power, 1., delta=1e-8)


if __name__ == '__main__':
    unittest.main()
