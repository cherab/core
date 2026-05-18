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

from raysect.core import World, Vector3D, translate

from cherab.core import Beam
from cherab.core.atomic import AtomicData, BeamStoppingRate
from cherab.core.atomic import deuterium
from cherab.tools.plasmas.slab import build_constant_slab_plasma
from cherab.core.model import SingleRayAttenuator

from cherab.core.utility import EvAmuToMS, EvToJ


class ConstantBeamStoppingRate(BeamStoppingRate):
    """
    Constant beam CX PEC for test purpose.
    """

    def __init__(self, donor_metastable, value):
        self.donor_metastable = donor_metastable
        self.value = value

    def evaluate(self, energy, density, temperature):

        return self.value


class MockAtomicData(AtomicData):
    """Fake atomic data for test purpose."""

    def beam_stopping_rate(self, beam_ion, plasma_ion, charge):

        return ConstantBeamStoppingRate(1, 1.e-13)


class TestBeam(unittest.TestCase):

    def setUp(self):

        self.atomic_data = MockAtomicData()

        self.world = World()

        self.plasma_density = 1.e19
        self.plasma_temperature = 1.e3
        plasma_species = [(deuterium, 1, self.plasma_density, self.plasma_temperature, Vector3D(0, 0, 0))]
        plasma = build_constant_slab_plasma(length=1, width=1, height=1,
                                            electron_density=self.plasma_density,
                                            electron_temperature=self.plasma_temperature,
                                            plasma_species=plasma_species)
        plasma.atomic_data = self.atomic_data
        plasma.parent = self.world

        beam = Beam(transform=translate(0.5, 0, 0))
        beam.atomic_data = self.atomic_data
        beam.plasma = plasma
        beam.attenuator = SingleRayAttenuator(clamp_to_zero=True)
        beam.energy = 50000
        beam.power = 1e6
        beam.temperature = 10
        beam.element = deuterium
        beam.parent = self.world
        beam.sigma = 0.2
        beam.divergence_x = 1.
        beam.divergence_y = 2.
        beam.length = 10.

        self.plasma = plasma
        self.beam = beam

    def test_beam_density(self):

        z0 = 0.8
        x0, y0 = 0.5, 0.5

        density_on_axis = self.beam.density(0, 0, z0)
        density_off_axis = self.beam.density(x0, y0, z0)
        density_outside_beam = self.beam.density(0, 0, -1)

        # validating

        speed = EvAmuToMS.to(self.beam.energy)
        # constant stopping rate
        stopping_rate = self.atomic_data.beam_stopping_rate(deuterium, deuterium, 1)(0, 0, 0)
        attenuation_factor = np.exp(-z0 * self.plasma_density * stopping_rate / speed)

        beam_particle_rate = self.beam.power / EvToJ.to(self.beam.energy * deuterium.atomic_weight)

        sigma0_sqr = self.beam.sigma**2
        tanxdiv = np.tan(np.deg2rad(self.beam.divergence_x))
        tanydiv = np.tan(np.deg2rad(self.beam.divergence_y))
        sigma_x = np.sqrt(sigma0_sqr + (z0 * tanxdiv)**2)
        sigma_y = np.sqrt(sigma0_sqr + (z0 * tanydiv)**2)

        norm_radius_sqr = ((x0 / sigma_x)**2 + (y0 / sigma_y)**2)

        gaussian_sample_on_axis = 1. / (2 * np.pi * sigma_x * sigma_y)
        gaussian_sample_off_axis = np.exp(-0.5 * norm_radius_sqr) / (2 * np.pi * sigma_x * sigma_y)

        test_density_on_axis = beam_particle_rate / speed * gaussian_sample_on_axis * attenuation_factor
        test_density_off_axis = beam_particle_rate / speed * gaussian_sample_off_axis * attenuation_factor

        self.assertAlmostEqual(density_on_axis / test_density_on_axis, 1., delta=1.e-12,
                               msg='Beam.density() gives a wrong value on the beam axis.')
        self.assertAlmostEqual(density_off_axis / test_density_off_axis, 1., delta=1.e-12,
                               msg='Beam.density() gives a wrong value off the beam axis.')
        self.assertEqual(density_outside_beam, 0,
                         msg='Beam.density() gives a non-zero value outside beam.')

    def test_beam_direction(self):
        # setting up the model

        z0 = 0.8
        x0, y0 = 0.5, 0.5

        direction_on_axis = self.beam.direction(0, 0, z0)
        direction_off_axis = self.beam.direction(x0, y0, z0)
        direction_outside_beam = self.beam.direction(0, 0, -1)

        # validating

        sigma0_sqr = self.beam.sigma**2
        z_tanx_sqr = (z0 * np.tan(np.deg2rad(self.beam.divergence_x)))**2
        z_tany_sqr = (z0 * np.tan(np.deg2rad(self.beam.divergence_y)))**2

        ex = x0 * z_tanx_sqr / (sigma0_sqr + z_tanx_sqr)
        ey = y0 * z_tany_sqr / (sigma0_sqr + z_tany_sqr)
        ez = z0

        test_direction_off_axis = Vector3D(ex, ey, ez).normalise()

        self.assertEqual(direction_on_axis, Vector3D(0, 0, 1),
                         msg='Beam.density() gives a wrong value on the beam axis.')
        for v, test_v in zip(direction_off_axis, test_direction_off_axis):
            self.assertAlmostEqual(v, test_v, delta=1.e-12,
                                   msg='Beam.direction() gives a wrong value off the beam axis.')
        self.assertEqual(direction_outside_beam, Vector3D(0, 0, 1),
                         msg='Beam.density() gives a non-zero value outside beam.')


if __name__ == '__main__':
    unittest.main()
