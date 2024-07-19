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
from raysect.optical import World, Ray

from cherab.core.atomic import AtomicData, MaxwellianFreeFreeGauntFactor
from cherab.core.math.integrators import GaussianQuadrature
from cherab.core.atomic import deuterium, nitrogen
from cherab.tools.plasmas.slab import build_constant_slab_plasma
from cherab.core.model import Bremsstrahlung

import scipy.constants as const


class TestBremsstrahlung(unittest.TestCase):

    def setUp(self):

        self.world = World()

        plasma_species = [(deuterium, 1, 1.e19, 2000., Vector3D(0, 0, 0)),
                          (nitrogen, 7, 1.e18, 2000., Vector3D(0, 0, 0))]
        self.plasma = build_constant_slab_plasma(length=1, width=1, height=1,
                                                 electron_density=1e19,
                                                 electron_temperature=2000.,
                                                 plasma_species=plasma_species)
        self.plasma.parent = self.world
        self.plasma.atomic_data = AtomicData()

    def test_bremsstrahlung_model(self):
        # setting up the model
        gaunt_factor = MaxwellianFreeFreeGauntFactor()
        bremsstrahlung = Bremsstrahlung(gaunt_factor=gaunt_factor)
        self.plasma.models = [bremsstrahlung]

        # observing
        origin = Point3D(1.5, 0, 0)
        direction = Vector3D(-1, 0, 0)
        ray = Ray(origin=origin, direction=direction,
                  min_wavelength=400., max_wavelength=800., bins=128)
        brems_spectrum = ray.trace(self.world)

        # validating
        brems_const = (const.e**2 * 0.25 / np.pi / const.epsilon_0)**3
        brems_const *= 32 * np.pi**2 / (3 * np.sqrt(3) * const.m_e**2 * const.c**3)
        brems_const *= np.sqrt(2 * const.m_e / (np.pi * const.e))
        brems_const *= const.c * 1e9 * 0.25 / np.pi
        exp_factor = const.h * const.c * 1.e9 / const. e

        ne = self.plasma.electron_distribution.density(0.5, 0, 0)
        te = self.plasma.electron_distribution.effective_temperature(0.5, 0, 0)

        def brems_func(wvl):
            ni_gff_z2 = 0
            for species in self.plasma.composition:
                z = species.charge
                ni = self.plasma.composition[(species.element, species.charge)].distribution.density(0.5, 0, 0)
                ni_gff_z2 += ni * gaunt_factor(z, te, wvl) * z * z

            return brems_const * ni_gff_z2 * ne / (np.sqrt(te) * wvl * wvl) * np.exp(- exp_factor / (te * wvl))

        integrator = GaussianQuadrature(brems_func)

        test_samples = np.zeros(brems_spectrum.bins)
        delta_wavelength = (brems_spectrum.max_wavelength - brems_spectrum.min_wavelength) / brems_spectrum.bins
        lower_wavelength = brems_spectrum.min_wavelength
        for i in range(brems_spectrum.bins):
            upper_wavelength = brems_spectrum.min_wavelength + delta_wavelength * (i + 1)
            bin_integral = integrator(lower_wavelength, upper_wavelength)
            test_samples[i] = bin_integral / delta_wavelength
            lower_wavelength = upper_wavelength

        for i in range(brems_spectrum.bins):
            self.assertAlmostEqual(brems_spectrum.samples[i], test_samples[i], delta=1e-10,
                                   msg='BeamCXLine model gives a wrong value at {} nm.'.format(brems_spectrum.wavelengths[i]))


if __name__ == '__main__':
    unittest.main()
