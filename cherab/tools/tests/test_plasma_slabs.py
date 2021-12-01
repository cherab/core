import unittest

import numpy as np

from raysect.core import Vector3D

from cherab.core.atomic.elements import hydrogen, deuterium, neon
from cherab.tools.plasmas.slab import build_constant_slab_plasma


class TestBuildConstantPlasmaSlab(unittest.TestCase):

    slab_length = 3
    slab_width = 1
    slab_height = 2

    margin = 1e-3

    x = np.linspace(margin, slab_length - margin, 5)
    y = np.linspace(-0.5 * slab_width + margin, 0.5 * slab_width - margin, 5)
    z = np.linspace(-0.5 * slab_height + margin, 0.5 * slab_width - margin, 5)

    def test_electrons_only(self):
        """ Tests the case of plasma containing only electrons with constant parameters"""

        electron_density = 5e19
        electron_temperature = 3e3

        # passing empty list as plasma_species should result in plasma containing only enectrons
        plasma = build_constant_slab_plasma(length=self.slab_length, width=self.slab_width, height=self.slab_height,
                                            plasma_species=[], electron_density=electron_density,
                                            electron_temperature=electron_temperature)

        # test if tere are no ions in the plasma
        composition = plasma.composition
        
        self.assertEqual(len(composition), 0, msg="Plasma should contain only electrons.")

        # test electron parameters across the slab

        for ix, vx in enumerate(self.x):
            for iy, vy in enumerate(self.y):
                for iz, vz in enumerate(self.z):
                    self.assertEqual(plasma.electron_distribution.density(vx, vy, vz), electron_density,
                                     msg="Electron density value is incorrect.")
                    self.assertEqual(plasma.electron_distribution.effective_temperature(vx, vy, vz),
                                     electron_temperature, msg="Electron temperature value is incorrect.")

    def test_default_species(self):
        """Tests the default species parameters"""

        electron_density = 5e19
        electron_temperature = 3e3

        plasma = build_constant_slab_plasma(length=self.slab_length, width=self.slab_width, height=self.slab_height,
                                            electron_density=electron_density, electron_temperature=electron_temperature)

        composition = list(plasma.composition)

        self.assertEqual(len(composition), 1, msg="Plasma can contain only a single species.")
        self.assertIs(composition[0].element, hydrogen, msg="Composition element should be hydrogen.")
        self.assertEqual(composition[0].charge, 1, msg="Element charge should be 1.")

        for ix, vx in enumerate(self.x):
            for iy, vy in enumerate(self.y):
                for iz, vz in enumerate(self.z):
                    self.assertEqual(plasma.ion_density(vx, vz, vz), plasma.electron_distribution.density(vx, vy, vz),
                                     msg="Ion and electron density should be equal")
                    self.assertEqual(composition[0].distribution.density(vx, vz, vz), electron_density,
                                     msg="Ion and electron density should be equal")
                    self.assertEqual(composition[0].distribution.effective_temperature(vx, vy, vz),
                                     electron_temperature, msg="Ion and electron temperatur should be equal")

    def test_species_set(self):
        """Test addition of a set of ion/neutral species"""

        electron_density = 5e19
        electron_temperature = 4e3

        species_temperature = [1e2, 2.5e3, 2e3]
        species_density = [1e16, 4e19, 5e18]
        species_elements = [deuterium, deuterium, neon]
        species_charge = [0, 1, 3]
        species_velocity = [Vector3D(3e3, 0, 0), Vector3D(1e4, 0, 0), Vector3D(5e3, 0, 0)]

        species = zip(species_elements, species_charge, species_density, species_temperature, species_velocity)

        plasma = build_constant_slab_plasma(length=self.slab_length, width=self.slab_width, height=self.slab_height,
                                            plasma_species=species, electron_density=electron_density,
                                            electron_temperature=electron_temperature)

        composition = list(plasma.composition)
        for ix, vx in enumerate(self.x):
            for iy, vy in enumerate(self.y):
                for iz, vz in enumerate(self.z):
                    for ie, ve in enumerate(composition):
                        self.assertIs(ve.element, species_elements[ie], msg="Elements don't match input.")
                        self.assertEqual(ve.charge, species_charge[ie], msg="Element Charges don't match input.")
                        self.assertEqual(ve.distribution.density(vx, vy, vz), species_density[ie],
                                         msg="Element densities don't match input.")
                        self.assertEqual(ve.distribution.effective_temperature(vx, vy, vz), species_temperature[ie],
                                         msg="Element temperatures don't match input.")
