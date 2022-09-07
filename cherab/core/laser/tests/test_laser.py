import unittest

from raysect.optical import World
from raysect.optical.material.emitter.inhomogeneous import NumericalIntegrator

from cherab.core import Plasma
from cherab.core.laser.node import Laser
from cherab.core.model.laser.laserspectrum import ConstantSpectrum
from cherab.core.model.laser.model import SeldenMatobaThomsonSpectrum
from cherab.core.model.laser.profile import UniformEnergyDensity


class TestLaser(unittest.TestCase):

    def test_laser_init(self):
        """
        Test correct initialisation of a laser instance.
        """

        world = World()
        laser = Laser(parent=world)

        with self.assertRaises(ValueError, msg="Model was attached before Plasma, Profile and LaserSpectrum were specified."):
            laser.models = [SeldenMatobaThomsonSpectrum()]

        laser.laser_profile = UniformEnergyDensity()
        with self.assertRaises(ValueError, msg="Model was attached before Plasma, Profile and LaserSpectrum were specified."):
            laser.models = [SeldenMatobaThomsonSpectrum()]

        laser.laser_spectrum = ConstantSpectrum(min_wavelength=1059, max_wavelength=1061, bins=10)
        with self.assertRaises(ValueError, msg="Model was attached before Plasma, Profile and LaserSpectrum were specified."):
            laser.models = [SeldenMatobaThomsonSpectrum()]

        laser.plasma = Plasma(parent=world)
        laser.models = [SeldenMatobaThomsonSpectrum()]

    def test_reference_change(self):

        world = World()

        laser_profile = UniformEnergyDensity(laser_length=1, laser_radius=0.1)
        laser_spectrum = ConstantSpectrum(min_wavelength=1059, max_wavelength=1061, bins=10)
        plasma = Plasma(parent=world)
        models = [SeldenMatobaThomsonSpectrum()]

        laser_profile2 = UniformEnergyDensity()
        laser_spectrum2 = ConstantSpectrum(min_wavelength=1059, max_wavelength=1061, bins=10)
        plasma2 = Plasma(parent=world)
        models2 = [SeldenMatobaThomsonSpectrum()]

        laser = Laser(parent=world)

        laser.laser_spectrum = laser_spectrum
        laser.plasma = plasma
        laser.laser_profile = laser_profile
        laser.models = models

        for mod in list(laser.models):
            self.assertIs(mod.laser_profile, laser_profile, msg="laser_profile reference in emission model"
                          "is not set correctly.")
            self.assertIs(mod.plasma, plasma, msg="plasma reference in emission model"
                                                            "is not set correctly.")
            self.assertIs(mod.laser_spectrum, laser_spectrum, msg="laser_spectrum reference in emission model"
                                                            "is not set correctly.")

        laser.laser_spectrum = laser_spectrum2
        laser.plasma = plasma2
        laser.laser_profile = laser_profile2

        for mod in list(laser.models):
            self.assertIs(mod.laser_profile, laser_profile2, msg="laser_profile reference in emission model"
                                                            "is not set correctly.")
            self.assertIs(mod.plasma, plasma2, msg="plasma reference in emission model"
                                                  "is not set correctly.")
            self.assertIs(mod.laser_spectrum, laser_spectrum2, msg="laser_spectrum reference in emission model"
                                                                  "is not set correctly.")

        laser.models = models + models2

        for mod in list(laser.models):
            self.assertIs(mod.laser_profile, laser_profile2, msg="laser_profile reference in emission model"
                                                             "is not set correctly.")
            self.assertIs(mod.plasma, plasma2, msg="plasma reference in emission model"
                                                   "is not set correctly.")
            self.assertIs(mod.laser_spectrum, laser_spectrum2, msg="laser_spectrum reference in emission model"
                                                                   "is not set correctly.")

    def test_integrator_change(self):

        world = World()

        laser_profile = UniformEnergyDensity(laser_length=1, laser_radius=0.1)
        laser_spectrum = ConstantSpectrum(min_wavelength=1059, max_wavelength=1061, bins=10)
        plasma = Plasma(parent=world)
        models = [SeldenMatobaThomsonSpectrum()]

        laser = Laser(parent=world)

        laser.laser_spectrum = laser_spectrum
        laser.plasma = plasma
        laser.laser_profile = laser_profile
        laser.models = models

        integrator = NumericalIntegrator(1e-4)

        laser.integrator = integrator

        for i in laser.get_geometry():
            self.assertIs(i.material.integrator, integrator, msg="Integrator not updated properly")
             
