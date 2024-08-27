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

from raysect.core import Point3D, Vector3D, translate
from raysect.optical import World, Spectrum, Ray

from cherab.core import Beam
from cherab.core.atomic import Line, AtomicData, BeamCXPEC, BeamStoppingRate
from cherab.core.atomic import deuterium
from cherab.tools.plasmas.slab import build_constant_slab_plasma
from cherab.core.model import SingleRayAttenuator, BeamCXLine, GaussianLine, ZeemanTriplet


class ConstantBeamCXPEC(BeamCXPEC):
    """
    Constant beam CX PEC for test purpose.
    """

    def __init__(self, donor_metastable, value):
        super().__init__(donor_metastable)
        self.value = value

    def evaluate(self, energy, temperature, density, z_effective, b_field):

        return self.value


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

    def beam_cx_pec(self, donor_ion, receiver_ion, receiver_charge, transition):

        return [ConstantBeamCXPEC(1, 3.4e-34)]

    def beam_stopping_rate(self, beam_ion, plasma_ion, charge):

        return ConstantBeamStoppingRate(1, 0)

    def wavelength(self, ion, charge, transition):

        return 656.104


class TestBeamCXLine(unittest.TestCase):

    def setUp(self):

        self.world = World()

        self.atomic_data = MockAtomicData()

        plasma_species = [(deuterium, 1, 1.e19, 200., Vector3D(0, 0, 0))]
        plasma = build_constant_slab_plasma(length=1, width=1, height=1,
                                            electron_density=1e19,
                                            electron_temperature=200.,
                                            plasma_species=plasma_species,
                                            b_field=Vector3D(0, 10., 0))
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

        self.plasma = plasma
        self.beam = beam

    def test_default_lineshape(self):
        # setting up the model
        line = Line(deuterium, 0, (3, 2))  # D-alpha line
        self.beam.models = [BeamCXLine(line)]

        # observing
        origin = Point3D(1.5, 0, 0)
        direction = Vector3D(-1, 0, 0)
        ray = Ray(origin=origin, direction=direction,
                  min_wavelength=655.1, max_wavelength=657.1, bins=512)
        cx_spectrum = ray.trace(self.world)

        # validating
        dx = self.beam.integrator.step
        rate = self.atomic_data.beam_cx_pec(deuterium, deuterium, 1, (3, 2))[0].value
        ni = self.plasma.ion_density(0.5, 0, 0)  # constant slab
        nd_beam = 0  # beam density
        for i in range(-int(0.5 / dx), int(0.5 / dx)):
            x = dx * (i + 0.5)
            nd_beam += self.beam.density(x, 0, 0)  # in internal beam coordinates
        radiance = 0.25 * rate * ni * nd_beam * dx / np.pi

        target_species = self.plasma.composition.get(line.element, line.charge + 1)
        wavelength = self.atomic_data.wavelength(line.element, line.charge, line.transition)
        gaussian_line = GaussianLine(line, wavelength, target_species, self.plasma, self.atomic_data)
        spectrum = Spectrum(ray.min_wavelength, ray.max_wavelength, ray.bins)
        spectrum = gaussian_line.add_line(radiance, Point3D(0.5, 0, 0), direction, spectrum)

        for i in range(ray.bins):
            self.assertAlmostEqual(cx_spectrum.samples[i], spectrum.samples[i], delta=1e-8,
                                   msg='BeamCXLine model gives a wrong value at {} nm.'.format(spectrum.wavelengths[i]))

    def test_custom_lineshape(self):
        # setting up the model
        line = Line(deuterium, 0, (3, 2))  # D-alpha line
        self.beam.models = [BeamCXLine(line, lineshape=ZeemanTriplet)]

        # observing
        origin = Point3D(1.5, 0, 0)
        direction = Vector3D(-1, 0, 0)
        ray = Ray(origin=origin, direction=direction,
                  min_wavelength=655.1, max_wavelength=657.1, bins=512)
        cx_spectrum = ray.trace(self.world)

        # validating
        dx = self.beam.integrator.step
        rate = self.atomic_data.beam_cx_pec(deuterium, deuterium, 1, (3, 2))[0].value
        ni = self.plasma.ion_density(0.5, 0, 0)  # constant slab
        nd_beam = 0  # beam density
        for i in range(-int(0.5 / dx), int(0.5 / dx)):
            x = dx * (i + 0.5)
            nd_beam += self.beam.density(x, 0, 0)  # in internal beam coordinates
        radiance = 0.25 * rate * ni * nd_beam * dx / np.pi

        target_species = self.plasma.composition.get(line.element, line.charge + 1)
        wavelength = self.atomic_data.wavelength(line.element, line.charge, line.transition)
        zeeman_line = ZeemanTriplet(line, wavelength, target_species, self.plasma, self.atomic_data)
        spectrum = Spectrum(ray.min_wavelength, ray.max_wavelength, ray.bins)
        spectrum = zeeman_line.add_line(radiance, Point3D(0.5, 0, 0), direction, spectrum)

        for i in range(ray.bins):
            self.assertAlmostEqual(cx_spectrum.samples[i], spectrum.samples[i], delta=1e-8,
                                   msg='BeamCXLine model gives a wrong value at {} nm.'.format(spectrum.wavelengths[i]))


if __name__ == '__main__':
    unittest.main()
