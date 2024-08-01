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

from cherab.core.atomic import Line, AtomicData, ImpactExcitationPEC, RecombinationPEC, ThermalCXPEC
from cherab.core.atomic import deuterium, carbon
from cherab.tools.plasmas.slab import build_constant_slab_plasma
from cherab.core.model import ExcitationLine, RecombinationLine, ThermalCXLine, GaussianLine, ZeemanTriplet


class ConstantImpactExcitationPEC(ImpactExcitationPEC):
    """
    Constant electron impact excitation PEC for test purpose.
    """

    def __init__(self, value):
        self.value = value

    def evaluate(self, density, temperature):

        return self.value


class ConstantRecombinationPEC(RecombinationPEC):
    """
    Constant recombination PEC for test purpose.
    """

    def __init__(self, value):
        self.value = value

    def evaluate(self, density, temperature):

        return self.value


class ConstantThermalCXPEC(ThermalCXPEC):
    """
    Constant recombination PEC for test purpose.
    """

    def __init__(self, value):
        self.value = value

    def evaluate(self, electron_density, electron_temperature, donor_temperature):

        return self.value


class MockAtomicData(AtomicData):
    """Fake atomic data for test purpose."""

    def impact_excitation_pec(self, ion, charge, transition):

        return ConstantImpactExcitationPEC(1.4e-39)

    def recombination_pec(self, ion, charge, transition):

        return ConstantRecombinationPEC(8.e-41)

    def thermal_cx_pec(self, donor_ion, donor_charge, receiver_ion, receiver_charge, transition):

        return ConstantThermalCXPEC(1.2e-46)

    def wavelength(self, ion, charge, transition):

        return 529.27


class TestExcitationLine(unittest.TestCase):

    def setUp(self):

        self.world = World()

        self.atomic_data = MockAtomicData()

        plasma_species = [(carbon, 5, 2.e18, 800., Vector3D(0, 0, 0))]
        self.slab_length = 1.2
        self.plasma = build_constant_slab_plasma(length=self.slab_length, width=1, height=1,
                                                 electron_density=1e19, electron_temperature=1000.,
                                                 plasma_species=plasma_species, b_field=Vector3D(0, 10., 0))
        self.plasma.atomic_data = self.atomic_data
        self.plasma.parent = self.world

    def test_default_lineshape(self):
        # setting up the model
        line = Line(carbon, 5, (8, 7))
        self.plasma.models = [ExcitationLine(line)]
        wavelength = self.atomic_data.wavelength(line.element, line.charge, line.transition)

        # observing
        origin = Point3D(1.5, 0, 0)
        direction = Vector3D(-1, 0, 0)
        ray = Ray(origin=origin, direction=direction,
                  min_wavelength=wavelength - 1.5, max_wavelength=wavelength + 1.5, bins=512)
        excit_spectrum = ray.trace(self.world)

        # validating
        ne = self.plasma.electron_distribution.density(0.5, 0, 0)  # constant slab
        te = self.plasma.electron_distribution.effective_temperature(0.5, 0, 0)
        rate = self.atomic_data.impact_excitation_pec(line.element, line.charge, line.transition)(ne, te)
        target_species = self.plasma.composition.get(line.element, line.charge)
        ni = target_species.distribution.density(0.5, 0, 0)
        radiance = 0.25 / np.pi * rate * ni * ne * self.slab_length

        gaussian_line = GaussianLine(line, wavelength, target_species, self.plasma, self.atomic_data)
        spectrum = Spectrum(ray.min_wavelength, ray.max_wavelength, ray.bins)
        spectrum = gaussian_line.add_line(radiance, Point3D(0.5, 0, 0), direction, spectrum)

        for i in range(ray.bins):
            self.assertAlmostEqual(excit_spectrum.samples[i], spectrum.samples[i], delta=1e-8,
                                   msg='ExcitationLine model gives a wrong value at {} nm.'.format(spectrum.wavelengths[i]))

    def test_custom_lineshape(self):
        # setting up the model
        line = Line(carbon, 5, (8, 7))
        self.plasma.models = [ExcitationLine(line, lineshape=ZeemanTriplet)]
        wavelength = self.atomic_data.wavelength(line.element, line.charge, line.transition)

        # observing
        origin = Point3D(1.5, 0, 0)
        direction = Vector3D(-1, 0, 0)
        ray = Ray(origin=origin, direction=direction,
                  min_wavelength=wavelength - 1.5, max_wavelength=wavelength + 1.5, bins=512)
        excit_spectrum = ray.trace(self.world)

        # validating
        ne = self.plasma.electron_distribution.density(0.5, 0, 0)  # constant slab
        te = self.plasma.electron_distribution.effective_temperature(0.5, 0, 0)
        rate = self.atomic_data.impact_excitation_pec(line.element, line.charge, line.transition)(ne, te)
        target_species = self.plasma.composition.get(line.element, line.charge)
        ni = target_species.distribution.density(0.5, 0, 0)
        radiance = 0.25 / np.pi * rate * ni * ne * self.slab_length

        zeeman_line = ZeemanTriplet(line, wavelength, target_species, self.plasma, self.atomic_data)
        spectrum = Spectrum(ray.min_wavelength, ray.max_wavelength, ray.bins)
        spectrum = zeeman_line.add_line(radiance, Point3D(0.5, 0, 0), direction, spectrum)

        for i in range(ray.bins):
            self.assertAlmostEqual(excit_spectrum.samples[i], spectrum.samples[i], delta=1e-8,
                                   msg='ExcitationLine model gives a wrong value at {} nm.'.format(spectrum.wavelengths[i]))


class TestRecombinationLine(unittest.TestCase):

    def setUp(self):

        self.world = World()

        self.atomic_data = MockAtomicData()

        plasma_species = [(carbon, 6, 1.67e18, 800., Vector3D(0, 0, 0))]
        self.slab_length = 1.2
        self.plasma = build_constant_slab_plasma(length=self.slab_length, width=1, height=1,
                                                 electron_density=1e19, electron_temperature=1000.,
                                                 plasma_species=plasma_species, b_field=Vector3D(0, 10., 0))
        self.plasma.atomic_data = self.atomic_data
        self.plasma.parent = self.world

    def test_default_lineshape(self):
        # setting up the model
        line = Line(carbon, 5, (8, 7))
        self.plasma.models = [RecombinationLine(line)]
        wavelength = self.atomic_data.wavelength(line.element, line.charge, line.transition)

        # observing
        origin = Point3D(1.5, 0, 0)
        direction = Vector3D(-1, 0, 0)
        ray = Ray(origin=origin, direction=direction,
                  min_wavelength=wavelength - 1.5, max_wavelength=wavelength + 1.5, bins=512)
        recomb_spectrum = ray.trace(self.world)

        # validating
        ne = self.plasma.electron_distribution.density(0.5, 0, 0)  # constant slab
        te = self.plasma.electron_distribution.effective_temperature(0.5, 0, 0)
        rate = self.atomic_data.recombination_pec(line.element, line.charge, line.transition)(ne, te)
        target_species = self.plasma.composition.get(line.element, line.charge + 1)
        ni = target_species.distribution.density(0.5, 0, 0)
        radiance = 0.25 / np.pi * rate * ni * ne * self.slab_length

        gaussian_line = GaussianLine(line, wavelength, target_species, self.plasma, self.atomic_data)
        spectrum = Spectrum(ray.min_wavelength, ray.max_wavelength, ray.bins)
        spectrum = gaussian_line.add_line(radiance, Point3D(0.5, 0, 0), direction, spectrum)

        for i in range(ray.bins):
            self.assertAlmostEqual(recomb_spectrum.samples[i], spectrum.samples[i], delta=1e-8,
                                   msg='RecombinationLine model gives a wrong value at {} nm.'.format(spectrum.wavelengths[i]))

    def test_custom_lineshape(self):
        # setting up the model
        line = Line(carbon, 5, (8, 7))
        self.plasma.models = [RecombinationLine(line, lineshape=ZeemanTriplet)]
        wavelength = self.atomic_data.wavelength(line.element, line.charge, line.transition)

        # observing
        origin = Point3D(1.5, 0, 0)
        direction = Vector3D(-1, 0, 0)
        ray = Ray(origin=origin, direction=direction,
                  min_wavelength=wavelength - 1.5, max_wavelength=wavelength + 1.5, bins=512)
        recomb_spectrum = ray.trace(self.world)

        # validating
        ne = self.plasma.electron_distribution.density(0.5, 0, 0)  # constant slab
        te = self.plasma.electron_distribution.effective_temperature(0.5, 0, 0)
        rate = self.atomic_data.recombination_pec(line.element, line.charge, line.transition)(ne, te)
        target_species = self.plasma.composition.get(line.element, line.charge + 1)
        ni = target_species.distribution.density(0.5, 0, 0)
        radiance = 0.25 / np.pi * rate * ni * ne * self.slab_length

        zeeman_line = ZeemanTriplet(line, wavelength, target_species, self.plasma, self.atomic_data)
        spectrum = Spectrum(ray.min_wavelength, ray.max_wavelength, ray.bins)
        spectrum = zeeman_line.add_line(radiance, Point3D(0.5, 0, 0), direction, spectrum)

        for i in range(ray.bins):
            self.assertAlmostEqual(recomb_spectrum.samples[i], spectrum.samples[i], delta=1e-8,
                                   msg='RecombinationLine model gives a wrong value at {} nm.'.format(spectrum.wavelengths[i]))


class TestThermalCXLine(unittest.TestCase):

    def setUp(self):

        self.world = World()

        self.atomic_data = MockAtomicData()

        plasma_species = [(carbon, 6, 1.67e18, 800., Vector3D(0, 0, 0)),
                          (deuterium, 0, 1.e19, 100., Vector3D(0, 0, 0))]
        self.slab_length = 1.2
        self.plasma = build_constant_slab_plasma(length=self.slab_length, width=1, height=1,
                                                 electron_density=1e19, electron_temperature=1000.,
                                                 plasma_species=plasma_species, b_field=Vector3D(0, 10., 0))
        self.plasma.atomic_data = self.atomic_data
        self.plasma.parent = self.world

    def test_default_lineshape(self):
        # setting up the model
        line = Line(carbon, 5, (8, 7))
        self.plasma.models = [ThermalCXLine(line)]
        wavelength = self.atomic_data.wavelength(line.element, line.charge, line.transition)

        # observing
        origin = Point3D(1.5, 0, 0)
        direction = Vector3D(-1, 0, 0)
        ray = Ray(origin=origin, direction=direction,
                  min_wavelength=wavelength - 1.5, max_wavelength=wavelength + 1.5, bins=512)
        thermalcx_spectrum = ray.trace(self.world)

        # validating
        ne = self.plasma.electron_distribution.density(0.5, 0, 0)  # constant slab
        te = self.plasma.electron_distribution.effective_temperature(0.5, 0, 0)
        donor_species = self.plasma.composition.get(deuterium, 0)
        donor_density = donor_species.distribution.density(0.5, 0, 0)
        donor_temperature = donor_species.distribution.effective_temperature(0.5, 0, 0)
        rate = self.atomic_data.thermal_cx_pec(deuterium, 0, line.element, line.charge, line.transition)(ne, te, donor_temperature)
        target_species = self.plasma.composition.get(line.element, line.charge + 1)
        receiver_density = target_species.distribution.density(0.5, 0, 0)
        radiance = 0.25 / np.pi * rate * receiver_density * donor_density * self.slab_length

        gaussian_line = GaussianLine(line, wavelength, target_species, self.plasma, self.atomic_data)
        spectrum = Spectrum(ray.min_wavelength, ray.max_wavelength, ray.bins)
        spectrum = gaussian_line.add_line(radiance, Point3D(0.5, 0, 0), direction, spectrum)

        for i in range(ray.bins):
            self.assertAlmostEqual(thermalcx_spectrum.samples[i], spectrum.samples[i], delta=1e-8,
                                   msg='ThermalCXLine model gives a wrong value at {} nm.'.format(spectrum.wavelengths[i]))

    def test_custom_lineshape(self):
        # setting up the model
        line = Line(carbon, 5, (8, 7))
        self.plasma.models = [ThermalCXLine(line, lineshape=ZeemanTriplet)]
        wavelength = self.atomic_data.wavelength(line.element, line.charge, line.transition)

        # observing
        origin = Point3D(1.5, 0, 0)
        direction = Vector3D(-1, 0, 0)
        ray = Ray(origin=origin, direction=direction,
                  min_wavelength=wavelength - 1.5, max_wavelength=wavelength + 1.5, bins=512)
        thermalcx_spectrum = ray.trace(self.world)

        # validating
        ne = self.plasma.electron_distribution.density(0.5, 0, 0)  # constant slab
        te = self.plasma.electron_distribution.effective_temperature(0.5, 0, 0)
        donor_species = self.plasma.composition.get(deuterium, 0)
        donor_density = donor_species.distribution.density(0.5, 0, 0)
        donor_temperature = donor_species.distribution.effective_temperature(0.5, 0, 0)
        rate = self.atomic_data.thermal_cx_pec(deuterium, 0, line.element, line.charge, line.transition)(ne, te, donor_temperature)
        target_species = self.plasma.composition.get(line.element, line.charge + 1)
        receiver_density = target_species.distribution.density(0.5, 0, 0)
        radiance = 0.25 / np.pi * rate * receiver_density * donor_density * self.slab_length

        zeeman_line = ZeemanTriplet(line, wavelength, target_species, self.plasma, self.atomic_data)
        spectrum = Spectrum(ray.min_wavelength, ray.max_wavelength, ray.bins)
        spectrum = zeeman_line.add_line(radiance, Point3D(0.5, 0, 0), direction, spectrum)

        for i in range(ray.bins):
            self.assertAlmostEqual(thermalcx_spectrum.samples[i], spectrum.samples[i], delta=1e-8,
                                   msg='ThermalCXLine model gives a wrong value at {} nm.'.format(spectrum.wavelengths[i]))


if __name__ == '__main__':
    unittest.main()
