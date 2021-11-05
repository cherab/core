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

import sys

import numpy as np
from scipy.constants import electron_mass, atomic_mass
from matplotlib.pyplot import ion, ioff, plot, show

from raysect.optical.library import *
from raysect.primitive import Sphere, Box, Intersect
from raysect.optical import Ray, d65_white, World, Point3D, Vector3D, translate, rotate
from raysect.optical.observer import PinholeCamera
from raysect.optical.material.emitter import Checkerboard

from cherab.core import Plasma, Beam, Species, Maxwellian
from cherab.core.atomic import elements, Line
from cherab.openadas import OpenADAS
from cherab.core.model import SingleRayAttenuator, BeamCXLine
from cherab.tools.plasmas import GaussianVolume

integration_step = 0.1

# setup scenegraph
world = World()

# create atomic data source
adas = OpenADAS(permit_extrapolation=True)

# PLASMA ----------------------------------------------------------------------
plasma = Plasma(parent=world)

# define basic distributions
ion_density = 9e19
sigma = 0.25

d_density = GaussianVolume(0.94 * ion_density, sigma)
he2_density = GaussianVolume(0.04 * ion_density, sigma)
c6_density = GaussianVolume(0.01 * ion_density, sigma)
ne10_density = GaussianVolume(0.01 * ion_density, sigma)
e_density = GaussianVolume((0.94 + 0.04*2 + 0.01*6 + 0.01*10) * ion_density, sigma)
temperature = 10 + GaussianVolume(240, sigma)
# temperature = 1000 + GaussianVolume(4000, sigma)
bulk_velocity = Vector3D(200e3, 0, 0)

d_distribution = Maxwellian(d_density, temperature, bulk_velocity, elements.deuterium.atomic_weight * atomic_mass)
he2_distribution = Maxwellian(he2_density, temperature, bulk_velocity, elements.helium.atomic_weight * atomic_mass)
c6_distribution = Maxwellian(c6_density, temperature, bulk_velocity, elements.carbon.atomic_weight * atomic_mass)
ne10_distribution = Maxwellian(ne10_density, temperature, bulk_velocity, elements.neon.atomic_weight * atomic_mass)
e_distribution = Maxwellian(e_density, temperature, bulk_velocity, electron_mass)

d_species = Species(elements.deuterium, 1, d_distribution)
he2_species = Species(elements.helium, 2, he2_distribution)
c6_species = Species(elements.carbon, 6, c6_distribution)
ne10_species = Species(elements.neon, 10, ne10_distribution)

# define species
plasma.b_field = Vector3D(1.0, 1.0, 1.0)
plasma.electron_distribution = e_distribution
plasma.composition = [d_species, he2_species, c6_species, ne10_species]

print("plasma.z_effective(): ", plasma.z_effective(0, 0, 0))


from cherab.core.model import ExcitationLine, RecombinationLine

plasma.geometry = Sphere(sigma * 5.0)
plasma.geometry_transform = None
plasma.integrator.step = integration_step
plasma.integrator.min_samples = 5
plasma.atomic_data = adas

# Setup elements.deuterium lines
d_alpha = Line(elements.deuterium, 0, (3, 2))
d_beta = Line(elements.deuterium, 0, (4, 2))
d_gamma = Line(elements.deuterium, 0, (5, 2))
d_delta = Line(elements.deuterium, 0, (6, 2))
d_epsilon = Line(elements.deuterium, 0, (7, 2))

plasma.models = [
    # Bremsstrahlung(),
    RecombinationLine(d_alpha),
    RecombinationLine(d_beta),
    RecombinationLine(d_gamma),
    RecombinationLine(d_delta),
    RecombinationLine(d_epsilon)
]


# BEAM ------------------------------------------------------------------------
beam = Beam(parent=world, transform=translate(1.0, 0.0, 0) * rotate(90, 0, 0))
beam.plasma = plasma
beam.atomic_data = adas
beam.energy = 60000
beam.power = 1e4
beam.element = elements.deuterium
beam.sigma = 0.025
beam.divergence_x = 0.5
beam.divergence_y = 0.5
beam.length = 3.0
beam.attenuator = SingleRayAttenuator(clamp_to_zero=True)
beam.models = [
    BeamCXLine(Line(elements.helium, 1, (4, 3))),
    BeamCXLine(Line(elements.helium, 1, (6, 4))),
    BeamCXLine(Line(elements.carbon, 5, (8, 7))),
    BeamCXLine(Line(elements.carbon, 5, (9, 8))),
    BeamCXLine(Line(elements.carbon, 5, (10, 8))),
    BeamCXLine(Line(elements.neon, 9, (11, 10))),
    BeamCXLine(Line(elements.neon, 9, (12, 11))),
]
beam.integrator.step = integration_step
beam.integrator.min_samples = 5

beam = Beam(parent=world, transform=translate(1.0, 0.0, 0) * rotate(90, 0, 0))
beam.plasma = plasma
beam.atomic_data = adas
beam.energy = 60000 / 2
beam.power = 1e4
beam.element = elements.deuterium
beam.sigma = 0.025
beam.divergence_x = 0.5
beam.divergence_y = 0.5
beam.length = 3.0
beam.attenuator = SingleRayAttenuator(clamp_to_zero=True)
beam.models = [
    BeamCXLine(Line(elements.helium, 1, (4, 3))),
    BeamCXLine(Line(elements.helium, 1, (6, 4))),
    BeamCXLine(Line(elements.carbon, 5, (8, 7))),
    BeamCXLine(Line(elements.carbon, 5, (9, 8))),
    BeamCXLine(Line(elements.carbon, 5, (10, 8))),
    BeamCXLine(Line(elements.neon, 9, (11, 10))),
    BeamCXLine(Line(elements.neon, 9, (12, 11))),
]
beam.integrator.step = integration_step
beam.integrator.min_samples = 5

beam = Beam(parent=world, transform=translate(1.0, 0.0, 0) * rotate(90, 0, 0))
beam.plasma = plasma
beam.atomic_data = adas
beam.energy = 60000 / 3
beam.power = 1e4
beam.element = elements.deuterium
beam.sigma = 0.025
beam.divergence_x = 0.5
beam.divergence_y = 0.5
beam.length = 3.0
beam.attenuator = SingleRayAttenuator(clamp_to_zero=True)
beam.models = [
    BeamCXLine(Line(elements.helium, 1, (4, 3))),
    BeamCXLine(Line(elements.helium, 1, (6, 4))),
    BeamCXLine(Line(elements.carbon, 5, (8, 7))),
    BeamCXLine(Line(elements.carbon, 5, (9, 8))),
    BeamCXLine(Line(elements.carbon, 5, (10, 8))),
    BeamCXLine(Line(elements.neon, 9, (11, 10))),
    BeamCXLine(Line(elements.neon, 9, (12, 11))),
]
beam.integrator.step = integration_step
beam.integrator.min_samples = 5

# LENS ------------------------------------------------------------------------

# s1 = Sphere(1.0, transform=translate(0, 0, 1.0-0.01))
# s2 = Sphere(0.5, transform=translate(0, 0, -0.5+0.01))
# Intersect(s1, s2, world, translate(0, 0, -1.6)*rotate(-30,30,0), schott("N-BK7"))

# BACKGROUND ------------------------------------------------------------------

# Box(Point3D(-50, -50, 50), Point3D(50, 50, 50.1), world, material=Checkerboard(4, d65_white, d65_white, 0.001, 0.002))
# Box(Point(-100, -100, -100), Point(100, 100, 100), world, material=UniformSurfaceEmitter(d65_white, 0.001))

# OBSERVER --------------------------------------------------------------------

#import cProfile
#
# def profile_test(n=25000):
#     r = Ray(origin=Point(0.0, 0, 0), min_wavelength=526, max_wavelength=532, num_samples=100)
#     for i in range(n):
#         r.trace(world)
#
# cProfile.run("profile_test()", sort="tottime")

ion()

r = Ray(origin=Point3D(0.5, 0, -2.5), min_wavelength=440, max_wavelength=740, bins=800)
s = r.trace(world)
plot(s.wavelengths, s.samples)

r = Ray(origin=Point3D(0.5, 0, -2.5), min_wavelength=440, max_wavelength=740, bins=3200)
s = r.trace(world)
plot(s.wavelengths, s.samples)
show()

camera = PinholeCamera((128, 128), parent=world, transform=translate(0, 0, -2.5))
camera.spectral_rays = 1
camera.spectral_bins = 15
camera.pixel_samples = 10

ion()
camera.observe()

ioff()
camera.pipelines[0].display()
