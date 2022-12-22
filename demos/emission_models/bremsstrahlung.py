# Copyright 2016-2022 Euratom
# Copyright 2016-2022 United Kingdom Atomic Energy Authority
# Copyright 2016-2022 Centro de Investigaciones Energéticas, Medioambientales y Tecnológicas
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


# External imports
import matplotlib.pyplot as plt
from scipy.constants import electron_mass, atomic_mass
from raysect.optical import World, Vector3D, Point3D, Ray
from raysect.primitive import Sphere
from raysect.optical.material.emitter.inhomogeneous import NumericalIntegrator

# Cherab imports
from cherab.core import Species, Maxwellian, Plasma
from cherab.core.atomic.elements import deuterium, nitrogen
from cherab.core.model import Bremsstrahlung
from cherab.openadas import OpenADAS
from cherab.tools.plasmas import GaussianVolume


# tunables
ion_density = 1e20
sigma = 1.

# setup scenegraph
world = World()

# create atomic data source
adas = OpenADAS(permit_extrapolation=True)

# PLASMA ----------------------------------------------------------------------
plasma = Plasma(parent=world)
plasma.atomic_data = adas
plasma.geometry = Sphere(sigma)
plasma.geometry_transform = None
plasma.integrator = NumericalIntegrator(step=0.01 * sigma)

# define basic distributions
d_density = GaussianVolume(ion_density, sigma)
n_density = d_density * 0.01
e_density = GaussianVolume(ion_density, sigma)
temperature = GaussianVolume(1000, sigma)
bulk_velocity = Vector3D(0, 0, 0)

deuterium_mass = deuterium.atomic_weight * atomic_mass
d_distribution = Maxwellian(d_density, temperature, bulk_velocity, deuterium_mass)
nitrogen_mass = nitrogen.atomic_weight * atomic_mass
n_distribution = Maxwellian(n_density, temperature, bulk_velocity, nitrogen_mass)
e_distribution = Maxwellian(e_density, temperature, bulk_velocity, electron_mass)

d1_species = Species(deuterium, 1, d_distribution)
n1_species = Species(nitrogen, 1, n_distribution)

# define species
plasma.b_field = Vector3D(1.0, 1.0, 1.0)
plasma.electron_distribution = e_distribution
plasma.composition = [d1_species, n1_species]

# add Bremsstrahlung to the plasma
plasma.models = [Bremsstrahlung()]

# Ray-trace and plot the results
r = Ray(origin=Point3D(0, 0, -5), direction=Vector3D(0, 0, 1),
        min_wavelength=380, max_wavelength=800, bins=256)
s = r.trace(world)
plt.plot(s.wavelengths, s.samples)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Radiance (W/m^2/str/nm)')
plt.title('Observed Bremsstrahlung spectrum')
plt.show()
