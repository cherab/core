
import matplotlib.pyplot as plt
from scipy.constants import electron_mass, atomic_mass

from raysect.core import translate, rotate_basis, Point3D, Vector3D
from raysect.primitive import Box
from raysect.optical import World, Ray

from cherab.core import Plasma, Beam, Maxwellian, Species
from cherab.core.math import ScalarToVectorFunction3D
from cherab.core.atomic import hydrogen, deuterium, carbon, Line
from cherab.core.model import SingleRayAttenuator, BeamCXLine, ThermalCXLine
from cherab.tools.plasmas.slab import NeutralFunction, IonFunction
from cherab.openadas import OpenADAS
from cherab.openadas.install import install_adf15


###############
# Make Plasma #

width = 1.0
length = 1.0
height = 3.0
peak_density = 5e19
edge_density = 1e18
pedestal_top = 1
neutral_temperature = 0.5
peak_temperature = 2500
edge_temperature = 25
impurities = [(carbon, 6, 0.005)]

world = World()
adas = OpenADAS(permit_extrapolation=True, missing_rates_return_null=True)

plasma = Plasma(parent=world)
plasma.atomic_data = adas

# plasma slab along x direction
plasma.geometry = Box(Point3D(0, -width / 2, -height / 2), Point3D(length, width / 2, height / 2))

species = []

# make a non-zero velocity profile for the plasma
vy_profile = IonFunction(1E5, 0, pedestal_top=pedestal_top)
velocity_profile = ScalarToVectorFunction3D(0, vy_profile, 0)

# define neutral species distribution
h0_density = NeutralFunction(peak_density, 0.1, pedestal_top=pedestal_top)
h0_temperature = neutral_temperature
h0_distribution = Maxwellian(h0_density, h0_temperature, velocity_profile,
                             hydrogen.atomic_weight * atomic_mass)
species.append(Species(hydrogen, 0, h0_distribution))

# define hydrogen ion species distribution
h1_density = IonFunction(peak_density, edge_density, pedestal_top=pedestal_top)
h1_temperature = IonFunction(peak_temperature, edge_temperature, pedestal_top=pedestal_top)
h1_distribution = Maxwellian(h1_density, h1_temperature, velocity_profile,
                             hydrogen.atomic_weight * atomic_mass)
species.append(Species(hydrogen, 1, h1_distribution))

# add impurities
if impurities:
    for impurity, ionisation, concentration in impurities:
        imp_density = IonFunction(peak_density * concentration, edge_density * concentration, pedestal_top=pedestal_top)
        imp_temperature = IonFunction(peak_temperature, edge_temperature, pedestal_top=pedestal_top)
        imp_distribution = Maxwellian(imp_density, imp_temperature, velocity_profile,
                                      impurity.atomic_weight * atomic_mass)
        species.append(Species(impurity, ionisation, imp_distribution))

# define the electron distribution
e_density = IonFunction(peak_density, edge_density, pedestal_top=pedestal_top)
e_temperature = IonFunction(peak_temperature, edge_temperature, pedestal_top=pedestal_top)
e_distribution = Maxwellian(e_density, e_temperature, velocity_profile, electron_mass)

# define species
plasma.electron_distribution = e_distribution
plasma.composition = species

# add thermal CX emission model
cVI_5_4 = Line(carbon, 5, (5, 4))
plasma.models = [ThermalCXLine(cVI_5_4)]

# trace thermal CX spectrum along y direction
ray = Ray(origin=Point3D(0.4, -3.5, 0), direction=Vector3D(0, 1, 0),
          min_wavelength=112.3, max_wavelength=112.7, bins=512)
thermal_cx_spectrum = ray.trace(world)

###########################
# Inject beam into plasma #

integration_step = 0.0025
# injected along x direction
beam_transform = translate(-0.5, 0.0, 0) * rotate_basis(Vector3D(1, 0, 0), Vector3D(0, 0, 1))
beam_energy = 50000  # eV

beam = Beam(parent=world, transform=beam_transform)
beam.plasma = plasma
beam.atomic_data = adas
beam.energy = beam_energy
beam.power = 3e6
beam.element = deuterium
beam.sigma = 0.05
beam.divergence_x = 0.5
beam.divergence_y = 0.5
beam.length = 3.0
beam.attenuator = SingleRayAttenuator(clamp_to_zero=True)
beam.integrator.step = integration_step
beam.integrator.min_samples = 10

# remove thermal CX model and add beam CX model
plasma.models = []
beam.models = [BeamCXLine(cVI_5_4)]

# trace the spectrum again
beam_cx_spectrum = ray.trace(world)

# plot the spectra
plt.figure()
plt.plot(thermal_cx_spectrum.wavelengths, thermal_cx_spectrum.samples, label='thermal CX')
plt.plot(beam_cx_spectrum.wavelengths, beam_cx_spectrum.samples, label='beam CX')
plt.legend()
plt.xlabel('Wavelength (nm)')
plt.ylabel('Radiance (W/m^2/str/nm)')
plt.title('Sampled CX spectra')

plt.show()
