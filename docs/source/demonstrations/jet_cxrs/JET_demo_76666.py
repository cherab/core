

# Demo for pulse 79666 at t=61s
PULSE = 79666
TIME = 61.

# External imports
from math import isnan

import numpy as np
from cherab_contrib.jet.corentin.data_source import DataSource
from cherab_contrib.jet.corentin.neutralbeam import PINIParameters
from cherab_contrib.jet.corentin.ppf_access import *
from cherab_contrib.jet.corentin.scenegraph import *
from core import Species
# Internal imports
from core.distribution import Maxwellian
from core.math.mapping.interpolators.interpolators1d import Interpolate1DCubic
from core.math.mapping.mappers import IsoMapper3D
from core.model.beam.singleray import SingleRayAttenuator
from core.model.cxs.beaminteraction import CXSBeamPlasmaIntersection
from matplotlib.pyplot import plot, show
from scipy.constants import electron_mass, atomic_mass
from scipy.integrate import cumtrapz

from cherab.core.atomic import ADAS
from cherab.core.atomic import elements, Line
from raysect.core import Vector3D

# ########################### SCENEGRAPH CREATION ########################### #
print('Scenegraph creation')

atomic_data = ADAS(permit_extrapolation=True)

# C1 = Cylinder(4, 4, transform=translate(0, 0, -2.0)*rotate(0, 0, 0))
# C2 = Cylinder(1.5, 4.2, transform=translate(0, 0, -2.1)*rotate(0, 0, 0))
# Subtract(C1, C2, world, transform=translate(0, 0, 0)*rotate(0, 0, 0),
#          material=DAlphaPlasma(PULSE, time=TIME, power=0.08, sigma=0.1, step=0.1), name="Plasma primitive")

attenuation_instruction = (SingleRayAttenuator, {'step': 0.01})
emission_instructions = [(CXSBeamPlasmaIntersection, {'line': Line(elements.carbon, 5, (8, 7)),  'step': 0.01}),
                         # (CXSBeamPlasmaIntersection, {'line': Line(elements.carbon, 5, (10, 8)), 'step': 0.01})
                        ]

world, plasma, components = build_jet_scenegraph(PULSE,
                                             atomic_data,
                                             attenuation_instruction,
                                             emission_instructions)

octant8 = components['nib8']
ks5_oct1 = components['ks5_oct1']
ks7_oct8 = components['ks7_oct8']

ks5_oct1.min_wavelength = 526
ks5_oct1.max_wavelength = 532

# print_scenegraph(world)

# ########################### PLASMA CONFIGURATION ########################## #
print('Plasma configuration')
# /!\ Plasma configuration is from pulse 79503!
PULSE_PLASMA = 79503

src = DataSource()
src.time = TIME
src.n_pulse = PULSE_PLASMA
psi = src.get_psi_normalised(cached2d=True)
inside = lambda x, y, z: not isnan(psi(x, y, z))

ppfsetdevice("JET")
ppfuid('cgiroud', rw='R')
ppfgo(pulse=PULSE_PLASMA, seq=0)

psi_coord = np.array(ppfget(PULSE_PLASMA, 'PRFL', 'C6')[3], dtype=np.float64)
mask = psi_coord <= 1.0
psi_coord = psi_coord[mask]

flow_velocity_tor_data = np.array(ppfget(PULSE_PLASMA, 'PRFL', 'VT')[2], dtype=np.float64)[mask]
flow_velocity_tor_psi = Interpolate1DCubic(psi_coord, flow_velocity_tor_data)
flow_velocity_tor = IsoMapper3D(psi, flow_velocity_tor_psi)
flow_velocity = lambda x, y, z: Vector3D(y * flow_velocity_tor(x, y, z), - x * flow_velocity_tor(x, y, z), 0.) / np.sqrt(x*x + y*y)

ion_temperature_data = np.array(ppfget(PULSE_PLASMA, 'PRFL', 'TI')[2], dtype=np.float64)[mask]
print("Ti between {} and {} eV".format(ion_temperature_data.min(), ion_temperature_data.max()))
ion_temperature_psi = Interpolate1DCubic(psi_coord, ion_temperature_data)
ion_temperature = IsoMapper3D(psi, ion_temperature_psi)

electron_density_data = np.array(ppfget(PULSE_PLASMA, 'PRFL', 'NE')[2], dtype=np.float64)[mask]
print("Ne between {} and {} m-3".format(electron_density_data.min(), electron_density_data.max()))
electron_density_psi = Interpolate1DCubic(psi_coord, electron_density_data)
electron_density = IsoMapper3D(psi, electron_density_psi)

density_c6_data = np.array(ppfget(PULSE_PLASMA, 'PRFL', 'C6')[2], dtype=np.float64)[mask]
density_c6_psi = Interpolate1DCubic(psi_coord, density_c6_data)
density_c6 = IsoMapper3D(psi, density_c6_psi)

density_d = lambda x, y, z: electron_density(x, y, z) - 6 * density_c6(x, y, z)

d_distribution = Maxwellian(density_d, ion_temperature, flow_velocity, elements.deuterium.atomic_weight * atomic_mass)
c6_distribution = Maxwellian(density_c6, ion_temperature, flow_velocity, elements.carbon.atomic_weight * atomic_mass)
e_distribution = Maxwellian(electron_density, ion_temperature, flow_velocity, electron_mass)

d_species = Species(elements.deuterium, 1, d_distribution)
c6_species = Species(elements.carbon, 6, c6_distribution)

plasma.b_field = lambda x, y, z: Vector3D(y, -x, 0.).normalise()
plasma.inside = inside
plasma.electron_distribution = e_distribution
plasma.set_species([d_species, c6_species])

# ########################### PINIS CONFIGURATION ########################### #
# (no information on PINI 6, assumed off)
print('PINIs configuration')

# energy must not be 0 if the beam is turned on!
# energy in eV/amu:
energy = np.array([109717./2, 109717./2, 100000./2, 100000./2, 109863./2, 109179./2, 99267.3/2, 99657.9/2], dtype=np.float64)
# powers in W:
powers = np.array([[1.17340e+06,  1.19512e+06, 0., 0.,  1.08606e+06,  1.00185e+06,  1.02248e+06,  1.00373e+06],  # main component
                   [237300.,      241693.,     0., 0.,  219749.,      202222.,      196378.,      193255.],  # half component
                   [175050.,      178290.,     0., 0.,  161481.,      151293.,      189440.,      184596.]],  # third component
                  dtype=np.float64)
turned_on = np.array([True, True, False, False, True, True, False, False], dtype=bool)

pini_parameters = PINIParameters(energy, powers, turned_on, elements.deuterium)
octant8.pini_parameters = pini_parameters

# ############################### OBSERVATION ############################### #
print('Observation')

ks5_oct1.observe()
ks5_oct1.display()
ks5_oct1.get_los_group('D Lines').display()
ks5_oct1.get_los_group('D Lines').get_los(5).display()


# ion()
# # camera = PinholeCamera(fov=90, parent=world, transform=rotate(0,0,0)*translate(0,0,-5)) # below
# # camera = PinholeCamera(scale=500, fov=90, parent=world, transform=translate(0,0,0)*rotate(0,180,180)*translate(0,0,-6)) # above
# camera = PinholeCamera(fov=90, parent=world, transform=translate(0,-2,0)*rotate(0,135,180)*translate(0,0,-4)) # plongee
# # camera = PinholeCamera(scale=5000000, fov=90, parent=world, transform=translate(0,0,0)*rotate(0,90,180)*translate(0,0,-6)) # horizon
#
# # camera = OrthographicCamera(width=10, parent=world, transform=translate(2,-2.5,0)*rotate(0,180,180)*translate(0,0,-2)) # above
#
# camera.ray_max_depth = 20
# camera.rays = 1
# camera.spectral_samples = 20
# camera.pixels = (200, 200)
# camera.display_progress = True
# camera.display_update_time = 10
# camera.observe()
#
# ioff()
# camera.save("render.png")
# camera.display()
# show()


nb_tracks = len(ks5_oct1.spectra['D Lines'])

# Calculated data
d_lines_c = ks5_oct1.spectra['D Lines']
d_samples_c = np.zeros((nb_tracks, len(d_lines_c[0])), dtype=np.float64)
d_wavelengths_all_c = np.array(d_lines_c[0].wavelengths, dtype=np.float64)

for i in range(len(d_lines_c)):
    # calculated radiance in J/m^2/str/s/nm
    d_samples_c[i, :] = d_lines_c[i].samples

    # turn it into ph/m^2/str/s/nm
    for j in range(len(d_wavelengths_all_c)):
        d_samples_c[i, j] = PhotonToJ.inv(d_samples_c[i, j], d_wavelengths_all_c[j])

plot(d_wavelengths_all_c, d_samples_c[4, :])
show()

# calculated intensity in ph/m^2/str/s
d_intensity_c = cumtrapz(d_samples_c, d_wavelengths_all_c, axis=-1)[:, -1]

d_wavelengths_c = d_wavelengths_all_c[d_samples_c.argmax(axis=1)]
print(d_wavelengths_c[0])

ind = np.arange(nb_tracks) + 1

plot(ind, d_wavelengths_c, 'x')
show()

plot(ind, d_intensity_c, 'x')
show()