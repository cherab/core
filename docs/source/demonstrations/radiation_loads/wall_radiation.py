
# This code measures integrated power around the machine walls.

import os
import time
import matplotlib.pyplot as plt
from math import sqrt, pi

from raysect.core import translate, rotate_basis
from raysect.optical import World
from raysect.optical.observer.nonimaging.pixel import Pixel
from raysect.optical.observer import PowerPipeline0D
from raysect.primitive.mesh import Mesh
from raysect.optical.material.absorber import AbsorbingSurface

from cherab_contrib.simulation_data.solps.models.radiated_power import solps_total_radiated_power
from cherab_contrib.simulation_data.solps.solps_plasma import SOLPSSimulation

from cherab_aug.integrated_power.wall_detector_geometry import aug_wall_detectors

world = World()

MESH_PATH = '/projects/cadmesh/aug/'

# Load all parts of mesh with chosen material
MESH_PARTS = ['vessel_s02-03.rsm', 'vessel_s04-05.rsm', 'vessel_s06-07.rsm', 'vessel_s08-09.rsm',
              'vessel_s10-11.rsm', 'vessel_s12-13.rsm', 'vessel_s14-15.rsm', 'vessel_s16-01.rsm',
              'divertor_s02-03.rsm', 'divertor_s04-05.rsm', 'divertor_s06-07.rsm', 'divertor_s08-09.rsm',
              'divertor_s10-11.rsm', 'divertor_s12-13.rsm', 'divertor_s14-15.rsm', 'divertor_s16-01.rsm',
              'inner_heat_shield_s01.rsm', 'inner_heat_shield_s02.rsm', 'inner_heat_shield_s03.rsm',
              'inner_heat_shield_s04.rsm', 'inner_heat_shield_s05.rsm', 'inner_heat_shield_s06.rsm',
              'inner_heat_shield_s07.rsm', 'inner_heat_shield_s08.rsm', 'inner_heat_shield_s09.rsm',
              'inner_heat_shield_s10.rsm', 'inner_heat_shield_s11.rsm', 'inner_heat_shield_s12.rsm',
              'inner_heat_shield_s13.rsm', 'inner_heat_shield_s14.rsm', 'inner_heat_shield_s15.rsm',
              'inner_heat_shield_s16.rsm']

machine_material = AbsorbingSurface()  # Mesh with perfect absorber

for path in MESH_PARTS:
    path = MESH_PATH + path
    print("importing {}  ...".format(os.path.split(path)[1]))
    directory, filename = os.path.split(path)
    name, ext = filename.split('.')
    Mesh.from_file(path, parent=world, material=machine_material, name=name)


# Load simulation from MDSplus
mds_server = 'solps-mdsplus.aug.ipp.mpg.de:8001'
ref_number = 40195
sim = SOLPSSimulation.load_from_mdsplus(mds_server, ref_number)

# Load simulation from raw output files
# SIM_PATH = '/home/mcarr/mst1/aug_2016/solps_testcase/'
# sim = SOLPSSimulation.load_from_output_files(SIM_PATH)

plasma = sim.plasma
mesh = sim.mesh
plasma_cylinder = solps_total_radiated_power(world, sim, step=0.001)

X_WIDTH = 0.01
powers = []
power_errors = []
detector_numbers = []
distance = []

running_distance = 0
cherab_total_power = 0

start = time.time()


for i, detector in enumerate(aug_wall_detectors):

    print()
    print("detector {}".format(i))

    y_width = detector[2]
    centre_point = detector[3]
    normal_vector = detector[4]
    y_vector = detector[5]
    pixel_area = X_WIDTH * y_width

    power_data = PowerPipeline0D()

    pixel_transform = translate(centre_point.x, centre_point.y, centre_point.z) * rotate_basis(normal_vector, y_vector)
    pixel = Pixel([power_data], x_width=X_WIDTH, y_width=y_width, name='pixel-{}'.format(i),
                  spectral_bins=1, transform=pixel_transform, parent=world, pixel_samples=500)

    pixel.observe()

    powers.append(power_data.value.mean / pixel_area)
    power_errors.append(power_data.value.error() / pixel_area)
    detector_numbers.append(i)

    running_distance += 0.5*y_width
    distance.append(running_distance)
    running_distance += 0.5*y_width

    pixel_radius = sqrt(centre_point.x**2 + centre_point.y**2)
    cherab_total_power += (power_data.value.mean / pixel_area) * y_width * 2 * pi * pixel_radius

running_time = time.time() - start
print("running time => {:.2f}".format(running_time/60))


total_rad_data = sim.total_rad_data
vol = mesh.vol
radius = mesh.cr

solps_total_power = 0
for i in range(mesh.nx):
    for j in range(mesh.ny):
        solps_total_power += total_rad_data[i, j] * vol[i, j]

print()
print("Cherab total radiated power => {:.4G} W".format(cherab_total_power))
print()
print("SOLPS total radiated power => {:.4G} W".format(solps_total_power))

plt.plot(distance, powers)
plt.xlabel("Distance around machine (m)")
plt.ylabel("Radiative power load (W/m^2)")

sim.plot_radiated_power()
plt.show()

