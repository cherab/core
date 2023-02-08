import numpy as np

from raysect.optical import World, translate, Point3D, rotate_basis, Vector3D
from raysect.optical.observer import FibreOptic

from cherab.core.model.laser import ConstantBivariateGaussian, ConstantSpectrum, SeldenMatobaThomsonSpectrum
from cherab.core.laser import Laser

from cherab.generomak.plasma import get_core_plasma
from cherab.generomak.equilibrium import load_equilibrium

import matplotlib.pyplot as plt

world = World()

plasma = get_core_plasma(parent=world)
equilibrium = load_equilibrium()

# set up the laser
laser = Laser(name="Thomson Scattering laser", parent=world)
laser.transform = translate(equilibrium.magnetic_axis[0], 0, -2)
laser.plasma = plasma
laser.laser_profile = ConstantBivariateGaussian(pulse_energy=2, pulse_length=1e-8,
                                                laser_length=4, laser_radius=1e-2,
                                                stddev_x=3e-3, stddev_y=2e-3)
laser.laser_spectrum = ConstantSpectrum(min_wavelength=1059.9, max_wavelength=1060.1, bins=1)
laser.models = [SeldenMatobaThomsonSpectrum()]

# generate points on laser in world space to measure on
laser_points = [Point3D(0, 0, round(z, 2)).transform(laser.to_root()) for z in np.linspace(2, 2.8, 5)]
te = [plasma.electron_distribution.effective_temperature(*point) for point in laser_points]
ne = [plasma.electron_distribution.density(*point) for point in laser_points]

# place fibres 0.1m outside sep on lfs
fibre_position = Point3D(equilibrium.psin_to_r(1) + 0.1, 0, 0)

# generate fibres list and observe
fibres = {}
for point in laser_points:

    direction = fibre_position.vector_to(point)
    transform = translate(*fibre_position) * rotate_basis(direction, Vector3D(0, 0, 1))

    fibre = FibreOptic(radius=1e-3, acceptance_angle=0.25,
                       parent=world, transform=transform,
                       min_wavelength=800, max_wavelength=1200,
                       spectral_bins=1000,
                       pixel_samples=1000)

    fibre.pipelines[0].display_progress = False
    fibre.observe()
    fibres[point.z] = fibre

_, ax = plt.subplots()
for z, fibre in fibres.items():
    pipeline = fibre.pipelines[0]
    ax.plot(pipeline.wavelengths, pipeline.samples.mean, label="z={:1.2f}m".format(z))
ax.set_xlabel("wavelength [nm]")
ax.set_ylabel("spectral power W/nm")
ax.legend()
plt.show()
