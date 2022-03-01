from random import gauss
from raysect.core import Vector3D

from cherab.core.math.samplers import sample3d_grid
from cherab.core.model.laser.profile import (UniformEnergyDensity, ConstantBivariateGaussian,
                                             TrivariateGaussian, GaussianBeamAxisymmetric)

import numpy as np
import matplotlib.pyplot as plt

def plot_profiles(profile, title=""):
    """Plot the laser profile
    
    Produces 2d energy density plot in the x-z plane at y=0.
    Produces plots of energy density profiles along x, y and z
    directions.
    """

    # set coordinate grid and zero indices                                            
    x = np.linspace(-0.01, 0.01, 31)
    z = np.linspace(-1, 1, 101)
    x_zero = np.abs(x).argmin()
    z_zero = np.abs(z).argmin()

    e_xz_profile = sample3d_grid(profile.get_energy_density, x, [0], z)[:, 0, :]
    e_y_profile = sample3d_grid(profile.get_energy_density, [0], x, [0])[0, :, 0]

    # plot
    fig = plt.figure(constrained_layout=True)
    fig.suptitle(title)
    spec = fig.add_gridspec(4, 3)

    # profile along y at z=0
    axz = fig.add_subplot(spec[0, 0:2])
    axz.plot(x, e_y_profile, color="C2", ls="dashed")
    axz.set_xlabel("y [m]")
    axz.set_ylabel("Energy [J/m^3]")

    # x-y plane
    ax2d = fig.add_subplot(spec[1:3, 0:2])
    im = ax2d.pcolormesh(x, z, e_xz_profile.T)
    fig.colorbar(im, ax=ax2d, label="Energy [J/m^3]")
    ax2d.axhline(z[z_zero], color="C0", ls="dashed")
    ax2d.axvline(x[x_zero], color="C1", ls="dashed")
    ax2d.plot([x[x_zero]], z[z_zero], marker="x", color="C2")
    ax2d.set_xlabel("x [m]")
    ax2d.set_ylabel("z [m]")

    # profile along z at x=0
    axz = fig.add_subplot(spec[1:3, -1])
    axz.plot(e_xz_profile[x_zero, :], z, color="C0", ls="dashed")
    axz.set_ylabel("z [m]")
    axz.set_xlabel("Energy [J/m^3]")

    # profile along x at z=0
    axz = fig.add_subplot(spec[-1, 0:2])
    axz.plot(x, e_xz_profile[:, z_zero].T, color="C1", ls="dashed")
    axz.set_xlabel("x [m]")
    axz.set_ylabel("Energy [J/m^3]")

# UniformEnergyDensity profile has constant parameters in the whole x, y, z space
uniform = UniformEnergyDensity(energy_density=1e3, laser_length=2., laser_radius=0.005,
                               polarization=Vector3D(0, 1, 0))
plot_profiles(uniform, "Uniform Energy Profile")

# Example of ConstantBivariteGaussian. With energy 2J, pulse temporal length 5 ns
# and different standard deviations in the x and y plane. The mean value in both
# dimensions is equal to 0. There is no correlation between x and y.

gauss2d = ConstantBivariateGaussian(pulse_energy=2, pulse_length=5e-9,
                                    stddev_x=2e-3, stddev_y=4e-3)
plot_profiles(gauss2d, "ConstantBivariateGaussian Energy Profile")

# Example of TrivariageGaussian profile. With the pulse mean at z=0.2
gauss3d = TrivariateGaussian(pulse_energy=2, pulse_length=2e-9,
                             mean_z=0.2, stddev_x=2e-3, stddev_y=4e-3 )
plot_profiles(gauss3d, "TrivariateGaussian Energy Profile")

# Example of GaussianBeamModel with wavelength 1e4 nm and 1 mm standard deviation
# in the waist. The wavelength is too high
# but was selected to produce nice plots in the defined x, y, z dimensions.

gaussbeam = GaussianBeamAxisymmetric(stddev_waist=1e-3, waist_z=0.2, laser_wavelength=1e4)
plot_profiles(gaussbeam, "GaussianBeam Energy Profile")

plt.show()