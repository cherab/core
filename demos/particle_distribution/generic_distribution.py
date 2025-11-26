from scipy.constants import atomic_mass, elementary_charge, pi

import numpy as np

import matplotlib.pyplot as plt

from raysect.core.math.function.float import Arg3D, Exp3D, Sqrt3D
from raysect.core.math.function.vector3d import FloatToVector3DFunction3D

from cherab.core.atomic import deuterium
from cherab.core.distribution import GenericDistribution
from cherab.core.math.function.float import Arg6D, Exp6D, Sqrt6D


# To set up a generic distribution, we need to define the following:
# - define 3D scalar function defining the spatial distribution of the effective temperature
# - define 3D vector function defining the spatial distribution of the bulk velocity
# - define 3D scalar function defining the spatial distribution of the density
# - define 6D scalar function defining the phase space density

# This example creates a toroidally symmetric distribution in R-Z coordinates
# where R = sqrt(X^2 + Y^2) and Z is the vertical coordinate.
# The distribution peaks at R=2, Z=0 with Gaussian-like profiles.

# initialise the spatial arguments for the 3D functions
x3d, y3d, z3d = Arg3D("x"), Arg3D("y"), Arg3D("z")

# Calculate R = sqrt(X^2 + Y^2) for 3D functions
r_3d = Sqrt3D(x3d**2 + y3d**2)

# Peak location in R-Z space
r_peak = 2.0  # meters
z_peak = 0.0  # meters

# set the properties of the temperature profile
maximum_temperature = 1000  # eV
temperature_peak_width_R = 0.5  # meters
temperature_peak_width_Z = 0.5  # meters

# set up a 3D gaussian-like temperature profile in R-Z
# The temperature is defined only as a 3D function,
# and is not used within the phase space density function.
# Consistency with the phase space density function is not checked
# and is the responsibility of the user, if required.
temperature_3d = maximum_temperature * Exp3D(
    -0.5
    * (
        ((r_3d - r_peak) ** 2 / temperature_peak_width_R**2)
        + ((z3d - z_peak) ** 2 / temperature_peak_width_Z**2)
    )
)

# set the properties of the density profile
maximum_density = 5e19  # m^-3
density_peak_width_r = 0.5  # meters
density_peak_width_z = 0.5  # meters

# set up the 3D function defining the spatial density in R-Z
# The density is defined only as a 3D function,
# and is not used within the phase space density function.
# Consistency with the phase space density function is not checked
# and is the responsibility of the user, if required.
density_3d = maximum_density * Exp3D(
    -0.5
    * (
        ((r_3d - r_peak) ** 2 / density_peak_width_r**2)
        + ((z3d - z_peak) ** 2 / density_peak_width_z**2)
    )
)

# set the properties of the toroidal rotation velocity profile
# The bulk velocity is defined only as a 3D vector function,
# and is not used within the phase space density function.
# Consistency with the phase space density function is not checked
# and is the responsibility of the user, if required.
maximum_toroidal_velocity = 1e5  # m/s
toroidal_velocity_peak_width_R = 0.5  # meters
toroidal_velocity_peak_width_Z = 0.5  # meters

# Toroidal velocity profile in R-Z (Gaussian-like)
# The toroidal direction is perpendicular to R and Z
# In Cartesian: v_toroidal * (-y/R, x/R, 0)
v_toroidal_profile_3d = maximum_toroidal_velocity * Exp3D(
    -0.5
    * (
        ((r_3d - r_peak) ** 2 / toroidal_velocity_peak_width_R**2)
        + ((z3d - z_peak) ** 2 / toroidal_velocity_peak_width_Z**2)
    )
)

# Convert toroidal velocity to Cartesian components
# vx = -v_toroidal * y/R, vy = v_toroidal * x/R, vz = 0
# Note: We need to handle the case where R=0, but for this example we assume R>0
vx_profile = -v_toroidal_profile_3d * y3d / (r_3d + 1e-10)  # small epsilon to avoid division by zero
vy_profile = v_toroidal_profile_3d * x3d / (r_3d + 1e-10)
vz_profile = 0.0
bulk_velocity_profile = FloatToVector3DFunction3D(vx_profile, vy_profile, vz_profile)

# initialise the arguments for the 6D function
x6d, y6d, z6d, vx6d, vy6d, vz6d = (
    Arg6D("x"),
    Arg6D("y"),
    Arg6D("z"),
    Arg6D("u"),
    Arg6D("w"),
    Arg6D("v"),
)

# Calculate R = sqrt(X^2 + Y^2) for 6D functions
r_6d = Sqrt6D(x6d**2 + y6d**2)

# set the missing parameters of the distribution function
deuterium_mass = deuterium.atomic_weight * atomic_mass

# re-define the spatial temperature profile with the 6D function parameters
te_6d = maximum_temperature * Exp6D(
    -0.5
    * (
        ((r_6d - r_peak) ** 2 / temperature_peak_width_R**2)
        + ((z6d - z_peak) ** 2 / temperature_peak_width_Z**2)
    )
)

# re-define the spatial density profile with the 6D function parameters
density_6d = maximum_density * Exp6D(
    -0.5
    * (
        ((r_6d - r_peak) ** 2 / density_peak_width_r**2)
        + ((z6d - z_peak) ** 2 / density_peak_width_z**2)
    )
)

# Toroidal velocity profile redefined with the 6D function parameters
v_toroidal_mean_6d = maximum_toroidal_velocity * Exp6D(
    -0.5
    * (
        ((r_6d - r_peak) ** 2 / toroidal_velocity_peak_width_R**2)
        + ((z6d - z_peak) ** 2 / toroidal_velocity_peak_width_Z**2)
    )
) * -1.0

# Convert toroidal velocity to Cartesian velocity components for the 6D function
# vx_mean = -v_toroidal * y/R, vy_mean = v_toroidal * x/R, vz_mean = 0
vx_mean_6d = -v_toroidal_mean_6d * y6d / (r_6d + 1e-10)
vy_mean_6d = v_toroidal_mean_6d * x6d / (r_6d + 1e-10)
vz_mean_6d = 0.0

# define the 6D distribution function for the bulk population
factor_6d = (deuterium_mass / (2 * pi * elementary_charge * te_6d)) ** 1.5

thermal_exponential_6d = Exp6D(
    -0.5
    * deuterium_mass
    * ((vx6d - vx_mean_6d) ** 2 + (vy6d - vy_mean_6d) ** 2 + (vz6d - vz_mean_6d) ** 2)
    / (elementary_charge * te_6d)
)
bulk_pdf_6d = (
    density_6d * factor_6d * thermal_exponential_6d
)  # bulk particle distribution function

# add a population of supra-thermal particles with higher toroidal rotation
# the 3D bulk velocity and temperature functions ignore this population,
# and is the responsibility of the user, if required.
supra_thermal_population_ratio = 0.005  # 10% of the particles are supra-thermal
suprathermal_toroidal_velocity_factor = 20.0  # Supra-thermal particles have 2x toroidal velocity
suprathermal_temperature = 100  # eV (higher temperature for supra-thermal particles)

# Supra-thermal toroidal velocity profile (higher than bulk)
v_toroidal_suprathermal_6d = (
    maximum_toroidal_velocity
    * suprathermal_toroidal_velocity_factor
    * Exp6D(
        -0.5
        * (
            ((r_6d - r_peak) ** 2 / toroidal_velocity_peak_width_R**2)
            + ((z6d - z_peak) ** 2 / toroidal_velocity_peak_width_Z**2)
        )
    )
)

# Convert supra-thermal toroidal velocity to Cartesian components
vx_suprathermal_6d = -v_toroidal_suprathermal_6d * y6d / (r_6d + 1e-10)
vy_suprathermal_6d = v_toroidal_suprathermal_6d * x6d / (r_6d + 1e-10)
vz_suprathermal_6d = 0.0

factor_st = (deuterium_mass / (2 * pi * elementary_charge * suprathermal_temperature)) ** 1.5

suprathermal_exponential_6d = Exp6D(
    -0.5
    * deuterium_mass
    * (
        (vx6d - vx_suprathermal_6d) ** 2
        + (vy6d - vy_suprathermal_6d) ** 2
        + (vz6d - vz_suprathermal_6d) ** 2
    )
    / (elementary_charge * suprathermal_temperature)
)
supra_thermal_pdf_6d = (
    supra_thermal_population_ratio * density_6d * factor_st * suprathermal_exponential_6d
)
phase_space_density_6d = bulk_pdf_6d + supra_thermal_pdf_6d


generic_distribution = GenericDistribution(
    phase_space_density_6d, density_3d, temperature_3d, bulk_velocity_profile
)

# Example evaluation: sample the distribution at a point in R-Z space
# At R=2, Z=0 (the peak), sample velocity distribution
# Convert R=2, Z=0 to Cartesian: x=2, y=0, z=0
sample_x, sample_y, sample_z = 2.0, 0.0, 0.0

# Sample velocity distribution along the toroidal and vertical directions (vy and vz components)
v_vals = np.linspace(-1e6, 3e6, 1000)
n_particles_vy = np.zeros_like(v_vals)
n_particles_vz = np.zeros_like(v_vals)
for i in range(len(v_vals)):
    n_particles_vy[i] = generic_distribution(sample_x, sample_y, sample_z, 0.0, v_vals[i], 0.0)
    n_particles_vz[i] = generic_distribution(sample_x, sample_y, sample_z, 0.0, 0.0, v_vals[i])


_, ax = plt.subplots()
ax.plot(v_vals, n_particles_vy, label="$\\mathrm{v}_\\mathrm{y}$")
ax.plot(v_vals, n_particles_vz, label="$\\mathrm{v}_\\mathrm{z}$")
ax.legend()
ax.set_xlabel("Velocity (m/s)")
ax.set_ylabel("Phase Space Density (s^3/m^6)")
ax.set_title("Velocity Distribution at R=2, Z=0 (Peak Location)")
ax.grid(True)

# sample the temperature distribution in the R-Z plane
r_vals = np.linspace(1, 3, 100)
z_vals = np.linspace(-2, 2, 210)

temperature_vals = np.zeros((r_vals.size, z_vals.size))
density_vals = np.zeros((r_vals.size, z_vals.size))
bulk_velocity_vals = np.zeros((r_vals.size, z_vals.size))
for i in range(r_vals.size):
    for j in range(z_vals.size):
        temperature_vals[i, j] = generic_distribution.effective_temperature(r_vals[i], 0.0, z_vals[j])
        density_vals[i, j] = generic_distribution.density(r_vals[i], 0.0, z_vals[j])
        vector_velocity = generic_distribution.bulk_velocity(r_vals[i], 0.0, z_vals[j])
        bulk_velocity_vals[i, j] = np.sqrt(vector_velocity.x**2 + vector_velocity.y**2 + vector_velocity.z**2)

_, ax = plt.subplots()
pcm = ax.pcolormesh(r_vals, z_vals, temperature_vals.transpose(), shading='gouraud')
plt.colorbar(pcm, ax=ax, label="Temperature (eV)")
ax.set_xlabel("R (m)")
ax.set_ylabel("Z (m)")
ax.set_title("Temperature Distribution")
ax.set_aspect('equal')
ax.grid(True)

_, ax = plt.subplots()
pcm = ax.pcolormesh(r_vals, z_vals, density_vals.transpose(), shading='gouraud')
plt.colorbar(pcm, ax=ax, label="Density (m^-3)")
ax.set_xlabel("R (m)")
ax.set_ylabel("Z (m)")
ax.set_title("Density Distribution")
ax.set_aspect('equal')
ax.grid(True)

_, ax = plt.subplots()
pcm = ax.pcolormesh(r_vals, z_vals, bulk_velocity_vals.transpose(), shading='gouraud')
plt.colorbar(pcm, ax=ax, label="velocity (m/s)")
ax.set_xlabel("R (m)")
ax.set_ylabel("Z (m)")
ax.set_title("Toroidal Bulk Velocity Distribution")
ax.set_aspect('equal')
ax.grid(True)

# sample the x, y velocity components in the X-Y plane using arrow vectors
x_vals = np.linspace(-3, 3, 21)  # Reduced resolution for clearer quiver plot
y_vals = np.linspace(-3, 3, 21)

X, Y = np.meshgrid(x_vals, y_vals)
x_velocity_vals = np.zeros_like(X)
y_velocity_vals = np.zeros_like(Y)

for i in range(x_vals.size):
    for j in range(y_vals.size):
        vector_velocity = generic_distribution.bulk_velocity(x_vals[i], y_vals[j], 0.0)
        x_velocity_vals[j, i] = vector_velocity.x
        y_velocity_vals[j, i] = vector_velocity.y

# Calculate velocity magnitude for colormap
velocity_magnitude = np.sqrt(x_velocity_vals**2 + y_velocity_vals**2)
        
_, ax = plt.subplots()
quiver = ax.quiver(X, Y, x_velocity_vals, y_velocity_vals, velocity_magnitude, 
                   cmap='viridis', scale=1e6, width=0.003)
plt.colorbar(quiver, ax=ax, label="Velocity Magnitude (m/s)")
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_title("Bulk x, y Velocity Cmponents Field in X-Y Plane (Z=0)")
ax.set_aspect('equal')
ax.grid(True)
