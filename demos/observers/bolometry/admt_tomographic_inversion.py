"""
This example demonstrates performing a tomographic reconstruction of a
radiation profile using Cherab's anisotropic diffusion (ADMT) regularisation
utilities. We use the machine geometry, sample bolometers and equilibrium
from Generomak.
"""
import matplotlib.pyplot as plt
import numpy as np

from raysect.core.math.function.float import Exp2D, Arg2D, Atan4Q2D
from raysect.core.math import translate
from raysect.optical import World
from raysect.optical.material import AbsorbingSurface, VolumeTransform
from raysect.primitive import Cylinder, Subtract

from cherab.generomak.machine import load_first_wall
from cherab.generomak.equilibrium import load_equilibrium
from cherab.generomak.diagnostics import load_bolometers
from cherab.core.math import sample2d, sample2d_grid, sample2d_points, AxisymmetricMapper
from cherab.tools.emitters import RadiationFunction
from cherab.tools.raytransfer import RayTransferCylinder, RayTransferPipeline0D
from cherab.tools.inversions import admt_utils as admt
from cherab.tools.inversions import invert_regularised_nnls


plt.ion()

################################################################################
# Define the emissivity profile.
################################################################################
# The emissivity profile consists of a blob, a ring and part of a ring on the LFS.
# The blob and the ring are Gaussian flux functions.
# The ring is Gaussian in flux and poloidal angle.
# All have equal maximum emissivities, but not necessarily equal total power.
# We use Raysect's function framework to specify an analytic form for the
# emissivity profile, as this is very quick to sample and ray trace.
eq = load_equilibrium()
psin = eq.psi_normalised
axis = eq.magnetic_axis
blob_centre_psin = 0
blob_width_psin = 0.1
blob = Exp2D(-0.5 * (psin - blob_centre_psin)**2 / (blob_width_psin**2))
ring_centre_psin = 0.5
ring_width_psin = 0.05
ring = Exp2D(-0.5 * (psin - ring_centre_psin)**2 / (ring_width_psin**2))
theta = Atan4Q2D(Arg2D('y') - axis.y, Arg2D('x') - axis.x)
lfs_centre_psin = 0.85
lfs_width_psin = 0.1
lfs_centre_theta = 0
lfs_width_theta = 0.5
lfs = Exp2D(-0.5 * (((psin - lfs_centre_psin) / lfs_width_psin)**2
                    + ((theta - lfs_centre_theta) / lfs_width_theta)**2))
emissivity = blob + ring + lfs
# Assume no emission from these contributors outside the separatrix.
emissivity = emissivity * eq.inside_lcfs

# Visualise the emissivity profile with the equilibrium overlayed.
plt.figure()
rsamp, zsamp, psisamp = sample2d(psin, (*eq.r_range, 500), (*eq.z_range, 1000))
plt.contour(rsamp, zsamp, psisamp.T, linewidths=0.5, alpha=0.3,
            levels=np.linspace(0, 1, 10), colors=['k']*9 + ['red'])
rsamp, zsamp, emsamp = sample2d(emissivity, (*eq.r_range, 500), (*eq.z_range, 1000))
im = plt.imshow(emsamp.T, extent=(rsamp[0], rsamp[-1], zsamp[0], zsamp[-1]), cmap='Purples')
plt.xlabel("R[m]")
plt.ylabel("Z[m]")
plt.colorbar(im, label="Model emissivity [W/m3]")
plt.xlim([rsamp[0], rsamp[-1]])
plt.ylim([zsamp[0], zsamp[-1]])
plt.gca().set_aspect('equal')
plt.pause(0.5)


################################################################################
# Load the machine wall and diagnostic.
################################################################################
print("Loading the geometry...")
world = World()
load_first_wall(world, material=AbsorbingSurface())
bolos = load_bolometers(world)
# Only consider the purely-poloidal cameras for now...
bolos = bolos[:3]

########################################################################
# Produce a voxel grid
########################################################################
print("Producing the voxel grid...")
# Define the centres of each voxel, as an (nx, ny, 2) array.
nx = 40
ny = 85
cell_r, cell_dx = np.linspace(0.7, 2.5, nx, retstep=True)
cell_z, cell_dz = np.linspace(-1.8, 1.6, ny, retstep=True)
cell_r_grid, cell_z_grid = np.broadcast_arrays(cell_r[:, None], cell_z[None, :])
cell_centres = np.stack((cell_r_grid, cell_z_grid), axis=-1)  # (nx, ny, 2) array

# Define the positions of the vertices of the voxels.
cell_vertices_r = np.linspace(cell_r[0] - 0.5 * cell_dx, cell_r[-1] + 0.5 * cell_dx, nx + 1)
cell_vertices_z = np.linspace(cell_z[0] - 0.5 * cell_dz, cell_z[-1] + 0.5 * cell_dz, ny + 1)

# Build a mask, only including cells within the wall.
mask_2d = sample2d_grid(eq.inside_limiter, cell_r, cell_z)
mask_3d = mask_2d[:, np.newaxis, :]
ncells = mask_3d.sum()

# We'll use the Ray Transfer frameworks as these voxels are rectangular
# and it's much faster than the Voxel framework for simple cases like this.
ray_transfer_grid = RayTransferCylinder(
    radius_outer=cell_vertices_r[-1],
    radius_inner=cell_vertices_r[0],
    height=cell_vertices_z[-1] - cell_vertices_z[0],
    n_radius=nx, n_height=ny, mask=mask_3d, n_polar=1,
    transform=translate(0, 0, cell_vertices_z[0]),
)

########################################################################
# Calculate the geometry matrix for the grid
########################################################################
print("Calculating the geometry matrix...")
# The ray transfer object must be in the same world as the bolometers
ray_transfer_grid.parent = world

sensitivity_matrix = []
for camera in bolos:
    for foil in camera:
        # Temporarily override foil pipelines for the sensitivity calculation.
        orig_pipelines = foil.pipelines
        foil.pipelines = [RayTransferPipeline0D(kind=foil.units)]
        # All objects in world have wavelength-independent material properties,
        # so it doesn't matter which wavelength range we use (as long as
        # max_wavelength - min_wavelength = 1)
        foil.min_wavelength = 1
        foil.max_wavelength = 2
        foil.spectral_bins = ray_transfer_grid.bins
        foil.observe()
        sensitivity_matrix.append(foil.pipelines[0].matrix)
        # Restore original pipelines for subsequent observe calls.
        foil.pipelines = orig_pipelines
sensitivity_matrix = np.asarray(sensitivity_matrix)


################################################################################
# Generate the regularisation operators.
################################################################################
print("Generating regularisation operators...")
# generate_derivative_operators requires two mappings, one from a flat list of
# voxels to the original 2D grid, and one for the 2D grid coordinates to the
# flat list of voxels. We could build these by hand, but the RayTransferCylinder
# object helpfully provides the data already so we just need to convert from
# arrays to dictionaries.
grid_index_1d_to_2d_map = {}
for k, idx2d in enumerate(ray_transfer_grid.invert_voxel_map()):
    # We want the x and z elements, as the Ray Transfer grid is 3D and this
    # inversion is going to be in 2D.
    grid_index_1d_to_2d_map[k] = (idx2d[0].item(), idx2d[2].item())
grid_index_2d_to_1d_map = {}
nx, _, ny = ray_transfer_grid.voxel_map.shape
for i in range(nx):
    for j in range(ny):
        voxel_index = ray_transfer_grid.voxel_map[i, 0, j]
        if voxel_index != -1:
            grid_index_2d_to_1d_map[(i, j)] = voxel_index
# We now need an (Nx4x2) array of voxel vertices, which can be easily calculated.
voxel_centres = np.array([cell_centres[grid_index_1d_to_2d_map[i]]
                          for i in range(ray_transfer_grid.bins)])
vertex_displacements = np.array([[-cell_dx/2, -cell_dz/2],
                                 [-cell_dx/2, cell_dz/2],
                                 [cell_dx/2, cell_dz/2],
                                 [cell_dx/2, -cell_dz/2]])
# Combine the (N,2) and (4,2) arrays to get an (N,4,2) array.
voxel_vertices = voxel_centres[:, None, :] + vertex_displacements[None, :, :]
derivative_operators = admt.generate_derivative_operators(
    voxel_vertices, grid_index_1d_to_2d_map, grid_index_2d_to_1d_map
)

# As described in the docstring for generate_derivative_operators, we can
# calculate a 2D laplacian operator for "isotropic" smoothing easily:
alpha = 1/3  # Optimal isotropy
aligned = derivative_operators['Dxx'] * cell_dx**2 + derivative_operators['Dyy'] * cell_dz**2
skewed = (derivative_operators['Dsp'] + derivative_operators['Dsm']) * (cell_dx**2 + cell_dz**2)
laplacian = (1 - alpha) * aligned + (alpha / 2) * skewed
# We could also use alpha = 2/3, which would produce an operator akin to the one
# used in Carr et. al. RSI 89, 083506 (2018).

# We can also derive an anistoropic regularisation operator, which calculates the
# amount of un-smoothness parallel and perpendicular to the magnetic field lines.
# For this we need the radii of the voxels and the magnetic flux at each voxel,
# along with a few other inputs.
voxel_radii = voxel_centres[:, 0]
psi_at_voxels = sample2d_points(eq.psi_normalised, voxel_centres)
# We also need to decide on the degree of anisotropy we expect, i.e. how much more
# smooth the radiation is along the field lines vs perpendicular to them.
# The optimal value will depend on the problem at hand.
anisotropy = 50
admt_operator = admt.calculate_admt(
    voxel_radii, derivative_operators, psi_at_voxels, cell_dx, cell_dz, anisotropy
)

################################################################################
# Forward model the measurements.
################################################################################
print("Modelling the measurement values...")
# Create an emitting object whose emission is defined by the analytic form we
# produced earlier. As the emission depends on the equilibrium, this object
# should have an extent no larger than the equilibrium reconstruction extent.
# We actually make the emitter slightly smaller than the equilibrium region to
# avoid numerical precision issues creating attempts to calculate the emissivity
# outside of the equlibrium domain.
CYLINDER_RADIUS = eq.r_range[-1] - 1e-6
CYLINDER_HEIGHT = eq.z_range[-1] - eq.z_range[0] - 2e-6
CYLINDER_SHIFT = eq.z_range[0] + 1e-6
emitter = Cylinder(radius=CYLINDER_RADIUS, height=CYLINDER_HEIGHT,
                   transform=translate(0, 0, CYLINDER_SHIFT))
# Cut out middle of cylinder as well: equilibrium not defined here.
emitter = Subtract(emitter, Cylinder(radius=eq.r_range[0] + 1e-6, height=10,
                                     transform=translate(0, 0, -5)))
emission_function_3d = AxisymmetricMapper(emissivity)
emitting_material = VolumeTransform(RadiationFunction(emission_function_3d),
                                    transform=emitter.transform.inverse())
emitter.material = emitting_material
emitter.parent = world

# Calculate the line-integral bolometer measurements by observing the emitter
# with all bolometers. The measurements should have the same channel order as
# the sensitivity matrix.
all_measurements = []
for camera in bolos:
    all_measurements.extend(camera.observe())


################################################################################
# Perform the inversions.
################################################################################
print("Performing inversions...")
# We'll use NNLS with regularisation. The hyperparameters have been chosen by
# hand but techniques such as the discrepancy principle or L curve optimisation
# could also be used to determine them. That is out of the scope of this demo.
isotropic_alpha = 1e-10
isotropic_inversion, _ = invert_regularised_nnls(
    sensitivity_matrix, all_measurements, alpha=isotropic_alpha,
    tikhonov_matrix=laplacian,
)

admt_alpha = 1e-10
admt_inversion, _ = invert_regularised_nnls(
    sensitivity_matrix, all_measurements, alpha=admt_alpha,
    tikhonov_matrix=admt_operator,
)


################################################################################
# Plot the inversion results.
################################################################################
emiss2d = np.zeros((nx, ny))

# Isotropic
for index1d, indices2d in grid_index_1d_to_2d_map.items():
    emiss2d[indices2d] = isotropic_inversion[index1d]
plt.figure()
im = plt.imshow(emiss2d.T, extent=(cell_r[0], cell_r[-1], cell_z[0], cell_z[-1]), cmap='Purples')
plt.contour(rsamp, zsamp, psisamp.T, linewidths=0.5, alpha=0.3,
            levels=np.linspace(0, 1, 10), colors=['k']*9 + ['red'])
plt.xlabel("R[m]")
plt.ylabel("Z[m]")
plt.colorbar(im, label="Inverted\nEmissivity [W/m3]")
plt.xlim([rsamp[0], rsamp[-1]])
plt.ylim([zsamp[0], zsamp[-1]])
plt.gca().set_aspect('equal')
plt.title("Isotropic regularisation")

# Anisotropic.
for index1d, indices2d in grid_index_1d_to_2d_map.items():
    emiss2d[indices2d] = admt_inversion[index1d]
plt.figure()
im = plt.imshow(emiss2d.T, extent=(cell_r[0], cell_r[-1], cell_z[0], cell_z[-1]), cmap='Purples')
plt.contour(rsamp, zsamp, psisamp.T, linewidths=0.5, alpha=0.3,
            levels=np.linspace(0, 1, 10), colors=['k']*9 + ['red'])
plt.xlabel("R[m]")
plt.ylabel("Z[m]")
plt.colorbar(im, label="Inverted\nEmissivity [W/m3]")
plt.xlim([rsamp[0], rsamp[-1]])
plt.ylim([zsamp[0], zsamp[-1]])
plt.gca().set_aspect('equal')
plt.title("Anisotropic regularisation")

plt.ioff()
plt.show()
