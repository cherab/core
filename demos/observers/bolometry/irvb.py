"""
This example demonstrates building a simple IR bolometer camera where
the position of the camera itself is known, and the slit and foil are
defined relative to the camera. This could be the case when working from
design office drawings, for example. Once the camera is instantiated, it
is used to observe an emission pattern.

This example also demonstrates the similarities between the IR and foil
bolometers.  Tasks such as sensitivity matrix calculation and
tomographic inversions will be very similar to the examples using the
foil bolometer: those demos should be used as a starting point for the
IRVB.
"""
from matplotlib.patches import Circle, Polygon
import matplotlib.pyplot as plt
import numpy as np

from raysect.core import Point3D, Vector3D, rotate_basis, translate
from raysect.core.math.function.float import Exp2D
from raysect.primitive import Box, Cylinder, Subtract
from raysect.optical import World
from raysect.optical.material import AbsorbingSurface, VolumeTransform

from cherab.core.math import AxisymmetricMapper, PolygonMask2D, sample2d
from cherab.tools.emitters import RadiationFunction
from cherab.tools.equilibrium import example_equilibrium
from cherab.tools.observers import BolometerCamera, BolometerSlit, BolometerIRVB
from cherab.tools.primitives import axisymmetric_mesh_from_polygon

# Convenient constants
XAXIS = Vector3D(1, 0, 0)
YAXIS = Vector3D(0, 1, 0)
ZAXIS = Vector3D(0, 0, 1)
ORIGIN = Point3D(0, 0, 0)

# Bolometer geometry
BOX_WIDTH = 0.1
BOX_HEIGHT = 0.15
BOX_DEPTH = 0.3
SLIT_WIDTH = 0.004
SLIT_HEIGHT = 0.005
FOIL_WIDTH = 0.06
FOIL_HEIGHT = 0.08
FOIL_CORNER_CURVATURE = 0
SLIT_SENSOR_SEPARATION = 0.2
PIXELS = (30, 40)

world = World()

########################################################################
# Build an IRVB bolometer camera.
########################################################################

# The camera consists of a box with a rectangular slit and foil.
# In its local coordinate system, the camera's slit is located at the
# origin and the foil below the z=0 plane, looking up towards the slit.

# To position the camera relative to its parent, set the `transform`
# property to produce the correct translation and rotation.
camera_box = Box(lower=Point3D(-BOX_WIDTH / 2, -BOX_HEIGHT / 2, -BOX_DEPTH),
                 upper=Point3D(BOX_WIDTH / 2, BOX_HEIGHT / 2, 0))
# Hollow out the box
outside_box = Box(lower=camera_box.lower - Vector3D(1e-5, 1e-5, 1e-5),
                  upper=camera_box.upper + Vector3D(1e-5, 1e-5, 1e-5))
camera_box = Subtract(outside_box, camera_box)
# The slit is a hole in the box.
aperture = Box(lower=Point3D(-SLIT_WIDTH / 2, -SLIT_HEIGHT / 2, -1e-4),
               upper=Point3D(SLIT_WIDTH / 2, SLIT_HEIGHT / 2, 1e-4))
camera_box = Subtract(camera_box, aperture, name="Camera box geometry")
camera_box.material = AbsorbingSurface()
# Instance of the bolometer camera.
bolometer_camera = BolometerCamera(camera_geometry=camera_box, parent=world,
                                   name="Demo camera")
# The bolometer slit in this instance just contains targeting information
# for the ray tracing, since we have already given our camera a geometry.
# The slit is defined in the local coordinate system of the camera.
slit = BolometerSlit(slit_id="Example slit", centre_point=ORIGIN,
                     basis_x=XAXIS, dx=SLIT_WIDTH, basis_y=YAXIS, dy=SLIT_HEIGHT,
                     parent=bolometer_camera)

# Make the foil. Its parent is the camera, so we specify the transform relative
# to the camera body rather than in world space.
foil_transform = translate(0, 0, -SLIT_SENSOR_SEPARATION)
foil = BolometerIRVB(
    name="IRVB foil", transform=foil_transform,
    width=FOIL_WIDTH, pixels=PIXELS, slit=slit, parent=bolometer_camera,
    units="power", accumulate=False, curvature_radius=FOIL_CORNER_CURVATURE)
bolometer_camera.add_foil_detector(foil)

# The camera is positioned at (x, y, z) = (1.5, 2.5, 0)m and looking along the -x axis.
bolometer_camera.transform = translate(1.5, 2.5, 0) * rotate_basis(-YAXIS, ZAXIS)

# Pick out just the foil, which we'll use for observing later.
irvb_foil = bolometer_camera[0]

########################################################################
# Show the lines of sight of this camera.
########################################################################
# Set up a representative tokamak geometry.
eq = example_equilibrium()
r_inner = eq.limiter_polygon[:, 0].min()
r_outer = eq.limiter_polygon[:, 0].max()
z_lower = eq.limiter_polygon[:, 1].min()
z_upper = eq.limiter_polygon[:, 1].max()
dr = r_outer - r_inner
dz = z_upper - z_lower
first_wall = axisymmetric_mesh_from_polygon(eq.limiter_polygon, 20)
irvb_port = Cylinder(radius=0.2, height=1.5,
                     transform=bolometer_camera.transform * translate(0, 0, -0.5))
vessel = Subtract(first_wall, irvb_port, name="Vessel")
vessel.name = "Vessel"
vessel.material = AbsorbingSurface()
vessel.parent = world

plt.ion()
print("Calculating sightlines...")
sightlines = irvb_foil.trace_sightlines()
# Row (nrows // 2) has a purely tangential line of sight.
tangential_sightlines = sightlines[:, 20]
fig, ax = plt.subplots()
ax.add_patch(Circle((0, 0), radius=r_inner, edgecolor='black', facecolor='none'))
ax.add_patch(Circle((0, 0), radius=r_outer, edgecolor='black', facecolor='none'))
for sightline in tangential_sightlines:
    los_start, los_end, _ = sightline
    ax.plot([los_start.x, los_end.x], [los_start.y, los_end.y], 'k', linewidth=0.2)
ax.set_xlabel("x / m")
ax.set_ylabel("y / m")
ax.set_title("Pixel lines of sight: plan view")
ax.axis('equal')
plt.pause(0.1)

print("Calculating poloidal sightlines...")
# Column -1 is the column closest to a poloidal view.
# Column (ncols // 2) nicely demonstrates some poloidal + tangential lines of sight.
poloidal_sightlines = sightlines[15, :]
fig, ax = plt.subplots()
ax.add_patch(Polygon(eq.limiter_polygon, closed=True, edgecolor='k', facecolor='none'))
for sightline in poloidal_sightlines:
    los_start, los_end, _ = sightline
    los_vector = los_start.vector_to(los_end).normalise()
    tmax = los_start.distance_to(los_end)
    los_xs = los_start.x + np.linspace(0, tmax, 100) * los_vector.x
    los_ys = los_start.y + np.linspace(0, tmax, 100) * los_vector.y
    los_zs = los_start.z + np.linspace(0, tmax, 100) * los_vector.z
    los_rs = np.hypot(los_xs, los_ys)
    ax.plot(los_rs, los_zs, 'k', linewidth=0.2)
ax.set_xlabel("R / m")
ax.set_ylabel("z / m")
ax.set_title("Pixel lines of sight: poloidal view")
ax.axis('equal')
plt.pause(0.1)

################################################################################
# Make a radiating annulus, constant within a given distance of (R0, Z0).
################################################################################
print("Example 1: a radiating annulus of constant emissivity.")
R0, Z0 = 1.7, 0.2
delta = 0.05
emiss_rz = PolygonMask2D([[R0 - delta/2, Z0 - delta/2], [R0 - delta/2, Z0 + delta/2],
                          [R0 + delta/2, Z0 + delta/2], [R0 + delta/2, Z0 - delta/2]])
resolution = 0.01
nr = dr / resolution
nz = dz / resolution

r, z, sampled_emiss = sample2d(emiss_rz, (r_inner, r_outer, nr), (z_lower, z_upper, nz))
fig, ax = plt.subplots()
emiss_plot = ax.pcolormesh(r, z, sampled_emiss.T, cmap="Purples")
ax.add_patch(Polygon(eq.limiter_polygon, closed=True, edgecolor='k', facecolor='none'))
ax.set_xlabel("R/m")
ax.set_ylabel("z/m")
ax.set_title("Radiating annulus emissivity [W/m3]")
fig.colorbar(emiss_plot)
ax.axis('equal')
plt.pause(0.1)

emitter = Cylinder(radius=3, height=4, transform=translate(0, 0, -2),
                   material=VolumeTransform(RadiationFunction(AxisymmetricMapper(emiss_rz)),
                                            translate(0, 0, 2)),
                   parent=world)
irvb_foil.quiet = False
irvb_foil.pixel_samples = 10000
irvb_foil.observe()
plt.pause(0.1)

print("Calculating etendues for conversion to radiance...")
empty_world = World()
bolometer_camera.parent = empty_world
etendue, _ = irvb_foil.calculate_etendue(ray_count=1000, batches=5)
bolometer_camera.parent = world
# Scale factor to convert from mean radiance over 2pi to mean radiance over
# the solid angle subtended by the aperture from the foil's pixels.
pixel = irvb_foil.pixels_as_foils[0][0]
pixel_area = pixel.x_width * pixel.y_width
radiance_scale_factor = etendue / (2 * np.pi * pixel_area)

if irvb_foil.units.lower() == "power":
    brightness = irvb_foil.pipelines[0].frame.mean / etendue * 4 * np.pi
else:
    brightness = irvb_foil.pipelines[0].frame.mean * radiance_scale_factor * 4 * np.pi

fig, ax = plt.subplots()
image = ax.imshow(brightness.T, interpolation="none")
ax.set_xlabel("Pixel column")
ax.set_ylabel("Pixel row")
ax.set_title("Brightness for radiating annulus [W/m2]")
fig.colorbar(image)

input("Hit return to start the next example.")

################################################################################
# An emissivity flux function, centrally-peaked.
################################################################################
print("Example 2: an emissivity flux-function, central peak.")
psi_mean = 0
psi_std = 0.2
emiss_rz = Exp2D(-(eq.psi_normalised - psi_mean)**2 / (2 * psi_std**2)) * eq.inside_lcfs

r, z, sampled_emiss = sample2d(emiss_rz, (r_inner, r_outer, nr), (z_lower, z_upper, nz))
fig, ax = plt.subplots()
emiss_plot = ax.pcolormesh(r, z, sampled_emiss.T, cmap="Purples")
ax.add_patch(Polygon(eq.limiter_polygon, closed=True, edgecolor='k', facecolor='none'))
ax.set_xlabel("R/m")
ax.set_ylabel("z/m")
ax.set_title("Central peak emissivity [W/m3]")
fig.colorbar(emiss_plot)
ax.axis('equal')
plt.pause(0.1)


emitter.parent = None  # Remove old emitter from world.
emitter = Cylinder(radius=r_outer, height=dz, transform=translate(0, 0, z_lower),
                   material=VolumeTransform(RadiationFunction(AxisymmetricMapper(emiss_rz)),
                                            translate(0, 0, -z_lower)),
                   parent=world)

irvb_foil.observe()

if irvb_foil.units.lower() == "power":
    brightness = irvb_foil.pipelines[0].frame.mean / etendue * 4 * np.pi
else:
    brightness = irvb_foil.pipelines[0].frame.mean * radiance_scale_factor * 4 * np.pi

fig, ax = plt.subplots()
image = ax.imshow(brightness.T, interpolation="none")
ax.set_xlabel("Pixel column")
ax.set_ylabel("Pixel row")
ax.set_title("Brightness for centrally-peaked profile [W/m2]")
fig.colorbar(image)

input("Hit return to start the next example.")

################################################################################
# An emissivity flux function, radiating ring.
################################################################################
print("Example 3: an emissivity flux-function, radiating ring.")
psi_mean = 0.6
psi_std = 0.2
emiss_rz = Exp2D(-(eq.psi_normalised - psi_mean)**2 / (2 * psi_std**2)) * eq.inside_lcfs

r, z, sampled_emiss = sample2d(emiss_rz, (r_inner, r_outer, nr), (z_lower, z_upper, nz))
fig, ax = plt.subplots()
emiss_plot = ax.pcolormesh(r, z, sampled_emiss.T, cmap="Purples")
ax.add_patch(Polygon(eq.limiter_polygon, closed=True, edgecolor='k', facecolor='none'))
ax.set_xlabel("R/m")
ax.set_ylabel("z/m")
ax.set_title("Hollow emissivity [W/m3]")
fig.colorbar(emiss_plot)
ax.axis('equal')
plt.pause(0.1)


emitter.parent = None  # Remove old emitter from world.
emitter = Cylinder(radius=r_outer, height=dz, transform=translate(0, 0, z_lower),
                   material=VolumeTransform(RadiationFunction(AxisymmetricMapper(emiss_rz)),
                                            translate(0, 0, -z_lower)),
                   parent=world)

irvb_foil.observe()

if irvb_foil.units.lower() == "power":
    brightness = irvb_foil.pipelines[0].frame.mean / etendue * 4 * np.pi
else:
    brightness = irvb_foil.pipelines[0].frame.mean * radiance_scale_factor * 4 * np.pi

fig, ax = plt.subplots()
image = ax.imshow(brightness.T, interpolation="none")
ax.set_xlabel("Pixel column")
ax.set_ylabel("Pixel row")
ax.set_title("Brightness for hollow profile [W/m2]")
fig.colorbar(image)

plt.ioff()

input("Hit return to finish")
