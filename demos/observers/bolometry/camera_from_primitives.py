"""
This example demonstrates building a simple bolometer camera where
the position of the camera itself is known, and the slits and foils
are defined relative to the camera. This could be the case when working
from design office drawings, for example.
"""
import math
import matplotlib.pyplot as plt

from raysect.core import Point2D, Point3D, Vector3D, Node, rotate_basis, translate
from raysect.primitive import Box, Subtract
from raysect.optical import World
from raysect.optical.material import AbsorbingSurface

from cherab.tools.observers import BolometerCamera, BolometerSlit, BolometerFoil

# Convenient constants
XAXIS = Vector3D(1, 0, 0)
YAXIS = Vector3D(0, 1, 0)
ZAXIS = Vector3D(0, 0, 1)
ORIGIN = Point3D(0, 0, 0)

# Bolometer geometry
BOX_WIDTH = 0.05
BOX_HEIGHT = 0.07
BOX_DEPTH = 0.2
SLIT_WIDTH = 0.004
SLIT_HEIGHT = 0.005
FOIL_WIDTH = 0.0013
FOIL_HEIGHT = 0.0038
FOIL_CORNER_CURVATURE = 0.0005
SLIT_SENSOR_SEPARATION = 0.02
FOIL_SEPARATION = 0.00508  # 0.2 inch between foils

world = World()

########################################################################
# Build a simple bolometer camera.
########################################################################

# The camera consists of a box with a rectangular slit and 4 foils.
# In its local coordinate system, the camera's slit is located at the
# origin and the foils below the z=0 plane, looking up towards the slit.

# To position the camera relative to its parent, set the `transform`
# property to produce the correct translation and rotation.
camera_box = Box(lower=Point3D(-BOX_WIDTH / 2, -BOX_HEIGHT / 2, -BOX_DEPTH),
                 upper=Point3D(BOX_WIDTH / 2, BOX_HEIGHT / 2, 0))
# Hollow out the box
outside_box = Box(lower=camera_box.lower - Vector3D(1e-5, 1e-5, 1e-5),
                  upper=camera_box.upper + Vector3D(1e-5, 1e-5, 1e-5))
camera_box = Subtract(outside_box, camera_box)
# The slit is a hole in the box
aperture = Box(lower=Point3D(-SLIT_WIDTH / 2, -SLIT_HEIGHT / 2, -1e-4),
               upper=Point3D(SLIT_WIDTH / 2, SLIT_HEIGHT / 2, 1e-4))
camera_box = Subtract(camera_box, aperture)
camera_box.material = AbsorbingSurface()
# Instance of the bolometer camera
bolometer_camera = BolometerCamera(camera_geometry=camera_box, parent=world,
                                   name="Demo camera")
# The bolometer slit in this instance just contains targeting information
# for the ray tracing, since we have already given our camera a geometry
# The slit is defined in the local coordinate system of the camera
slit = BolometerSlit(slit_id="Example slit", centre_point=ORIGIN,
                     basis_x=XAXIS, dx=SLIT_WIDTH, basis_y=YAXIS, dy=SLIT_HEIGHT,
                     parent=bolometer_camera)
# 4 bolometer foils, spaced at equal intervals along the local X axis
# The bolometer positions and orientations are given in the local coordinate
# system of the camera, just like the slit. All 4 foils are on a single
# sensor, so we define them relative to this sensor
sensor = Node(name="Bolometer sensor", parent=bolometer_camera,
              transform=translate(0, 0, -SLIT_SENSOR_SEPARATION))
# The foils are shifted relative to the centre of the sensor by -1.5, -0.5, 0.5 and 1.5
# times the foil-foil separation
for i, shift in enumerate([-1.5, -0.5, 0.5, 1.5]):
    foil_transform = translate(shift * FOIL_SEPARATION, 0, 0) * sensor.transform
    foil = BolometerFoil(detector_id="Foil {}".format(i + 1),
                         centre_point=ORIGIN.transform(foil_transform),
                         basis_x=XAXIS.transform(foil_transform), dx=FOIL_WIDTH,
                         basis_y=YAXIS.transform(foil_transform), dy=FOIL_HEIGHT,
                         slit=slit, parent=bolometer_camera, units="Power",
                         accumulate=False, curvature_radius=FOIL_CORNER_CURVATURE)
    bolometer_camera.add_foil_detector(foil)


# The camera is positioned at (x, y, z) = (1, 0, 0.5)m and looking straight down
bolometer_camera.transform = translate(1, 0, 0.5) * rotate_basis(-ZAXIS, YAXIS)

########################################################################
# Show the lines of sight of this camera, tracing them from the foils to
# a plane at z=0.4
########################################################################

Box(lower=Point3D(-10, -10, 0.39), upper=Point3D(10, 10, 0.4), parent=world,
    material=AbsorbingSurface(), name="Z=0.4 plane")


def _point3d_to_rz(point):
    return Point2D(math.hypot(point.x, point.y), point.z)


fig, ax = plt.subplots()
for foil in bolometer_camera:
    slit_centre = foil.slit.centre_point
    slit_centre_rz = _point3d_to_rz(slit_centre)
    ax.plot(slit_centre_rz[0], slit_centre_rz[1], 'ko')
    origin, hit, _ = foil.trace_sightline()
    centre_rz = _point3d_to_rz(foil.centre_point)
    ax.plot(centre_rz[0], centre_rz[1], 'kx')
    origin_rz = _point3d_to_rz(origin)
    hit_rz = _point3d_to_rz(hit)
    ax.plot([origin_rz[0], hit_rz[0]], [origin_rz[1], hit_rz[1]], 'k')
ax.set_xlabel("r")
ax.set_ylabel("z")
ax.axis('equal')
plt.show()
