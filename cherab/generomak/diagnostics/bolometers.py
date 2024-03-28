"""
Some foil bolometers for measuring total radiated power.
"""
from raysect.core import (Node, Point3D, Vector3D, rotate_basis,
                          rotate_x, rotate_y, rotate_z, translate)
from raysect.core.math import rotate
from raysect.optical.material import AbsorbingSurface
from raysect.primitive import Box, Subtract

from cherab.tools.observers import BolometerCamera, BolometerSlit, BolometerFoil


# Convenient constants
XAXIS = Vector3D(1, 0, 0)
YAXIS = Vector3D(0, 1, 0)
ZAXIS = Vector3D(0, 0, 1)
ORIGIN = Point3D(0, 0, 0)
# Bolometer geometry, independent of camera.
BOX_WIDTH = 0.05
BOX_WIDTH = 0.2
BOX_HEIGHT = 0.07
BOX_DEPTH = 0.2
SLIT_WIDTH = 0.004
SLIT_HEIGHT = 0.005
FOIL_WIDTH = 0.0013
FOIL_HEIGHT = 0.0038
FOIL_CORNER_CURVATURE = 0.0005
FOIL_SEPARATION = 0.00508  # 0.2 inch between foils


def _make_bolometer_camera(slit_sensor_separation, sensor_angles):
    """
    Build a single bolometer camera.

    The camera consists of a box with a rectangular slit and 4 sensors,
    each of which has 4 foils.
    In its local coordinate system, the camera's slit is located at the
    origin and the sensors below the z=0 plane, looking up towards the slit.
    """
    camera_box = Box(lower=Point3D(-BOX_WIDTH / 2, -BOX_HEIGHT / 2, -BOX_DEPTH),
                     upper=Point3D(BOX_WIDTH / 2, BOX_HEIGHT / 2, 0))
    # Hollow out the box
    inside_box = Box(lower=camera_box.lower + Vector3D(1e-5, 1e-5, 1e-5),
                     upper=camera_box.upper - Vector3D(1e-5, 1e-5, 1e-5))
    camera_box = Subtract(camera_box, inside_box)
    # The slit is a hole in the box
    aperture = Box(lower=Point3D(-SLIT_WIDTH / 2, -SLIT_HEIGHT / 2, -1e-4),
                   upper=Point3D(SLIT_WIDTH / 2, SLIT_HEIGHT / 2, 1e-4))
    camera_box = Subtract(camera_box, aperture)
    camera_box.material = AbsorbingSurface()
    bolometer_camera = BolometerCamera(camera_geometry=camera_box)
    # The bolometer slit in this instance just contains targeting information
    # for the ray tracing, since we have already given our camera a geometry
    # The slit is defined in the local coordinate system of the camera
    slit = BolometerSlit(slit_id="Example slit", centre_point=ORIGIN,
                         basis_x=XAXIS, dx=SLIT_WIDTH, basis_y=YAXIS, dy=SLIT_HEIGHT,
                         parent=bolometer_camera)
    for j, angle in enumerate(sensor_angles):
        # 4 bolometer foils, spaced at equal intervals along the local X axis
        sensor = Node(name="Bolometer sensor", parent=bolometer_camera,
                      transform=rotate_y(angle) * translate(0, 0, -slit_sensor_separation))
        for i, shift in enumerate([-1.5, -0.5, 0.5, 1.5]):
            # Note that the foils will be parented to the camera rather than the
            # sensor, so we need to define their transform relative to the camera.
            foil_transform = sensor.transform * translate(shift * FOIL_SEPARATION, 0, 0)
            foil = BolometerFoil(detector_id="Foil {} sensor {}".format(i + 1, j + 1),
                                 centre_point=ORIGIN.transform(foil_transform),
                                 basis_x=XAXIS.transform(foil_transform), dx=FOIL_WIDTH,
                                 basis_y=YAXIS.transform(foil_transform), dy=FOIL_HEIGHT,
                                 slit=slit, parent=bolometer_camera, units="Power",
                                 accumulate=False, curvature_radius=FOIL_CORNER_CURVATURE)
            bolometer_camera.add_foil_detector(foil)
    return bolometer_camera


def load_bolometers(parent=None):
    """
    Load the Generomak bolometers.

    The Generomak bolometer diagnostic consists of multiple 12-channel
    cameras. Each camera has 3 4-channel sensors inside.

    * 2 cameras are located at the midplane with purely-poloidal,
      horizontal views.
    * 1 camera is located at the top of the machine with purely-poloidal,
      vertical views.
    * 2 cameras have purely tangential views at the midplane.
    * 1 camera has combined poloidal+tangential views, which look like
      curved lines of sight in the poloidal plane. It looks at the lower
      divertor.

    :param parent: the scenegraph node the bolometers will belong to.
    :return: a list of BolometerCamera instances, one for each of the
             cameras described above.
    """
    poloidal_camera_rotations = [
        30, -30,  # Horizontal poloidal,
        -90,  # Vertical poloidal,
        0, # Tangential,
        0,  # Combined poloidal/tangential,
    ]
    toroidal_camera_rotations = [
        0, 0,  # Horizontal poloidal
        0,  # Vertical poloidal
        -40, # Tangential
        40,  # Combined poloidal/tangential
    ]
    radial_camera_rotations = [
        0, 0,  # Horizontal poloidal
        0,  # Vertical poloidal
        90,  # Tangential
        90,  # Combined poloidal/tangential
    ]
    camera_origins = [
        Point3D(2.5, 0.05, 0), Point3D(2.5, -0.05, 0),  # Horizontal poloidal
        Point3D(1.3, 0, 1.42),  # Vertical poloidal
        Point3D(2.5, 0, 0),  # Midplane tangential horizontal
        Point3D(2.5, 0, -0.2),  # Combined poloidal/tangential
    ]
    slit_sensor_separations = [
        0.08, 0.08,  # Horizontal poloidal
        0.05,  # Vertical poloidal
        0.1, # Tangential
        0.1, # Combined poloidal/tangential
    ]
    all_sensor_angles = [
        [-22.5, -7.5, 7.5, 22.5], [-22.5, -7.5, 7.5, 22.5],  # Horizontal poloidal
        [-36, -12, 12, 36],  # Vertical poloidal
        [-18, -6, 6, 18],  # Tangential
        [-18, -6, 6, 18],  # Combined poloidal/tangential
    ]
    toroidal_angles = [
        10, 10,  # Horizontal poloidal, need to avoid LFS limiters.
        0,  # Vertical poloidal, happy to hit LFS limiters.
        -15,  # Tangential, avoid LFS limiters.
        15,  # Combined poloidal/tangential, avoid LFS limiters.
    ]
    names = [
        'HozPol1', 'HozPol2',
        'VertPol',
        'TanMid1',
        'TanPol1',
    ]
    cameras = []
    # FIXME: this for loop definition is really ugly!
    for (
            poloidal_rotation, toroidal_rotation, radial_rotation, camera_origin,
            slit_sensor_separation, sensor_angles, toroidal_angle, name
    ) in zip(
        poloidal_camera_rotations, toroidal_camera_rotations, radial_camera_rotations,
        camera_origins,
        slit_sensor_separations, all_sensor_angles, toroidal_angles, names
    ):
        camera = _make_bolometer_camera(slit_sensor_separation, sensor_angles)
        # FIXME: work out how to combine tangential and poloidal rotations.
        camera.transform = (
            rotate_z(toroidal_angle)
            * translate(camera_origin.x, camera_origin.y, camera_origin.z)
            * rotate_z(toroidal_rotation)
            * rotate_x(radial_rotation)
            * rotate_y(poloidal_rotation + 90)
            * rotate_basis(-ZAXIS, YAXIS)
        )
        camera.parent = parent
        camera.name = "{} at angle {}".format(name, poloidal_rotation)
        cameras.append(camera)
    return cameras
