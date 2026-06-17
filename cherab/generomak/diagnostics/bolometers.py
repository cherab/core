"""
Some foil bolometers for measuring total radiated power.
"""
from raysect.core import (Node, Point3D, Vector3D, rotate_basis,
                          rotate_x, rotate_y, rotate_z, translate)
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
BOX_WIDTH = 0.1
BOX_HEIGHT = 0.07
BOX_DEPTH = 0.2
THICKNESS = 1e-3
SLIT_WIDTH = 0.004
SLIT_HEIGHT = 0.005
FOIL_WIDTH = 0.0013
FOIL_HEIGHT = 0.0038
FOIL_CORNER_CURVATURE = 0.0005
FOIL_SEPARATION = 0.00508  # 0.2 inch between foils


def _make_bolometer_camera(slit_sensor_separation, sensor_angles, sensor_rotations):
    """
    Build a single bolometer camera.

    The camera consists of a box with a rectangular slit and 4 sensors,
    each of which has 4 foils.

    In its local coordinate system, the camera's slit is located at the
    origin with its width along the X axis and its height along the y
    axis, and the sensors are below the z=0 plane looking up towards the
    slit.

    The sensors are rotated by sensor_angles about the y axis to form a
    fan, and by sensor_rotations about the axis defined by the line
    between the slit and the sensor. A rotation of 180 degrees flips
    the sensor upside down and therefore reverses the spatial ordering
    of lines of sight relative to a rotation of 0 degrees.
    """
    camera_box = Box(lower=Point3D(-BOX_WIDTH / 2, -BOX_HEIGHT / 2, -BOX_DEPTH),
                     upper=Point3D(BOX_WIDTH / 2, BOX_HEIGHT / 2, 0))
    # Hollow out the box: it has 1 mm thick walls.
    inside_box = Box(lower=camera_box.lower + Vector3D(THICKNESS, THICKNESS, THICKNESS),
                     upper=camera_box.upper - Vector3D(THICKNESS, THICKNESS, THICKNESS))
    camera_box = Subtract(camera_box, inside_box)
    # The slit is a hole in the box. Make it thicker than the wall.
    aperture = Box(lower=Point3D(-SLIT_WIDTH / 2, -SLIT_HEIGHT / 2, -1.1 * THICKNESS),
                   upper=Point3D(SLIT_WIDTH / 2, SLIT_HEIGHT / 2, 0.1 * THICKNESS))
    camera_box = Subtract(camera_box, aperture)
    camera_box.material = AbsorbingSurface()
    bolometer_camera = BolometerCamera(camera_geometry=camera_box)
    # The bolometer slit in this instance just contains targeting information
    # for the ray tracing, since we have already given our camera a geometry
    # The slit is defined in the local coordinate system of the camera
    slit = BolometerSlit(slit_id="Example slit", centre_point=ORIGIN,
                         basis_x=XAXIS, dx=SLIT_WIDTH, basis_y=YAXIS, dy=SLIT_HEIGHT,
                         parent=bolometer_camera)
    for j, (angle, rotation) in enumerate(zip(sensor_angles, sensor_rotations)):
        # 4 bolometer foils, spaced at equal intervals along the local X axis
        sensor = Node(name="Bolometer sensor", parent=bolometer_camera)
        sensor.transform = (
            rotate_y(angle)
            * rotate_z(rotation)
            * translate(0, 0, -slit_sensor_separation)
        )
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

    The Generomak bolometer diagnostic consists of multiple 16-channel
    cameras. Each camera has 4 4-channel sensors inside.

    * 2 cameras are located at the midplane with purely-poloidal,
      horizontal views.
    * 1 camera is located at the top of the machine with purely-poloidal,
      vertical views.
    * 2 cameras have purely tangential views at the midplane.
    * 1 camera has combined poloidal+tangential views, which look like
      curved lines of sight in the poloidal plane. It looks at the lower
      divertor.

    Channel ordering is as follows:
    * Poloidal channels are ordered anti-clockwise by line-of-sight:
      channel 1 of HozPol1 views the top of the machine and channel 16
      HozPol2 views the bottom of the machine. Similarly, channel 1 of
      VertPol views the high field side and channel 16 views the low
      field side.
    * Tangential channels are ordered by increasing tangency radius:
      channel 1 of TanMid1 has its tangency radius on the high field
      side and channel 16 has its tangency radius on the low field side.
    * The combined tangential/poloidal channels follow both conventions:
      channel 1 views the high field side and channel 16 views the low
      field side.

    :param parent: the scenegraph node the bolometers will belong to.
    :return: a list of BolometerCamera instances, one for each of the
             cameras described above.
    """
    # The coordinate system conventions are as follows. All angles are in
    # degrees and increase clockwise when viewing along the relevant axes:
    # y axis for poloidal rotation, z axis for toroidal rotation and x axis
    # for radial rotation.
    # - rotation_poloidal: viewing angle of the slit in the poloidal plane,
    #                      with 0 being horizontally inwards.
    # - rotation_toroidal: viewing angle of the slit in the toroidal plane,
    #                      with 0 being purely radial.
    # - rotation_radial: rotation about the radial axis, 0 being vertically upwards.
    # - origin: position of the slit relative to the (x, z) poloidal plane i.e. y=0.
    # - slit_sensor_separation: distance between slit and each 4-channel sensor.
    # - sensor_angles: angle between slit normal and sensor normal.
    # - sensor_rotations: rotation angle about the slit-sensor vector, enables
    #                     reversing the order of lines of sight spatially within
    #                     each sensor.
    # - toroidal_angle: the angle of the poloidal plane in which the origin is
    #                   definied, with 0 being the (x, z) plane.
    camera_properties = {
        'HozPol1': {},  # Horizontal poloidal
        'HozPol2': {},  # Horizontal poloidal,
        'VertPol': {},  # Vertical poloidal
        'TanMid1': {},  # Tangential
        'TanPol1': {}   # Combined poloidal/tangential
    }
    # poloidal rotations
    camera_properties['HozPol1']['rotation_poloidal'] = 30
    camera_properties['HozPol2']['rotation_poloidal'] = -30
    camera_properties['VertPol']['rotation_poloidal'] = -90
    camera_properties['TanMid1']['rotation_poloidal'] = 0
    camera_properties['TanPol1']['rotation_poloidal'] = -25
    # toroidal rotation
    camera_properties['HozPol1']['rotation_toroidal'] = 0
    camera_properties['HozPol2']['rotation_toroidal'] = 0
    camera_properties['VertPol']['rotation_toroidal'] = 0
    camera_properties['TanMid1']['rotation_toroidal'] = -40
    camera_properties['TanPol1']['rotation_toroidal'] = 40
    # radial rotation
    camera_properties['HozPol1']['rotation_radial'] = -90
    camera_properties['HozPol2']['rotation_radial'] = -90
    camera_properties['VertPol']['rotation_radial'] = -90
    camera_properties['TanMid1']['rotation_radial'] = 0
    camera_properties['TanPol1']['rotation_radial'] = 0
    # origins relative to the poloidal (x, z) plane
    camera_properties['HozPol1']['origin'] = Point3D(2.45, 0.05, 0)
    camera_properties['HozPol2']['origin'] = Point3D(2.45, -0.05, 0)
    camera_properties['VertPol']['origin'] = Point3D(1.3, 0, 1.42)
    camera_properties['TanMid1']['origin'] = Point3D(2.5, 0, 0)
    camera_properties['TanPol1']['origin'] = Point3D(2.2, 0, -0.8)
    # slit-sensor separations
    camera_properties['HozPol1']['slit_sensor_separation'] = 0.08
    camera_properties['HozPol2']['slit_sensor_separation'] = 0.08
    camera_properties['VertPol']['slit_sensor_separation'] = 0.05
    camera_properties['TanMid1']['slit_sensor_separation'] = 0.1
    camera_properties['TanPol1']['slit_sensor_separation'] = 0.15
    # sensor angles relative to the slit
    camera_properties['HozPol1']['sensor_angles'] = [22.5, 7.5, -7.5, -22.5]
    camera_properties['HozPol2']['sensor_angles'] = [22.5, 7.5, -7.5, -22.5]
    camera_properties['VertPol']['sensor_angles'] = [36, 12, -12, -36]
    camera_properties['TanMid1']['sensor_angles'] = [18, 6, -6, -18]
    camera_properties['TanPol1']['sensor_angles'] = [-12, -4, 4, 12]
    # sensor rotation relative to the slit
    camera_properties['HozPol1']['sensor_rotations'] = [0, 0, 0, 0]
    camera_properties['HozPol2']['sensor_rotations'] = [0, 0, 0, 0]
    camera_properties['VertPol']['sensor_rotations'] = [0, 0, 0, 0]
    camera_properties['TanMid1']['sensor_rotations'] = [0, 0, 0, 0]
    camera_properties['TanPol1']['sensor_rotations'] = [180, 180, 180, 180]
    # toroidal angles about which to rotate the poloidal plane
    camera_properties['HozPol1']['toroidal_angle'] = 10  # need to avoid LFS limiters
    camera_properties['HozPol2']['toroidal_angle'] = 10  # need to avoid LFS limiters
    camera_properties['VertPol']['toroidal_angle'] = 0  # happy to hit LFS limiters
    camera_properties['TanMid1']['toroidal_angle'] = -15  # avoid LFS limiters
    camera_properties['TanPol1']['toroidal_angle'] = 15  # avoid LFS limiters

    cameras = []
    for name, prop in camera_properties.items():
        camera = _make_bolometer_camera(
            prop['slit_sensor_separation'],
            prop['sensor_angles'],
            prop['sensor_rotations'],
        )
        # The transform is applied as follows:
        # 1. Point the camera along the inward radial direction in the (x, z) plane.
        # 2. Make the poloidal, radial and toroidal rotations while the camera is at
        #    the origin.
        # 3. Move the camera to its position relative to the (x, z) plane.
        # 4. Rotate the (x, z) plane to the correct toroidal angle.
        # Transforms are applied right-to-left (or bottom-to-top with one per line):
        camera.transform = (
            rotate_z(prop['toroidal_angle'])
            * translate(prop['origin'].x, prop['origin'].y, prop['origin'].z)
            * rotate_z(prop['rotation_toroidal'])
            * rotate_y(prop['rotation_poloidal'])
            * rotate_x(prop['rotation_radial'])
            * rotate_basis(-XAXIS, ZAXIS)
        )
        camera.parent = parent
        camera.name = name
        cameras.append(camera)
    return cameras
