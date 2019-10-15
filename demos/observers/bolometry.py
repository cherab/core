import math
import matplotlib.pyplot as plt
import numpy as np

from raysect.core import Node, Point2D, Point3D, Vector3D, rotate_basis, translate
from raysect.optical import World
from raysect.optical.material import AbsorbingSurface, VolumeTransform
from raysect.primitive import Box, Cylinder, Subtract

from cherab.core.math import AxisymmetricMapper
from cherab.tools.emitters import RadiationFunction
from cherab.tools.observers.bolometry import BolometerCamera, BolometerSlit, BolometerFoil


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
SLIT_SENSOR_SEPARATION = 0.1
FOIL_SEPARATION = 0.00508  # 0.2 inch between foils


def _point3d_to_rz(point):
    """
    Convert a Point3D from (x, y, z) to a Point2D in the (r, z) plane.
    """
    r = math.hypot(point.x, point.y)
    z = point.z
    return Point2D(r, z)


def make_bolometer_camera():
    """
    Build a simple bolometer camera.

    The camera consists of a box with a rectangular slit and 4 foils.
    In its local coordinate system, the camera's slit is located at the
    origin and the foils below the z=0 plane, looking up towards the slit.
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
    # Instance of the bolometer camera
    bolometer_camera = BolometerCamera(camera_geometry=camera_box)
    # The bolometer slit in this instance just contains targeting information
    # for the ray tracing, since we have already given our camera a geometry
    # The slit is defined in the local coordinate system of the camera
    slit = BolometerSlit(slit_id="Example slit", centre_point=ORIGIN,
                         basis_x=XAXIS, dx=SLIT_WIDTH, basis_y=YAXIS, dy=SLIT_HEIGHT,
                         parent=bolometer_camera, csg_aperture=False)
    # 4 bolometer foils, spaced at equal intervals along the local X axis
    # The bolometer positions and orientations are given in the local coordinate
    # system of the camera, just like the slit
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
    return bolometer_camera


def make_radiating_plasma():
    """
    Produce a simple radiating plasma.

    The plasma will be a cylindrical plasma which emits with a constant
    emissivity of 1 W/m3 in an annular ring.

    See the RadiationFunction example for more details of how this is
    set up.
    """
    MAJOR_RADIUS = 1
    MINOR_RADIUS = 0.5
    CENTRE_Z = -0.5
    CYLINDER_HEIGHT = 5
    EMISSIVITY = 1

    def emission_function(r, z):
        if (r - MAJOR_RADIUS)**2 + (z - CENTRE_Z)**2 < MINOR_RADIUS**2:
            return EMISSIVITY
        return 0

    emitter = Cylinder(radius=MAJOR_RADIUS + MINOR_RADIUS, height=CYLINDER_HEIGHT,
                       transform=translate(0, 0, CENTRE_Z - CYLINDER_HEIGHT / 2))
    emission_function_3d = AxisymmetricMapper(emission_function)
    emitting_material = VolumeTransform(RadiationFunction(emission_function_3d),
                                        transform=emitter.transform.inverse())
    emitter.material = emitting_material
    return emitter


def observe_radiation():
    """
    Add a radiating plasma and measure it with a bolometer camera.
    """
    world = World()
    bolometer = make_bolometer_camera()
    bolometer.transform = translate(1, 0, 1.5) * rotate_basis(-ZAXIS, YAXIS)
    bolometer.parent = world
    emitter = make_radiating_plasma()
    emitter.parent = world
    for foil in bolometer:
        foil.observe()
        if foil.units == "Power":
            emitter.parent = None
            etendue = foil.calculate_etendue()[0]
            emitter.parent = world
            radiance = foil.pipelines[0].value.mean / etendue * 4 * math.pi
        else:
            radiance = foil.pipelines[0].value.mean
        print("Measured radiance for {0}: {1:.4g}".format(foil.name, radiance))

def plot_bolometer_los():
    world = World()
    bolometer = make_bolometer_camera()
    bolometer.transform = translate(1, 0, 1.5) * rotate_basis(-ZAXIS, YAXIS)
    bolometer.parent = world
    # Put a plane at the origin to see where the LOS cross it
    Box(lower=Point3D(-10, -10, -0.1), upper=Point3D(10, 10, 0),
        material=AbsorbingSurface(), name="z=0 plane", parent=world)
    fig, ax = plt.subplots()
    for foil in bolometer:
        slit_centre = foil.slit.centre_point
        slit_centre_rz = _point3d_to_rz(slit_centre)
        ax.plot(slit_centre_rz[0], slit_centre_rz[1], 'ko')
        origin, hit, _ = foil.trace_sightline()
        centre_rz = _point3d_to_rz(foil.centre_point)
        ax.plot(centre_rz[0], centre_rz[1], 'kx')
        origin_rz = _point3d_to_rz(origin)
        hit_rz = _point3d_to_rz(hit)
        ax.plot([origin_rz[0], hit_rz[0]], [origin_rz[1], hit_rz[1]], 'k')
    ax.axis('equal')
    plt.show()


def calculate_etendue():
    """
    Demonstrate raytracing calculation of bolometer etendue vs analytical.
    """
    world = World()
    bolometer = make_bolometer_camera()
    bolometer.parent = world
    for foil in bolometer:
        raytraced_etendue, raytraced_error = foil.calculate_etendue()
        Adet = foil.x_width * foil.y_width
        Aslit = foil.slit.dx * foil.slit.dy
        costhetadet = foil.sightline_vector.normalise().dot(foil.normal_vector)
        costhetaslit = foil.sightline_vector.normalise().dot(foil.slit.normal_vector)
        distance = foil.centre_point.vector_to(foil.slit.centre_point).length
        analytic_etendue = Adet * Aslit * costhetadet * costhetaslit / distance**2
        print("{} raytraced etendue: {:.4g} +- {:.4g} analytic: {:.4g}".format(
            foil.name, raytraced_etendue, raytraced_error, analytic_etendue)
        )


def main():
    # plot_bolometer_los()
    # calculate_etendue()
    observe_radiation()


if __name__ == "__main__":
    main()
