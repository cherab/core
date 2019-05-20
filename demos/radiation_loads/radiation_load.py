
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from raysect.core import Point2D, Point3D, translate, Vector3D, rotate_basis
from raysect.core.math import Interpolator2DMesh
from raysect.optical import World, Spectrum
from raysect.primitive import Cylinder
from raysect.optical.observer import PinholeCamera, PowerPipeline2D

from cherab.core.math import sample2d, AxisymmetricMapper
from cherab.tools.emitters import RadiationFunction


plt.ion()

#############################
# define radiation function #

plasma_axis = Point2D(2.5, 0)
lcfs_radius = 1
ring_radius = 0.5

peak = 1
centre_peak_width = 0.05
ring_width = 0.025


def rad_function(r, z):

    sample_point = Point2D(r, z)
    direction = plasma_axis.vector_to(sample_point)
    bearing = np.arctan2(direction.y, direction.x)

    # calculate radius of coordinate from magnetic axis
    radius_from_axis = direction.length
    closest_ring_point = plasma_axis + (direction.normalise() * 0.5)
    radius_from_ring = sample_point.distance_to(closest_ring_point)

    # evaluate pedestal-> core function
    if radius_from_axis <= lcfs_radius:

        central_radiatior = peak * np.exp(-(radius_from_axis**2) / centre_peak_width)

        ring_radiator = peak * np.cos(bearing) * np.exp(-(radius_from_ring**2) / ring_width)
        ring_radiator = max(0, ring_radiator)

        return central_radiatior + ring_radiator
    else:
        return 0


####################
# 2D mesh creation #

# make a triangular mesh in the r-z plane

num_vertical_points = 100
vertical_points = np.linspace(-2, 2, num_vertical_points)
num_radial_points = 30
radial_points = np.linspace(0, 4, num_radial_points)

vertex_coords = np.empty((num_vertical_points * num_radial_points, 2))
for i in range(num_radial_points):
    for j in range(num_vertical_points):
        index = i * num_vertical_points + j
        vertex_coords[index, 0] = radial_points[i]
        vertex_coords[index, 1] = vertical_points[j]

# perform Delaunay triangulation to produce triangular mesh
triangles = Delaunay(vertex_coords).simplices

# sample our radiation function at the mesh vertices
vertex_powers = np.array([rad_function(r, z) for r, z in vertex_coords])


#################################
# add radiation source to world #

world = World()

rad_function_3d = AxisymmetricMapper(Interpolator2DMesh(vertex_coords, vertex_powers, triangles, limit=False))
radiation_emitter = RadiationFunction(rad_function_3d, vertical_offset=-1)

geom = Cylinder(4, 2, transform=translate(0, 0, -1), parent=world, material=radiation_emitter)


######################
# visualise emission #

# run some plots to check the distribution functions and emission profile are as expected
r, z, t_samples = sample2d(rad_function, (0, 4, 200), (-2, 2, 200))
plt.imshow(np.transpose(np.squeeze(t_samples)), extent=[0, 4, -2, 2])
plt.colorbar()
plt.axis('equal')
plt.xlabel('r axis')
plt.ylabel('z axis')
plt.title("Radiation profile in r-z plane")


camera = PinholeCamera((256, 256), pipelines=[PowerPipeline2D()], parent=world)
camera.transform = translate(-3.5, -2.5, 0)*rotate_basis(Vector3D(1, 0, 0), Vector3D(0, 0, 1))
camera.pixel_samples = 1

plt.ion()
camera.observe()
plt.ioff()
plt.show()



