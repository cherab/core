# Copyright 2016-2018 Euratom
# Copyright 2016-2018 United Kingdom Atomic Energy Authority
# Copyright 2016-2018 Centro de Investigaciones Energéticas, Medioambientales y Tecnológicas
#
# Licensed under the EUPL, Version 1.1 or – as soon they will be approved by the
# European Commission - subsequent versions of the EUPL (the "Licence");
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at:
#
# https://joinup.ec.europa.eu/software/page/eupl5
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the Licence is distributed on an "AS IS" basis, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied.
#
# See the Licence for the specific language governing permissions and limitations
# under the Licence.

"""
Demonstration of RayTransferCylinder in axisymmetrical case and applied voxel map
---------------------------------------------------------------------------------

This file will demonstrate how to:

 * calculate ray transfer matrix (geometry matrix) for axisymmetrical cylindrical emitter defined
   on a regular grid
 * map multiple grid cells to a single light source by applying a voxel map

"""
import os
import numpy as np
from matplotlib import pyplot as plt

from raysect.primitive import Cylinder, Subtract
from raysect.optical import World, translate, rotate
from raysect.optical.observer import PinholeCamera, FullFrameSampler2D

from cherab.tools.raytransfer import RayTransferPipeline2D, RayTransferCylinder
from cherab.tools.raytransfer import RoughNickel

world = World()

# creating the scene
cylinder_inner = Cylinder(radius=80., height=140.)
cylinder_outer = Cylinder(radius=220., height=140.)
wall = Subtract(cylinder_outer, cylinder_inner, material=RoughNickel(0.1), parent=world,
                transform=translate(0, 0, -70.))

# creating ray transfer cylinder with 200 (m) outer radius, 100 (m) inner radius, 140 (m) height
# for axisymmetric cylindrical emission profile defined on a 100 x 100 (R, Z) gird
rtc = RayTransferCylinder(200., 100., 100, 100, radius_inner=100.)
# n_polar=0 by default (axisymmetric case)
rtc.parent = world
rtc.transform = translate(0, 0, -50.)

# unlike the demo with a mask, here we not only cut out a circle but also
# create 50 ring-shaped light sources using the voxel map
rad_circle = 50.
xsqr = np.linspace(-49.5, 49.5, 100) ** 2
rad = np.sqrt(xsqr[:, None] + xsqr[None, :])
voxel_map = np.zeros((100, 100), dtype=np.int32)
voxel_map[rad > 50.] = -1  # removing the area outside the circle
for i in range(50):
    voxel_map[(rad < i + 1.) * (rad > i)] = i  # mapping multiple grid cells to a single light source
rtc.voxel_map = voxel_map[:, None, :]  # making 3D voxel map from 2D (RZ-plane) voxel map and applying it
# now we have only 50 light sources

# creating ray transfer pipeline
pipeline = RayTransferPipeline2D()

# setting up the camera
camera = PinholeCamera((256, 256), pipelines=[pipeline], frame_sampler=FullFrameSampler2D(),
                       transform=translate(219., 0, 0) * rotate(90., 0., -90.), parent=world)
camera.fov = 90
camera.pixel_samples = 500
camera.min_wavelength = 500.
camera.max_wavelength = camera.min_wavelength + 1.
camera.spectral_bins = rtc.bins

# starting ray tracing
camera.observe()

# uncomment this to save ray transfer matrix to file
# np.save('ray_transfer_map.npy', pipeline.matrix)

# let's collapse the ray transfer matrix with some emission profiles

# obtaining 30 images for 30 emission profiles
images = []
vmax = 0
rad_inside = np.linspace(0.5, 49.5, 50)  # we have only 50 light sources now
shifts = np.linspace(0, 2. / 3., 30, endpoint=False)
for shift in shifts:
    profile = ((1. + np.cos(3. * np.pi * (rad_inside / rad_circle - shift))) *
               np.exp(-rad_inside / rad_circle))
    image = np.dot(pipeline.matrix, profile)
    vmax = max(vmax, image.max())
    images.append(image)

# creating a folder for images
if not os.path.exists('images'):
    os.makedirs('images')

# now let's see all 30 ray-traced images obtained with only a single run of ray-tracing
fig = plt.figure(figsize=(4., 4.))
ax = fig.add_subplot(111)
fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
for i in range(30):
    ax.cla()
    ax.imshow(images[i].T, cmap='gray', vmax=0.75 * vmax)
    ax.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
    fig.savefig('images/ray_transfer_map_demo_%02d.png' % i, dpi=180)
    plt.pause(0.1)

# creating gif animation with ImageMagick
os.system("convert -delay 10 -loop 0 images/ray_transfer_map_demo_*.png ray_transfer_map_demo.gif")
# compare with ray_transfer_mask_demo.gif
