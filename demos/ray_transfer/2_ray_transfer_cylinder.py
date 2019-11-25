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
RayTransferCylinder Demonstration
---------------------------------

This file will demonstrate how to:

 * calculate ray transfer matrix (geometry matrix) for a cylindrical periodic emitter defined
   on a regular grid,
 * obtain images by collapsing calculated ray transfer matrix with various emission profiles.

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

# creating the scene, two cylinders act like a wall here
cylinder_inner = Cylinder(radius=100., height=200.)
cylinder_outer = Cylinder(radius=300., height=200.)
wall = Subtract(cylinder_outer, cylinder_inner, material=RoughNickel(0.1), parent=world,
                transform=translate(0, 0, -100.))

# creating ray transfer cylinder with 260 (m) outer radius, 140 (m) inner radius,
# 160 (m) height and 60 deg peroid for cylindrical periodic emission profile defined
# on a 12 x 16 x 16 (R, Phi, Z) gird
rtc = RayTransferCylinder(260., 160., 12, 16, radius_inner=140., n_polar=16, period=60.)
rtc.parent = world
rtc.transform = translate(0, 0, -80.)

# setting the integration step
rtc.step = 0.2

# creating ray transfer pipeline
pipeline = RayTransferPipeline2D()

# setting up the camera
camera = PinholeCamera((256, 256), pipelines=[pipeline], frame_sampler=FullFrameSampler2D(),
                       transform=translate(258.9415, 149.5, 0) * rotate(90., -30., -90.),
                       parent=world)
camera.fov = 90
camera.pixel_samples = 500
camera.min_wavelength = 500.
camera.max_wavelength = camera.min_wavelength + 1.
camera.spectral_bins = rtc.bins

# starting ray tracing
camera.observe()

# uncomment this to save ray transfer matrix to file
# np.save('ray_transfer_cylinder_matrix.npy', pipeline.matrix)

# let's collapse the ray transfer matrix with some emission profiles

# defining emission profiles on a 12 x 8 RZ grid
rz_profiles = [np.array([[0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0],
                         [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0],
                         [1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0],
                         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                         [0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0],
                         [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                         [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0]], dtype=np.float64).T,

               np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                         [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                         [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                         [1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0],
                         [1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0],
                         [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
                         [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0]], dtype=np.float64).T,

               np.array([[0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0],
                         [0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0],
                         [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
                         [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                         [0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0],
                         [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                         [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]], dtype=np.float64).T,

               np.array([[0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
                         [0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0],
                         [0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0],
                         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                         [1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1],
                         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                         [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                         [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]], dtype=np.float64).T]

# making 3d profiles from 2d profiles (RZ plane) and obtaining 16 different images
# for each profile by collapsing the ray transfer matrix with emission profiles
vmax = 0
images = []
for profile in rz_profiles:
    images_prof = []
    z_shifts = list(range(8, -1, -1)) + list(range(1, 8))
    for iphi, iz in enumerate(z_shifts):
        profile3d = np.zeros((12, 16, 16))
        profile3d[:, iphi, iz:iz + 8] = profile
        image = np.dot(pipeline.matrix, profile3d.flatten())
        vmax = max(vmax, image.max())
        images_prof.append(image)
    images.append(images_prof)

# creating a folder for images
if not os.path.exists('images'):
    os.makedirs('images')

# now let's see all 64 ray-traced images obtained with only a single run of ray-tracing
fig = plt.figure(figsize=(6., 6.))
for i in range(16):
    fig.clf()
    for j in range(4):
        ax = fig.add_subplot(2, 2, j + 1)
        ax.imshow(images[j][i].T, cmap='gray', vmax=0.15 * vmax)
        ax.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
    fig.subplots_adjust(left=0., right=1., bottom=0., top=1., wspace=0.01, hspace=0.01)
    fig.savefig('images/ray_transfer_cylinder_demo_%02d.png' % i, dpi=180)
    plt.pause(0.2)

# creating gif animation with ImageMagick
os.system("convert -delay 20 -loop 0 images/ray_transfer_cylinder_demo_*.png ray_transfer_cylinder_demo.gif")
