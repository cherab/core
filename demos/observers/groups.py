# Copyright 2014-2021 United Kingdom Atomic Energy Authority
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
Simple group observer use demonstartion. Includes also example use of plotting routines.
"""

import numpy as np
import matplotlib.pyplot as plt

from raysect.core.math import Point3D, Vector3D, rotate_z, rotate_basis, translate
from raysect.optical import World
from raysect.optical.observer import SpectralRadiancePipeline0D, FibreOptic

from cherab.core.model import ExcitationLine, RecombinationLine
from cherab.core.atomic import Line, hydrogen
from cherab.openadas import OpenADAS
from cherab.generomak.machine import load_first_wall
from cherab.tools.observers import FibreOpticGroup
from cherab.tools.observers.group.plotting import plot_group_spectra, plot_group_total
from cherab.generomak.plasma import get_edge_plasma

###############################################################################
# Load the simulation and create a plasma object from it.
###############################################################################
plasma = get_edge_plasma()

# Adding H-alpha excitation and recombination models
plasma.atomic_data = OpenADAS(permit_extrapolation=True)
h_alpha = Line(hydrogen, 0, (3, 2))
plasma.models = [ExcitationLine(h_alpha), RecombinationLine(h_alpha)]


###############################################################################
# Observe the plasma with a group of optical fibres.
###############################################################################
world = World()
plasma.parent = world

# Load the generomak first wall.
load_first_wall(world)

# Create a group of optical fibres observing the divertor.
group = FibreOpticGroup(parent=world, name='Divertor Fibre Optic Array')
group.transform = rotate_z(22.5)
origin = Point3D(2.3, 0, 1.25)
angles = [-63.8, -66.5, -69.2, -71.9, -74.6]
direction_r = -np.cos(np.deg2rad(angles))
direction_z = np.sin(np.deg2rad(angles))
for i in range(len(angles)):
    trans = translate(*origin)
    rot = rotate_basis(Vector3D(direction_r[i], 0, direction_z[i]), Vector3D(0, 1, 0))
    fibre = FibreOptic(name='{}'.format(i + 1), transform=trans*rot)
    group.add_observer(fibre)
group.connect_pipelines([SpectralRadiancePipeline0D], [{'name': 'SpectralRadiance'}])

# Set observer parameters for all observers in group
group.acceptance_angle = 1.4
group.radius = 0.001
group.pixel_samples = 5000
group.min_wavelength = 655.5
group.max_wavelength = 656.9
group.spectral_bins = 256

# Observe.
print('Observing plasma...')
group.observe()

###############################################################################
# Plot results using the plotting functions for groups
###############################################################################

plt.ion()
plot_group_spectra(group, item='SpectralRadiance', in_photons=True)
plot_group_total(group, item='SpectralRadiance')
plt.ioff()
plt.show()
