
# Copyright 2016-2022 Euratom
# Copyright 2016-2022 United Kingdom Atomic Energy Authority
# Copyright 2016-2022 Centro de Investigaciones Energéticas, Medioambientales y Tecnológicas
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

import numpy as np
import matplotlib.pyplot as plt
from raysect.optical import World

from cherab.core.math import sample3d
from cherab.core.atomic import hydrogen, carbon
from cherab.tools.plasmas.slab import build_slab_plasma
from cherab.atomic import AtomicData


# make a slab plasma
world = World()
plasma = build_slab_plasma(peak_density=5e19, impurities=[(carbon, 6, 0.005)], parent=world)
plasma.atomic_data = AtomicData(permit_extrapolation=True)

####################
# Visualise Plasma #

h0 = plasma.composition.get(hydrogen, 0)
h1 = plasma.composition.get(hydrogen, 1)
c6 = plasma.composition.get(carbon, 6)

r, _, z, t_samples = sample3d(h1.distribution.effective_temperature, (-1, 2, 200), (0, 0, 1), (-1, 1, 200))
plt.imshow(np.transpose(np.squeeze(t_samples)), extent=[-1, 2, -1, 1])
plt.colorbar()
plt.axis('equal')
plt.xlabel('x axis')
plt.ylabel('z axis')
plt.title("Ion temperature profile in x-z plane")

plt.figure()
r, _, z, t_samples = sample3d(h1.distribution.effective_temperature, (0, 0, 1), (-1, 1, 200), (-1, 1, 200))
plt.imshow(np.transpose(np.squeeze(t_samples)), extent=[-1, 1, -1, 1])
plt.colorbar()
plt.axis('equal')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title("Ion temperature profile in y-z plane")

plt.figure()
r, _, z, t_samples = sample3d(h0.distribution.density, (-1, 2, 200), (0, 0, 1), (-1, 1, 200))
plt.imshow(np.transpose(np.squeeze(t_samples)), extent=[-1, 2, -1, 1])
plt.colorbar()
plt.axis('equal')
plt.xlabel('x axis')
plt.ylabel('z axis')
plt.title("Neutral Density profile in x-z plane")

plt.show()
