
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

"""
This demo does the same plots as the plot_2d_profiles.py demo, but samples the profiles from the Plasma objects.
"""

import numpy as np
from matplotlib.colors import SymLogNorm
from matplotlib import pyplot as plt

from cherab.core.math import sample3d
from cherab.core.atomic.elements import hydrogen, carbon

from cherab.generomak.plasma import get_core_plasma, get_edge_plasma, get_plasma


def plot_profiles(core_profile, edge_profile, full_profile, r_range, z_range, label):

    # Sample core profile on a regular grid
    _, _, _, core_profile_samples = sample3d(core_profile, r_range, (0, 0, 1), z_range)
    core_profile_samples = core_profile_samples.squeeze()

    # Sample edge profile on a regular grid
    _, _, _, edge_profile_samples = sample3d(edge_profile, r_range, (0, 0, 1), z_range)
    edge_profile_samples = edge_profile_samples.squeeze()

    # Sample blended profile on a regular grid
    _, _, _, full_profile_samples = sample3d(full_profile, r_range, (0, 0, 1), z_range)
    full_profile_samples = full_profile_samples.squeeze()

    fig = plt.figure(figsize=(9.5, 5.), tight_layout=True)

    vmax = full_profile_samples.max()
    linthresh = 0.01 * min(core_profile_samples.max(), edge_profile_samples.max())
    color_norm = SymLogNorm(linthresh, vmin=0, vmax=vmax)

    core_profile_samples[core_profile_samples == 0] = np.nan
    edge_profile_samples[edge_profile_samples == 0] = np.nan
    full_profile_samples[full_profile_samples == 0] = np.nan

    ax_core = fig.add_subplot(131)
    ax_core.imshow(core_profile_samples.T, extent=[r_range[0], r_range[1], z_range[0], z_range[1]], origin='lower', norm=color_norm, cmap='gnuplot')
    ax_core.text(0.99, 0.99, 'Core', ha='right', va='top', transform=ax_core.transAxes)
    ax_core.set_xlim(r_range[0], r_range[1])
    ax_core.set_ylim(z_range[0], z_range[1])
    ax_core.set_xlabel('R, m')
    ax_core.set_ylabel('Z, m')

    ax_edge = fig.add_subplot(132, sharex=ax_core, sharey=ax_core)
    img = ax_edge.imshow(edge_profile_samples.T, extent=[r_range[0], r_range[1], z_range[0], z_range[1]], origin='lower', norm=color_norm, cmap='gnuplot')
    ax_edge.text(0.99, 0.99, 'Edge', ha='right', va='top', transform=ax_edge.transAxes)
    ax_edge.set_xlabel('R, m')

    ax_blend = fig.add_subplot(133, sharex=ax_core, sharey=ax_core)
    img = ax_blend.imshow(full_profile_samples.T, extent=[r_range[0], r_range[1], z_range[0], z_range[1]], origin='lower', norm=color_norm, cmap='gnuplot')
    ax_blend.text(0.99, 0.99, 'Blended', ha='right', va='top', transform=ax_blend.transAxes)
    ax_blend.set_xlabel('R, m')

    fig.colorbar(img, label=label)

    return fig


# get Generomak core plasma
core_plasma = get_core_plasma()

# get Generomak edge plasma
edge_plasma = get_edge_plasma()

# get Generomak full plasma (blended core and edge profiles)
full_plasma = get_plasma()

# plasma domain
r_range = (0.78, 2.23, 290)
z_range = (-1.74, 1.49, 647)

# Plotting plasma profiles
plot_profiles(core_plasma.electron_distribution.density,
              edge_plasma.electron_distribution.density,
              full_plasma.electron_distribution.density,
              r_range, z_range, 'Electron density, m-3')

plot_profiles(core_plasma.electron_distribution.effective_temperature,
              edge_plasma.electron_distribution.effective_temperature,
              full_plasma.electron_distribution.effective_temperature,
              r_range, z_range, 'Electron temperature, eV')

plot_profiles(core_plasma.composition.get(hydrogen, 1).distribution.effective_temperature,
              edge_plasma.composition.get(hydrogen, 1).distribution.effective_temperature,
              full_plasma.composition.get(hydrogen, 1).distribution.effective_temperature,
              r_range, z_range, 'Ion temperature, eV')

for element, charges in ((hydrogen, (0, 1)), (carbon, (0, 1, 2, 3, 4, 5, 6))):
    for charge in charges:
        state_str = ' {}+'.format(charge) if charge else ' 0'
        plot_profiles(core_plasma.composition.get(element, charge).distribution.density,
                      edge_plasma.composition.get(element, charge).distribution.density,
                      full_plasma.composition.get(element, charge).distribution.density,
                      r_range, z_range, '{} density, m-3'.format(element.name + state_str))

plot_profiles(core_plasma.composition.get(hydrogen, 0).distribution.effective_temperature,
              edge_plasma.composition.get(hydrogen, 0).distribution.effective_temperature,
              full_plasma.composition.get(hydrogen, 0).distribution.effective_temperature,
              r_range, z_range, 'neutral hydrogen effective temperature, eV')

plot_profiles(core_plasma.composition.get(carbon, 0).distribution.effective_temperature,
              edge_plasma.composition.get(carbon, 0).distribution.effective_temperature,
              full_plasma.composition.get(carbon, 0).distribution.effective_temperature,
              r_range, z_range, 'neutral carbon effective temperature, eV')

plt.show()
