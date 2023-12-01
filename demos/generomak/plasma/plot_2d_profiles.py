
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
This demo plots core, edge and blended Generomak 2D plasma profiles.
"""

import numpy as np
from matplotlib.colors import SymLogNorm
from matplotlib.collections import PolyCollection
from matplotlib import pyplot as plt

from cherab.core.math import sample2d
from cherab.core.utility import RecursiveDict

from cherab.generomak.equilibrium import load_equilibrium
from cherab.generomak.plasma.plasma import get_core_interpolators, load_edge_profiles, get_full_profiles


def plot_profiles(core_profile, edge_mesh, edge_data, full_profile, label):

    # get grid parameters
    vertex_coords = np.asarray(edge_mesh["vertex_coords"])
    triangles = np.asarray(edge_mesh["triangles"])
    rl, ru = (vertex_coords[:, 0].min(), vertex_coords[:, 0].max())
    zl, zu = (vertex_coords[:, 1].min(), vertex_coords[:, 1].max())
    nr = 288
    nz = 647

    edge_data = np.asarray(edge_data)

    # Sample core profile on a regular grid
    _, _, core_profile_samples = sample2d(core_profile, (rl, ru, nr), (zl, zu, nz))

    # Sample blended profile on a regular grid
    _, _, profile_samples = sample2d(full_profile, (rl, ru, nr), (zl, zu, nz))

    fig = plt.figure(figsize=(9.5, 5.), tight_layout=True)

    vmax = profile_samples.max()
    linthresh = 0.01 * min(core_profile_samples.max(), edge_data.max())
    color_norm = SymLogNorm(linthresh, vmin=0, vmax=vmax)

    core_profile_samples[core_profile_samples == 0] = np.nan
    profile_samples[profile_samples == 0] = np.nan

    ax_core = fig.add_subplot(131)
    ax_core.imshow(core_profile_samples.T, extent=[rl, ru, zl, zu], origin='lower', norm=color_norm, cmap='gnuplot')
    ax_core.text(0.99, 0.99, 'Core', ha='right', va='top', transform=ax_core.transAxes)
    ax_core.set_xlim(rl, ru)
    ax_core.set_ylim(zl, zu)
    ax_core.set_xlabel('R, m')
    ax_core.set_ylabel('Z, m')

    ax_edge = fig.add_subplot(132, sharex=ax_core, sharey=ax_core)
    collection = PolyCollection(vertex_coords[triangles], norm=color_norm, cmap='gnuplot')
    collection.set_array(edge_data)
    ax_edge.add_collection(collection)
    ax_edge.text(0.99, 0.99, 'Edge', ha='right', va='top', transform=ax_edge.transAxes)
    ax_edge.set_aspect(1)
    ax_edge.set_xlabel('R, m')

    ax_blend = fig.add_subplot(133, sharex=ax_core, sharey=ax_core)
    img = ax_blend.imshow(profile_samples.T, extent=[rl, ru, zl, zu], origin='lower', norm=color_norm, cmap='gnuplot')
    ax_blend.text(0.99, 0.99, 'Blended', ha='right', va='top', transform=ax_blend.transAxes)
    ax_blend.set_xlabel('R, m')

    fig.colorbar(img, label=label)

    return fig


# load Generomak equilibrium
equilibrium = load_equilibrium()

# load 1D core profiles, f(psi_norm)
core_profiles_1d = get_core_interpolators()

# load 2D edge profiles defined on a quadrilateral mesh
edge_data = load_edge_profiles()

# load 2D plasma profiles covering both core and edge regions
# see the source code for the get_full_profiles() to learn how to blend core and edge profiles using a mask function
full_profiles = get_full_profiles(equilibrium, core_profiles_1d)

# map core profiles to 2D using the equilibrium
core_profiles_2d = RecursiveDict()

core_profiles_2d["electron"]["temperature"] = equilibrium.map2d(core_profiles_1d["electron"]["f1d_temperature"])
core_profiles_2d["electron"]["density"] = equilibrium.map2d(core_profiles_1d["electron"]["f1d_density"])

for element, states in core_profiles_1d["composition"].items():
    for charge, state in states.items():
        core_profiles_2d["composition"][element][charge]["density"] = equilibrium.map2d(state["f1d_density"])
        core_profiles_2d["composition"][element][charge]["temperature"] = equilibrium.map2d(state["f1d_temperature"])

core_profiles_2d = core_profiles_2d.freeze()

# Plotting plasma profiles
plot_profiles(core_profiles_2d["electron"]["density"], edge_data["mesh"],
              edge_data["electron"]["density"], full_profiles["electron"]["density"],
              'Electron density, m-3')

plot_profiles(core_profiles_2d["electron"]["temperature"], edge_data["mesh"],
              edge_data["electron"]["temperature"], full_profiles["electron"]["temperature"],
              'Electron temperature, eV')

plot_profiles(core_profiles_2d["composition"]["hydrogen"][1]["temperature"], edge_data["mesh"],
              edge_data["composition"]["hydrogen"][1]["temperature"],
              full_profiles["composition"]["hydrogen"][1]["temperature"], 'Ion temperature, eV')

for element, states in core_profiles_2d["composition"].items():
    for charge, state in states.items():
        state_str = ' {}+'.format(charge) if charge else ' 0'
        plot_profiles(state["density"], edge_data["mesh"],
                      edge_data["composition"][element][charge]["density"],
                      full_profiles["composition"][element][charge]["density"],
                      '{} density, m-3'.format(element + state_str))

plot_profiles(core_profiles_2d["composition"]["hydrogen"][0]["temperature"], edge_data["mesh"],
              edge_data["composition"]["hydrogen"][0]["temperature"],
              full_profiles["composition"]["hydrogen"][0]["temperature"], 'neutral hydrogen effective temperature, eV')

plot_profiles(core_profiles_2d["composition"]["carbon"][0]["temperature"], edge_data["mesh"],
              edge_data["composition"]["carbon"][0]["temperature"],
              full_profiles["composition"]["carbon"][0]["temperature"], 'neutral carbon effective temperature, eV')

plt.show()
