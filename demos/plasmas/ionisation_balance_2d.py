from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
from cherab.core.atomic import neon, hydrogen, helium
from cherab.core.math import Interpolate1DCubic, Interpolate2DCubic
from cherab.openadas import OpenADAS
from cherab.tools.equilibrium import example_equilibrium
from cherab.tools.plasmas.ionisation_balance import (fractional_abundance, equilibrium_map3d_fractional,
                                                     equilibrium_map3d_from_elementdensity,
                                                     equilibrium_map3d_match_plasma_neutrality,
                                                     from_elementdensity)


def double_parabola(r, centre, edge, p, q, lim=None):
    """
    Double parabolic function to mimic plasma profiles

    :param r: free variable coordinate
    :param centre: Value profile reaches in the cenre
    :param edge: Value profile reaches at the edge
    :param p: Shape parameter
    :param q: Shape parameter
    :param lim: limit above which the profile is constant to treat the edge and prevent divergence of values
    """

    if lim is not None:
        dp = (centre - edge) * np.power((1 - np.power((r - r.min()) / (lim - r.min()), p)), q) + edge

        lm = np.where(r > lim)[0]
        dp[lm] = edge
    else:
        dp = (centre - edge) * np.power((1 - np.power((r - r.min()) / (r.max() - r.min()), p)), q) + edge

    return dp


def normal(x, mu, sd, height=1, offset=0):
    return height * np.exp(-1 * np.power(x - mu, 2) / (2 * sd ** 2)) + offset


def sumup_electrons(densities):
    if isinstance(densities, dict):
        if isinstance(densities[0], Iterable):
            total = np.zeros_like(densities[0])
        else:
            total = 0

        for index, values in densities.items():
            total += values * index

    elif isinstance(densities, np.ndarray):
        total = np.zeros_like(densities[0, ...])

        for index in np.ndindex(densities.shape):
            total[index] += densities[index] * index[0]

    return total


def get_electron_density_spot(densities):
    n_e = 0
    for spec in densities:
        for index, value in enumerate(spec):
            n_e += index * value

    return n_e


def exp_decay(r, lamb, max_val, lim=None):
    ed = max_val * np.exp((r - r.max()) * lamb)
    if lim is not None:
        lm = np.where(r > lim)[0]
        ed[lm] = max_val

    return ed


equilibrium = example_equilibrium()
# plot_equilibrium(equilibrium, detail=True)
# load adas atomic database and define elements
atomic_data = OpenADAS(permit_extrapolation=True)

element = neon
element2 = helium
element_bulk = hydrogen
donor_element = hydrogen

r = np.linspace(*equilibrium.r_range, 20)
z = np.linspace(*equilibrium.z_range, 25)

psin_2d = np.zeros((*r.shape, *z.shape))

for index0, value0 in enumerate(r):
    for index1, value1 in enumerate(z):
        psin_2d[index0, index1] = equilibrium.psi_normalised(value0, value1)

# create plasma profiles and interpolators
psin_1d = np.linspace(0, psin_2d.max(), 50, endpoint=True)

t_e_profile_1d = double_parabola(psin_1d, 5000, 10, 2, 2, 1)
n_e_profile_1d = double_parabola(psin_1d, 6e19, 5e18, 2, 2, 1)

t_element_profile_1d = double_parabola(psin_1d, 1500, 40, 2, 2, 1)
n_element_profile_1d = double_parabola(psin_1d, 1e17, 1e17, 2, 2, 1) + normal(psin_1d, 0.9, 0.1, 5e17)
n_element2_profile_1d = double_parabola(psin_1d, 5e17, 1e17, 2, 2, 1)
n_tcx_donor_profile_1d = exp_decay(psin_1d, 10, 3e16, 1)

t_e_1d = Interpolate1DCubic(psin_1d, t_e_profile_1d)
n_e_1d = Interpolate1DCubic(psin_1d, n_e_profile_1d)

t_element_1d = Interpolate1DCubic(psin_1d, t_element_profile_1d)
n_element_1d = Interpolate1DCubic(psin_1d, n_element_profile_1d)
n_element2_1d = Interpolate1DCubic(psin_1d, n_element2_profile_1d)
n_tcx_donor_1d = Interpolate1DCubic(psin_1d, n_tcx_donor_profile_1d)

psin_2d = np.zeros(equilibrium.psi_data.shape)

for index0, value0 in enumerate(equilibrium.r_data):
    for index1, value1 in enumerate(equilibrium.z_data):
        psin_2d[index0, index1] = equilibrium.psi_normalised(value0, value1)

t_e_profile_2d = np.zeros_like(psin_2d)
for index in np.ndindex(*t_e_profile_2d.shape):
    t_e_profile_2d[index] = t_e_1d(psin_2d[index])

t_e_2d = Interpolate2DCubic(equilibrium.r_data, equilibrium.z_data, t_e_profile_2d)

n_e_profile_2d = np.zeros_like(psin_2d)
for index in np.ndindex(*n_e_profile_2d.shape):
    n_e_profile_2d[index] = n_e_1d(psin_2d[index])

n_e_2d = Interpolate2DCubic(equilibrium.r_data, equilibrium.z_data, n_e_profile_2d)

t_element_profile_2d = np.zeros_like(psin_2d)
for index in np.ndindex(*t_element_profile_2d.shape):
    t_element_profile_2d[index] = t_element_1d(psin_2d[index])

t_element_2d = Interpolate2DCubic(equilibrium.r_data, equilibrium.z_data, t_element_profile_2d)

n_element_profile_2d = np.zeros_like(psin_2d)
for index in np.ndindex(*n_element_profile_2d.shape):
    n_element_profile_2d[index] = n_element_1d(psin_2d[index])

n_element_2d = Interpolate2DCubic(equilibrium.r_data, equilibrium.z_data, n_element_profile_2d)

n_element2_profile_2d = np.zeros_like(psin_2d)
for index in np.ndindex(*n_element2_profile_2d.shape):
    n_element2_profile_2d[index] = n_element2_1d(psin_2d[index])

n_element2_2d = Interpolate2DCubic(equilibrium.r_data, equilibrium.z_data, n_element2_profile_2d)

n_tcx_donor_profile_2d = np.zeros_like(psin_2d)
for index in np.ndindex(*n_tcx_donor_profile_2d.shape):
    n_tcx_donor_profile_2d[index] = n_tcx_donor_1d(psin_2d[index])

n_tcx_donor_2d = Interpolate2DCubic(equilibrium.r_data, equilibrium.z_data, n_element_profile_2d)


########################################################################################################################
# calculate fractional abundance profiles

element_fractional_abundance = fractional_abundance(atomic_data, element, n_e_profile_2d,
                                                    t_e_profile_2d, tcx_donor=donor_element,
                                                    tcx_donor_n=n_tcx_donor_profile_2d, tcx_donor_charge=0)


# plot
plot_ne = plt.subplots()
ax = plot_ne[1]
ax.contourf(equilibrium.r_data, equilibrium.z_data, t_e_profile_2d.T)
ax.contour(equilibrium.r_data, equilibrium.z_data, equilibrium.psi_data.T, colors="white")
ax.plot(equilibrium.limiter_polygon[:, 0], equilibrium.limiter_polygon[:, 1], "k-")
ax.plot(equilibrium.lcfs_polygon[:, 0], equilibrium.lcfs_polygon[:, 1], "r-")
ax.set_title("Te [eV]")
ax.set_aspect(1)

for key, item in element_fractional_abundance.items():
    plot_fractional_abundance = plt.subplots()
    ax = plot_fractional_abundance[1]
    ax.contourf(equilibrium.r_data, equilibrium.z_data, item.T)
    ax.contour(equilibrium.r_data, equilibrium.z_data, equilibrium.psi_data.T, colors="white")
    ax.plot(equilibrium.limiter_polygon[:, 0], equilibrium.limiter_polygon[:, 1], "k-")
    ax.plot(equilibrium.lcfs_polygon[:, 0], equilibrium.lcfs_polygon[:, 1], "r-")
    ax.set_title("{} {}+".format(element.symbol, key))
    ax.set_aspect(1)


########################################################################################################################
#use equilibrium and map3d

element_abundance_equilibrium = equilibrium_map3d_from_elementdensity(atomic_data, element, equilibrium, psin_1d, n_element_1d,
                                                                      n_e_1d, t_e_1d, donor_element, n_tcx_donor_1d,
                                                                      tcx_donor_charge=0)
element2_abundance_equilibrium = equilibrium_map3d_from_elementdensity(atomic_data, element2, equilibrium, psin_1d, n_element2_1d,
                                                                      n_e_1d, t_e_1d, donor_element, n_tcx_donor_1d,
                                                                      tcx_donor_charge=0)

profile = np.zeros((*equilibrium.r_data.shape, *equilibrium.z_data.shape))
for key, item in element_abundance_equilibrium.items():
    for index0, value0 in enumerate(equilibrium.r_data):
        for index1, value1 in enumerate(equilibrium.z_data):
            profile[index0, index1] = item(value0, 0, value1)

    plot_fractional_abundance = plt.subplots()
    ax = plot_fractional_abundance[1]
    ax.contourf(equilibrium.r_data, equilibrium.z_data, profile.T)
    ax.contour(equilibrium.r_data, equilibrium.z_data, equilibrium.psi_data.T, colors="white")
    ax.plot(equilibrium.limiter_polygon[:, 0], equilibrium.limiter_polygon[:, 1], "k-")
    ax.plot(equilibrium.lcfs_polygon[:, 0], equilibrium.lcfs_polygon[:, 1], "r-")
    ax.set_title("{} {}+".format(element.symbol, key))
    ax.set_aspect(1)


profile = np.zeros((*equilibrium.r_data.shape, *equilibrium.z_data.shape))
for key, item in element2_abundance_equilibrium.items():
    for index0, value0 in enumerate(equilibrium.r_data):
        for index1, value1 in enumerate(equilibrium.z_data):
            profile[index0, index1] = item(value0, 0, value1)

    plot_fractional_abundance = plt.subplots()
    ax = plot_fractional_abundance[1]
    ax.contourf(equilibrium.r_data, equilibrium.z_data, profile.T)
    ax.contour(equilibrium.r_data, equilibrium.z_data, equilibrium.psi_data.T, colors="white")
    ax.plot(equilibrium.limiter_polygon[:, 0], equilibrium.limiter_polygon[:, 1], "k-")
    ax.plot(equilibrium.lcfs_polygon[:, 0], equilibrium.lcfs_polygon[:, 1], "r-")
    ax.set_title("{} {}+".format(element2.symbol, key))
    ax.set_aspect(1)

# fill with deuterium back ground

element_abundance_1d = from_elementdensity(atomic_data, element, n_element_1d, n_e_1d, t_e_1d, donor_element,
                                           n_tcx_donor_1d, 0, psin_1d)
element2_abundance_1d = from_elementdensity(atomic_data, element2, n_element2_1d, n_e_1d, t_e_1d, donor_element,
                                           n_tcx_donor_1d, 0, psin_1d)


element_bulk_abundance = equilibrium_map3d_match_plasma_neutrality(atomic_data, element_bulk, equilibrium, psin_1d,
                                                         [element_abundance_1d, element2_abundance_1d], n_e_1d, t_e_1d,
                                                         donor_element, n_tcx_donor_1d, 0)


profile = np.zeros((*equilibrium.r_data.shape, *equilibrium.z_data.shape))
for key, item in element_bulk_abundance.items():
    for index0, value0 in enumerate(equilibrium.r_data):
        for index1, value1 in enumerate(equilibrium.z_data):
            profile[index0, index1] = item(value0, 0, value1)

    plot_fractional_abundance = plt.subplots()
    ax = plot_fractional_abundance[1]
    ax.contourf(equilibrium.r_data, equilibrium.z_data, profile.T)
    ax.contour(equilibrium.r_data, equilibrium.z_data, equilibrium.psi_data.T, colors="white")
    ax.plot(equilibrium.limiter_polygon[:, 0], equilibrium.limiter_polygon[:, 1], "k-")
    ax.plot(equilibrium.lcfs_polygon[:, 0], equilibrium.lcfs_polygon[:, 1], "r-")
    ax.set_title("{} {}+".format(element.symbol, key))
    ax.set_aspect(1)

