
from collections.abc import Iterable
import matplotlib._color_data as mcd
import matplotlib.pyplot as plt
import numpy as np

from cherab.core.atomic import neon, hydrogen, helium
from cherab.core.math import Interpolate1DCubic
from cherab.openadas import OpenADAS
from cherab.tools.plasmas.ionisation_balance import (fractional_abundance,
                                                     interpolators1d_fractional, from_elementdensity,
                                                     match_plasma_neutrality, interpolators1d_from_elementdensity,
                                                     interpolators1d_match_plasma_neutrality)


def double_parabola(r, centre, edge, p, q):
    return (centre - edge) * np.power((1 - np.power((r - r.min()) / (r.max() - r.min()), p)), q) + edge


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


def exp_decay(r, lamb, max_val):
    return max_val * np.exp((r - r.max()) * lamb)


colors = list(mcd.XKCD_COLORS)  # load color list to iterate over

# create plasma profiles and interpolators
psin_1d = np.linspace(0, 1.1, 50, endpoint=True)
psin_1d_detailed = np.linspace(0, 1.1, 450, endpoint=True)

t_e_profile = double_parabola(psin_1d, 5000, 10, 2, 2)
n_e_profile = double_parabola(psin_1d, 6e19, 5e18, 2, 2)

t_element_profile = double_parabola(psin_1d, 1500, 40, 2, 2)
n_element_profile = double_parabola(psin_1d, 1e17, 1e17, 2, 2) + normal(psin_1d, 0.9, 0.1, 5e17)
n_element2_profile = double_parabola(psin_1d, 5e17, 1e17, 2, 2)
n_tcx_donor_profile = exp_decay(psin_1d, 10, 3e16)

t_e = Interpolate1DCubic(psin_1d, t_e_profile)
n_e = Interpolate1DCubic(psin_1d, n_e_profile)

t_element = Interpolate1DCubic(psin_1d, t_element_profile)
n_element = Interpolate1DCubic(psin_1d, n_element_profile)
n_element2 = Interpolate1DCubic(psin_1d, n_element2_profile)
n_tcx_donor = Interpolate1DCubic(psin_1d, n_tcx_donor_profile)

# load adas atomic database and define elements
adas = OpenADAS(permit_extrapolation=True)

element = neon
element2 = helium
element_bulk = hydrogen
donor_element = hydrogen

# calculate profiles of fractional abundance for the element
abundance_fractional_profile = fractional_abundance(adas, element, n_e_profile, t_e_profile)
abundance_fractional_profile_tcx = fractional_abundance(adas, element, n_e_profile, t_e_profile,
                                                        tcx_donor=donor_element, tcx_donor_n=n_tcx_donor,
                                                        tcx_donor_charge=0, free_variable=psin_1d)

fig_abundance = plt.subplots()
ax = fig_abundance[1]
for key in abundance_fractional_profile.keys():
    ax.plot(psin_1d, abundance_fractional_profile[key], label="{0} {1}+".format(element.symbol, key), color=colors[key])
    ax.plot(psin_1d, abundance_fractional_profile_tcx[key], "--", label="{0} {1}+ (tcx)".format(element.symbol, key),
            color=colors[key])

ax.legend(loc=6)
ax.set_xlabel("$\Psi_n$")
ax.set_ylabel("fractional abundance [a.u.]")
plt.title('Fractional Abundance VS $\Psi_n$')

# calculate charge state density profiles by specifying element density
density_element_profiles = from_elementdensity(adas, element, n_element, n_e_profile,
                                               t_e, free_variable=psin_1d)
density_element_profiles_tcx = from_elementdensity(adas, element, n_element, n_e_profile,
                                                   t_e, tcx_donor=donor_element, tcx_donor_n=n_tcx_donor_profile,
                                                   tcx_donor_charge=0, free_variable=psin_1d)

fig_abundance = plt.subplots()
ax = fig_abundance[1]
ax.plot(psin_1d, n_element_profile, "k", label="n_element")
for key in density_element_profiles.keys():
    ax.plot(psin_1d, density_element_profiles[key], label="{0} {1}+".format(element.symbol, key), color=colors[key])
    ax.plot(psin_1d, density_element_profiles_tcx[key], "--", label="{0} {1}+ (tcx)".format(element.symbol, key),
            color=colors[key])

ax.legend(loc=6)
ax.set_xlabel("$\Psi_n$")
ax.set_ylabel("ion density [m$^{-3}]$")

# calculate fill the plasma with bulk element to match plasma neutrality condition

# calculate ion densities for a 2nd element
density_element2_profiles_tcx = from_elementdensity(adas, element2, n_element2, n_e_profile,
                                                    t_e, tcx_donor=donor_element, tcx_donor_n=n_tcx_donor_profile,
                                                    tcx_donor_charge=0, free_variable=psin_1d)

# fill plasma with 3rd element to match plasma neutrality
density_element3_profiles_tcx = match_plasma_neutrality(adas, element_bulk,
                                                        [density_element_profiles_tcx, density_element2_profiles_tcx],
                                                        n_e, t_e, tcx_donor=donor_element,
                                                        tcx_donor_n=n_tcx_donor_profile,
                                                        tcx_donor_charge=0, free_variable=psin_1d)

n_e_recalculated = sumup_electrons(density_element_profiles_tcx)
n_e_recalculated += sumup_electrons(density_element2_profiles_tcx)
n_e_recalculated += sumup_electrons(density_element3_profiles_tcx)

fig_plasma = plt.subplots()
ax = fig_plasma[1]
for key in density_element_profiles.keys():
    ax.plot(psin_1d, density_element_profiles_tcx[key], "--", label="{0} {1}+".format(element.symbol, key),
            color=colors[key])

for key2 in density_element2_profiles_tcx.keys():
    ax.plot(psin_1d, density_element2_profiles_tcx[key2], "--", label="{0} {1}+ (tcx)".format(element2.symbol, key2),
            color=colors[key2 + key])

for key3 in density_element3_profiles_tcx.keys():
    ax.plot(psin_1d, density_element3_profiles_tcx[key3], "--",
            label="{0} {1}+ (tcx)".format(element_bulk.symbol, key3), color=colors[key3 + key2 + key])

ax.plot(psin_1d, n_e_profile, "kx", label="input n_e")
ax.plot(psin_1d, n_e_recalculated, "k-", label="recalculated n_e")

ax.legend(loc=6)
ax.set_xlabel("$\Psi_n$")
ax.set_ylabel("ion density [m$^{-3}]$")

# create ion density 1d interpolators
interpolators_element_1d_fractional = interpolators1d_fractional(adas, element, psin_1d, n_e, t_e,
                                                                 tcx_donor=donor_element,
                                                                 tcx_donor_n=n_tcx_donor, tcx_donor_charge=0)
interpolators_element_1d_density = interpolators1d_from_elementdensity(adas, element, psin_1d, n_element, n_e, t_e,
                                                                       tcx_donor=donor_element,
                                                                       tcx_donor_n=n_tcx_donor, tcx_donor_charge=0)
interpolators_element2_1d_density = interpolators1d_from_elementdensity(adas, element2, psin_1d, n_element2, n_e, t_e,
                                                                        tcx_donor=donor_element,
                                                                        tcx_donor_n=n_tcx_donor, tcx_donor_charge=0)

# also it is possible to combine different kinds of parameter types (profiles. numbers and interpolators)
interpolators_element3_1d_density = interpolators1d_match_plasma_neutrality(adas, element_bulk, psin_1d,
                                                                            [interpolators_element_1d_density,
                                                                             density_element2_profiles_tcx],
                                                                            n_e, t_e, tcx_donor=donor_element,
                                                                            tcx_donor_n=n_tcx_donor,
                                                                            tcx_donor_charge=0)
