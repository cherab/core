from cherab.core.math import Interpolate1DCubic
from cherab.core.atomic import neon, hydrogen, helium
import matplotlib._color_data as mcd
from cherab.tools.plasmas.ionisationbalance import (_calculate_fractional_abundance, calculate_fractional_abundance,
                                                    from_element_density, _from_element_density, _profile1d_fractional,
                                                    profile1d_fractional, interpolators1d_fractional, _profile1d_from_elementdensity,
                                                    profile1d_from_elementdensity, interpolators1d_from_elementdensity,
                                                    _match_element_density, match_element_density, _profile1d_match_density,
                                                    profile1d_match_density, interpolators1d_match_element_density)

from cherab.openadas import OpenADAS

import numpy as np
import matplotlib.pyplot as plt

def doubleparabola(r, Centre, Edge, p, q):
    return (Centre - Edge) * np.power((1 - np.power((r - r.min()) / (r.max() - r.min()), p)), q) +Edge

def normal(x, mu, sd, height=1, offset=0):
    return height * np.exp(-1 * np.power(x - mu, 2) / (2 * sd**2)) + offset

def get_electron_density_profile(abundances):
    n_e = np.zeros((abundances[0].shape[1]))
    for abundance in abundances:
        for rownumber, row in enumerate(abundance.T):
            n_e[rownumber] += np.sum(row * np.arange(row.shape[0]))

    return n_e

def get_electron_density_spot(densities):

    n_e = 0
    for spec in densities:
        n_e += np.sum(spec * np.arange(spec.shape[0]))

    return n_e

colors = list(mcd.XKCD_COLORS)

psin_1d = np.linspace(0, 1.1, 50, endpoint=True)
psin_1d_detailed = np.linspace(0, 1.1, 450, endpoint=True)

t_e_profile = doubleparabola(psin_1d, 5000, 10, 2, 2)
n_e_profile = doubleparabola(psin_1d, 6e19, 5e18, 2, 2)

t_element_profile = doubleparabola(psin_1d, 1500, 40, 2, 2)
n_element_profile = doubleparabola(psin_1d, 1e17, 1e17, 2, 2) + normal(psin_1d, 0.9, 0.1, 5e17)
n_element2_profile = doubleparabola(psin_1d, 5e17, 1e17, 2, 2)


t_e = Interpolate1DCubic(psin_1d, t_e_profile)
n_e = Interpolate1DCubic(psin_1d, n_e_profile)

t_element = Interpolate1DCubic(psin_1d, t_element_profile)
n_element = Interpolate1DCubic(psin_1d, n_element_profile)
n_element2 = Interpolate1DCubic(psin_1d, n_element2_profile)

adas = OpenADAS(permit_extrapolation=True)

psi_value = 0.9
element = neon
element2 = helium
element_bulk = hydrogen

if False:
    _abundance_fractional_spot = _calculate_fractional_abundance(adas, element, n_e(psi_value), t_e(psi_value))
    abundance_fractional_spot = calculate_fractional_abundance(adas, element, n_e(psi_value), t_e(psi_value))
    _abundance_fractional_spot_tcx = _calculate_fractional_abundance(adas, element, n_e(psi_value), t_e(psi_value), hydrogen, 3e15, 0)
    abundance_fractional_spot_tcx = calculate_fractional_abundance(adas, element, n_e(psi_value), t_e(psi_value), hydrogen, 3e15, 0)

    _abundance_profile = _profile1d_fractional(adas, element, n_e_profile, t_e_profile)
    abundance_profile = profile1d_fractional(adas, element, n_e_profile, t_e_profile)

    abundance_fractional_interpolators = interpolators1d_fractional(adas, element, psin_1d, n_e, t_e)

    #calculate total abundance for consistency check
    abundance_spot_total = np.sum(_abundance_fractional_spot)
    abundance_profile_total = np.sum(_abundance_profile, axis=0)
    abundance_profile_total_interpolators = np.zeros_like(abundance_profile_total)
    for key, item in abundance_fractional_interpolators.items():
        for index, value in enumerate(psin_1d):
            abundance_profile_total_interpolators[index] += item(value)

    fig_fractional = plt.subplots()
    ax = fig_fractional[1]
    for index, value in enumerate(_abundance_profile):
        ax.plot(psin_1d, value, "x", color = colors[15+index])
    for index, value in enumerate(_abundance_fractional_spot):
        ax.plot(psi_value, value, "o", color = colors[15+index])

    for key, item in abundance_fractional_interpolators.items():
        tmp = np.zeros_like(psin_1d)
        for index, value in enumerate(psin_1d):
            tmp[index] = item(value)
        ax.plot(psin_1d, tmp, color = colors[15+key])


    ax.plot(psin_1d, abundance_profile_total, "x", color ="xkcd:red")
    ax.plot(psi_value, abundance_spot_total, "o", color ="xkcd:red")
    ax.plot(psin_1d, abundance_profile_total_interpolators, color ="xkcd:red", label="total")

    for i in range(element.atomic_number+1):
        ax.plot([], [], color = colors[15+i], label="{0}{1}+".format(element.symbol, i))

    ax.legend()

if True:

    density_element_spot = _from_element_density(adas, element, n_element(psi_value), n_e(psi_value), t_e(psi_value))
    density_element2_spot = _from_element_density(adas, element2, n_element2(psi_value), n_e(psi_value), t_e(psi_value))
    density_bulk_spot = _match_element_density(adas, element_bulk, [density_element_spot, density_element2_spot],
                                               n_e(psi_value), t_e(psi_value))

    density_total_spot = get_electron_density_spot([density_element_spot, density_element2_spot, density_bulk_spot])

    density_element_profiles = _profile1d_from_elementdensity(adas, element, n_element_profile, n_e_profile, t_e_profile)
    density_element2_profiles = _profile1d_from_elementdensity(adas, element2, n_element2_profile, n_e_profile, t_e_profile)
    density_bulk_profiles = _profile1d_match_density(adas, element_bulk, [density_element_profiles, density_element2_profiles],
                                                     n_e_profile, t_e_profile)

    density_total_profile = get_electron_density_profile([density_element_profiles, density_element2_profiles,
                                                          density_bulk_profiles])


    density_element_interpolators = interpolators1d_from_elementdensity(adas, element, psin_1d, n_element, n_e, t_e)
    density_element2_interpolators = interpolators1d_from_elementdensity(adas, element2, psin_1d, n_element2, n_e, t_e)
    density_bulk_interpolators = interpolators1d_match_element_density(adas, element_bulk, psin_1d_detailed,
                                                                       [density_element_interpolators,
                                                                             density_element2_interpolators],
                                                                       n_e, t_e)



    fig_abundance = plt.subplots()
    ax = fig_abundance[1]
    # Element 1
    density_total_interpolators = np.zeros_like(psin_1d)

    for rownumber, row in enumerate(density_element_profiles):
        ax.plot(psin_1d, row, "x", color = colors[15 + rownumber])
    for index, value in enumerate(density_element_spot):
        ax.plot(psi_value, value, "o", color = colors[15+index])

    for key, item in density_element_interpolators.items():
        tmp = np.zeros_like(psin_1d)
        for index, value in enumerate(psin_1d):
            tmp[index] = item(value)
        density_total_interpolators = density_total_interpolators + tmp * key
        ax.plot(psin_1d, tmp, color = colors[15+key])

    for i in range(element.atomic_number + 1):
        ax.plot([],[], color = colors[15 + i], lw = 3, label= "{0} {1}+".format(element.symbol, i))

    # Element 2
    for rownumber, row in enumerate(density_element2_profiles):
        ax.plot(psin_1d, row, "x", color = colors[5 + rownumber])
    for index, value in enumerate(density_element2_spot):
        ax.plot(psi_value, value, "o", color = colors[5+index])

    for key, item in density_element2_interpolators.items():
        tmp = np.zeros_like(psin_1d)
        for index, value in enumerate(psin_1d):
            tmp[index] = item(value)
        density_total_interpolators = density_total_interpolators + tmp * key
        ax.plot(psin_1d, tmp, color = colors[5+key])

    for i in range(element2.atomic_number + 1):
        ax.plot([],[], color = colors[5 + i], lw = 3, label= "{0} {1}+".format(element2.symbol, i))

    # Element bulk
    for rownumber, row in enumerate(density_bulk_profiles):
        ax.plot(psin_1d, row, "x", color = colors[30 + rownumber])
    for index, value in enumerate(density_bulk_spot):
        ax.plot(psi_value, value, "o", color = colors[30+index])

    for key, item in density_bulk_interpolators.items():
        tmp = np.zeros_like(psin_1d)
        for index, value in enumerate(psin_1d):
            tmp[index] = item(value)
        density_total_interpolators = density_total_interpolators + tmp * key
        ax.plot(psin_1d, tmp, color = colors[30+key])

    for i in range(element_bulk.atomic_number + 1):
        ax.plot([],[], color = colors[30 + i], lw = 3, label= "{0} {1}+".format(element_bulk.symbol, i))



    ax.plot(psin_1d, n_e_profile, color="xkcd:red", label="e input")
    ax.plot(psin_1d, density_total_profile, ":", color="xkcd:red", label="e profiles")
    ax.plot(psin_1d, density_total_interpolators, ".-", color="xkcd:red", label="e profiles")
    ax.plot(psi_value, density_total_spot, "o", color =  "xkcd:red")
    ax.legend()
