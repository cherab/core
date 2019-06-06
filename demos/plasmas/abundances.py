from cherab.core.math import Interpolate1DCubic
from cherab.core.atomic import neon, hydrogen, helium

from cherab.tools.plasmas.ionisationbalance import (match_element_density, from_element_density, interpolators1d_fractional,
                                                    interpolators1d_from_elementdensity, interpolators1d_match_element_density)

from cherab.openadas import OpenADAS

import numpy as np
import matplotlib.pyplot as plt

def doubleparabola(r, Centre, Edge, p, q):
    return (Centre - Edge) * np.power((1 - np.power((r - r.min()) / (r.max() - r.min()), p)), q) +Edge

def normal(x, mu, sd, height=1, offset=0):
    return height * np.exp(-1 * np.power(x - mu, 2) / (2 * sd**2)) + offset

psin_1d = np.linspace(0, 1.1, 50, endpoint=True)
t_e = Interpolate1DCubic(psin_1d, doubleparabola(psin_1d, 5000, 10, 2, 2))
n_e = Interpolate1DCubic(psin_1d, doubleparabola(psin_1d, 6e19, 5e18, 2, 2))

t_ne = Interpolate1DCubic(psin_1d, doubleparabola(psin_1d, 1500, 40, 2, 2))
n_ne = Interpolate1DCubic(psin_1d, doubleparabola(psin_1d, 1e17, 1e17, 2, 2) + normal(psin_1d, 0.9, 0.1, 5e17))

t_ne = Interpolate1DCubic(psin_1d, doubleparabola(psin_1d, 1500, 40, 2, 2))

t_e_profile = np.zeros_like(psin_1d)
n_e_profile = np.zeros_like(psin_1d)
n_ne_profile = np.zeros_like(psin_1d)
t_ne_profile = np.zeros_like(psin_1d)

for index, value in enumerate(psin_1d):
    t_e_profile[index] = t_e(value)
    n_e_profile[index] = n_e(value)
    n_ne_profile[index] = n_ne(value)
    t_ne_profile[index] = t_ne(value)


adas = OpenADAS(permit_extrapolation=True)

psi_value = 0.9
ne_density = from_element_density(adas, neon, n_ne(psi_value), n_e(psi_value), t_e(psi_value))
matched_density = match_element_density(adas, helium, ne_density, n_e(psi_value), t_e(psi_value))
n_e_density_fromions = 0
for key, item in ne_density.items():
    n_e_density_fromions += key * item
for key, item in matched_density.items():
    n_e_density_fromions += key * item

fractional_interpolators = interpolators1d_fractional(adas, neon, psin_1d, n_e, t_e)

density_interpolators = interpolators1d_from_elementdensity(adas, neon, psin_1d, n_ne, n_e, t_e)
matched_interpolators = interpolators1d_match_element_density(adas, helium, psin_1d, density_interpolators, n_e, t_e)

fig_profiles = plt.subplots()
ax = fig_profiles[1]
ax.plot(psin_1d, n_e_profile)
ax.plot(psin_1d, t_e_profile)
ax.plot(psin_1d, n_ne_profile)

ne_fromspecies = np.zeros_like(psin_1d)

fractional_profiles = {}
for key, item in fractional_interpolators.items():
    fractional_profiles[key] = np.zeros_like(psin_1d)
    for index, value in enumerate(psin_1d):
        fractional_profiles[key][index] = item(value)

density_profiles = {}
for key, item in density_interpolators.items():
    density_profiles[key] = np.zeros_like(psin_1d)
    for index, value in enumerate(psin_1d):
        density_profiles[key][index] = item(value)
        ne_fromspecies[index] += key * item(value)

matched_profiles = {}
for key, item in matched_interpolators.items():
    matched_profiles[key] = np.zeros_like(psin_1d)
    for index, value in enumerate(psin_1d):
        matched_profiles[key][index] = item(value)
        ne_fromspecies[index] += key * item(value)

fig_fracprof = plt.subplots()
ax = fig_fracprof[1]
for key, item in fractional_profiles.items():
    ax.plot(psin_1d, item)

fig_densprof = plt.subplots()
ax = fig_densprof[1]
for key, item in density_profiles.items():
    ax.plot(psin_1d, item)
for key, item in matched_profiles.items():
    ax.plot(psin_1d, item)
ax.plot(psin_1d, n_e_profile, "--", color="xkcd:red")
ax.plot(psin_1d, ne_fromspecies, "-.", color="xkcd:blue")