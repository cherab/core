
# Copyright 2016-2021 Euratom
# Copyright 2016-2021 United Kingdom Atomic Energy Authority
# Copyright 2016-2021 Centro de Investigaciones Energéticas, Medioambientales y Tecnológicas
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

from cherab.core.math.samplers import sample1d_points
from cherab.generomak.plasma.plasma import get_core_interpolators

profiles = get_core_interpolators()

# setup temperature plot
_, ax_t = plt.subplots()
ax_t.set_title("Species Core Temperature Profiles")
ax_t.set_xlabel("Psin")
ax_t.set_ylabel("eV")

# setup density plot
_, ax_n = plt.subplots()
ax_n.set_yscale("log")
ax_n.set_ylim(1.e-1, 1.e21)
ax_n.set_title("Species Core Density Profiles")
ax_n.set_xlabel("psin")
ax_n.set_ylabel("m^-3")

psin = np.append(1. - np.geomspace(1.e-4, 1, 127)[::-1], [1.])
# add hydrogen curves
for chrg, desc in profiles["composition"]["hydrogen"].items():
    vals = sample1d_points(desc["f1d_temperature"], psin)
    ax_t.plot(psin, vals, label="H{:d}+".format(chrg))
    vals = sample1d_points(desc["f1d_density"], psin)
    ax_n.plot(psin, vals, label="H{:d}+".format(chrg))

# add carbon curves
for chrg, desc in profiles["composition"]["carbon"].items():
    vals = sample1d_points(desc["f1d_temperature"], psin)
    ax_t.plot(psin, vals, label="C{:d}+".format(chrg))
    vals = sample1d_points(desc["f1d_density"], psin)
    ax_n.plot(psin, vals, label="C{:d}+".format(chrg))

# add electrons
vals = sample1d_points(profiles["electron"]["f1d_temperature"], psin)
ax_t.plot(psin, vals, label="electron", ls="dashed")
vals = sample1d_points(profiles["electron"]["f1d_density"], psin)
ax_n.plot(psin, vals, label="electron", ls="dashed")

ax_t.legend()
ax_n.legend(ncol=2)
plt.show()
