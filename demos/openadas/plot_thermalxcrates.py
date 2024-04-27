
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
from cherab.core.atomic import neon, carbon, helium, hydrogen
from cherab.atomic import AtomicData

atomic_data = AtomicData(permit_extrapolation=True)

electron_temperatures = [10**x for x in np.linspace(np.log10(1), np.log10(10000), num=100)]
electron_density = 1e19

elem = neon

numstates = elem.atomic_number + 1

# Collect rate coefficients
coef_tcx = {}
for i in np.arange(1, elem.atomic_number+1):
    coef_tcx[i] = atomic_data.thermal_cx_rate(hydrogen, 0, neon, int(i))

    # test correctness of available charge numbers
    try:
        atomic_data.thermal_cx_rate(hydrogen, 0, neon, i)
    except RuntimeError:
        print("Check that thermal charge exchange between a neutral element and neutral hydrogen "
              "is not allowed.")


fig_tcxrates = plt.subplots()
ax = fig_tcxrates[1]
for i in range(1, elem.atomic_number+1):
    tcx_rate = [coef_tcx[i](electron_density, x) for x in electron_temperatures]
    ax.loglog(electron_temperatures, tcx_rate, "-x", label = "Ne{}+".format(i))

    plt.xlabel("Electron Temperature (eV)")
    ax.set_ylabel("rate [m^3s^-1]")
    plt.title("Thermal Charge Exchange Rates")


plt.show()


# test loading rates for CX between neutrals is not allowed
try:
    coef_notallowed = atomic_data.thermal_cx_rate(hydrogen, 0, neon, 0)
except RuntimeError:
    print("All correct")
