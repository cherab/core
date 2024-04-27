
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

from cherab.core.atomic import deuterium, helium, carbon
from cherab.atomic import AtomicData


atomic_data = AtomicData()

# load the PEC rate objects for transitions of interest
dalpha = atomic_data.impact_excitation_pec(deuterium, 0, (3, 2))
heliumii_468 = atomic_data.impact_excitation_pec(helium, 1, (4, 3))
carbonii_515 = atomic_data.impact_excitation_pec(carbon, 1, ("2s1 2p1 3d1 2D4.5", "2s2 4d1 2D4.5"))
carboniii_465 = atomic_data.impact_excitation_pec(carbon, 2, ("2s1 3p1 3P4.0", "2s1 3s1 3S1.0"))

# settings for plot range
temp_low = 1
temp_high = 1000
num_points = 100
electron_density = 1E19
electron_temperatures = [10**x for x in np.linspace(np.log10(temp_low), np.log10(temp_high), num=num_points)]

# sample the PECs
dalpha_pecs = []
heliumii_468_pecs = []
carbonii_515_pecs = []
carboniii_465_pecs = []
for te in electron_temperatures:
    dalpha_pecs.append(dalpha(electron_density, te))
    heliumii_468_pecs.append(heliumii_468(electron_density, te))
    carbonii_515_pecs.append(carbonii_515(electron_density, te))
    carboniii_465_pecs.append(carboniii_465(electron_density, te))

plt.figure()
plt.loglog(electron_temperatures, dalpha_pecs, '.-', label="Dalpha")
plt.loglog(electron_temperatures, heliumii_468_pecs, '.-', label="HeliumII_468")
plt.loglog(electron_temperatures, carbonii_515_pecs, '.-', label="CarbonII_515")
plt.loglog(electron_temperatures, carboniii_465_pecs, '.-', label="CarbonIII_465")
plt.xlabel("Temperature (eV)")
plt.ylabel("PECs (W m^-3)")
plt.legend()
plt.show()
