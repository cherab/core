
import numpy as np
import matplotlib.pyplot as plt

from cherab.core.atomic import deuterium, helium, carbon
from cherab.openadas import OpenADAS


adas = OpenADAS()

# load the PEC rate objects for transitions of interest
dalpha = adas.impact_excitation_pec(deuterium, 0, (3, 2))
heliumii_468 = adas.impact_excitation_pec(helium, 1, (4, 3))
carbonii_515 = adas.impact_excitation_pec(carbon, 1, ("2s1 2p1 3d1 2D4.5", "2s2 4d1 2D4.5"))
carboniii_465 = adas.impact_excitation_pec(carbon, 2, ("2s1 3p1 3P4.0", "2s1 3s1 3S1.0"))

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
