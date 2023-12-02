
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
import matplotlib
import matplotlib.pyplot as plt
from cherab.core.atomic import deuterium, carbon
from cherab.atomic import AtomicData

# Make Latex available in matplotlib figures
matplotlib.rcParams.update({'font.size': 12})
matplotlib.rc('text', usetex=True)
matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})

# initialise the atomic data provider
atomic_data = AtomicData(permit_extrapolation=True)

# Request beam stopping rate and sample at three different electron temperatures
bms = atomic_data.beam_stopping_rate(deuterium, carbon, 6)
beam_energies = [10**x for x in np.linspace(np.log10(5000), np.log10(125000), num=512)]
bms_rates_1 = [bms(x, 1E19, 1) for x in beam_energies]
bms_rates_2 = [bms(x, 1E19, 100) for x in beam_energies]
bms_rates_3 = [bms(x, 1E19, 1000) for x in beam_energies]

# plot the beam stopping rates
plt.figure()
plt.semilogx(beam_energies, bms_rates_1, '.-', label=r'$t_e$ = 1eV')
plt.semilogx(beam_energies, bms_rates_2, '.-', label=r'$t_e$ = 100eV')
plt.semilogx(beam_energies, bms_rates_3, '.-', label=r'$t_e$ = 1000eV')
plt.xlabel("Interaction Energy (eV/amu)")
plt.ylabel(r"$S^{e, i}_{CR}$ ($m^3s^{-1}$)")
plt.title("Beam Stopping Rates")
plt.legend()

# Sample the beam population rates
bmp = atomic_data.beam_population_rate(deuterium, 2, carbon, 6)
bmp_rates_1 = [bmp(x, 1E19, 1) for x in beam_energies]
bmp_rates_2 = [bmp(x, 1E19, 100) for x in beam_energies]
bmp_rates_3 = [bmp(x, 1E19, 1000) for x in beam_energies]

# plot the beam population rates
plt.figure()
plt.semilogx(beam_energies, bmp_rates_1, '.-', label=r'$t_e$ = 1eV')
plt.semilogx(beam_energies, bmp_rates_2, '.-', label=r'$t_e$ = 100eV')
plt.semilogx(beam_energies, bmp_rates_3, '.-', label=r'$t_e$ = 1000eV')
plt.xlabel("Interaction Energy (eV/amu)")
plt.ylabel(r"Beam population rate (dimensionless)")
plt.title("Beam Population Rates")
plt.legend()

# Sample the beam emission rates
bme = atomic_data.beam_emission_pec(deuterium, deuterium, 1, (3, 2))
bme_rates_1 = [bme(x, 1E19, 1) for x in beam_energies]
bme_rates_2 = [bme(x, 1E19, 100) for x in beam_energies]
bme_rates_3 = [bme(x, 1E19, 1000) for x in beam_energies]

# plot the beam emission rates
plt.figure()
plt.semilogx(beam_energies, bme_rates_1, '.-', label=r'$t_e$ = 1eV')
plt.semilogx(beam_energies, bme_rates_2, '.-', label=r'$t_e$ = 100eV')
plt.semilogx(beam_energies, bme_rates_3, '.-', label=r'$t_e$ = 1000eV')
plt.xlabel("Interaction Energy (eV/amu)")
plt.ylabel(r"Beam emission rate [$W m^{3}$]")
plt.title("Beam Emission Rates")
plt.legend()

# Sample the effective CX emission rates
cxr = atomic_data.beam_cx_pec(deuterium, carbon, 6, (8, 7))
cxr_n1, cxr_n2 = cxr
cxr_rate_1 = [cxr[0](x, 100, 1E19, 1, 1) for x in beam_energies]
cxr_rate_2 = [cxr[1](x, 1000, 1E19, 1, 1) for x in beam_energies]

# plot the effective CX emission rates
plt.figure()
plt.loglog(beam_energies, cxr_rate_1, '.-', label='n=1 donor')
plt.loglog(beam_energies, cxr_rate_2, '.-', label='n=2 donor')
plt.xlabel("Interaction Energy (eV/amu)")
plt.ylabel(r"$q^{eff}_{n\rightarrow n'}$ [$W m^{3}$]")
plt.title("Effective CX rate")
plt.legend()

plt.show()
