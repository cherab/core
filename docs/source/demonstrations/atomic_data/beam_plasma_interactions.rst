
.. _beam_plasma_interaction_rates:

Beam-Plasma Interaction Rates
=============================

Some example code for requesting beam rate objects and sampling them with the __call__()
method.


.. code-block:: pycon

   >>> import numpy as np
   >>> import matplotlib
   >>> import matplotlib.pyplot as plt
   >>> from cherab.core.atomic import deuterium, carbon
   >>> from cherab.openadas import OpenADAS
   >>>
   >>> # Make Latex available in matplotlib figures
   >>> matplotlib.rcParams.update({'font.size': 12})
   >>> matplotlib.rc('text', usetex=True)
   >>> matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
   >>>
   >>> # initialise the atomic data provider
   >>> adas = OpenADAS(permit_extrapolation=True)
   >>>
   >>> # Request beam stopping rate and sample at three different electron temperatures
   >>> bms = adas.beam_stopping_rate(deuterium, carbon, 6)
   >>> beam_energies = [10**x for x in np.linspace(np.log10(5000), np.log10(125000), num=num_points)]
   >>> bms_rates_1 = [bms(x, 1E19, 1) for x in beam_energies]
   >>> bms_rates_2 = [bms(x, 1E19, 100) for x in beam_energies]
   >>> bms_rates_3 = [bms(x, 1E19, 1000) for x in beam_energies]
   >>>
   >>> # plot the beam stopping rates
   >>> plt.figure()
   >>> plt.semilogx(beam_energies, bms_rates_1, '.-', label=r'$t_e$ = 1eV')
   >>> plt.semilogx(beam_energies, bms_rates_2, '.-', label=r'$t_e$ = 100eV')
   >>> plt.semilogx(beam_energies, bms_rates_3, '.-', label=r'$t_e$ = 1000eV')
   >>> plt.xlabel("Interaction Energy (eV/amu)")
   >>> plt.ylabel(r"$S^{e, i}_{CR}$ ($m^3s^{-1}$)")
   >>> plt.title("Beam Stopping Rates")
   >>> plt.legend()
   >>>
   >>> # Sample the beam population rates
   >>> bmp = adas.beam_population_rate(deuterium, 2, carbon, 6)
   >>> bmp_rates_1 = [bmp(x, 1E19, 1) for x in beam_energies]
   >>> bmp_rates_2 = [bmp(x, 1E19, 100) for x in beam_energies]
   >>> bmp_rates_3 = [bmp(x, 1E19, 1000) for x in beam_energies]
   >>>
   >>> # plot the beam population rates
   >>> plt.figure()
   >>> plt.semilogx(beam_energies, bmp_rates_1, '.-', label=r'$t_e$ = 1eV')
   >>> plt.semilogx(beam_energies, bmp_rates_2, '.-', label=r'$t_e$ = 100eV')
   >>> plt.semilogx(beam_energies, bmp_rates_3, '.-', label=r'$t_e$ = 1000eV')
   >>> plt.xlabel("Interaction Energy (eV/amu)")
   >>> plt.ylabel(r"Beam population rate (dimensionless)")
   >>> plt.title("Beam Population Rates")
   >>> plt.legend()
   >>>
   >>> # Sample the beam emission rates
   >>> bme = adas.beam_emission_pec(deuterium, deuterium, 1, (3, 2))
   >>> bme_rates_1 = [bme(x, 1E19, 1) for x in beam_energies]
   >>> bme_rates_2 = [bme(x, 1E19, 100) for x in beam_energies]
   >>> bme_rates_3 = [bme(x, 1E19, 1000) for x in beam_energies]
   >>>
   >>> # plot the beam emission rates
   >>> plt.figure()
   >>> plt.semilogx(beam_energies, bme_rates_1, '.-', label=r'$t_e$ = 1eV')
   >>> plt.semilogx(beam_energies, bme_rates_2, '.-', label=r'$t_e$ = 100eV')
   >>> plt.semilogx(beam_energies, bme_rates_3, '.-', label=r'$t_e$ = 1000eV')
   >>> plt.xlabel("Interaction Energy (eV/amu)")
   >>> plt.ylabel(r"$q^{eff}_{n\rightarrow n'}$ [$W m^{3} s^{-1} str^{-1}$]")
   >>> plt.title("Beam Emission Rates")
   >>> plt.legend()
   >>>
   >>> # Sample the effective CX emission rates
   >>> cxr = adas.beam_cx_rate(deuterium, carbon, 6, (8, 7))
   >>> cxr_n1, cxr_n2 = cxr
   >>> cxr_rate_1 = [cxr[0](x, 100, 1E19, 1, 1) for x in beam_energies]
   >>> cxr_rate_2 = [cxr[1](x, 1000, 1E19, 1, 1) for x in beam_energies]
   >>>
   >>> # plot the effective CX emission rates
   >>> plt.figure()
   >>> plt.loglog(beam_energies, cxr_rate_1, '.-', label='n=1 donor')
   >>> plt.loglog(beam_energies, cxr_rate_2, '.-', label='n=2 donor')
   >>> plt.xlabel("Interaction Energy (eV/amu)")
   >>> plt.ylabel(r"$q^{eff}_{n\rightarrow n'}$ [$W m^{3} s^{-1} str^{-1}$]")
   >>> plt.title("Effective CX rate")
   >>> plt.legend()

.. figure:: beam_stopping_rates.png
   :align: center
   :width: 450px

.. figure:: beam_population_rates.png
   :align: center
   :width: 450px

.. figure:: beam_emission_rates.png
   :align: center
   :width: 450px

.. figure:: effective_cx_rates.png
   :align: center
   :width: 450px
