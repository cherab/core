
Rate Coefficients
-----------------

All atomic rate coefficients specify a calling signature that must be used for this rate,
e.g. `__call__(arg1, arg2, ...)`. The calculations associated with this calling signature
are actually implemented in a separate function called `evaluate(arg1, arg2, ...)`. No other
information or data about the rate is specified in the core API, instead all other
implementation details are deferred to the atomic data provider.

The reason for this design is that it allows rate objects to be used throughout all the
Cherab emission models without knowing how this data will be provided or calculated.
For example, some atomic data providers might use interpolated data while others could
provide theoretical equations. Cherab emission models only need to know how to call
them after they have been instantiated.


Photon Emissivity Coefficients
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: cherab.core.atomic.rates.ImpactExcitationPEC

.. autoclass:: cherab.core.atomic.rates.RecombinationPEC

.. autoclass:: cherab.core.atomic.rates.ThermalCXPEC

The `ImpactExcitationPEC`, `RecombinationPEC` and `ThermalCXPEC` classes all share
the same call signatures.

.. function:: __call__(density, temperature)

   Returns a photon emissivity coefficient at the specified plasma conditions.

   This function just wraps the cython evaluate() method.

.. function:: evaluate(density, temperature)

   Returns a photon emissivity coefficient at the specified plasma conditions.

   This function needs to be implemented by the atomic data provider.

   :param float temperature: Receiver ion temperature in eV.
   :param float density: Receiver ion density in m^-3
   :return: The effective PEC rate [Wm^3].

Some example code for requesting PEC objects and sampling them with the __call__()
method.

.. code-block:: pycon

   >>> import numpy as np
   >>> import matplotlib.pyplot as plt
   >>> from cherab.core.atomic import deuterium
   >>> from cherab.openadas import OpenADAS
   >>>
   >>> # initialise the atomic data provider
   >>> adas = OpenADAS()
   >>>
   >>> # request d-alpha instance of ImpactExcitationRate
   >>> dalpha_excit = adas.impact_excitation_pec(deuterium, 0, (3, 2))
   >>> # request d-alpha instance of RecombinationRate
   >>> dalpha_recom = adas.recombination_pec(deuterium, 0, (3, 2))
   >>>
   >>> # evaluate D-alpha ImpactExcitationRate PEC at n_e = 1E19 m^-3 and t_e = 2 eV
   >>> dalpha_excit(1E19, 2)
   2.50352900-36
   >>>
   >>> # evaluate D-alpha ImpactExcitationRate PEC at n_e = 1E19 m^-3 and t_e = 2 eV
   >>> dalpha_recom(1E19, 2)
   1.09586154-38


Beam-Plasma Interaction Rates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: cherab.core.atomic.rates.BeamStoppingRate

.. autoclass:: cherab.core.atomic.rates.BeamPopulationRate

.. autoclass:: cherab.core.atomic.rates.BeamEmissionPEC

The `BeamStoppingRate`, `BeamPopulationRate` and `BeamEmissionPEC` classes all share
the same call signatures.

.. function:: __call__(energy, density, temperature)

   Returns the associated beam reaction rate at the specified plasma conditions.

   This function just wraps the cython evaluate() method.

.. function:: evaluate(energy, density, temperature)

   Returns the beam coefficient for the supplied parameters.

   :param float energy: Interaction energy in eV/amu.
   :param float density: Target electron density in m^-3
   :param float temperature: Target temperature in eV.
   :return: The beam coefficient

Some example code for requesting beam rate objects and sampling them with the __call__()
method.

.. code-block:: pycon

   >>> from cherab.core.atomic import deuterium, carbon
   >>> from cherab.openadas import OpenADAS
   >>>
   >>> # initialise the atomic data provider
   >>> adas = OpenADAS(permit_extrapolation=True)
   >>>
   >>> # Request beam stopping rate and sample
   >>> bms = adas.beam_stopping_rate(deuterium, carbon, 6)
   >>> bms(50000, 1E19, 1)
   1.777336e-13
   >>>
   >>> # Sample the beam population rate
   >>> bmp = adas.beam_population_rate(deuterium, 2, carbon, 6)
   >>> bmp(50000, 1E19, 1)
   7.599066e-4
   >>>
   >>> # Sample the beam emission rate
   >>> bme = adas.beam_emission_pec(deuterium, deuterium, 1, (3, 2))
   >>> bme(50000, 1E19, 1)
   8.651598e-34

.. autoclass:: cherab.core.atomic.rates.BeamCXPEC
   :members: __call__, evaluate

Some example code for requesting beam CX rate object and sampling it with the __call__() method.

.. code-block:: pycon

   >>> from cherab.core.atomic import deuterium, carbon
   >>> from cherab.openadas import OpenADAS
   >>>
   >>> # initialise the atomic data provider
   >>> adas = OpenADAS(permit_extrapolation=True)
   >>>
   >>> cxr = adas.beam_cx_pec(deuterium, carbon, 6, (8, 7))
   >>> cxr_n1, cxr_n2 = cxr
   >>> cxr_n1(50000, 100, 1E19, 1, 1)
   5.826619e-33
   >>> cxr_n2(50000, 100, 1E19, 1, 1)
   1.203986e-32


Abundances
^^^^^^^^^^

.. class:: cherab.core.atomic.rates.FractionalAbundance

   Rate provider for fractional abundances in thermodynamic equilibrium.

   .. function:: __call__(electron_density, electron_temperature)

      Evaluate the fractional abundance of this ionisation stage at the given plasma conditions.

      This function just wraps the cython evaluate() method.

      :param float electron_density: electron density in m^-3
      :param float electron_temperature: electron temperature in eV

.. code-block:: pycon

   >>> from cherab.core.atomic import neon
   >>> from cherab.adas import ADAS
   >>>
   >>> atomic_data = ADAS()
   >>>
   >>> ne0_frac = atomic_data.fractional_abundance(neon, 0)
   >>> ne0_frac(1E19, 1.0)
   0.999899505093943


Radiated Power
^^^^^^^^^^^^^^

.. class:: cherab.core.atomic.rates.RadiatedPower

   Total radiated power for a given species and radiation type.

   Radiation type can be:
   - 'total' (line + recombination + bremsstrahlung + charge exchange)
   - 'line' radiation
   - 'continuum' (recombination + bremsstrahlung)
   - 'cx' charge exchange

   .. function:: __call__(electron_density, electron_temperature)

      Evaluate the total radiated power of this species at the given plasma conditions.

      This function just wraps the cython evaluate() method.

      :param float electron_density: electron density in m^-3
      :param float electron_temperature: electron temperature in eV


.. code-block:: pycon

   >>> from cherab.core.atomic import neon
   >>> from cherab.adas import ADAS
   >>>
   >>> atomic_data = ADAS()
   >>>
   >>> ne_total_rad = atomic_data.radiated_power_rate(neon, 'total')
   >>> ne_total_rad(1E19, 10) * 1E19
   9.2261136594e-08
   >>>
   >>> ne_continuum_rad = atomic_data.radiated_power_rate(neon, 'continuum')
   >>> ne_continuum_rad(1E19, 10) * 1E19
   3.4387672228e-10


.. class:: cherab.core.atomic.rates.StageResolvedLineRadiation

   Total ionisation state resolved line radiated power rate.

   :param Element element: the radiating element
   :param int ionisation: the integer charge state for this ionisation stage
   :param str name: optional label identifying this rate

   .. function:: __call__(electron_density, electron_temperature)

      Evaluate the total radiated power of this species at the given plasma conditions.

      This function just wraps the cython evaluate() method.

      :param float electron_density: electron density in m^-3
      :param float electron_temperature: electron temperature in eV

.. code-block:: pycon

   >>> from cherab.core.atomic import neon
   >>> from cherab.adas import ADAS
   >>>
   >>> atomic_data = ADAS()
   >>>
   >>> ne0_line_rad = atomic_data.stage_resolved_line_radiation_rate(neon, 0)
   >>> ne0_line_rad(1E19, 10) * 1E19
   6.1448254527e-16
   >>>
   >>> ne1_line_rad = atomic_data.stage_resolved_line_radiation_rate(neon, 1)
   >>> ne1_line_rad(1E19, 10) * 1E19
   1.7723122151e-11

