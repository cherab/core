Atomic data interpolators
=========================

Classes that interpolate numerical atomic data.

Rate Coefficients
-----------------

Atomic Processes
^^^^^^^^^^^^^^^^

.. autoclass:: cherab.atomic.rates.atomic.IonisationRate
   :members:

.. autoclass:: cherab.atomic.rates.atomic.NullIonisationRate
   :members:

.. autoclass:: cherab.atomic.rates.atomic.RecombinationRate
   :members:

.. autoclass:: cherab.atomic.rates.atomic.NullRecombinationRate
   :members:

.. autoclass:: cherab.atomic.rates.atomic.ThermalCXRate
   :members:

.. autoclass:: cherab.atomic.rates.atomic.NullThermalCXRate
   :members:

Photon Emissivity Coefficients
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: cherab.atomic.rates.pec.ImpactExcitationPEC
   :members:

.. autoclass:: cherab.atomic.rates.pec.NullImpactExcitationPEC
   :members:

.. autoclass:: cherab.atomic.rates.pec.RecombinationPEC
   :members:

.. autoclass:: cherab.atomic.rates.pec.NullRecombinationPEC
   :members:

Beam-Plasma Interaction Rates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: cherab.atomic.rates.cx.BeamCXPEC
   :members:

.. autoclass:: cherab.atomic.rates.cx.NullBeamCXPEC
   :members:

.. autoclass:: cherab.atomic.rates.beam.BeamStoppingRate
   :members:

.. autoclass:: cherab.atomic.rates.beam.NullBeamStoppingRate
   :members:

.. autoclass:: cherab.atomic.rates.beam.BeamPopulationRate
   :members:

.. autoclass:: cherab.atomic.rates.beam.NullBeamPopulationRate
   :members:

.. autoclass:: cherab.atomic.rates.beam.BeamEmissionPEC
   :members:

.. autoclass:: cherab.atomic.rates.beam.NullBeamEmissionPEC
   :members:

Abundances
^^^^^^^^^^

.. autoclass:: cherab.atomic.rates.fractional_abundance.FractionalAbundance
   :members:

Radiated Power
^^^^^^^^^^^^^^

.. autoclass:: cherab.atomic.rates.radiated_power.LineRadiationPower
   :members:

.. autoclass:: cherab.atomic.rates.radiated_power.NullLineRadiationPower
   :members:

.. autoclass:: cherab.atomic.rates.radiated_power.ContinuumPower
   :members:

.. autoclass:: cherab.atomic.rates.radiated_power.NullContinuumPower
   :members:

.. autoclass:: cherab.atomic.rates.radiated_power.CXRadiationPower
   :members:

.. autoclass:: cherab.atomic.rates.radiated_power.NullCXRadiationPower
   :members:

.. autoclass:: cherab.atomic.rates.radiated_power.TotalRadiatedPower
   :members:

.. autoclass:: cherab.atomic.rates.radiated_power.NullTotalRadiatedPower
   :members:

Gaunt Factors
-------------

.. autoclass:: cherab.atomic.gaunt.gaunt.FreeFreeGauntFactor
   :members:

Zeeman structure
----------------

.. autoclass:: cherab.atomic.zeeman.zeeman.ZeemanStructure
   :members:
