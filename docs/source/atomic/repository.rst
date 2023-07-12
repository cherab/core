
Atomic data repository
----------------------

The following functions allow to manipulate the local atomic data repository.

The default atomic data repository is created at `~/.cherab/atomicdata/default_repository`.

To create the new atomic data repository at the default location and populate it with a typical
set of rates and wavelengths from Open-ADAS, do:

.. code-block:: pycon

   >>> from cherab.atomic.repository import populate
   >>> populate()


.. autofunction:: cherab.atomic.repository.create.populate

Wavelength
^^^^^^^^^^

.. automodule:: cherab.atomic.repository.wavelength
    :members:

Ionisation
^^^^^^^^^^

.. autofunction:: cherab.atomic.repository.atomic.add_ionisation_rate

.. autofunction:: cherab.atomic.repository.atomic.get_ionisation_rate

.. autofunction:: cherab.atomic.repository.atomic.update_ionisation_rates

Recombination
^^^^^^^^^^^^^

.. autofunction:: cherab.atomic.repository.atomic.add_recombination_rate

.. autofunction:: cherab.atomic.repository.atomic.get_recombination_rate

.. autofunction:: cherab.atomic.repository.atomic.update_recombination_rates

Thermal Charge Exchange
^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: cherab.atomic.repository.atomic.add_thermal_cx_rate

.. autofunction:: cherab.atomic.repository.atomic.get_thermal_cx_rate

.. autofunction:: cherab.atomic.repository.atomic.update_thermal_cx_rates

Photon Emissivity Coefficients
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: cherab.atomic.repository.pec
    :members:

Radiated Power
^^^^^^^^^^^^^^

.. automodule:: cherab.atomic.repository.radiated_power
    :members:

Fractional Abundance
^^^^^^^^^^^^^^^^^^^^

.. automodule:: cherab.atomic.repository.fractional_abundance
    :members:

Beam
^^^^

.. automodule:: cherab.atomic.repository.beam.cx
    :members:

.. automodule:: cherab.atomic.repository.beam.emission
    :members:

.. automodule:: cherab.atomic.repository.beam.population
    :members:

.. automodule:: cherab.atomic.repository.beam.stopping
    :members:

