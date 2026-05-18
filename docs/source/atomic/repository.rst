
Atomic data repository
----------------------

The following functions allow to manipulate the local atomic data repository:
add the rates of the atomic processes, update existing ones or get the data
already present in the repository.

The default repository is created at `~/.cherab/openadas/repository`.
Cherab supports multiple atomic data repositories. The user can configure different
repositories by setting the `repository_path` parameter.
The data in these repositories can be accessed through the `OpenADAS` atomic data provider
by specifying the `data_path` parameter.

To create the new atomic data repository at the default location and populate it with a typical
set of rates and wavelengths from Open-ADAS, do:

.. code-block:: pycon

   >>> from cherab.openadas.repository import populate
   >>> populate()

.. autofunction:: cherab.openadas.repository.create.populate

Wavelength
^^^^^^^^^^

.. automodule:: cherab.openadas.repository.wavelength
    :members:

Ionisation
^^^^^^^^^^

.. autofunction:: cherab.openadas.repository.atomic.add_ionisation_rate

.. autofunction:: cherab.openadas.repository.atomic.get_ionisation_rate

.. autofunction:: cherab.openadas.repository.atomic.update_ionisation_rates

Recombination
^^^^^^^^^^^^^

.. autofunction:: cherab.openadas.repository.atomic.add_recombination_rate

.. autofunction:: cherab.openadas.repository.atomic.get_recombination_rate

.. autofunction:: cherab.openadas.repository.atomic.update_recombination_rates

Thermal Charge Exchange
^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: cherab.openadas.repository.atomic.add_thermal_cx_rate

.. autofunction:: cherab.openadas.repository.atomic.get_thermal_cx_rate

.. autofunction:: cherab.openadas.repository.atomic.update_thermal_cx_rates

Photon Emissivity Coefficients
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: cherab.openadas.repository.pec
    :members:

Radiated Power
^^^^^^^^^^^^^^

.. automodule:: cherab.openadas.repository.radiated_power
    :members:

Beam
^^^^

.. automodule:: cherab.openadas.repository.beam.cx
    :members:

.. automodule:: cherab.openadas.repository.beam.emission
    :members:

.. automodule:: cherab.openadas.repository.beam.population
    :members:

.. automodule:: cherab.openadas.repository.beam.stopping
    :members:
