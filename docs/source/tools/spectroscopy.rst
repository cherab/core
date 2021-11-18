
Spectroscopy
============

The tools for plasma spectroscopy.

.. _spectroscopy_instruments:

Spectroscopic instruments
-------------------------

Spectroscopic instruments such as polychromators and spectrometers
simplify the setup of properties of the observers and rendering pipelines. The instruments
are not connected to the scenegraph, so they cannot observe the world. However, the instruments
have properties, such as `min_wavelength`, `max_wavelength`, `spectral_bins`,
`pipeline_properties`, with which the observer can be configured.
The Cherab core package provides base classes for spectroscopic instruments,
so machine-specific packages can build more advance instruments from them, such as instruments
with spectral properties based on the actual experimental setup for a given shot/pulse.

.. autoclass:: cherab.tools.spectroscopy.SpectroscopicInstrument
   :members:

.. autoclass:: cherab.tools.spectroscopy.PolychromatorFilter
   :members:

.. autoclass:: cherab.tools.spectroscopy.TrapezoidalFilter
   :show-inheritance:
   :members:

.. autoclass:: cherab.tools.spectroscopy.Polychromator
   :show-inheritance:
   :members:

.. autoclass:: cherab.tools.spectroscopy.Spectrometer
   :show-inheritance:
   :members:

.. autoclass:: cherab.tools.spectroscopy.CzernyTurnerSpectrometer
   :show-inheritance:
   :members:

