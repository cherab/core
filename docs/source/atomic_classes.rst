
Atomic Classes
==============

Importable elements:

* hydrogen
* helium
* lithium
* beryllium
* boron
* carbon
* nitrogen
* oxygen
* fluorine
* neon

Importable isotopes:

* protium
* deuterium
* tritium
* helium3
* helium4


Elements
--------

.. autoclass:: cherab.core.atomic.elements.Element
   :members:

Isotopes
--------

.. autoclass:: cherab.core.atomic.elements.Isotope
   :members:


Reading atomic coefficients
---------------------------

Classes to read atomic coefficients.

.. Commented out because of module refactoring
   autoclass:: cherab.atomic.adas.ADAS
   :members:
   autoclass:: cherab.atomic.adas.adas.CXRate
   :members:
   :special-members: __call__
   autoclass:: cherab.atomic.adas.adas.BeamCoefficient
   :members:
   :special-members: __call__
