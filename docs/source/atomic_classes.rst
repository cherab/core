
Atomic Classes
==============


Elements
--------

.. autoclass:: cherab.core.atomic.elements.Element
   :members:


The importable pre-defined elements are:

    >>> from cherab.core.atomic import hydrogen, helium, lithium, beryllium, boron, \
    >>>     carbon, nitrogen, oxygen, fluorine, neon, argon, krypton, xenon


Isotopes
--------

.. autoclass:: cherab.core.atomic.elements.Isotope
   :members:


The importable pre-defined isotopes are:

    >>> from cherab.core.atomic import protium, deuterium, tritium, helium3, helium4


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
