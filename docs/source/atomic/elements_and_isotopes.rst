
Elements and Isotopes
---------------------

.. autoclass:: cherab.core.atomic.elements.Element
   :members:


Some examples of commonly used pre-defined elements:

    >>> from cherab.core.atomic import hydrogen, helium, lithium, beryllium, boron, \
    >>>     carbon, nitrogen, oxygen, fluorine, neon, argon, krypton, xenon

For the full list of available elements please consult the
`source file <https://github.com/cherab/core/blob/master/cherab/core/atomic/elements.pyx>`_ .

.. autofunction:: cherab.core.atomic.elements.lookup_element


.. autoclass:: cherab.core.atomic.elements.Isotope
   :members:


Some examples of commonly used pre-defined isotopes:

    >>> from cherab.core.atomic import protium, deuterium, tritium, helium3, helium4

For the full list of available isotopes please consult the
`source file <https://github.com/cherab/core/blob/master/cherab/core/atomic/elements.pyx>`_ .

.. autofunction:: cherab.core.atomic.elements.lookup_isotope

