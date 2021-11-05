
.. _multiplet_lines:

Multiplet Line Ratios
=====================

Some lines split into multiple components, requiring a higher spectroscopy
resolution than is normally modelled in Cherab. It is possible to add the
experimentally measured multiplet ratios by using the MultipletLineShape()
class. The ratios between the multiplet lines will be constant, but the total
emissivity of the multiplet will match the original emissivity as specified
by your atomic data.

In this example we specify a Nitrogen II multiplet using experimental line
ratios from Table I of the paper: Henderson, S.S., et al. "Determination of
volumetric plasma parameters from spectroscopic N II and N III line ratio
measurements in the ASDEX Upgrade divertor." Nuclear Fusion 58.1 (2017):
016047.


.. literalinclude:: ../../../../demos/emission_models/multiplet.py

.. figure:: multiplet_spectrum.png
   :align: center
   :width: 450px

   **Caption:** An observed NII multiplet in the region of 402-407nm. The multiplet
   is surrounded by two brighter hydrogen Balmer series lines.
