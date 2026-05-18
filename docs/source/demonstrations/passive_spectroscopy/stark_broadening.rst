:orphan:


.. _stark_broadening:

Stark Broadening
================

Normally, the dominant factor in determining the lineshape is the thermal
doppler broadening. However, in certain high density plasma scenarios a
secondary effect can take over, known as pressure broadening. This effect
results from the fact that radiating ions experience a change in the
electric field due to the presence of neighbouring ions. In normal tokamak
operations this effect is negligible, except in the divertor region.

It is possible to override the default doppler broadened line shape by
specifying the StarkBroadenedLine() lineshape class.
This class suppors Balmer and Paschen series and is based on
the Stark-Doppler-Zeeman line shape model from B. Lomanowski, et al.
"Inferring divertor plasma properties from hydrogen Balmer
and Paschen series spectroscopy in JET-ILW." Nuclear Fusion 55.12 (2015)
`123028 <https://doi.org/10.1088/0029-5515/55/12/123028>`_.
In this example we can see two stark broadened balmer series lines surrounding a
Nitrogen II multiplet feature. The logarithmic scale is chosen to illustrate
the power-law decay of the Stark-broadened line wings.

.. literalinclude:: ../../../../demos/emission_models/stark_broadening.py

.. figure:: stark_spectrum.png
   :align: center
   :width: 450px

   **Caption:** The observed spectrum with two stark broadened balmer lines
   (397nm and 410nm) surrounding a NII multiplet feature in the logarithmic scale.
