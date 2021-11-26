
Spectral Line Shapes
====================

Cherab contains Doppler-broadened, Stark-broadened and Doppler-Zeeman line shapes of
atomic spectra. Stark-Doppler and Stark-Doppler-Zeeman line shapes of hydrogen spectra
will be added in the future.

**Assumption: Maxwellian distribution of emitting species is assumed.**
**A general model of Doppler broadening will be implemented in the future.**

.. autoclass:: cherab.core.model.lineshape.add_gaussian_line

.. autoclass:: cherab.core.model.lineshape.LineShapeModel
   :members:

.. autoclass:: cherab.core.model.lineshape.GaussianLine
   :members:

.. autoclass:: cherab.core.model.lineshape.MultipletLineShape
   :members:

.. autoclass:: cherab.core.model.lineshape.StarkBroadenedLine
   :members:

.. autoclass:: cherab.core.model.lineshape.ZeemanLineShapeModel
   :members:

.. autoclass:: cherab.core.model.lineshape.ZeemanTriplet
   :members:

.. autoclass:: cherab.core.model.lineshape.ParametrisedZeemanTriplet
   :members:

.. autoclass:: cherab.core.model.lineshape.ZeemanMultiplet
   :members:

