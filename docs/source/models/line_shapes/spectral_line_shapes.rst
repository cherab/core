
Spectral Line Shapes
====================

Cherab contains Doppler-broadened, Doppler-Zeeman and Stark-Doppler-Zeeman line shapes of
atomic spectra.

**Assumption: Maxwellian distribution of emitting species is assumed.**
**A general model of Doppler broadening will be implemented in the future.**

.. autoclass:: cherab.core.model.lineshape.doppler.doppler_shift

.. autoclass:: cherab.core.model.lineshape.doppler.thermal_broadening

.. autoclass:: cherab.core.model.lineshape.gaussian.add_gaussian_line

.. autoclass:: cherab.core.model.lineshape.stark.add_lorentzian_line

.. autoclass:: cherab.core.model.lineshape.base.LineShapeModel
   :members:

.. autoclass:: cherab.core.model.lineshape.gaussian.GaussianLine
   :members:

.. autoclass:: cherab.core.model.lineshape.multiplet.MultipletLineShape
   :members:

.. autoclass:: cherab.core.model.lineshape.zeeman.ZeemanLineShapeModel
   :members:

.. autoclass:: cherab.core.model.lineshape.zeeman.ZeemanTriplet
   :members:

.. autoclass:: cherab.core.model.lineshape.zeeman.ParametrisedZeemanTriplet
   :members:

.. autoclass:: cherab.core.model.lineshape.zeeman.ZeemanMultiplet
   :members:

.. autoclass:: cherab.core.model.lineshape.stark.StarkBroadenedLine
   :members:

.. autoclass:: cherab.core.model.lineshape.beam.base.BeamLineShapeModel
   :members:

.. autoclass:: cherab.core.model.lineshape.beam.mse.BeamEmissionMultiplet
   :members:
