
Interpolators
-------------

Historically Cherab provided 1D, 2D and 3D interpolators.
New codes are encouraged to use Raysect's `interpolators <https://www.raysect.org/api_reference/core/functions.html#interpolators>`_ instead.
For backwards compatibility the Cherab interpolators are retained for now.


.. autoclass:: cherab.core.math.interpolators.interpolators1d.Interpolate1DLinear
   :members:

.. autoclass:: cherab.core.math.interpolators.interpolators1d.Interpolate1DCubic
   :members:

.. autoclass:: cherab.core.math.interpolators.interpolators2d.Interpolate2DLinear
   :members:

.. autoclass:: cherab.core.math.interpolators.interpolators2d.Interpolate2DCubic
   :members:

.. autoclass:: cherab.core.math.interpolators.interpolators3d.Interpolate3DLinear
   :members:

.. autoclass:: cherab.core.math.interpolators.interpolators3d.Interpolate3DCubic
   :members:
