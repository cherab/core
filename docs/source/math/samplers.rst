
Samplers
========

This module provides a set of sampling functions for rapidly generating samples
of 1D, 2D and 3D functions.

These functions use C calls when sampling Function1D, Function2D and Function3D
objects and are therefore considerably faster than the equivalent Python code.

1D Sampling
-----------

.. autoclass:: cherab.core.math.samplers.sample1d
   :members:

.. autoclass:: cherab.core.math.samplers.sample1d_points
   :members:


2D Sampling
-----------

.. autoclass:: cherab.core.math.samplers.sample2d
   :members:

.. autoclass:: cherab.core.math.samplers.sample2d_points
   :members:

.. autoclass:: cherab.core.math.samplers.sample2d_grid
   :members:

.. autoclass:: cherab.core.math.samplers.samplevector2d
   :members:

.. autoclass:: cherab.core.math.samplers.samplevector2d_points
   :members:

.. autoclass:: cherab.core.math.samplers.samplevector2d_grid
   :members:


3D Sampling
-----------

.. autoclass:: cherab.core.math.samplers.sample3d
   :members:

.. autoclass:: cherab.core.math.samplers.sample3d_points
   :members:

.. autoclass:: cherab.core.math.samplers.sample3d_grid
   :members:

.. autoclass:: cherab.core.math.samplers.samplevector3d
   :members:

.. autoclass:: cherab.core.math.samplers.samplevector3d_points
   :members:

.. autoclass:: cherab.core.math.samplers.samplevector3d_grid
   :members:
