
Samplers
========

This module provides a set of sampling functions for rapidly generating samples
of 1D, 2D and 3D functions.

These functions use C calls when sampling Function1D, Function2D and Function3D
objects and are therefore considerably faster than the equivalent Python code.

1D Sampling
-----------

.. autofunction:: cherab.core.math.samplers.sample1d

.. autofunction:: cherab.core.math.samplers.sample1d_points


2D Sampling
-----------

.. autofunction:: cherab.core.math.samplers.sample2d

.. autofunction:: cherab.core.math.samplers.sample2d_points

.. autofunction:: cherab.core.math.samplers.sample2d_grid

.. autofunction:: cherab.core.math.samplers.samplevector2d

.. autofunction:: cherab.core.math.samplers.samplevector2d_points

.. autofunction:: cherab.core.math.samplers.samplevector2d_grid


3D Sampling
-----------

.. autofunction:: cherab.core.math.samplers.sample3d

.. autofunction:: cherab.core.math.samplers.sample3d_points

.. autofunction:: cherab.core.math.samplers.sample3d_grid

.. autofunction:: cherab.core.math.samplers.samplevector3d

.. autofunction:: cherab.core.math.samplers.samplevector3d_points

.. autofunction:: cherab.core.math.samplers.samplevector3d_grid

