
Functions
=========

The Cherab function framework is built on the core Raysect 1D, 2D and 3D
`function framework <https://raysect.github.io/documentation/api_reference/core/functions.html>`_.
For more details on the base functionality refer to the Raysect
documentation and the Cherab function tutorials.

Cherab previously provided vector functions which were not present in Raysect.
New codes should prefer the Raysect vector functions, but the old aliases are preserved for backwards compatibility.

2D Vector Functions
-------------------

.. autoclass:: cherab.core.math.function.VectorFunction2D
   :members:

.. autoclass:: cherab.core.math.function.ScalarToVectorFunction2D
   :members:
   :show-inheritance:


3D Vector Functions
-------------------

.. autoclass:: cherab.core.math.function.VectorFunction3D
   :members:

.. autoclass:: cherab.core.math.function.ScalarToVectorFunction3D
   :members:
   :show-inheritance:
