
Functions
=========

The Cherab function framework is built on the core Raysect 1D, 2D and 3D
`function framework <https://raysect.github.io/documentation/api_reference/core/functions.html>`_.
For more details on the base functionality refer to the Raysect
documentation and the Cherab function tutorials.

Cherab previously provided vector functions which were not present in Raysect.
New codes should prefer the Raysect vector functions, but the old aliases are preserved for backwards compatibility.

The Function6D framework in Cherab aims to provide a framework for building six-dimensional distribution functions.
It follows closely Raysect's function framework.

6D Scalar Functions
-------------------

.. autoclass:: cherab.core.math.function.float.function6d.base.Function6D
   :members:
   :special-members: __call__

.. autoclass:: cherab.core.math.function.float.function6d.constant.Constant6D
   :show-inheritance:

.. autoclass:: cherab.core.math.function.float.function6d.arg.Arg6D
   :show-inheritance:

.. autoclass:: cherab.core.math.function.float.function6d.cmath.Exp6D
   :show-inheritance:

.. autoclass:: cherab.core.math.function.float.function6d.cmath.Sin6D
   :show-inheritance:

.. autoclass:: cherab.core.math.function.float.function6d.cmath.Cos6D
   :show-inheritance:

.. autoclass:: cherab.core.math.function.float.function6d.cmath.Tan6D
   :show-inheritance:

.. autoclass:: cherab.core.math.function.float.function6d.cmath.Asin6D
   :show-inheritance:

.. autoclass:: cherab.core.math.function.float.function6d.cmath.Acos6D
   :show-inheritance:

.. autoclass:: cherab.core.math.function.float.function6d.cmath.Atan6D
   :show-inheritance:

.. autoclass:: cherab.core.math.function.float.function6d.cmath.Atan4Q6D
   :show-inheritance:

.. autoclass:: cherab.core.math.function.float.function6d.cmath.Sqrt6D
   :show-inheritance:

.. autoclass:: cherab.core.math.function.float.function6d.cmath.Erf6D
   :show-inheritance:

.. autoclass:: cherab.core.math.function.float.function6d.blend.Blend6D
   :show-inheritance:

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
