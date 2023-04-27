
Function Framework
==================


One of Cherab's most powerful features is the way it is designed to support mathematical
functions. Most of the physics interfaces in Cherab are specified in terms of math functions
of a specified dimension. I.e. density is a 3D function f(x, y, z), while :math:`\psi_n(r,z)`
is a 2D function in the r-z plane. Cherab leverages Raysect's function framework which provides
arithmetic, blending, interpolation and wrapping capabilities.

This section of the documentation describes the various additional
utilities that Cherab provides for slicing, dicing and projecting these functions.

.. toctree::
   caching
   clamp
   function
   interpolators
   mappers
   transform
   mask
   samplers
   slice
