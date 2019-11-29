
.. _ray_transfer_map:

Axisymmetrical (toroidal) regular grid with custom mapping of light sources
===========================================================================

This example shows how to:

 * calculate ray transfer matrix (geometry matrix) for axisymmetrical cylindrical emitter defined on a regular grid
 * map multiple grid cells to a single light source by applying a voxel map


.. literalinclude:: ../../../../demos/ray_transfer/4_ray_transfer_map.py

.. figure:: ray_transfer_map_demo.gif
   :align: center
   :width: 450px

   **Caption:** The result of collapsing geometry matrix with 30 different emission profiles.
