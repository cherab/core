
.. _ray_transfer_cylinder:

Cylindrical regular grid
========================

This example shows how to:

 * calculate ray transfer matrix (geometry matrix) for a cylindrical periodic emitter defined on a regular grid,
 * obtain images using calculated ray transfer matrix by collapsing it with various emission profiles.


.. literalinclude:: ../../../../demos/ray_transfer/2_ray_transfer_cylinder.py

.. figure:: ray_transfer_cylinder_demo.gif
   :align: center
   :width: 450px

   **Caption:** The result of collapsing geometry matrix with 64 different emission profiles.
