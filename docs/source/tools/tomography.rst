
Tomography
==========

Tomographic inversion problems often arise in the study of plasma diagnostics, especially
when making line integrated measurements. Tomographic inversion techniques allow us to
try recovering the underlying plasma properties from a reduced set of measurements. In CHERAB
we implement a basic set of tomography algorithms, for a wider range of options please
consult the dedicated tomography libraries such as `ToFu <https://github.com/ToFuProject/tofu>`_.

In general, recovering a plasma emission profile with tomography is an ill-posed problem.
It is customary to describe the system in terms of a sensitivity matrix :math:`\mathbf{W}`. The
elements :math:`W_{k,l}` describe the coupling between the :math:`N_s` plasma emission voxels
:math:`x_l` and measured power :math:`\Phi_k` at :math:`N_d` detectors. The whole detector set
is typically represented as the matrix equation

.. math::

   \mathbf{\Phi} = \mathbf{W} \mathbf{x}.

The power for the *k* th detector can be expressed as

.. math::
   \Phi_k = \sum_{l=1}^{N_s} W_{k,l} \, x_l,

where :math:`k` and :math:`l` are the indices for the detectors and source voxels respectively.


In this module we implement a number of basic inversion algorithms for recovering the emissivity
vector :math:`\mathbf{x}` from a set of measurements :math:`\mathbf{\Phi}` and sensitivity
matrix :math:`\mathbf{W}`.


Inversion Methods
-----------------

.. autofunction:: cherab.tools.inversions.sart.invert_sart

.. autofunction:: cherab.tools.inversions.sart.invert_constrained_sart

.. autoclass:: cherab.tools.inversions.opencl.sart_opencl.SartOpencl
   :members: __call__, clean, update_laplacian_matrix

.. autofunction:: cherab.tools.inversions.nnls.invert_regularised_nnls

.. autofunction:: cherab.tools.inversions.svd.invert_svd


Voxels
------

.. autoclass:: cherab.tools.inversions.voxels.Voxel
   :members:

.. autoclass:: cherab.tools.inversions.voxels.AxisymmetricVoxel
   :members:

.. autoclass:: cherab.tools.inversions.voxels.VoxelCollection
   :members:

.. autoclass:: cherab.tools.inversions.voxels.ToroidalVoxelGrid
   :members:


Ray Transfer Objects
--------------------

Ray transfer objects accelerate the calculation of geometry matrices (or Ray Transfer Matrices as they were called 
in `S. Kajita, et al. Contrib. Plasma Phys., 2016, 1-9 <https://onlinelibrary.wiley.com/doi/abs/10.1002/ctpp.201500124>`_) 
in the case of regular spatial grids. As in the case of Voxels, the spectral array is used to store the data 
for individual light sources (in this case the grid cells or their unions), however no voxels are created at all. 
Instead, a custom integration along the ray is implemented. Use `RayTransferBox` class for Cartesian grids 
and `RayTransferCylinder` class for cylindrical grids (3D or axisymmetrical).

Performance tips:

* The best performance is achieved when Ray Transfer Objects are used with special pipelines and optimised materials (currently only rough metals are optimised, see the demos).
* When the number of individual light sources and respective bins in the spectral array is higher than ~50-70 thousands, the lack of CPU cache memory becomes a serious factor affecting performance. Therefore, it is not recommended to use hyper-threading when calculating geometry matrices for a large number of light sources. It is also recommended to divide the calculation into several parts and to calculate partial geometry matrices for not more than ~50-70 thousands of light sources in a single run. Partial geometry matrices can easily be combined into one when all computations are complete.

.. autoclass:: cherab.tools.raytransfer.raytransfer.RayTransferBox
   :members: invert_voxel_map

.. autoclass:: cherab.tools.raytransfer.raytransfer.RayTransferCylinder
   :members: invert_voxel_map

.. autoclass:: cherab.tools.raytransfer.pipelines.RayTransferPipeline0D

.. autoclass:: cherab.tools.raytransfer.pipelines.RayTransferPipeline1D

.. autoclass:: cherab.tools.raytransfer.pipelines.RayTransferPipeline2D
