
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


