
Observers
=========

Most plasma diagnostics can be easily modelled using the base observer types provided in
Raysect. In CHERAB we only provide a few specialist observers with extra functionality.


Bolometers
----------

.. autoclass:: cherab.tools.observers.bolometry.BolometerCamera
   :members:
   :special-members: __len__, __iter__, __getitem__

.. autoclass:: cherab.tools.observers.bolometry.BolometerSlit
   :members:

.. autoclass:: cherab.tools.observers.bolometry.BolometerFoil
   :members:
