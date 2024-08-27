:orphan:


.. _analytic_function_plasma:

Analytic Function Plasma
========================

This demonstration shows how to define a set of plasma distributions using analytic functions.
Each function must by implemented as a python callable. The rest of the code
shows how to use these functions in a plasma and visualises the results.

Note that while it is possible to use pure python functions for development, they are typically
~100 times slower than their cython counterparts. Therefore, for use cases where speed is important
we recommend moving these functions to cython classes. An alternative solution which may not require
writing and compiling any additional cython code is to use Raysect's
`function framework <https://www.raysect.org/api_reference/core/functions.html>`_ to build up
expressions which will be evaluated like Python functions. These will typically run slightly slower
than a hand-coded cython implementation but still significantly faster than a pure python
implementation.

Two examples are provided, one using a pure python implementation of analytic forms for neutral and
ion plasma species distributions, and one using objects from Raysect's function framework.

.. tabs::

   .. tab:: Pure python

       .. literalinclude:: ../../../../demos/plasmas/analytic_plasma.py

   .. tab:: Function framework

       .. literalinclude:: ../../../../demos/plasmas/analytic_plasma_function_framework.py

.. figure:: analytic_plasma_slices.png
   :align: center
   :width: 650px

.. figure:: analytic_plasma.png
   :align: center
   :width: 450px
