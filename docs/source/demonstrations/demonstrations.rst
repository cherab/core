
Atomic Data
===========

.. list-table::
   :widths: 28 50 22
   :header-rows: 1

   * - Name
     - Description
     - Preview
   * - :ref:`Photon Emissivity Coefficients <photon_emissivity_coefficients>`
     - Sampling and plotting PEC rates provided by OPEN-ADAS.
     - .. image:: ./atomic_data/D_alpha_PECs.png
          :height: 150px
          :width: 150px
   * - :ref:`Beam-Plasma Interaction Rates <beam_plasma_interaction_rates>`
     - Sampling and plotting various beam-plasma interaction rates provided by OPEN-ADAS.
     - .. image:: ./atomic_data/effective_cx_rates.png
          :height: 150px
          :width: 150px
   * - :ref:`Fractional Abundances <fractional_abundances>`
     - Sampling and plotting Neon fractional abundances with the ADAS subscription package.
     - .. image:: ./atomic_data/fractional_abundance.png
          :height: 150px
          :width: 150px
   * - :ref:`Radiated Powers <radiated_powers>`
     - Sampling and plotting total and stage resolved radiated powers with the ADAS
       subscription package.
     - .. image:: ./atomic_data/stage_resolved_radiation.png
          :height: 150px
          :width: 150px


Creating Plasmas
================

.. list-table::
   :widths: 28 50 22
   :header-rows: 1

   * - Name
     - Description
     - Preview
   * - :ref:`Analytic Functions <analytic_function_plasma>`
     - Specifying plasma distributions with analytic functions.
     - .. image:: ./plasmas/analytic_plasma.png
          :height: 150px
          :width: 150px
   * - :ref:`Flux Function Plasmas <flux_function_plasmas>`
     - Loading the example EFIT equilibrium and making 1D flux functions.
     - .. image:: ./plasmas/equilibrium_surfaces.png
          :height: 150px
          :width: 150px
   * - :ref:`2D Mesh Plasmas <mesh2d_plasma>`
     - Specifying a plasma distribution with a 2D r-z triangular mesh.
     - .. image:: ./plasmas/mesh_plasma_column.png
          :height: 150px
          :width: 150px
   * - :ref:`Beams into Plasmas <beams_into_plasmas>`
     - Specifying a mono-energetic neutral beam that interacts with a plasma.
     - .. image:: ./plasmas/beam_into_plasma.png
          :height: 150px
          :width: 150px


Surface Radiation Loads
=======================

.. list-table::
   :widths: 28 50 22
   :header-rows: 1

   * - Name
     - Description
     - Preview
   * - :ref:`Defining A Radiation Function <radiation_function>`
     - Defining an example radiation function.
     - .. image:: ./radiation_loads/radiation_function.png
          :height: 150px
          :width: 150px
   * - :ref:`Defining A Wall From A 2D Polygon <wall_from_polygon>`
     - A toroidal mesh representing the tokamak wall is made from a
       2D polygon outline.
     - .. image:: ./radiation_loads/toroidal_wall.png
          :height: 150px
          :width: 150px
   * - :ref:`Symmetric Power Load Calculation <symmetric_power_load>`
     - Calculating the power load by exploiting symmetry. We manually
       create an array of detectors for sampling.
     - .. image:: ./radiation_loads/symmetric_power_load.png
          :height: 150px
          :width: 150px
   * - `Mesh Observer <https://raysect.github.io/documentation/demonstrations/observers/mesh_observers.html>`_
     - Calculating powers on an arbitrary 3D surface (Raysect docs).
     - .. image:: https://raysect.github.io/documentation/_images/mesh_observers.jpg
          :height: 150px
          :width: 150px
   * - :ref:`AUG - SOLPS radiation load example <aug_solps_radiation_load>`
     - An older demonstration of the tutorials above using a SOLPS simulation
       and an AUG wall outline.
     - .. image:: ./radiation_loads/AUG_wall_outline.png
          :height: 150px
          :width: 150px

Code examples gallery
=====================

.. list-table::
   :widths: 28 50 22
   :header-rows: 1

   * - Name
     - Description
     - Preview
   * - :ref:`CXRS Quickstart <jet_cxrs_quickstart>`
     - Commented demo file about how to use CHERAB for JET CX simulations.
     - .. image:: ./jet_cxrs/JET_CXRS_d5lines.png
          :height: 150px
          :width: 150px
   * - :ref:`#76666 sample analysis <jet_cxrs_76666>`
     - Demo CX analysis for pulse 76666 at t=61s
     -
   * - :ref:`MAST-U filtered cameras <mastu_forward_cameras>`
     - Example of using SOLPS simulation and ADAS rates to model filtered cameras.
     - .. image:: ./line_emission/mastu_bulletb_midplane_dgamma.png
          :height: 150px
          :width: 150px
   * - :ref:`MAST-U SOLPS plasma <mastu_solps_plasma>`
     - Example of loading a plasma from a SOLPS simulation and inspecting the various
       plasma species parameters.
     - .. image:: ./solps/species_narrow.png
          :height: 150px
          :width: 150px
   * - :ref:`Custom Emission Model <custom_emitter>`
     - Example of making a custom emitter class in CHERAB. D-alpha impact excitation
       is used for the example.
     - .. image:: ./line_emission/mastu_bulletb_midplane_dalpha.png
          :height: 150px
          :width: 150px
   * - :ref:`Measuring line of sight spectra <balmer_series_spectra>`
     - Basic balmer series measurement in the MAST-U divertor with an optical fibre.
       Localisation of the plasma emission is examined by plotting profiles of parameters
       such as density and temperature along the ray trajectory.
     - .. image:: ./line_emission/balmer_series_spectra.png
          :height: 150px
          :width: 150px
