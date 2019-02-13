
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


Code examples gallery
=====================

.. list-table:: CHERAB Examples
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
   * - :ref:`Surface radiation loads <surface_radiation_loads>`
     - Demo of loading a plasma from SOLPS and using its radiation data to calculate
       total radiation arriving at surfaces.
     - .. image:: ./radiation_wall_loads/AUG_wall_outline.png
          :height: 150px
          :width: 150px
