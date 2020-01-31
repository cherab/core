============================
Beam attenuation calcualtion
============================

.. *Date: 03/12/2014*

In this section, the calculation of beam attenuation in Cherab is
presented. This documentation is meant to be a concise compilation
of the main aspects needed to carry out a beam attenuation calculation
highlighting what could be implemented in Cherab and what is implemented
at present and how.

.. WARNING::
    This documentation is useful only if it is correct and up-to-date.
    The reader and code developer is asked to report any mistake to the
    documentation manager. Thanks.

.. csv-table:: Notations for this section
    :header: "Notation", "Meaning", "Unit"

    ":math:`n_e`", "Electron density", ":math:`m^{-3}`"
    ":math:`n_{H^0}`", "Density of atom :math:`H^0`", ":math:`m^{-3}`"
    ":math:`n_I`", "Total ion density", ":math:`m^{-3}`"
    ":math:`n_I^{i, equiv}`", "Equivalent ion density of ion *i*", ":math:`m^{-3}`"
    ":math:`n_i`", "Density of plasma ion *i*", ":math:`m^{-3}`"
    ":math:`n_e^i`", "Electron density corresponding to ion *i*", ":math:`m^{-3}`"
    ":math:`n_e^{i, equiv}`", "Equivalent electron density of ion *i*, to be used in the extraction of the stopping coefficient.", ":math:`m^{-3}`"
    ":math:`\alpha_i`", "Charge of plasma ion *i*", "dimensionless (integer)"
    ":math:`f_i`", "Ionic fraction of ion *i*", "dimensionless"
    ":math:`T_i`", "Temperature of ion *i*", "*eV*"
    ":math:`E_c^i`", "Collision energy between plasma ions *i* and beam neutrals", "*eV/amu*"
    ":math:`Q_{ST}`", "Ionisation rate of atom :math:`H^0`", ":math:`m^{-3}.s^{-1}`"
    ":math:`Q_{ST}^i`", "Ionisation rate of atom :math:`H^0` by ions *i* and their electrons", ":math:`m^{-3}.s^{-1}`"
    ":math:`S_{ST}`", "Composite beam stopping coefficient for atom :math:`H^0` defined with respect to electron density.", ":math:`m^3/s`"
    ":math:`S_{ST}^i`", "Beam stopping coefficient for atom :math:`H^0` by ions *i* and their electrons defined with respect to ion density.", ":math:`m^3/s`"
    ":math:`S_{ST}^{i, e}`", "Beam stopping coefficient for atom :math:`H^0` by ions *i* and their electrons defined with respect to electron density.", ":math:`m^3/s`"
    ":math:`Z_{eff}`", "Plasma effective charge", "dimensionless"

The rate :math:`Q_{ST}^i` at which neutrals are ionised in the beam by one
ionic species *i* can be calculated from the beam stopping coefficient
:math:`S_{ST}^i` with the formula:

.. math::
    Q_{ST}^i = n_i.n_{H^0}.S_{ST}^i \quad \text{with} \quad \left.\frac{n_{H^0}}{dt}\right|_{\text{(ionisation by ion i)}} = - Q_{ST}^i

:math:`S_{ST}^i` corresponds closely to the effective ionisation coefficient.
It depends on the collisional energy between ions and neutrals, ion temperature
and ion density: :math:`S_{ST}^i = S_{ST}^i[E_c^i, n_i, T_i]`. Here we consider
both main ion and impurities.

Stopping coefficients with respect to electron density
------------------------------------------------------

Although ion impact ionisation and charge transfer are the most efficient
reactions causing beam stopping, it is usual in fusion to define the stopping
coefficient with respect to the electron density:

.. math::
    Q_{ST}^i = n_e^i.n_{H^0}.S_{ST}^{i, e}

Still considering only one ionic species *i*, quasi-neutrality gives: :math:`n_i.\alpha_i = n_e^i`

So: :math:`Q_{ST}^i = n_i.n_{H^0}.\alpha_i.S_{ST}^{i, e}`

Which means: :math:`S_{ST}^{i, e} = S_{ST}^i/\alpha_i`

In reality, there is more than one ion specie in the plasma, and therefore the
calculation of the stopping coefficient is dependent on the plasma composition.
We will see below how an accurate calculation can be made for the effective
stopping coefficient and which simplification can be made for a more
time-efficient calculation.

The collisional-radiative model for beam stopping
-------------------------------------------------

It was found advantageous to embed the calculation of beam stopping (and beam
emission) by hydrogen in a more general picture of neutral hydrogen as a
radiating, ionizing and recombining species in the fusion plasma.

Following collisional-radiative theory, the model calculates the **quasi-static
equilibrium** excited population structure relative to the instantaneous hydrogen
ground-state and ionised-state populations in a very many n-shell bundled-n
approximation.

**Quasi-static equilibrium collisional-radiative assumption under these conditions**

    In a nutshell:

    * The time in which the population of the excited levels comes into
      equilibrium with the ground population density and plasma conditions is
      so short that the population density of the excited energy levels can be
      considered to be in equilibrium with the population density of the ground
      levels. However, metastable states represent an exception, these states
      have an equilibration time with the ground state comparable to the
      characteristic time of ionisation and recombination processes. So a
      quasi-steady state approximation is adopted where the equilibrium
      population densities of ordinary excited states are functions of the
      population density of the ground and metastable levels. The rate at which
      the population density of the ground and metastable states evolves is
      related to the collisional radiative and ionisation rate coefficients.
    * The upper levels approach Saha-Boltzmann population density because the
      probability for radiative transition, A, decreases with increasing
      quantum number whilst the collisional rate increases. Therefore, for
      energy levels higher than a level :math:`n_s`, the population densities are
      governed by collisional processes, leading to the Saha-Boltzmann
      population.

**Assumption: there is no need to take into account the effect of spatial transport of metastable**

    Probably ok for beam stopping, not always for beam emission.

    - The localisation of beam emission measurements of the confined plasma,
      analysed with adoption of the theoretical quasi-static equilibrium
      assumption, cannot be better than :math:`3.10^{-2} m` in the JET experiment [Anderson2000].
    - Neutral beam diagnostic probes of edge/scrape-off-layer plasma with
      temperature and density scale lengths of about :math:`3.10^{-2} m` must be modelled
      in the full , spatially non-equilibrium picture.
    - As has been discussed in previous work, hydrogen with its high first
      excitation energy is suited as an equilibrium, fast, deep neutral beam
      probe while a medium/slow lithium beam is suited as a non-equlibrium edge
      probe.
    - A helium beam, because of the presence of the triplet metastable, shows
      a mixed character in which the detailed interplay of relaxation times,
      quasi-static equilibrium assumptions and plasma scale lengths are
      demonstrated. Analysis of the helium diagnostic beam thus requires
      generalized collisional-radiative theory.

For hydrogenic systems only populations of complete n shells need to be
evaluated (assume relative statistical population for the l states). Thus for
hydrogenic systems only populations of complete n-shells need to be evaluated,
the bundle-n approximation.

.. topic:: ADAS310

    Full implementation of the collisional-radiative model for hydrogen in the
    bundled-n approximation. It iterates through sets of plasma conditions,
    which include the plasma density, temperature and neutral beam energy.
    Output from ADAS310 includes all collisional-radiative ionization and
    recombination coefficients :math:`S_{CR}` and :math:`\alpha_{CR}`, with
    and without the influence of charge-transfer, at each set if plasma
    conditions. :math:`S_{CR}` are in fact the :math:`S_{ST}^{i, e}` mentioned
    above. This is a structured output organised according to ADAS data format
    adf26. ADAS310 accepts as input the definitions of these scans. Established
    an extended list of cases required to achieve the latter and then executes
    repeated population calculations at each set of plasma conditions in the
    list.

ADAS310 can compute the populations for any mixture of light impurities (H+ to
Ne10+) in the plasma. It is impractical to tabulate all possible mixtures of
impurities. It is our usual practise to execute ADAS310 in turn for each light
impurity species from hydrogen to neon treated as a pure species. The mixed
species effective coefficients are constructed from these pure impurity
solutions as a linear superposition by the theoretical data acquisition
routines in CHEAP.

Building-up the beam stopping coefficient from the pure impurity solutions: simplification adopted to make calculation faster
-----------------------------------------------------------------------------------------------------------------------------

.. topic:: ADAS312

    It is the interactive post-processing code designed to extract effective
    stopping and emission coefficients from the comprehensive adf26 file and
    archive the data in condensed rapid-look-up tables in their respective ADAS
    data formats of adf21 and adf22. The effective coefficients are most
    sensitive to the beam particle energy and the plasma ion density and less
    sensitive to plasma ion temperature. Suitable tabulations can therefore be
    built on a reference set of plasma and beam conditions, namely a
    two-dimensional array of coefficients as functions of beam energy and
    plasma density at the reference conditions of the plasma ion temperature
    and then a one-dimensional vector of coefficients as a function of the
    plasma ion temperature at the reference conditions of the other parameters.

If now instead of one ion species, we consider n ion species, the rate at which
atoms :math:`H^0` in a beam are ionised :math:`Q_{ST}` can be written as:

.. math::
    Q_{ST}=\sum_iQ_{ST}^i=\sum_i n_in_{H^0}S_{ST}^i\quad\text{where}\quad\left.\frac{n_{H^0}}{dt}\right|_{\text{(all ionisations)}}=-Q_{ST}

Now replacing :math:`S_{ST}^i` with :math:`S_{ST}^{i, e}`:

.. math::
    Q_{ST}=\sum_in_in_{H^0}\alpha_iS_{ST}^{i, e}=n_{H^0}\sum_in_i^eS_{ST}^{i, e}

Defining the composite stopping coefficient with respect to electron density :math:`S_{ST}` by:

.. math::
    Q_{ST}=n_en_{H^0}S_{ST}

This leads to:

.. math::
    n_e.S_{ST}=n_{H^0}\sum_in_i^eS_{ST}^{i, e}

The coefficient is written in terms of the primary parameters :math:`E_c^i, n_i, T_i`.
It remains to define what should be taken as the parameter for electron density
which is equivalent electron density to be used in the extraction of the
stopping coefficient contribution from the :math:`i^{th}` pure impurity archive
equivalent to the ion density :math:`n_i` in the case of this multi-ion plasma.

In this composite ion plasma, the effective charge is:

.. math::
    Z_{eff}=\frac{\sum_jn_j\alpha_j^2}{\sum_jn_j\alpha_j}=\frac{\sum_jn_j\alpha_j^2}{n_e}

The following assumption is made, that the quantity we are keeping constant for
the plasma conditions is :math:`n_e.Z_{eff}`. Then, the equivalent ion density is defined by:

.. math::
    n_e.Z_{eff}=n_I^{i, equiv}\alpha_i^2

Therefore, the equivalent electron density is:

.. math::
    n_e^{i, equiv}=\alpha_in_I^{i, equiv}=\frac{n_e}{\alpha_i}Z_{eff}

Finally:

.. math::
    n_e^{i, equiv}=\frac{\sum_jn_j\alpha_j^2}{\alpha_i}

A detailed error analysis of the effect of this approximations has been carried
out by Anderson[1999] by comparison with ADAS310 calculations using the true
mixtures. The errors from the compact tabulations of the beam emission
coefficient are more substantial, but less than 1% for reasonable ranges of
plasma and beam conditons. [Delabie also in annex].

.. WARNING::
    It is appropriate to re-tune the tabulations of reference beam and plasma
    conditions for application to other plasma/injection systems. (He beams, …..)

Features implemented in Cherab
------------------------------

**Assumption: no need to take into account the spatial transport of metastables in the plasma**

    (more relevant to beam emission)

Litterature on which this note is based
---------------------------------------

Anderson 2000
