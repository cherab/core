.. Substitution definitions
.. |impurity| replace:: :math:`Z^{(\alpha +1)+}`


========================================================================
Calculation of predicted charge-exchange spectra seen by a line-of-sight
========================================================================

.. *Date: 24/11/2014*

In this section, the calculation of a predicted spectral line
:math:`I_{obs}(\lambda)` resulting from a charge-exchange process is presented
together with the different possible simplifications and details of what is
currently implemented in Cherab. This documentation is not meant to be a
course on charge-exchange but a concise compilation of the main aspects in
order to understand the assumptions made in Cherab for an educated use.

.. WARNING::
    This documentation is useful only if it is correct and up-to-date. It is
    asked of the reader and code developer to report any mistake to the
    documentation manager. Thanks.

.. NOTE::
    :math:`I_{obs}(\lambda)` in ray-tracer and therefore Cherab is in :math:`W.m^{-2}.str^{-1}.nm^{-1}`.

.. csv-table:: Notations for this section
    :header: "Notation", "Meaning", "Unit"

    ":math:`n_e`", "Electron density", ":math:`m^{-3}`"


Definition of charge-exchange coefficients
------------------------------------------

The charge exchange reaction between an ion |impurity| and a
donor :math:`H^0` (can be in this section deuterium, tritium, hydrogen or helium) is:

.. math:: H^0(m) + Z^{(\alpha +1)+} \longrightarrow H^+ + Z^{* \alpha+}(n, l, j) \longrightarrow H^+ + Z^{\alpha+} + h\nu_1 + h\nu_2 + ...

Where |impurity| is the ion capturing the donor electron into a
specific shell (n,l,j). The excited ions, :math:`Z^{* \alpha+}(n, l, j)`, decay
in one or more steps to their ground state under photon emission. It is
possible for the ion |impurity| to capture a donor electron from
the ground-state :math:`H^0(m=1)` but also from high-lying shells if the
electron donor is not ground-state but excited :math:`H^0(m=2)`.

A charge-exchange line observed from an ion :math:`Z^{\alpha+}` will be the
result of individual primary capture of donor electron onto specific shell of
the ion |impurity| which then will undergo a redistribution
process and cascade process affecting the excited populations of the recombined
ion :math:`Z^{* \alpha+}` in a thermal plasma. The quantity linking the
fundamental cross-sections and the charge-exchange observations is the
effective emission coefficient. The effective emission coefficient :math:`q^{CX, Z^{(\alpha+1)+}, H^0(m)}_{n\rightarrow n'}`
for a particular charge-exchange line (such as CVI :math:`n=8\rightarrow n'=7`)
incorporate these effects. The coefficient is defined such that the number of
:math:`n\rightarrow n'` photons emitted per unit volume per second per
:math:`4\pi` steradian is :math:`q^{CX, Z^{(\alpha+1)+}, H^0(m)}_{n\rightarrow n'}.n_{Z^{(\alpha +1)+}}.n_{H^0(m)}`
with :math:`n_{Z^{(\alpha +1)+}}` the number density of impurity receiver ions
of charge :math:`\alpha+1` in the plasma and :math:`n_{H^0(m)}` the number
density of beam donor atoms in the state :math:`m`. For example for D(n=2)
donors, m=2 and for the ground state D(n=1), m=1. The methods used to calculate
the effective emission coefficients, :math:`q^{CX, Z^{(\alpha+1)+}, H^0(m)}_{n\rightarrow n'}`
is calculated in a collisional-radiative nl-resolved l-mixing model of all the
excited populations of the :math:`\alpha+1` times ionized receiver ion. The
effective coefficients depend on the collision energy with the neutral beam
:math:`E_c`, plasma electron and ion temperatures, plasma impurity ion species
mix and number densities and electron density. The strongest dependences are on
collisional energy :math:`E_c` and ion density :math:`n_I`. (For more specific
use of ADAS effective charge-exchange coefficient look at Cherab atomic data
note). For a mixed impurity plasma of known fractions with charge neutrality,
the electron and ion densities are related and it is usual to use the electron
density as a parameter.

The composite charge-exchange emission coefficient with both :math:`H^0(m=1)`
and :math:`H^0(m=2)` beam donors is defined as:

.. math::
    q^{CX, Z^{(\alpha+1)+}}_{n\rightarrow n'} &= \frac{n_{H^0(1)} q^{CX, Z^{(\alpha+1)+}, H^0(1)}_{n\rightarrow n'} + n_{H^0(2)}.q^{CX, Z^{(\alpha+1)+}, H^0(2)}_{n\rightarrow n'}}{n_{H^0(1)} + n_{H^0(2)}} \\
                                              &= \frac{q^{CX, Z^{(\alpha+1)+}, H^0(1)}_{n\rightarrow n'} + bmp(2).q^{CX, Z^{(\alpha+1)+}, H^0(2)}_{n\rightarrow n'}}{1 + bmp(2)} \quad \text{with} \quad bmp(m) = \frac{n_{H^0(m)}}{n_{H^0(1)}}

Or in a more general way:

.. math::
    q^{CX, Z^{(\alpha+1)+}}_{n\rightarrow n'} = \frac{q^{CX, Z^{(\alpha+1)+}, H^0(1)}_{n\rightarrow n'} + \sum_{m=2}^M bmp(m).q^{CX, Z^{(\alpha+1)+}, H^0(m)}_{n\rightarrow n'}}{1 + \sum_{m=2}^M bmp(m)}

The ratio of population between :math:`n_{H^0(m)}` and :math:`n_{H^0(1)}` is
called :math:`bmp(m)`. The calculation of the donor population requires a
detailed collisional-radiative model including many ion and electron
collisional processes. For information, the fraction of excited beam atoms
depends not only on the beam energy but also on :math:`Z_{eff}`, plasma
temperature and densities. Note that D(n=1) donor dominates at high beam energy
and the small fraction of D(n=2) donor dominates at low beam energy
[Hoekstra1998]. For more practical information, look at the Cherab atomic data
note.

Calculation of the line intensity of CX spectral lines (without predicting observed spectra)
--------------------------------------------------------------------------------------------

As already mentioned above the number of :math:`n\rightarrow n'` photons
emitted per unit volume per second per :math:`4\pi` steradian can be calculated
by:

.. math:: I_{n\rightarrow n'} = n_{H^0}.n_{Z^{(\alpha +1)+}}.q^{CX, Z^{(\alpha+1)+}}_{n\rightarrow n'}

The light emitted by a specie :math:`Z^{\alpha+}` due to a transition of
electron from quantum :math:`n\rightarrow n'` has a natural wavelength :math:`\lambda_n`
associated and a natural line width :math:`\Delta\lambda_n`. It can be calculated
that this width is small and can be neglected. The spectral line intensity as
a function of wavelength is then given by:

.. math:: I_{n\rightarrow n'}(\lambda) = I_{n\rightarrow n'}.\delta(\lambda - \lambda_n)

**Assumption: natural width of the line neglected**

An observer looking at a direction :math:`\vec{l}` (:math:`\|\vec{l}\|=1`) at a photon with a velocity
:math:`\vec{u}` will see this photon at a wavelength :math:`\lambda_D = \lambda_n + \Delta\lambda_D`
with a Doppler shift :math:`\Delta\lambda_D = \lambda_n \frac{\vec{u}.\vec{l}}{c}`
with c the speed of light.

The observed spectral line intensity can then be defined as (ignoring all effects
but Doppler effect):

.. math::
    I_{n\rightarrow n'}(\lambda) = I_{n\rightarrow n'}.\delta(\lambda - \lambda_D)
                                 = I_{n\rightarrow n'}.\delta\left(\lambda - \lambda_n \left(1 + \frac{\vec{u}.\vec{l}}{c}\right)\right)

In more details charge-exchange cross-section
---------------------------------------------

Primary charge-exchange cross-section plays a principal role in this calculation,
since final derived impurity ion densities depend absolutely on their reliability.
Theoretical charge-exchange calculations especially for capture to the highly
excited levels of interest here show variation from one to another. Firstly,
the total charge exchange capture cross-section, that is summed over all levels,
is the most reliable quantity [Boileau89]. Very few experimental measurements
exist even of charge-exchange capture cross-section summed over all levels.
One such measurement was made with [Phaneuf82, Isler94], where a collimating
beam of multiply charge ions of charge :math:`\alpha+1` through a calibrated
atomic hydrogen gas target and detecting scattered product ions of charge :math:`\alpha`
separately from the primary ions of charge :math:`\alpha+1`.

When charged ions :math:`Z^{(\alpha +1)+}` are thrown against an atomic hydrogen
gas, the cross-section :math:`\sigma^{CX}_{Z^{(\alpha+1)+}}` is a hypothetical
area measure around the target particle of atoms that represents a surface. If
an ion crosses this surface, there will be some kind of interaction. In the
charge-exchange reaction, the cross-section is specific to a donor electron in
the level :math:`H^0(m)` and capture in the shell (n,l) of the ion :math:`Z^{(\alpha+1)+}`
(abandoning j for now), hence :math:`\sigma^{CX}_{Z^{(\alpha +1)+}}(m,n,l)`.

In general, the rate :math:`Q` at which a specific reaction occurs is a physical
quantity measuring the number of reactions per unit of time which can be written
as follow:

.. math:: Q = N.J.\sigma

with :math:`N` the number of target particles, illuminated by the beam containing
:math:`n` particles per unit volume (number density of particles) travelling with
the average velocity :math:`v` in the rest frame of the target and these two
quantities combine in the particle current density of the beam :math:`J=n.v`.
So the rate can be rewritten as:

.. math:: Q = N.n.v.\sigma

Now, we want to get back to the specific equation for the charge-exchange rate.
For our charge-exchange reaction, the target ions :math:`Z^{(\alpha+1)+}` have
a certain distribution :math:`f_{Z^{(\alpha+1)+}(n,l)}(\vec{x}, \vec{u}, t)`
that needs to be taken into account in the calculation:

.. math:: Q^{CX, Z^{(\alpha+1)+}(n,l), H^0(m)}_{n\rightarrow n'}(E_c) =
    n_{H^0(m)}.n_{Z^{(\alpha +1)+}}(n,l).\int_{\vec{u}}f_{Z^{(\alpha+1)+}(n,l)}(\vec{u}-\vec{v_B})
    .\|\vec{u}-\vec{v_B}\|.\sigma^{CX, Z^{(\alpha+1)+}(n,l), H^0(m)}_{n\rightarrow n'}(\|\vec{u}-\vec{v_B}\|).d\vec{u}

With the collision velocity between the impurity ion and the neutral atom :math:`E_c = \|\vec{u}-\vec{v_B}\|`

The above equation point out that the charge-exchange rate coefficient needs the
distribution of the impurity. The general case is that a Maxwellian distribution
function can be assumed in the velocity space, but there are cases such as for
the He-ash where a Maxwellian distribution does not apply and a more general
calculation of the charge-exchange rate would be needed and a new effective
emission coefficients would be needed :math:`q^{CX, Z^{(\alpha+1)+}(n,l), H^0(m)}_{n\rightarrow n', \text{non Max}}`.

As a result there are possibly three different calculations for the effective
charge-exchange emission coefficient, of which only two are used in practise.

**Two different composite charge exchange emission coefficients:**

    1. A Maxwellian distribution function in velocity space is assumed for the ion :math:`Z^{(\alpha+1)+}`:
       :math:`q^{CX, Z^{(\alpha+1)+}(n,l), H^0(m)}_{n\rightarrow n'}`

    2. A Maxwellian distribution function in velocity space is not assumed for the ion :math:`Z^{(\alpha+1)+}`:
       this requires a new set of effective charge-exchange emission coefficient
       :math:`q^{CX, Z^{(\alpha+1)+}(n,l), H^0(m)}_{n\rightarrow n', \text{non Max}}`,
       additional parameters in respect to 1. will be needed.

In more details calculation of excited level populations of impurity ions :math:`Z^{* \alpha+}(n, l, j)`
--------------------------------------------------------------------------------------------------------

The calculation of excited level populations of impurity ions in plasma has been
discussed in details in [Spence and Summer86]. A concise summary is given here.
Subsequent to the primary direct capture process, it is supposed that four
further reaction processes may redistribute the excited level populations.
There are:

* Spontaneous emission:
.. math:: Z^{*, \alpha+}(n,l,j) \longrightarrow Z^{*, \alpha+}(n',l',j') + h\nu

* Collisional ionisation by electrons:
.. math:: Z^{\alpha+}(n,l,j) + e \longrightarrow Z^{(\alpha+1)+} + e + e

* Collisional transitions between nearly degenerate levels by electron and positive ion impact:
.. math:: Z^{\alpha+}(n,l,j) + \left\{\begin{array}{ll}e\\Z_{\mu} \end{array} \right\}
    \longrightarrow Z^{\alpha+}(n,l',j') + \left\{\begin{array}{ll}e\\Z_{\mu} \end{array} \right\}

* Transitions between nearly degenerate levels due to ion motion and magnetic field:
.. math:: Z^{\alpha+}(n,l,j) \overset{B_{mag}}{\longrightarrow} Z^{\alpha+}(n,l',j')

A statistical view of excited ion level populations is appropriate, the number
densities of ions in various excited levels being determined by the balance of
populating and depopulating reaction rates. Levels of principal quantum number
substantially larger than the upper levels of the observed emitted transitions
are included to allow for cascade contributions. The upper limit is determined
properly by collisional ionisation but in practise is mostly influenced by the
decay with n of the primary capture processes. The main effect of the particle
collisions and magnetic fields is to cause transitions between levels of the
same principal quantum number and so these alone are included in the calculations.
For ions in high orbital angular momentum states l, induced transitions between
whole l state populations are of most importance. Whereas, at low l, the
transitions between separate j sublevels are important. For these reasons two
populations models are used, namely:

a. **the nl picture:** in which the populations of nl levels are calculated and
   the populations of j sublevels are assumed to be in proportion to their statistical weights.

b. **the nlj picture:** in which the populations of j sublevels are calculated
   in full magnetic field effects are included only in the nlj picture.

**Two different composite charge exchange emission coefficients depending on the model chosen for the excited population model:**

    3. nl picture: :math:`q^{CX, Z^{(\alpha+1)+}(n,l), H^0(m)}_{n\rightarrow n'}`
    4. nlj picture: :math:`q^{CX, Z^{(\alpha+1)+}(n,l), H^0(m)}_{n\rightarrow n'}(\vec{B})`
       these coefficients exist in ADAS but are mostly not used.

**Justification of assumption usually made on the excited population calculation:**

    Electron impact ionisation:

        Collisional transitions between nearly degenerate levels play a significant
        role for the densities, ions and principal quantum shells of concern in
        JET studies of charge exchange by visible spectroscopy. The transitions
        are of the form :math:`nl\rightarrow nl\pm 1` in the nl picture and
        :math:`nlj\rightarrow nl\pm 1, j\pm 1` in the nlj picture and are induced
        by both electron and positive ion impact. As the transition energies
        approach zero, ions become relatively more efficient than electrons in
        causing transitions. Detailed expressions for collisions rate coefficients
        are given in Spence[1986] are are adopted here. For greatest precision,
        rate coefficients for the different positive ions present in the plasma
        should be combined, weighted by their fractional number densities. Since
        the cross-section is essentially proportional to :math:`Z^2_{\mu}` where
        :math:`Z_{\mu}` is the impacting ion charge number, an error less than
        the intrinsic uncertainty in the cross-sections is introduced by considering
        a single ion species to be present of charge :math:`Z_{eff}` (the usually
        defined plasma effective ion charge). The rate expressions depend sensitively
        on the transition energies and so the latter must be evaluated quite precisely.
        Rates are calculated in the nlj picture initially. Since at high l, where
        the nl picture is often most useful, the transitions have line strengths
        which tend to zero, it is suitable to sum and average over final and initial
        j states to obtain rate coefficients in the nl picture. It is usual to
        estimate approximately the levels at which mixing by magnetic fields matter
        by a simple consideration of static energy level shifts. This is not appropriate
        for a population treatment in which a detailed balance of rates is followed
        to obtain actual ion populations in cases where any process may not be
        fully dominant. It is evident that the field processes matter most for
        low l un vuv and xuv measurements and then the nlj picture is appropriate.

        There is a critical plasma density at which a given transition is fully
        mixed: this is usually a problem for the edge densities of plasmas.

        The sigma component are dominant for observation angles perpendicular to
        the magnetic field, while the :math:`\pi`-component dominates in the case
        of observation angles perpendicular to the magnetic field. In summary,
        in the standard analysis of the CXRS measurements on AUG the corrections
        due to the CX cross-section and gyro-motion effects are not taken into
        account as they are found to be small, while the corrections due to the
        Zeeman effects are included.

        Practically, the l-mixing and Zeeman effect correspond to a broadening
        of the charge-exchange line that is independent of ion temperature. Due
        to the fine structure each allowed transition has a slightly different
        wavelength, and the total emission spectrum consists of a set of lines
        instead of one line.

    Collisional l-mixing:

        Which of the l-levels ae populated in a charge-exchange reactions depends
        on the beam energy. If the lifetime :math:`\tau` of the excited states is considerably
        larger than the ion-ion collision time, collisions will cause a transfer
        between the different l-states of n=8 shell before the charge-exchange
        electron drops to a lower level and emits a photon. This phenomenon is
        called collisional l-mixing. It means that even if the population of the
        l-levels would not be statistic, collisional l-mixing would make sure that
        the l-levels are statistically populated. The spectrum of these lies can
        be calculated in ADAS. In the case of the C6+ CXRS lines, the line broadening
        would correspond to a Doppler broadening of 4eV.

    Zeeman splitting:

        Without a magnetic field the energy levels within the same n shell differ
        slightly due to the fine-structure. The presence of a magnetic field will
        cause Zeeman splitting of one j-level into 2J+1 energy levels separated.
        This is a stronger effect than the collisional l-mixing. For the carbon
        lines, this correspond to a Doppler broadening of 90eV.

    The total CX spectrum, where non-thermal broadening due to l-mixing and Zeeman
    splitting is included, is the sum of the Doppler spectra for every emission
    line, where the relative intensity of every transition and the population of
    every sublevel is taken into account. When we treat every emission line separately
    the analysis of a CX spectrum gets quite complicated. Therefore the total profile
    of all transition lines is presented as a single, but broadened Gaussian, of
    which the peak position depends on the l-mixing and Zeeman splitting. This
    Gaussian replace the :math:`\delta`-function in equation.

**Assumption: at the moment the collisional l-mixing and Zeeman splitting is not taken into account**

    It can however be easily done.

Calculation of the line intensity of CX spectral lines using a statistical description (prediction of spectra)
--------------------------------------------------------------------------------------------------------------

We want to be able to calculate the spectra observed in the case of a charge-exchange
reaction between an ion :math:`Z^{(\alpha+1)+}` and neutral atom present in the plasma
whatever the direction of observation :math:`\vec{l}` and velocity :math:`\vec{u}`
of the ion.

The most general description for the the ion :math:`Z^{\alpha+}` is to have a distribution
function associated to it, :math:`f_{Z^{\alpha+}}(\vec{x}, \vec{u}, t)`. The most
general description for the statistical behaviour of specie :math:`Z^{\alpha+}`
is described by the Boltzmann equation :eq:`boltzman`:

.. math::
    :label: boltzman

    \frac{\partial f_{Z^{\alpha+}}}{\partial t} + \vec{u}.\vec{\nabla} f_{Z^{\alpha+}}
    + \frac{\vec{F}}{m_{Z^{\alpha+}}}.\vec{\nabla_{\vec{u}}} f_{Z^{\alpha+}}
    = \left(\frac{\partial f_{Z^{\alpha+}}}{\partial t}\right)_{col}


The distribution function changes as a result of the forces :math:`F` and collisions
:math:`\left(\frac{\partial f_{Z^{\alpha+}}}{\partial t}\right)_{col}`.

In a plasma the main forces are long-range Lorentz force :math:`\vec{F} = q_{\alpha}(\vec{E} + \vec{u}\times\vec{B})`

The distribution of an ion :math:`Z^{*,\alpha+}` experiencing a charge-exchange
reaction can be written as follows:

.. math::

    \frac{\partial f_{Z^{*,\alpha+}}}{\partial t} + \vec{u}.\vec{\nabla} f_{Z^{*,\alpha+}}
    + \frac{\vec{F}}{m_{Z^{*,\alpha+}}}.\vec{\nabla_{\vec{u}}} f_{Z^{*,\alpha+}}
    = source\left(Z^{*,\alpha+}\right) - sink\left(Z^{*,\alpha+}\right)

As source for ion :math:`Z^{*,\alpha+}`, the charge exchange reaction with a
neutral is considered and for sink it is the re-ionisation taking place in the plasma.

**Assumption: we ignore the transport of ion in space:** :math:`\vec{u}.\vec{\nabla} f_{Z^{*,\alpha+}} \approx 0` **, local calculation.**

    In the case of :math:`He^{*,+}` (plume), the transport in space cannot be ignored.

**Assumption: we neglect the electric field in the Lorentz force** :math:`q_{\alpha}\vec{E} \approx \vec{0}`

.. math::

    \frac{\partial f_{Z^{*,\alpha+}}}{\partial t}
    + \frac{q_{\alpha}\vec{u}\times\vec{B}}{m_{Z^{*,\alpha+}}}.\vec{\nabla_{\vec{u}}} f_{Z^{*,\alpha+}}
    = source\left(Z^{*,\alpha+}\right) - sink\left(Z^{*,\alpha+}\right)

* With a sink :math:`sink\left(Z^{*,\alpha+}\right) = \frac{f_{Z^{*,\alpha+}}}{\tau_{Z^{*,\alpha+}}}`
  With :math:`\tau_{Z^{*,\alpha+}}` being the lifetime of the excited ion.

* And the source :math:`source\left(Z^{*,\alpha+}\right) = Q^{CX, Z^{(\alpha+1)+}, H^0}_{n\rightarrow n'}(E_c).f_{Z^{(\alpha+1)+}}.n_{H^0}`
  due to the charge-exchange reaction.

**Assumption: no gyro-motion of excited ions** :math:`Z^{*,\alpha+}`

    This is only important in tokamak for poloidal views and more information will
    be given in section on how to take this into account.

**Assumption: no lifetime of excited ions taken into account**

    This is only important for poloidal views and if gyro-effects are taken into account.

Having made a little detour, we can now get back to the calculation of the spectra
from a transition :math:`n\rightarrow n'` due to a charge-exchange reaction between
an ion :math:`Z^{(\alpha+1)+}` and neutral atom present in the plasma whatever
the direction of observation and velocity of the ion.

The assumption here is that we have a distribution function in the velocity space
:math:`f_{Z^{(\alpha+1)+}}(\vec{u})` of ion :math:`Z^{(\alpha+1)+}`, and we assume
that the excited population of ion :math:`f_{Z^{*,\alpha+}}(\vec{u})` after the
electron capture is of the same description.

The spectrum of light coming from a small volume dV and observed in the direction
:math:`\vec{l}` can be written as:

.. math::
    dI_{obs, n\rightarrow n'}(\lambda) = n_{H^0}.n_{Z^{(\alpha+1)+}}.\int_{\vec{u}}
    f_{Z^{(\alpha+1)+}}(\vec{u}-\vec{v}).q^{CX, Z^{(\alpha+1)+}, H^0}_{n\rightarrow n'}(E_c).
    \delta(\lambda-\lambda_D).d\vec{u}