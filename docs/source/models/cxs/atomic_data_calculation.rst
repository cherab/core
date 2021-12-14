Atomic data calculation
=======================

.. *Date: 02/12/2014*

Aim of this note: Below is given the details on how ADAS atomic data are
used and implemented in Cherab at present. The aim here is to give a very
concise information. More details can be found in ADAS manual on line.

.. WARNING::
    We will need in each section another paragraph on how these data
    are being used within the atomic model in Cherab and modified to be used
    within the main Cherab.

.. csv-table:: Important quantities
    :header: "Name Cherab", "Description", "ADAS notation", "ADAS file", "ADAS program", "Units in ADAS"

    ":math:`\sigma^{CX}_{Z^{(\alpha+1)+}}(H^0(m_i), n, l, E_{c})`", "Charge-Exchange cross-section between electron donor :math:`H^0` in metastable state :math:`m_{i}` and ion :math:`Z^{(\alpha+1)+}` in shell :math:`(n,l)` with a collision energy :math:`E_{c}`", ":math:`\sigma_{n, l}(E_{c})`", "", "", ":math:`cm^{2}`"
    ":math:`q^{CX, Z^{(\alpha+1)+}, H^0(m_i)}_{n\rightarrow n'}`", "Effective emission coefficient (or rate) for a charge-exchange line corresponding to a transition :math:`n\rightarrow n'` of ion :math:`Z^{(\alpha+1)+}` with electron donor :math:`H^0` in metastable state :math:`m_{i}`", ":math:`q^{eff}_{n\rightarrow n'}`", "adf12", "adas303", ":math:`photon.cm^{3}.s^{-1}.(4\pi steradian)^{-1}`"
    ":math:`bmp(H^0(m_i))`", "Relative beam population of excited state :math:`m_i` over ground state for atom :math:`H^0`", "BMP", "adf22", "adas304", "dimensionless"
    ":math:`Q^{CX}_{Z^{(\alpha+1)+}}(H^0(m_i), n, l, E_{c})`", "Charge-Exchange rate between electron donor :math:`H^0` in metastable state :math:`m_{i}` and ion :math:`Z^{(\alpha+1)+}` in shell :math:`(n,l)` with a collision energy :math:`E_{c}`", "", "", "", ":math:`reaction.s^{-1}.cm^{-3}`"
    ":math:`I_{n\rightarrow n'}`", "Intensity of charge exchange line due to transition :math:`n\rightarrow n'`.", "", "", "", ":math:`photon.s^{-1}.cm^{-3}.(4\pi steradian)^{-1}`"
    ":math:`\lambda_n`", "Natural wavelength of transition :math:`n\rightarrow n'` from ion :math:`Z^{(\alpha+1)+, *}`", "", "", "", ":math:`nm`"
    ":math:`S^{e, i}_{CR}(E_H, n_e, T_i)`", "Effective collisional radiative stopping coefficient for atom :math:`H^0` in a beam by fully stripped ions :math:`X^i` and their electrons.", ":math:`S^{e, i}_{CR}`", "adf21", "adas304", ":math:`cm^3.s^{-1}`"

.. WARNING::
    Make it clear: all information clearly available in ADAS

--------------------------------------------------------------------------------
Effective emission charge-exchange coefficient :math:`q^{eff}_{n\rightarrow n'}`
--------------------------------------------------------------------------------

Open ADAS:

    * ADAS303 manual: http://www.adas.ac.uk/man/chap3-03.pdf
    * ADAS file: adf12
    * Adas idl routine to read the file : /u/adas/idl/adaslib/readadf/read_adf12.pro (possible to read the file or do the calculation of :math:`q^{eff}_{n\rightarrow n'}` within read_adf12.pro)
    * Adas idl routine to calculate the :math:`q^{eff}_{n\rightarrow n'}`: /u/adas/idl/adaslib/cxsqef.pro
    * Type of adf12 file:

        * /u/adas/adas/adf12/qef93#h/qef93#h_c6.dat for in H(1s)
        * /u/adas/adas/adf12/qef97#h/qef97#h_en2_kvi#c6.dat for donor in H(2s)

.. csv-table:: ADAS inputs for effective emission charge-exchange coefficient
    :header: "Input in ADAS", "Description", "Unit in ADAS"

    ":math:`E_c`", "Collision energy: :math:`\frac{1}{2}\left(\vec{v_H}-\vec{v}\right)^2` with :math:`\vec{v_H}` the velocity of the neutral H and :math:`\vec{v}` the velocity of the target ion :math:`Z^{(\alpha +1)+}`.", ":math:`eV/amu`"
    ":math:`n_Z`", "Target ion density", ":math:`cm^{-3}`"
    ":math:`T_Z`", "Target ion temperature", ":math:`eV`"
    ":math:`Z_{eff}`", "Plasma effective charge", "dimensionless"
    ":math:`B_{mag}`", "Magnetic field", ":math:`T`"
    "isel", "Identification of line transition :math:`n\rightarrow n'`", "dimensionless (integer)"

The adf12 data file have the following format:

    * 1 grid in collision energy :math:`E_c` at reference values of :math:`n_Z`, :math:`T_Z`, :math:`Z_{eff}`, :math:`B_{mag}`
    * 1 grid in target ion density :math:`n_Z` at reference values of :math:`E_c`, :math:`T_Z`, :math:`Z_{eff}`, :math:`B_{mag}`
    * 1 grid in target ion temperature :math:`T_Z` at reference values of :math:`E_c`, :math:`n_Z`, :math:`Z_{eff}`, :math:`B_{mag}`
    * 1 grid in plasma effective charge :math:`Z_{eff}` at reference values of :math:`E_c`, :math:`n_Z`, :math:`T_Z`, :math:`B_{mag}`
    * 1 grid in magnetic field :math:`B_{mag}` at reference values of :math:`E_c`, :math:`n_Z`, :math:`T_Z`, :math:`Z_{eff}`

We do not use the l-resolve effective emission charge-exchange coefficient although the detail of the calculation is taking this into account.

> Best method to calculate :math:`q^{eff}_{n\rightarrow n'}` from adf12 files?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Note from Alfonso Baciero / Ephrem Delabie: what is the best method for doing interpolation? x vs y, log(x) vs y, x vs log(y) or log(x) vs log(y)? It is not clear what the best interpolation is, it could be better to use a new interpolated grid in the range of interest.

> Best method to calculate simulated spectral line with Maxwellian distribution of velocity for target ion :math:`Z^{(\alpha+1)+}` but taking into account the cross-section effet
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We need to calculate the following:

.. math::

    I_{n\rightarrow n'}^{obs}(\lambda) = \iiint_V d\vec{x} n^H n^{Z^{(\alpha+1)+}} \iiint d\vec{u} f^{Z^{(\alpha+1)+}}(\vec{u}-\vec{v}) q^{CX, Z^{(\alpha+1)+}}_{n\rightarrow n'} \delta(\lambda-\lambda_n\left(1+\frac{\vec{u}.\vec{l}}{c}\right))

We can perform this calculation by calculating the composite charge-exchange emission coefficient with the lowest plasma ion temperature. We would need to simulate the new ADAS adf12 files for velocity cross-sections where, instead of :math:`T_i`, :math:`T_e` is the input. (Baciero / Delabie)

-------------------------------------
Effective beam population coefficient
-------------------------------------

Open ADAS: http://open.adas.ac.uk/adf22

ADAS304: http://www.adas.ac.uk/man/chap3-04.pdf

ADAS file: adf22

Type of adf22 file:

To obtain the :math:`bmp(H, m_{i=2})[E_c, n_e, T_i]`

.. csv-table:: ADAS inputs for effective beam population coefficient
    :header: "Input in ADAS", "Description", "Unit in ADAS"

    "Filename", "Target ion fraction: same as number of adf21 files"
    "Fraction", "Target ion fraction. 1st dimension: fraction for each ion. 2nd dimension: list of adf21 files to use for each ion given in 1st dimension."
    ":math:`T_i`", "Target ion temperature", ":math:`eV`"
    ":math:`E_c`", "Collision energy between the target ion and neutral atom.", ":math:`eV/amu`"
    ":math:`n_e^{i, equiv}`", "Equivalent electron density to be used in the extraction of the stopping coefficient contribution from the ith pure impurity archive for a plasma with multi ions", ":math:`cm^{-3}`"

These data are build up in the same way as the beam stopping coefficient. From a known composition of the plasma. Build up the beam population. Should write the equation down properly even if it is case.

------------------------------------------------------
Effective beam stopping coefficient :math:`S_{CR}^{Z}`
------------------------------------------------------

Open ADAS: http://open.adas.ac.uk/adf21

ADAS304: http://www.adas.ac.uk/man/chap3-04.pdf

ADAS file: adf21

Type of adf21 file: /u/adas/idl/adaslib/readadf/read_adf21.pro ( this is however to read the archive data and build up the composite beam stopping I think )

.. csv-table:: ADAS inputs for effective beam stopping coefficient
    :header: "Input in ADAS", "Description", "Unit in ADAS"

    ":math:`S_{CR}^{e, i}[E_c, n_e, T_i]`", "Stopping coefficient contribution from the ith pure impurity archive for a plasma with multi-ions", ":math:`cm^3/s`"
    ":math:`E_c`", "Collision energy between the target ion and the neutral atom.", ":math:`eV/amu`"
    ":math:`n_e^{i, equiv}`", "Equivalent electron density to be used in the extraction of the stopping coefficient contribution from the ith pure impurity archive for a plasma with multi ions", ":math:`cm^{-3}`"
    ":math:`T_i`", "Target ion temperature", ":math:`eV`"

More details can be found on how to read the data :

The adf21 data: (I am fed up and need a break)

------------------------------------------
List of ADAS file used at present in CHEAP
------------------------------------------

----------------------------------------------------
List of CX lines, transition and accurate wavelength
----------------------------------------------------

Fundamental data in ADAS

Adf02: fundamental ion-atom collision cross-section

Adf04: fundamental electron collisional rate coefficient data and A values

Adf07: fundamental electron impact ionisation collision data

Derived data in ADAS

Adf21 effective beam stopping coefficients

Adf22 effective Dalpha beam emission coefficients

Adf26 collisional-radiative bundled-n population tabulations.
