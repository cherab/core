
Observers
=========

Most plasma diagnostics can be easily modelled using the base observer types provided in
Raysect. In Cherab we only provide a few specialist observers with extra functionality.

.. _observers_bolometers:

Bolometers
----------

Bolometer systems are modelled as a collection of foils viewing emission through
a collection of apertures. The foils and slits are grouped together into a
:class:`BolometerCamera` object, which acts as the parent node for the apertures
and detectors. Bolometer cameras contain one or more slits (apertures), which
provide targetting information for the foils (detectors) to improve
computational efficiency.  Bolometer cameras also contain one or more foils,
which are modelled as pixels and which fire rays in the direction of the slit.

A note on units: the :class:`BolometerFoil` class can be used to measure
emission in units of power [W] or radiance [W/m²/sr]. In the latter case, the
radiance is defined as the mean radiance over the entire solid angle
:math:`\Omega = 2 \pi` subtended by the foil:

.. math:: \overline{L} &= \frac{\Phi}{A \Omega} \\ \Phi &= \int_0^{\Omega}
    \mathrm{d}\omega \int_0^A \mathrm{d}A \, L(\mathbf{x}, \omega) \cos(\theta)

When a bolometer is approximated by a single line of sight, the radiance
measured is taken at a single angle and position: it is equivalent to the above
equations in the limits :math:`\Omega \to 0` and :math:`A \to 0`. A real
bolometer needs some finite area and solid angle to measure signal, and as long
as the solid angle :math:`\omega_s` subtended by the slit at the foil surface is
small, a meaningful comparison to a sightline can be made using:

.. math:: L_B &= \frac{\Phi_B}{A \omega_s} \\
          \Phi_B &= \int_0^{\omega_s} \mathrm{d}\omega \int_0^A \mathrm{d}A \, L(\mathbf{x}, \omega) \cos(\theta)

Note that :math:`\Phi_B = \Phi`, since no power is incident on the bolometer for
solid angles :math:`\omega > \omega_s`, and this allows us to directly relate
the radiance reported by Cherab with the radiance expected from a finite
bolometer system being modelled as a sightline:

.. math:: \frac{L_B}{\overline{L}} = \frac{\omega_s}{\Omega}

When comparing radiance measurements from Cherab with other modelling tools
which treat the bolometer as a single sightline, applying this conversion factor
is necessary. The fractional solid angle can be easily determined from the
bolometer etendue :math:`G`, which is given by:

.. math:: G = A \omega_s = A \Omega \frac{\omega_s}{\Omega}.

CHERAB also provides an Infra-Red Video Bolometer (IRVB) class. These are typically
large foil bolometers which are imaged with an IR camera. CHERAB models only the
incoming radiation incident on the foil, not the subsequent IR emission on the IR
camera-facing side of the foil: solving the heat transfer equation to calculate this is
left to downstream codes.

Practically, the IRVB foil is typically divided up into separate regions, and the
incident power on each of these individual regions is the quantity of interest. The foil
can then be treated as a 2D array of pixels, where each pixel corresponds to a separate
region on the foil. In this sense, the IRVB is used in a similar manner to a CCD array:
observations will return a 2D array of measured powers. The pixels the foil is divided
into are specified at instantiation time, along with the width of the foil: the height
is calculated from the width and the pixel dimensions.

Although the IRVB observations produce data similar to that of a CCD array,
instantiation of the :class:`BolometerIRVB` is done in much the same way as for the
:class:`BolometerFoil`. The two have similar methods too, with the caveat that some
methods of the :class:`BolometerIRVB` return 2D arrays in place of a single return value
for the :class:`BolometerFoil`.

.. autoclass:: cherab.tools.observers.bolometry.BolometerCamera
   :members:
   :special-members: __len__, __iter__, __getitem__

.. autoclass:: cherab.tools.observers.bolometry.BolometerSlit
   :members:

.. autoclass:: cherab.tools.observers.bolometry.BolometerFoil
   :members:

.. autoclass:: cherab.tools.observers.bolometry.BolometerIRVB
   :members:

.. _observers_spectroscopic:

Spectroscopic lines of sight
----------------------------

.. deprecated:: 1.4.0
   Use Raysect's observer classes instead

Spectroscopic line of sight allows to control main parameters of the pipeline
without accessing the pipeline directly. Multiple spectroscopic line of sight can be
combined into a group.

.. autoclass:: cherab.tools.observers.spectroscopy.base._SpectroscopicObserver0DBase
   :members:

.. autoclass:: cherab.tools.observers.spectroscopy.SpectroscopicSightLine
   :members:

.. autoclass:: cherab.tools.observers.spectroscopy.SpectroscopicFibreOptic
   :members:

.. _observers_group:

Group observers
---------------

Group observer is a collection of observers of the same type. All Observer0D classes 
defined in Raysect are supoorted. The parameters of individual observers in a group 
may differ. Group observer allows combined observation, namely, calling the observe
function for a group leads to a sequential call of this function for each observer 
in the group.

.. autoclass:: cherab.tools.observers.group.base.Observer0DGroup
   :members:

.. autoclass:: cherab.tools.observers.group.SightLineGroup
   :members:

.. autoclass:: cherab.tools.observers.group.FibreOpticGroup
   :members:

.. autoclass:: cherab.tools.observers.group.PixelGroup
   :members:

.. autoclass:: cherab.tools.observers.group.TargettedPixelGroup
   :members:

Spectroscopic Groups
^^^^^^^^^^^^^^^^^^^^

.. deprecated:: 1.4.0
   Use groups based on Raysect's observer classes instead

These groups take control of spectroscopic lines of sight observers. They support 
direction and origin positioning and contain methods for plotting the power and 
spectrum. Originally, these were called group observers and did not include the 
Spectroscopic prefix in class name.

.. autoclass:: cherab.tools.observers.SpectroscopicSightLine
   :members

.. autoclass:: cherab.tools.observers.SpectroscopicFibreOptic
   :members

