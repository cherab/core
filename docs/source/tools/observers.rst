
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



.. autoclass:: cherab.tools.observers.bolometry.BolometerCamera
   :members:
   :special-members: __len__, __iter__, __getitem__

.. autoclass:: cherab.tools.observers.bolometry.BolometerSlit
   :members:

.. autoclass:: cherab.tools.observers.bolometry.BolometerFoil
   :members:
