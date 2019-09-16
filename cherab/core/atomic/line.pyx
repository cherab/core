# Copyright 2016-2018 Euratom
# Copyright 2016-2018 United Kingdom Atomic Energy Authority
# Copyright 2016-2018 Centro de Investigaciones Energéticas, Medioambientales y Tecnológicas
#
# Licensed under the EUPL, Version 1.1 or – as soon they will be approved by the
# European Commission - subsequent versions of the EUPL (the "Licence");
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at:
#
# https://joinup.ec.europa.eu/software/page/eupl5
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the Licence is distributed on an "AS IS" basis, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied.
#
# See the Licence for the specific language governing permissions and limitations
# under the Licence.


cdef class Line:
    """
    A class fully specifies an observed spectroscopic emission line.

    Note that wavelengths are not arguments to this class. This is because in
    principle the transition has already been fully specified with the other three
    arguments. The wavelength is looked up in the wavelength database of the
    atomic data provider.

    :param Element element: The atomic element/isotope to which this emission line belongs.
    :param int charge: The charge state of the element/isotope that emits this line.
    :param tuple transition: A two element tuple that defines the upper and lower electron
      configuration states of the transition. For hydrogen-like ions it may be enough to
      specify the n-levels with integers (e.g. (3,2)). For all other ions the full spectroscopic
      configuration string should be specified for both states. It is up to the atomic data
      provider package to define the exact notation.

    .. code-block:: pycon

        >>> from cherab.core.atomic import Line, deuterium, carbon
        >>>
        >>> # Specifying the d-alpha and d-gamma balmer lines
        >>> d_alpha = Line(deuterium, 0, (3, 2))
        >>> d_beta = Line(deuterium, 0, (4, 2))
        >>>
        >>> # Specifying a CIII line at 465nm
        >>> ciii_465 = Line(carbon, 2, ('2s1 3p1 3P4.0', '2s1 3s1 3S1.0'))
    """

    def __init__(self, Element element, int charge, tuple transition):

        if charge > element.atomic_number - 1:
            raise ValueError("Charge state cannot be larger than one less than the atomic number.")

        if charge < 0:
            raise ValueError("Charge state cannot be less than zero.")

        self.element = element
        self.charge = charge
        self.transition = transition

    def __repr__(self):
        return '<Line: {}, {}, {}>'.format(self.element.name, self.charge, self.transition)