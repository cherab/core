# cython: language_level=3

# Copyright 2016-2019 Euratom
# Copyright 2016-2019 United Kingdom Atomic Energy Authority
# Copyright 2016-2019 Centro de Investigaciones Energéticas, Medioambientales y Tecnológicas
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

import sys

# search indices for elements and isotopes
_element_index = {}
_isotope_index = {}


cdef class Element:
    """
    Class representing an atomic element.

    :param str name: Element name.
    :param str symbol: Element symbol, e.g. 'H'.
    :param int atomic_number: Number of protons.
    :param float atomic_weight: average atomic weight in amu

    :ivar str name: Element name.
    :ivar str symbol: Element symbol, e.g. 'H'.
    :ivar int atomic_number: Number of protons.
    :ivar float atomic_weight: average atomic weight in amu

    .. code-block:: pycon

       >>> from cherab.core.atomic import Element
       >>> helium = Element("helium", "He", 2, 4.002602)
    """

    def __init__(self, str name, str symbol, int atomic_number, double atomic_weight):

        self.name = name
        self.symbol = symbol
        self.atomic_number = atomic_number
        self.atomic_weight = atomic_weight

    def __repr__(self):
        return '<Element: {}>'.format(self.name)


cdef class Isotope(Element):
    """
    Class representing an atomic isotope.

    :param str name: Isotope name.
    :param str symbol: Isotope symbol, e.g. 'T'.
    :param Element element: The parent element of this isotope,
      e.g. for Tritium it would be Hydrogen.
    :param int mass_number: Atomic mass number, which is total number of protons
      and neutrons. Allows identification of specific isotopes.
    :param float atomic_weight: atomic weight in amu

    :ivar str name: Isotope name.
    :ivar str symbol: Isotope symbol, e.g. 'T'.
    :param Element element: The parent element of this isotope,
      e.g. for Tritium it would be Hydrogen.
    :ivar int atomic_number: Number of protons.
    :ivar int mass_number: Atomic mass number, which is total number of protons
      and neutrons. Allows identification of specific isotopes.
    :ivar float atomic_weight: atomic weight in amu

    .. code-block:: pycon

       >>> from cherab.core.atomic import Isotope, hydrogen
       >>> tritium = Isotope("tritium", "T", hydrogen, 3, 3.0160492777)
    """

    def __init__(self, str name, str symbol, Element element, int mass_number, double atomic_weight):

        super().__init__(name, symbol, element.atomic_number, atomic_weight)
        self.mass_number = mass_number
        self.element = element


    def __repr__(self):
        return '<Isotope: {}>'.format(self.name)


def _build_element_index():
    """
    Populates an element search dictionary.

    Populates the element index so users can search for elements by name,
    symbol or atomic number.
    """

    module = sys.modules[__name__]
    for name in dir(module):
        obj = getattr(module, name)
        if type(obj) is Element:
            # lookup by name, symbol or atomic number
            _element_index[obj.symbol.lower()] = obj
            _element_index[obj.name.lower()] = obj
            _element_index[str(obj.atomic_number)] = obj


def _build_isotope_index():
    """
    Populates an isotope search dictionary.

    Populates the isotope index so users can search for isotopes by name or
    symbol.
    """

    module = sys.modules[__name__]
    for name in dir(module):
        obj = getattr(module, name)
        if type(obj) is Isotope:
            # lookup by name or symbol including variations e.g. D and H2 refer to deuterium)
            _isotope_index[obj.symbol.lower()] = obj
            _isotope_index[obj.name.lower()] = obj
            _isotope_index[obj.element.symbol.lower() + str(obj.mass_number)] = obj
            _isotope_index[obj.element.name.lower() + str(obj.mass_number)] = obj


def lookup_element(v):
    """
    Finds an element by name, symbol or atomic number.

    .. code-block:: pycon

       >>> from cherab.core.atomic import lookup_element
       >>> hydrogen = lookup_element('hydrogen')
       >>> neon = lookup_element('Ne')
       >>> argon = lookup_element(18)

    :param v: Search string or integer.
    :return: Element object.
    """

    if type(v) is Element:
        return v

    key = str(v).lower()
    try:
        return _element_index[key]
    except KeyError:
        raise ValueError('Could not find an element object for the key \'{}\'.'.format(v))


def lookup_isotope(v, number=None):
    """
    Finds an isotope by name, symbol or number.

    Isotopes are uniquely determined by the element type and mass number. These
    can be specified as a single string or a combination of element and mass number.

    .. code-block:: pycon

       >>> from cherab.core.atomic import lookup_isotope
       >>> deuterium = lookup_element('deuterium')
       >>> tritium = lookup_element(1, number=3)
       >>> helium3 = lookup_element('he3')
       >>> helium4 = lookup_element('he', number=4)

    :param v: Search string, integer or element.
    :param number: Integer mass number
    :return: Element object.
    """

    if type(v) is Isotope:
        return v

    if number:
        # mass number supplied, so only need the element
        element = lookup_element(v)
        key = (element.symbol + str(number)).lower()
    else:
        # full information contained in string
        key = str(v).lower()

    try:
        return _isotope_index[key]
    except KeyError:
        if number:
            raise ValueError('Could not find an isotope object for the element \'{}\' and number \'{}\'.'.format(v, number))
        else:
            raise ValueError('Could not find an isotope object for the key \'{}\'.'.format(v))


# Atomic data sourced from www.ciaaw.org and wikipedia.org on 25/1/2015

# elements
hydrogen = Element("hydrogen", "H", 1, (1.00784 + 1.00811) / 2)
helium = Element("helium", "He", 2, 4.002602)
lithium = Element("lithium", "Li", 3, (6.938 + 6.997) / 2)
beryllium = Element("beryllium", "Be", 4, 9.0121831)
boron = Element("boron", "B", 5, (10.806 + 10.821) / 2)
carbon = Element("carbon", "C", 6, (12.0096 + 12.0116) / 2)
nitrogen = Element("nitrogen", "N", 7, (14.00643 + 14.00728) / 2)
oxygen = Element("oxygen", "O", 8, (15.99903 + 15.99977) / 2)
fluorine = Element("fluorine", "F", 9, 18.998403163)
neon = Element("neon", "Ne", 10, 20.1797)
argon = Element("argon", "Ar", 18, 39.948)
krypton = Element("krypton", "Kr", 36, 83.798)
xenon = Element("xenon", "Xe", 54, 131.293)

# select isotopes
protium = Isotope("protium", "H", hydrogen, 1, 1.00782503207)
deuterium = Isotope("deuterium", "D", hydrogen, 2, 2.0141017778)
tritium = Isotope("tritium", "T", hydrogen, 3, 3.0160492777)

helium3 = Isotope("helium3", "He3", helium, 3, 3.0160293191)
helium4 = Isotope("helium4", "He4", helium, 4, 4.00260325415)

# once objects created build an indices for reverse lookup (string instancing of element)
_build_element_index()
_build_isotope_index()