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
       >>> helium = Element('helium', 'He', 2, 4.002602)
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
       >>> tritium = Isotope('tritium', 'T', hydrogen, 3, 3.0160492777)
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


# Atomic data sourced from www.ciaaw.org and wikipedia.org on 25/1/2015 and 21/1/2019
hydrogen = Element('hydrogen', 'H', 1, (1.00784 + 1.00811) / 2)
helium = Element('helium', 'He', 2, 4.002602)
lithium = Element('lithium', 'Li', 3, (6.938 + 6.997) / 2)
beryllium = Element('beryllium', 'Be', 4, 9.0121831)
boron = Element('boron', 'B', 5, (10.806 + 10.821) / 2)
carbon = Element('carbon', 'C', 6, (12.0096 + 12.0116) / 2)
nitrogen = Element('nitrogen', 'N', 7, (14.00643 + 14.00728) / 2)
oxygen = Element('oxygen', 'O', 8, (15.99903 + 15.99977) / 2)
fluorine = Element('fluorine', 'F', 9, 18.998403163)
neon = Element('neon', 'Ne', 10, 20.1797)
sodium = Element('sodium', 'Na', 11, 22.990)
magnesium = Element('magnesium', 'Mg', 12, (24.304 + 24.307) / 2)
aluminium = Element('aluminium', 'Al', 13, 26.982)
silicon = Element('silicon', 'Si', 14, (28.084 + 28.086) / 2)
phosphorus = Element('phosphorus', 'P', 15, 30.974)
sulfur = Element('sulfur', 'S', 16, (32.059 + 32.076) / 2)
chlorine = Element('chlorine', 'Cl', 17, (35.446 + 35.457) / 2)
argon = Element('argon', 'Ar', 18, (39.792 + 39.963) / 2)
potassium = Element('potassium', 'K', 19, 39.098)
calcium = Element('calcium', 'Ca', 20, 40.078)
scandium = Element('scandium', 'Sc', 21, 44.956)
titanium = Element('titanium', 'Ti', 22, 47.867)
vanadium = Element('vanadium', 'V', 23, 50.942)
chromium = Element('chromium', 'Cr', 24, 51.996)
manganese = Element('manganese', 'Mn', 25, 54.938)
iron = Element('iron', 'Fe', 26, 55.845)
cobalt = Element('cobalt', 'Co', 27, 58.933)
nickel = Element('nickel', 'Ni', 28, 58.693)
copper = Element('copper', 'Cu', 29, 63.546)
zinc = Element('zinc', 'Zn', 30, 65.38)
gallium = Element('gallium', 'Ga', 31, 69.723)
germanium = Element('germanium', 'Ge', 32, 72.630)
arsenic = Element('arsenic', 'As', 33, 74.922)
selenium = Element('selenium', 'Se', 34, 78.971)
bromine = Element('bromine', 'Br', 35, (79.901 + 79.907) / 2)
krypton = Element('krypton', 'Kr', 36, 83.798)
rubidium = Element('rubidium', 'Rb', 37, 85.468)
strontium = Element('strontium', 'Sr', 38, 87.62)
yttrium = Element('yttrium', 'Y', 39, 88.906)
zirconium = Element('zirconium', 'Zr', 40, 91.224)
niobium = Element('niobium', 'Nb', 41, 92.906)
molybdenum = Element('molybdenum', 'Mo', 42, 95.95)
# technetium = Element('technetium', 'Tc', 43, None)
ruthenium = Element('ruthenium', 'Ru', 44, 101.07)
rhodium = Element('rhodium', 'Rh', 45, 102.91)
palladium = Element('palladium', 'Pd', 46, 106.42)
silver = Element('silver', 'Ag', 47, 107.87)
cadmium = Element('cadmium', 'Cd', 48, 112.41)
indium = Element('indium', 'In', 49, 114.82)
tin = Element('tin', 'Sn', 50, 118.71)
antimony = Element('antimony', 'Sb', 51, 121.76)
tellurium = Element('tellurium', 'Te', 52, 127.60)
iodine = Element('iodine', 'I', 53, 126.9)
xenon = Element('xenon', 'Xe', 54, 131.293)
caesium = Element('caesium', 'Cs', 55, 132.91)
barium = Element('barium', 'Ba', 56, 137.33)
lanthanum = Element('lanthanum', 'La', 57, 138.91)
cerium = Element('cerium', 'Ce', 58, 140.12)
praseodymium = Element('praseodymium', 'Pr', 59, 140.91)
neodymium = Element('neodymium', 'Nd', 60, 144.24)
# promethium = Element('promethium', 'Pm', 61, None)
samarium = Element('samarium', 'Sm', 62, 150.36)
europium = Element('europium', 'Eu', 63, 151.96)
gadolinium = Element('gadolinium', 'Gd', 64, 157.25)
terbium = Element('terbium', 'Tb', 65, 158.93)
dysprosium = Element('dysprosium', 'Dy', 66, 162.5)
holmium = Element('holmium', 'Ho', 67, 164.93)
erbium = Element('erbium', 'Er', 68, 167.26)
thulium = Element('thulium', 'Tm', 69, 168.93)
ytterbium = Element('ytterbium', 'Yb', 70, 173.05)
lutetium = Element('lutetium', 'Lu', 71, 174.97)
hafnium = Element('hafnium', 'Hf', 72, 178.49)
tantalum = Element('tantalum', 'Ta', 73, 180.95)
tungsten = Element('tungsten', 'W', 74, 183.84)
rhenium = Element('rhenium', 'Re', 75, 186.21)
osmium = Element('osmium', 'Os', 76, 190.23)
iridium = Element('iridium', 'Ir', 77, 192.22)
platinum = Element('platinum', 'Pt', 78, 195.08)
gold = Element('gold', 'Au', 79, 196.97)
mercury = Element('mercury', 'Hg', 80, 200.59)
thallium = Element('thallium', 'Tl', 81, (204.38 + 204.39) / 2)
lead = Element('lead', 'Pb', 82, 207.2)
bismuth = Element('bismuth', 'Bi', 83, 208.98)
# polonium = Element('polonium', 'Po', 84, None)
# astatine = Element('astatine', 'At', 85, None)
# radon = Element('radon', 'Rn', 86, None)
# francium = Element('francium', 'Fr', 87, None)
# radium = Element('radium', 'Ra', 88, None)
# actinium = Element('actinium', 'Ac', 89, None)
thorium = Element('thorium', 'Th', 90, 232.04)
protactinium = Element('protactinium', 'Pa', 91, 231.04)
uranium = Element('uranium', 'U', 92, 238.03)
# neptunium = Element('neptunium', 'Np', 93, None)
# plutonium = Element('plutonium', 'Pu', 94, None)
# americium = Element('americium', 'Am', 95, None)
# curium = Element('curium', 'Cm', 96, None)
# berkelium = Element('berkelium', 'Bk', 97, None)
# californium = Element('californium', 'Cf', 98, None)
# einsteinium = Element('einsteinium', 'Es', 99, None)
# fermium = Element('fermium', 'Fm', 100, None)
# mendelevium = Element('mendelevium', 'Md', 101, None)
# nobelium = Element('nobelium', 'No', 102, None)
# lawrencium = Element('lawrencium', 'Lr', 103, None)
# rutherfordium = Element('rutherfordium', 'Rf', 104, None)
# dubnium = Element('dubnium', 'Db', 105, None)
# seaborgium = Element('seaborgium', 'Sg', 106, None)
# bohrium = Element('bohrium', 'Bh', 107, None)
# hassium = Element('hassium', 'Hs', 108, None)
# meitnerium = Element('meitnerium', 'Mt', 109, None)
# darmstadtium = Element('darmstadtium', 'Ds', 110, None)
# roentgenium = Element('roentgenium', 'Rg', 111, None)
# copernicium = Element('copernicium', 'Cn', 112, None)
# nihonium = Element('nihonium', 'Nh', 113, None)
# flerovium = Element('flerovium', 'Fl', 114, None)
# moscovium = Element('moscovium', 'Mc', 115, None)
# livermorium = Element('livermorium', 'Lv', 116, None)
# tennessine = Element('tennessine', 'Ts', 117, None)
# oganesson = Element('oganesson', 'Og', 118, None)

# select isotopes
protium = Isotope('protium', 'H', hydrogen, 1, 1.00782503207)
deuterium = Isotope('deuterium', 'D', hydrogen, 2, 2.0141017778)
tritium = Isotope('tritium', 'T', hydrogen, 3, 3.0160492777)

helium3 = Isotope('helium3', 'He3', helium, 3, 3.0160293191)
helium4 = Isotope('helium4', 'He4', helium, 4, 4.00260325415)

# once objects created build an indices for reverse lookup (string instancing of element)
_build_element_index()
_build_isotope_index()