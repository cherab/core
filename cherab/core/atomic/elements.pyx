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

# Atomic data sourced from www.ciaaw.org and wikipedia.org on 25/1/2015 and 21/1/2019

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




# elements
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

# isotopes
protium = Isotope('protium', 'H', hydrogen, 1, 1.00782503207)
deuterium = Isotope('deuterium', 'D', hydrogen, 2, 2.0141017778)
tritium = Isotope('tritium', 'T', hydrogen, 3, 3.0160492777)
helium3 = Isotope('helium3', 'He3', helium, 3, 3.016029322)
helium4 = Isotope('helium4', 'He4', helium, 4, 4.0026032545)
lithium6 = Isotope('lithium6', 'Li6', lithium, 6, 6.01512289)
lithium7 = Isotope('lithium7', 'Li7', lithium, 7, 7.01600344)
beryllium9 = Isotope('beryllium9', 'Be9', beryllium, 9, 9.0121831)
boron10 = Isotope('boron10', 'B10', boron, 10, 10.0129369)
boron11 = Isotope('boron11', 'B11', boron, 11, 11.00930517)
carbon12 = Isotope('carbon12', 'C12', carbon, 12, 12)
carbon13 = Isotope('carbon13', 'C13', carbon, 13, 13.003354835)
nitrogen14 = Isotope('nitrogen14', 'N14', nitrogen, 14, 14.003074004)
nitrogen15 = Isotope('nitrogen15', 'N15', nitrogen, 15, 15.000108899)
oxygen16 = Isotope('oxygen16', 'O16', oxygen, 16, 15.994914619)
oxygen17 = Isotope('oxygen17', 'O17', oxygen, 17, 16.999131757)
oxygen18 = Isotope('oxygen18', 'O18', oxygen, 18, 17.999159613)
fluorine19 = Isotope('fluorine19', 'F19', fluorine, 19, 18.998403163)
neon20 = Isotope('neon20', 'Ne20', neon, 20, 19.99244018)
neon21 = Isotope('neon21', 'Ne21', neon, 21, 20.9938467)
neon22 = Isotope('neon22', 'Ne22', neon, 22, 21.9913851)
sodium23 = Isotope('sodium23', 'Na23', sodium, 23, 22.98976928)
magnesium24 = Isotope('magnesium24', 'Mg24', magnesium, 24, 23.9850417)
magnesium25 = Isotope('magnesium25', 'Mg25', magnesium, 25, 24.985837)
magnesium26 = Isotope('magnesium26', 'Mg26', magnesium, 26, 25.982593)
aluminium27 = Isotope('aluminium27', 'Al27', aluminium, 27, 26.9815384)
silicon28 = Isotope('silicon28', 'Si28', silicon, 28, 27.976926535)
silicon29 = Isotope('silicon29', 'Si29', silicon, 29, 28.976494665)
silicon30 = Isotope('silicon30', 'Si30', silicon, 30, 29.9737701)
phosphorus31 = Isotope('phosphorus31', 'P31', phosphorus, 31, 30.973761998)
sulfur32 = Isotope('sulfur32', 'S32', sulfur, 32, 31.972071174)
sulfur33 = Isotope('sulfur33', 'S33', sulfur, 33, 32.97145891)
sulfur34 = Isotope('sulfur34', 'S34', sulfur, 34, 33.967867)
sulfur36 = Isotope('sulfur36', 'S36', sulfur, 36, 35.967081)
chlorine35 = Isotope('chlorine35', 'Cl35', chlorine, 35, 34.9688527)
chlorine37 = Isotope('chlorine37', 'Cl37', chlorine, 37, 36.9659026)
argon36 = Isotope('argon36', 'Ar36', argon, 36, 35.9675451)
argon38 = Isotope('argon38', 'Ar38', argon, 38, 37.962732)
argon40 = Isotope('argon40', 'Ar40', argon, 40, 39.96238312)
potassium39 = Isotope('potassium39', 'K39', potassium, 39, 38.96370649)
potassium40 = Isotope('potassium40', 'K40', potassium, 40, 39.9639982)
potassium41 = Isotope('potassium41', 'K41', potassium, 41, 40.96182526)
calcium40 = Isotope('calcium40', 'Ca40', calcium, 40, 39.9625909)
calcium42 = Isotope('calcium42', 'Ca42', calcium, 42, 41.958618)
calcium43 = Isotope('calcium43', 'Ca43', calcium, 43, 42.958766)
calcium44 = Isotope('calcium44', 'Ca44', calcium, 44, 43.955481)
calcium46 = Isotope('calcium46', 'Ca46', calcium, 46, 45.95369)
calcium48 = Isotope('calcium48', 'Ca48', calcium, 48, 47.9525229)
scandium45 = Isotope('scandium45', 'Sc45', scandium, 45, 44.955908)
titanium46 = Isotope('titanium46', 'Ti46', titanium, 46, 45.952627)
titanium47 = Isotope('titanium47', 'Ti47', titanium, 47, 46.9517577)
titanium48 = Isotope('titanium48', 'Ti48', titanium, 48, 47.9479409)
titanium49 = Isotope('titanium49', 'Ti49', titanium, 49, 48.9478646)
titanium50 = Isotope('titanium50', 'Ti50', titanium, 50, 49.9447858)
vanadium50 = Isotope('vanadium50', 'V50', vanadium, 50, 49.947156)
vanadium51 = Isotope('vanadium51', 'V51', vanadium, 51, 50.943957)
chromium50 = Isotope('chromium50', 'Cr50', chromium, 50, 49.946041)
chromium52 = Isotope('chromium52', 'Cr52', chromium, 52, 51.940505)
chromium53 = Isotope('chromium53', 'Cr53', chromium, 53, 52.940647)
chromium54 = Isotope('chromium54', 'Cr54', chromium, 54, 53.938878)
manganese55 = Isotope('manganese55', 'Mn55', manganese, 55, 54.938043)
iron54 = Isotope('iron54', 'Fe54', iron, 54, 53.939608)
iron56 = Isotope('iron56', 'Fe56', iron, 56, 55.934936)
iron57 = Isotope('iron57', 'Fe57', iron, 57, 56.935392)
iron58 = Isotope('iron58', 'Fe58', iron, 58, 57.933274)
cobalt59 = Isotope('cobalt59', 'Co59', cobalt, 59, 58.933194)
nickel58 = Isotope('nickel58', 'Ni58', nickel, 58, 57.935342)
nickel60 = Isotope('nickel60', 'Ni60', nickel, 60, 59.930785)
nickel61 = Isotope('nickel61', 'Ni61', nickel, 61, 60.931055)
nickel62 = Isotope('nickel62', 'Ni62', nickel, 62, 61.928345)
nickel64 = Isotope('nickel64', 'Ni64', nickel, 64, 63.927966)
copper63 = Isotope('copper63', 'Cu63', copper, 63, 62.929597)
copper65 = Isotope('copper65', 'Cu65', copper, 65, 64.92779)
zinc64 = Isotope('zinc64', 'Zn64', zinc, 64, 63.929142)
zinc66 = Isotope('zinc66', 'Zn66', zinc, 66, 65.926034)
zinc67 = Isotope('zinc67', 'Zn67', zinc, 67, 66.927127)
zinc68 = Isotope('zinc68', 'Zn68', zinc, 68, 67.924844)
zinc70 = Isotope('zinc70', 'Zn70', zinc, 70, 69.92532)
gallium69 = Isotope('gallium69', 'Ga69', gallium, 69, 68.925573)
gallium71 = Isotope('gallium71', 'Ga71', gallium, 71, 70.924702)
germanium70 = Isotope('germanium70', 'Ge70', germanium, 70, 69.924249)
germanium72 = Isotope('germanium72', 'Ge72', germanium, 72, 71.9220758)
germanium73 = Isotope('germanium73', 'Ge73', germanium, 73, 72.923459)
germanium74 = Isotope('germanium74', 'Ge74', germanium, 74, 73.92117776)
germanium76 = Isotope('germanium76', 'Ge76', germanium, 76, 75.9214027)
arsenic75 = Isotope('arsenic75', 'As75', arsenic, 75, 74.921595)
selenium74 = Isotope('selenium74', 'Se74', selenium, 74, 73.9224759)
selenium76 = Isotope('selenium76', 'Se76', selenium, 76, 75.9192137)
selenium77 = Isotope('selenium77', 'Se77', selenium, 77, 76.9199141)
selenium78 = Isotope('selenium78', 'Se78', selenium, 78, 77.917309)
selenium80 = Isotope('selenium80', 'Se80', selenium, 80, 79.916522)
selenium82 = Isotope('selenium82', 'Se82', selenium, 82, 81.916699)
bromine79 = Isotope('bromine79', 'Br79', bromine, 79, 78.918338)
bromine81 = Isotope('bromine81', 'Br81', bromine, 81, 80.916288)
krypton78 = Isotope('krypton78', 'Kr78', krypton, 78, 77.920366)
krypton80 = Isotope('krypton80', 'Kr80', krypton, 80, 79.916378)
krypton82 = Isotope('krypton82', 'Kr82', krypton, 82, 81.91348115)
krypton83 = Isotope('krypton83', 'Kr83', krypton, 83, 82.91412652)
krypton84 = Isotope('krypton84', 'Kr84', krypton, 84, 83.91149773)
krypton86 = Isotope('krypton86', 'Kr86', krypton, 86, 85.91061063)
rubidium85 = Isotope('rubidium85', 'Rb85', rubidium, 85, 84.91178974)
rubidium87 = Isotope('rubidium87', 'Rb87', rubidium, 87, 86.90918053)
strontium84 = Isotope('strontium84', 'Sr84', strontium, 84, 83.913419)
strontium86 = Isotope('strontium86', 'Sr86', strontium, 86, 85.90926073)
strontium87 = Isotope('strontium87', 'Sr87', strontium, 87, 86.9088775)
strontium88 = Isotope('strontium88', 'Sr88', strontium, 88, 87.90561226)
yttrium89 = Isotope('yttrium89', 'Y89', yttrium, 89, 88.90584)
zirconium90 = Isotope('zirconium90', 'Zr90', zirconium, 90, 89.9046988)
zirconium91 = Isotope('zirconium91', 'Zr91', zirconium, 91, 90.9056402)
zirconium92 = Isotope('zirconium92', 'Zr92', zirconium, 92, 91.9050353)
zirconium94 = Isotope('zirconium94', 'Zr94', zirconium, 94, 93.906313)
zirconium96 = Isotope('zirconium96', 'Zr96', zirconium, 96, 95.9082776)
niobium93 = Isotope('niobium93', 'Nb93', niobium, 93, 92.90637)
molybdenum92 = Isotope('molybdenum92', 'Mo92', molybdenum, 92, 91.906807)
molybdenum94 = Isotope('molybdenum94', 'Mo94', molybdenum, 94, 93.905084)
molybdenum95 = Isotope('molybdenum95', 'Mo95', molybdenum, 95, 94.9058374)
molybdenum96 = Isotope('molybdenum96', 'Mo96', molybdenum, 96, 95.9046748)
molybdenum97 = Isotope('molybdenum97', 'Mo97', molybdenum, 97, 96.906017)
molybdenum98 = Isotope('molybdenum98', 'Mo98', molybdenum, 98, 97.905404)
molybdenum100 = Isotope('molybdenum100', 'Mo100', molybdenum, 100, 99.907468)
ruthenium96 = Isotope('ruthenium96', 'Ru96', ruthenium, 96, 95.907589)
ruthenium98 = Isotope('ruthenium98', 'Ru98', ruthenium, 98, 97.90529)
ruthenium99 = Isotope('ruthenium99', 'Ru99', ruthenium, 99, 98.90593)
ruthenium100 = Isotope('ruthenium100', 'Ru100', ruthenium, 100, 99.904211)
ruthenium101 = Isotope('ruthenium101', 'Ru101', ruthenium, 101, 100.905573)
ruthenium102 = Isotope('ruthenium102', 'Ru102', ruthenium, 102, 101.90434)
ruthenium104 = Isotope('ruthenium104', 'Ru104', ruthenium, 104, 103.90543)
rhodium103 = Isotope('rhodium103', 'Rh103', rhodium, 103, 102.90549)
palladium102 = Isotope('palladium102', 'Pd102', palladium, 102, 101.905632)
palladium104 = Isotope('palladium104', 'Pd104', palladium, 104, 103.90403)
palladium105 = Isotope('palladium105', 'Pd105', palladium, 105, 104.905079)
palladium106 = Isotope('palladium106', 'Pd106', palladium, 106, 105.90348)
palladium108 = Isotope('palladium108', 'Pd108', palladium, 108, 107.903892)
palladium110 = Isotope('palladium110', 'Pd110', palladium, 110, 109.905173)
silver107 = Isotope('silver107', 'Ag107', silver, 107, 106.90509)
silver109 = Isotope('silver109', 'Ag109', silver, 109, 108.904756)
cadmium106 = Isotope('cadmium106', 'Cd106', cadmium, 106, 105.90646)
cadmium108 = Isotope('cadmium108', 'Cd108', cadmium, 108, 107.904184)
cadmium110 = Isotope('cadmium110', 'Cd110', cadmium, 110, 109.903008)
cadmium111 = Isotope('cadmium111', 'Cd111', cadmium, 111, 110.904184)
cadmium112 = Isotope('cadmium112', 'Cd112', cadmium, 112, 111.902764)
cadmium113 = Isotope('cadmium113', 'Cd113', cadmium, 113, 112.904408)
cadmium114 = Isotope('cadmium114', 'Cd114', cadmium, 114, 113.903365)
cadmium116 = Isotope('cadmium116', 'Cd116', cadmium, 116, 115.904763)
indium113 = Isotope('indium113', 'In113', indium, 113, 112.90406)
indium115 = Isotope('indium115', 'In115', indium, 115, 114.90387877)
tin112 = Isotope('tin112', 'Sn112', tin, 112, 111.904825)
tin114 = Isotope('tin114', 'Sn114', tin, 114, 113.9027801)
tin115 = Isotope('tin115', 'Sn115', tin, 115, 114.9033447)
tin116 = Isotope('tin116', 'Sn116', tin, 116, 115.9017428)
tin117 = Isotope('tin117', 'Sn117', tin, 117, 116.902954)
tin118 = Isotope('tin118', 'Sn118', tin, 118, 117.901607)
tin119 = Isotope('tin119', 'Sn119', tin, 119, 118.903311)
tin120 = Isotope('tin120', 'Sn120', tin, 120, 119.902202)
tin122 = Isotope('tin122', 'Sn122', tin, 122, 121.90344)
tin124 = Isotope('tin124', 'Sn124', tin, 124, 123.905277)
antimony121 = Isotope('antimony121', 'Sb121', antimony, 121, 120.90381)
antimony123 = Isotope('antimony123', 'Sb123', antimony, 123, 122.90421)
tellurium120 = Isotope('tellurium120', 'Te120', tellurium, 120, 119.90406)
tellurium122 = Isotope('tellurium122', 'Te122', tellurium, 122, 121.90304)
tellurium123 = Isotope('tellurium123', 'Te123', tellurium, 123, 122.90427)
tellurium124 = Isotope('tellurium124', 'Te124', tellurium, 124, 123.90282)
tellurium125 = Isotope('tellurium125', 'Te125', tellurium, 125, 124.90443)
tellurium126 = Isotope('tellurium126', 'Te126', tellurium, 126, 125.90331)
tellurium128 = Isotope('tellurium128', 'Te128', tellurium, 128, 127.904461)
tellurium130 = Isotope('tellurium130', 'Te130', tellurium, 130, 129.90622275)
iodine127 = Isotope('iodine127', 'I127', iodine, 127, 126.90447)
xenon124 = Isotope('xenon124', 'Xe124', xenon, 124, 123.90589)
xenon126 = Isotope('xenon126', 'Xe126', xenon, 126, 125.9043)
xenon128 = Isotope('xenon128', 'Xe128', xenon, 128, 127.903531)
xenon129 = Isotope('xenon129', 'Xe129', xenon, 129, 128.90478086)
xenon130 = Isotope('xenon130', 'Xe130', xenon, 130, 129.90350935)
xenon131 = Isotope('xenon131', 'Xe131', xenon, 131, 130.90508414)
xenon132 = Isotope('xenon132', 'Xe132', xenon, 132, 131.90415509)
xenon134 = Isotope('xenon134', 'Xe134', xenon, 134, 133.90539303)
xenon136 = Isotope('xenon136', 'Xe136', xenon, 136, 135.90721448)
caesium133 = Isotope('caesium133', 'Cs133', caesium, 133, 132.90545196)
barium130 = Isotope('barium130', 'Ba130', barium, 130, 129.90632)
barium132 = Isotope('barium132', 'Ba132', barium, 132, 131.905061)
barium134 = Isotope('barium134', 'Ba134', barium, 134, 133.904508)
barium135 = Isotope('barium135', 'Ba135', barium, 135, 134.905689)
barium136 = Isotope('barium136', 'Ba136', barium, 136, 135.904576)
barium137 = Isotope('barium137', 'Ba137', barium, 137, 136.905827)
barium138 = Isotope('barium138', 'Ba138', barium, 138, 137.905247)
lanthanum138 = Isotope('lanthanum138', 'La138', lanthanum, 138, 137.90712)
lanthanum139 = Isotope('lanthanum139', 'La139', lanthanum, 139, 138.90636)
cerium136 = Isotope('cerium136', 'Ce136', cerium, 136, 135.907129)
cerium138 = Isotope('cerium138', 'Ce138', cerium, 138, 137.90599)
cerium140 = Isotope('cerium140', 'Ce140', cerium, 140, 139.90545)
cerium142 = Isotope('cerium142', 'Ce142', cerium, 142, 141.90925)
praseodymium141 = Isotope('praseodymium141', 'Pr141', praseodymium, 141, 140.90766)
neodymium142 = Isotope('neodymium142', 'Nd142', neodymium, 142, 141.90773)
neodymium143 = Isotope('neodymium143', 'Nd143', neodymium, 143, 142.90982)
neodymium144 = Isotope('neodymium144', 'Nd144', neodymium, 144, 143.91009)
neodymium145 = Isotope('neodymium145', 'Nd145', neodymium, 145, 144.91258)
neodymium146 = Isotope('neodymium146', 'Nd146', neodymium, 146, 145.91312)
neodymium148 = Isotope('neodymium148', 'Nd148', neodymium, 148, 147.9169)
neodymium150 = Isotope('neodymium150', 'Nd150', neodymium, 150, 149.920902)
samarium144 = Isotope('samarium144', 'Sm144', samarium, 144, 143.91201)
samarium147 = Isotope('samarium147', 'Sm147', samarium, 147, 146.9149)
samarium148 = Isotope('samarium148', 'Sm148', samarium, 148, 147.91483)
samarium149 = Isotope('samarium149', 'Sm149', samarium, 149, 148.917191)
samarium150 = Isotope('samarium150', 'Sm150', samarium, 150, 149.917282)
samarium152 = Isotope('samarium152', 'Sm152', samarium, 152, 151.919739)
samarium154 = Isotope('samarium154', 'Sm154', samarium, 154, 153.92222)
europium151 = Isotope('europium151', 'Eu151', europium, 151, 150.919857)
europium153 = Isotope('europium153', 'Eu153', europium, 153, 152.921237)
gadolinium152 = Isotope('gadolinium152', 'Gd152', gadolinium, 152, 151.919799)
gadolinium154 = Isotope('gadolinium154', 'Gd154', gadolinium, 154, 153.920873)
gadolinium155 = Isotope('gadolinium155', 'Gd155', gadolinium, 155, 154.92263)
gadolinium156 = Isotope('gadolinium156', 'Gd156', gadolinium, 156, 155.922131)
gadolinium157 = Isotope('gadolinium157', 'Gd157', gadolinium, 157, 156.923968)
gadolinium158 = Isotope('gadolinium158', 'Gd158', gadolinium, 158, 157.924112)
gadolinium160 = Isotope('gadolinium160', 'Gd160', gadolinium, 160, 159.927062)
terbium159 = Isotope('terbium159', 'Tb159', terbium, 159, 158.925354)
dysprosium156 = Isotope('dysprosium156', 'Dy156', dysprosium, 156, 155.924284)
dysprosium158 = Isotope('dysprosium158', 'Dy158', dysprosium, 158, 157.92441)
dysprosium160 = Isotope('dysprosium160', 'Dy160', dysprosium, 160, 159.925203)
dysprosium161 = Isotope('dysprosium161', 'Dy161', dysprosium, 161, 160.926939)
dysprosium162 = Isotope('dysprosium162', 'Dy162', dysprosium, 162, 161.926804)
dysprosium163 = Isotope('dysprosium163', 'Dy163', dysprosium, 163, 162.928737)
dysprosium164 = Isotope('dysprosium164', 'Dy164', dysprosium, 164, 163.929181)
holmium165 = Isotope('holmium165', 'Ho165', holmium, 165, 164.930328)
erbium162 = Isotope('erbium162', 'Er162', erbium, 162, 161.928787)
erbium164 = Isotope('erbium164', 'Er164', erbium, 164, 163.929207)
erbium166 = Isotope('erbium166', 'Er166', erbium, 166, 165.930299)
erbium167 = Isotope('erbium167', 'Er167', erbium, 167, 166.932054)
erbium168 = Isotope('erbium168', 'Er168', erbium, 168, 167.932376)
erbium170 = Isotope('erbium170', 'Er170', erbium, 170, 169.93547)
thulium169 = Isotope('thulium169', 'Tm169', thulium, 169, 168.934218)
ytterbium168 = Isotope('ytterbium168', 'Yb168', ytterbium, 168, 167.933889)
ytterbium170 = Isotope('ytterbium170', 'Yb170', ytterbium, 170, 169.93476725)
ytterbium171 = Isotope('ytterbium171', 'Yb171', ytterbium, 171, 170.93633152)
ytterbium172 = Isotope('ytterbium172', 'Yb172', ytterbium, 172, 171.93638666)
ytterbium173 = Isotope('ytterbium173', 'Yb173', ytterbium, 173, 172.93821622)
ytterbium174 = Isotope('ytterbium174', 'Yb174', ytterbium, 174, 173.93886755)
ytterbium176 = Isotope('ytterbium176', 'Yb176', ytterbium, 176, 175.9425747)
lutetium175 = Isotope('lutetium175', 'Lu175', lutetium, 175, 174.940777)
lutetium176 = Isotope('lutetium176', 'Lu176', lutetium, 176, 175.942692)
hafnium174 = Isotope('hafnium174', 'Hf174', hafnium, 174, 173.94005)
hafnium176 = Isotope('hafnium176', 'Hf176', hafnium, 176, 175.94141)
hafnium177 = Isotope('hafnium177', 'Hf177', hafnium, 177, 176.94323)
hafnium178 = Isotope('hafnium178', 'Hf178', hafnium, 178, 177.94371)
hafnium179 = Isotope('hafnium179', 'Hf179', hafnium, 179, 178.94583)
hafnium180 = Isotope('hafnium180', 'Hf180', hafnium, 180, 179.94656)
tantalum180 = Isotope('tantalum180', 'Ta180', tantalum, 180, 179.94747)
tantalum181 = Isotope('tantalum181', 'Ta181', tantalum, 181, 180.948)
tungsten180 = Isotope('tungsten180', 'W180', tungsten, 180, 179.94671)
tungsten182 = Isotope('tungsten182', 'W182', tungsten, 182, 181.948206)
tungsten183 = Isotope('tungsten183', 'W183', tungsten, 183, 182.950224)
tungsten184 = Isotope('tungsten184', 'W184', tungsten, 184, 183.950933)
tungsten186 = Isotope('tungsten186', 'W186', tungsten, 186, 185.954365)
rhenium185 = Isotope('rhenium185', 'Re185', rhenium, 185, 184.952958)
rhenium187 = Isotope('rhenium187', 'Re187', rhenium, 187, 186.955752)
osmium184 = Isotope('osmium184', 'Os184', osmium, 184, 183.952493)
osmium186 = Isotope('osmium186', 'Os186', osmium, 186, 185.953838)
osmium187 = Isotope('osmium187', 'Os187', osmium, 187, 186.95575)
osmium188 = Isotope('osmium188', 'Os188', osmium, 188, 187.955837)
osmium189 = Isotope('osmium189', 'Os189', osmium, 189, 188.958146)
osmium190 = Isotope('osmium190', 'Os190', osmium, 190, 189.958446)
osmium192 = Isotope('osmium192', 'Os192', osmium, 192, 191.96148)
iridium191 = Isotope('iridium191', 'Ir191', iridium, 191, 190.960591)
iridium193 = Isotope('iridium193', 'Ir193', iridium, 193, 192.962924)
platinum190 = Isotope('platinum190', 'Pt190', platinum, 190, 189.95995)
platinum192 = Isotope('platinum192', 'Pt192', platinum, 192, 191.96104)
platinum194 = Isotope('platinum194', 'Pt194', platinum, 194, 193.962683)
platinum195 = Isotope('platinum195', 'Pt195', platinum, 195, 194.964794)
platinum196 = Isotope('platinum196', 'Pt196', platinum, 196, 195.964955)
platinum198 = Isotope('platinum198', 'Pt198', platinum, 198, 197.9679)
gold197 = Isotope('gold197', 'Au197', gold, 197, 196.96657)
mercury196 = Isotope('mercury196', 'Hg196', mercury, 196, 195.96583)
mercury198 = Isotope('mercury198', 'Hg198', mercury, 198, 197.966769)
mercury199 = Isotope('mercury199', 'Hg199', mercury, 199, 198.968281)
mercury200 = Isotope('mercury200', 'Hg200', mercury, 200, 199.968327)
mercury201 = Isotope('mercury201', 'Hg201', mercury, 201, 200.970303)
mercury202 = Isotope('mercury202', 'Hg202', mercury, 202, 201.970644)
mercury204 = Isotope('mercury204', 'Hg204', mercury, 204, 203.973494)
thallium203 = Isotope('thallium203', 'Tl203', thallium, 203, 202.972344)
thallium205 = Isotope('thallium205', 'Tl205', thallium, 205, 204.974427)
lead204 = Isotope('lead204', 'Pb204', lead, 204, 203.973043)
lead206 = Isotope('lead206', 'Pb206', lead, 206, 205.974465)
lead207 = Isotope('lead207', 'Pb207', lead, 207, 206.975897)
lead208 = Isotope('lead208', 'Pb208', lead, 208, 207.976652)
bismuth209 = Isotope('bismuth209', 'Bi209', bismuth, 209, 208.9804)
thorium230 = Isotope('thorium230', 'Th230', thorium, 230, 230.033132)
thorium232 = Isotope('thorium232', 'Th232', thorium, 232, 232.03805)
protactinium231 = Isotope('protactinium231', 'Pa231', protactinium, 231, 231.03588)
uranium234 = Isotope('uranium234', 'U234', uranium, 234, 234.04095)
uranium235 = Isotope('uranium235', 'U235', uranium, 235, 235.043928)
uranium238 = Isotope('uranium238', 'U238', uranium, 238, 238.05079)

# once objects created build an indices for reverse lookup (string instancing of element)
_build_element_index()
_build_isotope_index()
