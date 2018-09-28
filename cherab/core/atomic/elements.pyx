# cython: language_level=3

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


cdef class Element:
    """
    Class representing an atomic element.

    :param str name: Element name.
    :param str symbol: Element symbol, e.g. 'H'.
    :param int atomic_number: Number of protons.
    :param float atomic_weight: average atomic weight in amu
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

    :param str name: Element name.
    :param str symbol: Element symbol, e.g. 'H'.
    :param int atomic_number: Number of protons.
    :param int mass_number: Atomic mass number, which is total number of protons
      and neutrons. Allows identification of specific isotopes.
    :param float atomic_weight: atomic weight in amu
    """

    def __init__(self, str name, str symbol, Element element, int mass_number, double atomic_weight):

        super().__init__(name, symbol, element.atomic_number, atomic_weight)
        self.mass_number = mass_number
        self.element = element


    def __repr__(self):
        return '<Isotope: {}>'.format(self.name)

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

