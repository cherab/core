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

from cherab.core.atomic cimport Element
from cherab.core.distribution cimport DistributionFunction


# immutable, so the plasma doesn't have to track changes
cdef class Species:
    """
    A class representing a given plasma species.

    A plasma in Cherab will be composed of 1 or more Species objects. A species
    can be uniquely identified through its element and charge state.

    When instantiating a Species object a 6D distribution function (3 space, 3 velocity)
    must be defined. The DistributionFunction object provides the base interface for
    defining a distribution function, it could be a reduced analytic representation
    (such as a Maxwellian for example) or a fully numerically interpolated 6D function.

    :param Element element: The element object of this species.
    :param int charge: The charge state of the species.
    :param DistributionFunction distribution: A distribution function for this species.

    .. code-block:: pycon

       >>> # In this example we define a single plasma species with spatially homogeneous properties
       >>>
       >>> from scipy.constants import atomic_mass
       >>> from raysect.core.math import Vector3D
       >>> from cherab.core import Species, Maxwellian
       >>> from cherab.core.atomic import deuterium
       >>>
       >>> # Setup a distribution function for the species
       >>> density = 1E18
       >>> temperature = 10
       >>> bulk_velocity = Vector3D(-1e6, 0, 0)
       >>> d1_distribution = Maxwellian(density, temperature, bulk_velocity, deuterium.atomic_weight * atomic_mass)
       >>>
       >>> # create the plasma Species object
       >>> d1_species = Species(deuterium, 1, d1_distribution)
       >>>
       >>> # Request some properties from the species' distribution function.
       >>> print(d1_species)
       <Species: element=deuterium, charge=1>
       >>> d1_species.distribution.density(1, -2.5, 7)
       1e+18
    """

    def __init__(self, Element element, int charge, DistributionFunction distribution):

        if charge > element.atomic_number:
            raise ValueError("Charge state cannot be larger than the atomic number.")

        if charge < 0:
            raise ValueError("Charge state cannot be less than zero.")

        self.element = element
        self.charge = charge
        self.distribution = distribution

    def __repr__(self):
        return '<Species: element={}, charge={}>'.format(self.element.name, self.charge)


# todo: move to a common exception module
class SpeciesNotFound(Exception):
    pass
