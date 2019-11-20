#
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
#

"""
This is almost a copy of the raysect/optical/library/metal/roughmetal.py file.
The only difference is that RoughConductor class is replaces with RtmOptimisedRoughConductor class.

The data used to define the following metal materials was sourced from http://refractiveindex.info.
This data is licensed as public domain (CC0 1.0 - https://creativecommons.org/publicdomain/zero/1.0/).
"""

from os import path
import json
from numpy import array
import raysect.optical.library.metal as libmetal
from raysect.optical import InterpolatedSF
from .roughconductor import RToptimisedRoughConductor


class _DataLoader(RToptimisedRoughConductor):

    def __init__(self, filename, roughness):

        with open(path.join(path.dirname(libmetal.__file__), "data", filename + ".json")) as f:
            data = json.load(f)

        wavelength = array(data['wavelength'])
        index = InterpolatedSF(wavelength, array(data['index']))
        extinction = InterpolatedSF(wavelength, array(data['extinction']))

        super().__init__(index, extinction, roughness)


class RoughAluminium(_DataLoader):
    """Aluminium metal material."""
    def __init__(self, roughness):
        super().__init__("aluminium", roughness)


class RoughBeryllium(_DataLoader):
    """Beryllium metal material."""
    def __init__(self, roughness):
        super().__init__("beryllium", roughness)


class RoughCobolt(_DataLoader):
    """Cobolt metal material."""
    def __init__(self, roughness):
        super().__init__("cobolt", roughness)


class RoughCopper(_DataLoader):
    """Copper metal material."""
    def __init__(self, roughness):
        super().__init__("copper", roughness)


class RoughGold(_DataLoader):
    """Gold metal material."""
    def __init__(self, roughness):
        super().__init__("gold", roughness)


class RoughIron(_DataLoader):
    """Iron metal material."""
    def __init__(self, roughness):
        super().__init__("iron", roughness)


class RoughLithium(_DataLoader):
    """Lithium metal material."""
    def __init__(self, roughness):
        super().__init__("lithium", roughness)


class RoughMagnesium(_DataLoader):
    """Magnesium metal material."""
    def __init__(self, roughness):
        super().__init__("magnesium", roughness)


class RoughManganese(_DataLoader):
    """Manganese metal material."""
    def __init__(self, roughness):
        super().__init__("manganese", roughness)


class RoughMercury(_DataLoader):
    """Mercury metal material."""
    def __init__(self, roughness):
        super().__init__("mercury", roughness)


class RoughNickel(_DataLoader):
    """Nickel metal material."""
    def __init__(self, roughness):
        super().__init__("nickel", roughness)


class RoughPalladium(_DataLoader):
    """Palladium metal material."""
    def __init__(self, roughness):
        super().__init__("palladium", roughness)


class RoughPlatinum(_DataLoader):
    """Platinum metal material."""
    def __init__(self, roughness):
        super().__init__("platinum", roughness)


class RoughSilicon(_DataLoader):
    """Silicon metal material."""
    def __init__(self, roughness):
        super().__init__("silicon", roughness)


class RoughSilver(_DataLoader):
    """Silver metal material."""
    def __init__(self, roughness):
        super().__init__("silver", roughness)


class RoughSodium(_DataLoader):
    """Sodium metal material."""
    def __init__(self, roughness):
        super().__init__("sodium", roughness)


class RoughTitanium(_DataLoader):
    """Titanium metal material."""
    def __init__(self, roughness):
        super().__init__("titanium", roughness)


class RoughTungsten(_DataLoader):
    """Tungsten metal material."""
    def __init__(self, roughness):
        super().__init__("tungsten", roughness)
