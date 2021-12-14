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

"""
Utility functions shared for two or more modules
"""

import numpy as np
from scipy.constants import atomic_mass, elementary_charge
from scipy.constants import Planck, speed_of_light


class EvAmuToMS:
    """Converts from eV/amu to velocity (m/s)
    """
    conversion_factor = 2 * elementary_charge / atomic_mass

    @classmethod
    def to(cls, x):
        """Direct conversion"""
        return np.sqrt(x * cls.conversion_factor)

    @classmethod
    def inv(cls, x):
        """Inverse conversion"""
        return (x ** 2) / cls.conversion_factor


class PhotonToJ:
    """Converts from photon to Jules
    """
    conversion_factor = Planck * speed_of_light * 1e9

    @classmethod
    def to(cls, x, wavelength):
        """Direct conversion; wavelength in nm"""
        return x / wavelength * cls.conversion_factor 

    @classmethod
    def inv(cls, x, wavelength):
        """Inverse conversion; wavelength in nm"""
        return x * wavelength / cls.conversion_factor


class BaseFactorConversion:
    """Base class for conversion based on factor
    """
    @classmethod
    def to(cls, x):
        """Direct conversion"""
        return x * cls.conversion_factor

    @classmethod
    def inv(cls, x):
        """Inverse conversion"""
        return x / cls.conversion_factor


class AmuToKg(BaseFactorConversion):
    """Converts from amu to kg
    """
    conversion_factor = atomic_mass


class EvToJ(BaseFactorConversion):
    """Converts from eV to Jules
    """
    conversion_factor = elementary_charge


class Cm3ToM3(BaseFactorConversion):
    """Converts from cm3 to m3
    """
    conversion_factor = 1e-6


class PerCm3ToPerM3(BaseFactorConversion):
    """Converts from cm-3 to m-3
    """
    conversion_factor = 1e6


class AngstromToNm(BaseFactorConversion):
    """Converts from Angstroms to nm.
    """
    conversion_factor = 0.1
