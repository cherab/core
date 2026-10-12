# Copyright 2016-2022 Euratom
# Copyright 2016-2022 United Kingdom Atomic Energy Authority
# Copyright 2016-2022 Centro de Investigaciones Energéticas, Medioambientales y Tecnológicas
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
from types import ModuleType

from libc.math cimport M_PI


cdef:

    # sourced c standard maths library
    double RECIP_2_PI = 1 / (2 * M_PI)
    double RECIP_4_PI = 1 / (4 * M_PI)
    double DEGREES_TO_RADIANS = M_PI / 180
    double RADIANS_TO_DEGREES = 180 / M_PI

    # sourced from NIST, CODATA 2018: https://physics.nist.gov/cuu/Constants/Table/allascii.txt
    double ATOMIC_MASS = 1.66053906660e-27
    double ELEMENTARY_CHARGE = 1.602176634e-19
    double SPEED_OF_LIGHT = 299792458.0
    double PLANCK_CONSTANT = 6.62607015e-34
    double HC_EV_NM = 1239.8419738620933  # (Planck constant in eV s) x (speed of light in nm/s)
    double ELECTRON_CLASSICAL_RADIUS = 2.8179403262e-15
    double ELECTRON_REST_MASS = 9.1093837015e-31
    double RYDBERG_CONSTANT_EV = 13.605693122994
    double VACUUM_PERMITTIVITY = 8.8541878128e-12
    double BOHR_MAGNETON = 5.78838180123e-5  # in eV/T


# Make the constants available to Python too.
# To ensure the Python and Cython constants do not got out of sync the exported
# Python attributes of the module are made read only using module getattr.
cdef dict _CONSTANTS = {
    # c stdlib
    "RECIP_2_PI": RECIP_2_PI,
    "RECIP_4_PI": RECIP_4_PI,
    "DEGREES_TO_RADIANS": DEGREES_TO_RADIANS,
    "RADIANS_TO_DEGREES": RADIANS_TO_DEGREES,
    # NIST 2018
    "ATOMIC_MASS": ATOMIC_MASS,
    "ELEMENTARY_CHARGE": ELEMENTARY_CHARGE,
    "SPEED_OF_LIGHT": SPEED_OF_LIGHT,
    "PLANCK_CONSTANT": PLANCK_CONSTANT,
    "HC_EV_NM": HC_EV_NM,
    "ELECTRON_CLASSICAL_RADIUS": ELECTRON_CLASSICAL_RADIUS,
    "ELECTRON_REST_MASS": ELECTRON_REST_MASS,
    "RYDBERG_CONSTANT_EV": RYDBERG_CONSTANT_EV,
    "VACUUM_PERMITTIVITY": VACUUM_PERMITTIVITY,
    "BOHR_MAGNETON": BOHR_MAGNETON,
}


def __getattr__(name):
    if name not in _CONSTANTS:
        raise AttributeError()
    return _CONSTANTS[name]


def __dir__():
    return list(_CONSTANTS.keys())


class ReadOnlyModule(ModuleType):
    def __setattr__(self, attr, value):
        raise AttributeError("Constants are read-only")

    def __delattr__(self, attr):
        raise AttributeError("Constants are read-only")


sys.modules[__name__].__class__ = ReadOnlyModule
