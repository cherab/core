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

from cherab.core cimport ImpactExcitationPEC as CoreImpactExcitationPEC
from cherab.core cimport RecombinationPEC as CoreRecombinationPEC
from cherab.core cimport ThermalCXPEC as CoreThermalCXPEC
from cherab.core.math cimport Interpolate2DCubic


cdef class ImpactExcitationPEC(CoreImpactExcitationPEC):

    cdef:
        readonly dict raw_data
        readonly double wavelength
        readonly tuple density_range, temperature_range
        Interpolate2DCubic _rate


cdef class NullImpactExcitationPEC(CoreImpactExcitationPEC):
    pass


cdef class RecombinationPEC(CoreRecombinationPEC):

    cdef:
        readonly dict raw_data
        readonly double wavelength
        readonly tuple density_range, temperature_range
        Interpolate2DCubic _rate


cdef class NullRecombinationPEC(CoreRecombinationPEC):
    pass


# cdef class CXThermalRate(CoreCXThermalRate):
#     pass
#
# cdef class ThermalCXRate(CoreThermalCXRate):
#     pass
