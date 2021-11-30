# Copyright 2016-2021 Euratom
# Copyright 2016-2021 United Kingdom Atomic Energy Authority
# Copyright 2016-2021 Centro de Investigaciones Energéticas, Medioambientales y Tecnológicas
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

from cherab.core cimport IonisationRate as CoreIonisationRate
from cherab.core cimport RecombinationRate as CoreRecombinationRate
from cherab.core cimport ThermalCXRate as CoreThermalCXRate
from cherab.core.math cimport Function2D


cdef class IonisationRate(CoreIonisationRate):

    cdef:
        readonly dict raw_data
        readonly tuple density_range, temperature_range
        Function2D _rate


cdef class NullIonisationRate(CoreIonisationRate):
    pass


cdef class RecombinationRate(CoreRecombinationRate):

    cdef:
        readonly dict raw_data
        readonly tuple density_range, temperature_range
        Function2D _rate


cdef class NullRecombinationRate(CoreRecombinationRate):
    pass


cdef class ThermalCXRate(CoreThermalCXRate):

    cdef:
        readonly dict raw_data
        readonly tuple density_range, temperature_range
        Function2D _rate


cdef class NullThermalCXRate(CoreThermalCXRate):
    pass
