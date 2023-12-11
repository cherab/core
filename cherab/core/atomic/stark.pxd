# Copyright 2016-2023 Euratom
# Copyright 2016-2023 United Kingdom Atomic Energy Authority
# Copyright 2016-2023 Centro de Investigaciones Energéticas, Medioambientales y Tecnológicas
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


from numpy cimport ndarray


cdef class StarkStructure:

    cdef:
        readonly ndarray index
        readonly ndarray polarisation
        const int[::1] index_mv
        const int[::1] polarisation_mv

    cdef double[::1] evaluate(self, double energy, double density, double b_field)


cdef class InterpolatedStarkStructure(StarkStructure):

    cdef:
        readonly dict raw_data
        readonly tuple beam_energy_range
        readonly tuple density_range
        readonly tuple b_field_range
        list _ratio_functions
        double _cached_energy, _cached_density, _cached_b_field
        double[::1] _ratios_mv
