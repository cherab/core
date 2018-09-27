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

# TODO: requires reimplementation in future release

# from cherab.core.atomic.elements cimport Element
# from cherab.core.atomic.rates cimport StageResolvedLineRadiation
# from cherab.core.plasma cimport PlasmaModel
# from cherab.core.species cimport Species
#
#
# cdef class TotalRadiatedPower(PlasmaModel):
#
#     cdef:
#         Element _element
#         int _ionisation
#         Species _target_species
#         StageResolvedLineRadiation _plt_rate, _prb_rate
#
#     cdef inline int _populate_cache(self) except -1
