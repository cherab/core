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

cdef:
    double RECIP_2_PI = 1 / (2 * M_PI)
    double RECIP_4_PI = 1 / (4 * M_PI)
    double ATOMIC_MASS = 1.66053892e-27
    double ELEMENTARY_CHARGE = 1.6021766208e-19
    double SPEED_OF_LIGHT = 299792458.0
    double DEGREES_TO_RADIANS = M_PI / 180
    double RADIANS_TO_DEGREES = 180 / M_PI
    double PLANCK_CONSTANT = 6.6260700400e-34
