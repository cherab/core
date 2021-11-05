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
    double ELECTRON_CLASSICAL_RADIUS = 2.8179403262e-15
    double ELECTRON_REST_MASS = 9.1093837015e-31

