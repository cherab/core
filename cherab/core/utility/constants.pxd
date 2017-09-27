# Copyright 2014-2017 United Kingdom Atomic Energy Authority
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

from libc.math cimport M_PI

cdef:
    public double RECIP_2_PI = 0.15915494309189535
    public double RECIP_4_PI = 0.07957747154594767
    public double AMU = 1.66053892e-27
    public double ELEMENTARY_CHARGE = 1.6021766208e-19
    public double SPEED_OF_LIGHT = 299792458.0
    public double DEGREES_TO_RADIANS = 2 * M_PI / 360
    public double RADIANS_TO_DEGREES = 360 / (2 * M_PI)
    public double PLANCK_CONSTANT = 6.6260700400e-34
