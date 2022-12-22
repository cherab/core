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


from raysect.optical cimport Vector3D, Point3D
from raysect.optical.spectrum cimport Spectrum

from cherab.core cimport Plasma
from cherab.core.laser.profile cimport LaserProfile
from cherab.core.laser.laserspectrum cimport LaserSpectrum


cdef class LaserModel:
    """
    Laser spectrum base class.

    This is an abstract class and cannot be used for observing.

    Calculates the contribution to a spectrum caused by a laser.

    :param laser_profile: LaserProfile object
    :param plasma: Plasma object
    :param laser_spectrum: LaserSpectrum object

    :ivar laser_profile: LaserProfile object
    :ivar plasma: Plasma object
    :ivar laser_spectrum: LaserSpectrum object
    """
    def __init__(self, LaserProfile laser_profile=None, LaserSpectrum laser_spectrum=None, Plasma plasma=None):

        self._laser_profile = laser_profile
        self._laser_spectrum = laser_spectrum
        self._plasma = plasma

    cpdef Spectrum emission(self, Point3D point_plasma, Vector3D observation_plasma, Point3D point_laser, Vector3D observation_laser,
                            Spectrum spectrum):

        raise NotImplementedError('Virtual method must be implemented in a sub-class.')

    @property
    def laser_profile(self):
        return self._laser_profile

    @laser_profile.setter
    def laser_profile(self, LaserProfile value):
        self._laser_profile = value

    @property
    def plasma(self):
        return self._plasma

    @plasma.setter
    def plasma(self, Plasma value):
        self._plasma = value

    @property
    def laser_spectrum(self):
        return self._laser_spectrum

    @laser_spectrum.setter
    def laser_spectrum(self, LaserSpectrum value):

        self._laser_spectrum = value
