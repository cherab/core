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

from cherab.core.laser cimport Laser, LaserModel, LaserSpectrum, LaserProfile


cdef class SeldenMatobaThomsonSpectrum(LaserModel):
    
    cdef:
        double _CONST_ALPHA, _RATE_TS, _RECIP_M_PI

    cpdef Spectrum emission(self, Point3D point_plasma, Vector3D observation_plasma, Point3D point_laser,
                            Vector3D observation_laser, Spectrum spectrum)

    cdef double seldenmatoba_spectral_shape(self, double epsilon, double cos_theta, double alpha)

    cdef Spectrum _add_spectral_contribution(self, double ne, double te, double laser_energy, double angle_pointing,
                                             double angle_polarization, double laser_wavelength, Spectrum spectrum)

    cpdef Spectrum calculate_spectrum(self, double ne, double te, double laser_energy_density, double laser_wavelength,
                                      double observation_angle, double angle_polarization, Spectrum spectrum)