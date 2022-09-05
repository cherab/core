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


from raysect.core.math.function.float cimport Function3D
from raysect.optical cimport Spectrum, Point3D, Vector3D

from cherab.core.laser cimport LaserProfile


cdef class UniformEnergyDensity(LaserProfile):

    cdef:
        double _energy_density, _laser_length, _laser_radius


cdef class ConstantAxisymmetricGaussian(LaserProfile):

    cdef:
        double _stddev, _pulse_energy, _pulse_length, _laser_length, _laser_radius
        Function3D _distribution


cdef class ConstantBivariateGaussian(LaserProfile):

    cdef:
        double _stddev_x, _stddev_y, _pulse_energy, _pulse_length, _laser_length, _laser_radius
        Function3D _distribution


cdef class TrivariateGaussian(LaserProfile):

    cdef:
        double _stddev_x, _stddev_y, _stddev_z, _mean_z, _pulse_energy, _pulse_length, _laser_length, _laser_radius
        Function3D _distribution


cdef class GaussianBeamAxisymmetric(LaserProfile):

    cdef:
        double _pulse_energy, _pulse_length, _stddev_waist, _waist_z, _laser_wavelength, _laser_length, _laser_radius
        Function3D _distribution
