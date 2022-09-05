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

cdef class ConstantAxisymmetricGaussian3D(Function3D):

    cdef:
        double _stddev, _normalisation, _kr

    cdef double evaluate(self, double x, double y, double z) except? -1e999


cdef class ConstantBivariateGaussian3D(Function3D):

    cdef:
        double _stddev_x, _stddev_y, _kx, _ky, _normalisation


cdef class TrivariateGaussian3D(Function3D):

    cdef:
        double _mean_z, _stddev_x, _stddev_y, _stddev_z, _kx, _ky
        double _kz, _normalisation


cdef class GaussianBeamModel(Function3D):

    cdef:
        double _waist_z, _stddev_waist, _stddev_waist2, _wavelength, _rayleigh_range

    cdef double evaluate(self, double x, double y, double z) except? -1e999
