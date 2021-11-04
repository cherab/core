# cython: language_level=3

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
# under the Licence

from numpy cimport ndarray
from raysect.optical cimport World, Primitive, Ray, Spectrum, Point3D, Vector3D, AffineMatrix3D
from raysect.optical.material cimport VolumeIntegrator, InhomogeneousVolumeEmitter


cdef class RayTransferIntegrator(VolumeIntegrator):

    cdef:
        double _step
        int _min_samples


cdef class CylindricalRayTransferIntegrator(RayTransferIntegrator):

    cpdef Spectrum integrate(self, Spectrum spectrum, World world, Ray ray, Primitive primitive,
                             InhomogeneousVolumeEmitter material, Point3D start_point, Point3D end_point,
                             AffineMatrix3D world_to_primitive, AffineMatrix3D primitive_to_world)


cdef class CartesianRayTransferIntegrator(RayTransferIntegrator):

    cpdef Spectrum integrate(self, Spectrum spectrum, World world, Ray ray, Primitive primitive,
                             InhomogeneousVolumeEmitter material, Point3D start_point, Point3D end_point,
                             AffineMatrix3D world_to_primitive, AffineMatrix3D primitive_to_world)


cdef class RayTransferEmitter(InhomogeneousVolumeEmitter):

    cdef:
        int[3] _grid_shape
        double[3] _grid_steps
        int _bins
        ndarray _voxel_map
        public:
            int[:, :, ::1] voxel_map_mv

    cdef ndarray _map_from_mask(self, mask)


cdef class CylindricalRayTransferEmitter(RayTransferEmitter):

    cdef:
        double _dr, _dphi, _dz, _period, _rmin

    cpdef Spectrum emission_function(self, Point3D point, Vector3D direction, Spectrum spectrum,
                                     World world, Ray ray, Primitive primitive,
                                     AffineMatrix3D world_to_primitive, AffineMatrix3D primitive_to_world)


cdef class CartesianRayTransferEmitter(RayTransferEmitter):

    cdef:
        double _dx, _dy, _dz

    cpdef Spectrum emission_function(self, Point3D point, Vector3D direction, Spectrum spectrum,
                                     World world, Ray ray, Primitive primitive,
                                     AffineMatrix3D world_to_primitive, AffineMatrix3D primitive_to_world)