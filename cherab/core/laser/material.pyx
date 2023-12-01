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


from raysect.core cimport Primitive
from raysect.optical cimport World, Primitive, Ray, Spectrum, Point3D, Vector3D, AffineMatrix3D
from raysect.optical.material.emitter cimport InhomogeneousVolumeEmitter
from raysect.optical.material.emitter.inhomogeneous cimport VolumeIntegrator

from cherab.core.laser.node cimport Laser
from cherab.core.laser.model cimport LaserModel


cdef class LaserMaterial(InhomogeneousVolumeEmitter):

    def __init__(self, Laser laser not None, Primitive laser_segment not None, list models, VolumeIntegrator integrator not None):

        super().__init__(integrator)

        self._laser = laser
        self._primitive = laser_segment
        self.importance = laser.importance
        
        #validate and set models
        for model in models:
            if not isinstance(model, LaserModel):
                raise TypeError("Model supplied to laser are not LaserMaterial is not LaserModel")
            model.plasma = laser.plasma
            model.laser_profile = laser.laser_profile
            model.laser_spectrum = laser.laser_spectrum

        self._models = models

    cpdef Spectrum emission_function(self, Point3D point, Vector3D direction, Spectrum spectrum,
                                     World world, Ray ray, Primitive primitive,
                                     AffineMatrix3D to_local, AffineMatrix3D to_world):

        cdef:
            Point3D point_plasma, point_laser
            Vector3D direction_plasma, direction_laser
            LaserModel model
        
        # cache the important transforms
        if self._laser_segment_to_laser_node is None or self._laser_to_plasma is None:
            self._cache_transforms()

        point_laser = point.transform(self._laser_segment_to_laser_node)
        direction_laser = direction.transform(self._laser_segment_to_laser_node) # observation vector in the laser frame
        point_plasma = point.transform(self._laser_to_plasma)
        direction_plasma = direction.transform(self._laser_to_plasma)

        for model in self._models:
            spectrum = model.emission(point_plasma, direction_plasma, point_laser, direction_laser, spectrum)

        return spectrum

    cdef void _cache_transforms(self):
        """
        cache transforms from laser primitive to laser and plasma
        """

        # if transforms are cached, the material should be used only for one primitive for safety
        if not len(self.primitives) == 1:
            raise ValueError("LaserMaterial must be attached to exactly one primitive.")

        self._laser_segment_to_laser_node = self._primitive.to(self._laser)
        self._laser_to_plasma = self._primitive.to(self._laser.get_plasma())
