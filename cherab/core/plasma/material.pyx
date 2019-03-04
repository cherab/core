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

from raysect.optical cimport World, Primitive, Ray, Spectrum, SpectralFunction, Point3D, Vector3D, AffineMatrix3D
from raysect.optical.material.emitter cimport InhomogeneousVolumeEmitter
from raysect.optical.material.emitter.inhomogeneous cimport VolumeIntegrator
from cherab.core.plasma.model cimport PlasmaModel


cdef class PlasmaMaterial(InhomogeneousVolumeEmitter):
    """Raysect Material that handles the integration of the plasma model emission."""

    def __init__(self, Plasma plasma not None, AtomicData atomic_data not None, list models not None, VolumeIntegrator integrator not None, AffineMatrix3D local_to_plasma):

        super().__init__(integrator)

        self._plasma = plasma
        self._atomic_data = atomic_data
        self._local_to_plasma = local_to_plasma

        # validate
        for model in models:
            if not isinstance(model, PlasmaModel):
                raise TypeError('Model supplied to PlasmaMaterial is not a PlasmaModel.')

        # configure models
        for model in models:
            model.plasma = plasma
            model.atomic_data = atomic_data

        self._models = models

    cpdef Spectrum emission_function(self, Point3D point, Vector3D direction, Spectrum spectrum,
                                     World world, Ray ray, Primitive primitive,
                                     AffineMatrix3D to_local, AffineMatrix3D to_world):

        cdef PlasmaModel model

        # perform coordinate transform to plasma space if required
        if self._local_to_plasma:
            point = point.transform(self._local_to_plasma)
            direction = direction.transform(self._local_to_plasma)

        # call each model and accumulate spectrum
        for model in self._models:
            spectrum = model.emission(point, direction, spectrum)

        return spectrum


