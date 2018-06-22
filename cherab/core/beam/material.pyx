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
from cherab.core.beam.model cimport BeamModel


cdef class BeamMaterial(InhomogeneousVolumeEmitter):

    def __init__(self, Beam beam not None, Plasma plasma not None, AtomicData atomic_data not None,
                 list models not None, VolumeIntegrator integrator not None):

        super().__init__(integrator)

        self._beam = beam
        self._plasma = plasma
        self._atomic_data = atomic_data

        # validate
        for model in models:
            if not isinstance(model, BeamModel):
                raise TypeError('Model supplied to BeamMaterial is not a BeamModel.')

        # configure models
        for model in models:
            model.beam = beam
            model.plasma = plasma
            model.atomic_data = atomic_data

        self._models = models

    cpdef Spectrum emission_function(self, Point3D point, Vector3D direction, Spectrum spectrum,
                                     World world, Ray ray, Primitive primitive,
                                     AffineMatrix3D to_local, AffineMatrix3D to_world):

        cdef:
            BeamModel model
            Point3D plasma_point
            Vector3D beam_direction, observation_direction

        beam_direction = self._beam.direction(point.x, point.y, point.z)

        # transform points and directions
        # todo: cache this transform and rebuild if beam or plasma notifies
        beam_to_plasma = self._beam.to(self._plasma)
        plasma_point = point.transform(beam_to_plasma)
        beam_direction = beam_direction.transform(beam_to_plasma)
        observation_direction = direction.transform(beam_to_plasma)

        # call each model and accumulate spectrum
        for model in self._models:
            spectrum = model.emission(point, plasma_point, beam_direction, observation_direction, spectrum)

        return spectrum


