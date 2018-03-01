
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

from raysect.optical.material.emitter.inhomogeneous import NumericalIntegrator
from raysect.optical cimport World, Primitive, Ray, Spectrum, SpectralFunction, Point3D, Vector3D, AffineMatrix3D
from raysect.optical.material.emitter cimport InhomogeneousVolumeEmitter

from cherab.core.math.function cimport Function3D


cdef class SimplePowerEmitter(InhomogeneousVolumeEmitter):

    cdef public Function3D emitter

    def __init__(self, emission_function, step=0.01):

        super().__init__(NumericalIntegrator(step))

        self.emitter = emission_function

    cpdef Spectrum emission_function(self, Point3D point, Vector3D direction, Spectrum spectrum,
                                     World world, Ray ray, Primitive primitive,
                                     AffineMatrix3D to_local, AffineMatrix3D to_world):

        cdef:
            int i
            double x, y, z, radiance, wavelength_range

        x, y, z = point.transform(to_world)

        wavelength_range = spectrum.max_wavelength - spectrum.min_wavelength
        radiance = self.emitter.evaluate(x, y, z) / wavelength_range

        for i in range(spectrum.bins):
            spectrum.samples_mv[i] += radiance

        return spectrum

