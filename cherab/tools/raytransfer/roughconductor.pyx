#
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
#

from raysect.optical cimport Point3D, Vector3D, AffineMatrix3D, World, Ray, Spectrum, new_vector3d
from libc.math cimport M_PI, sqrt
from raysect.optical.material cimport RoughConductor
cimport cython


cdef class RToptimisedRoughConductor(RoughConductor):
    """
    A `RoughConductor` optimised for calculation of ray transfer matrix (geometry matrix).
    The spectral array in this case contains ~ 10^5 - 10^6 spectral bins but the wavelengths for all of them are equal.
    The Fresnel indeces are equal for all spectral bins, so the unnecessary calculations are avoided.
    """

    @cython.cdivision(True)
    cpdef Spectrum evaluate_shading(self, World world, Ray ray, Vector3D s_incoming, Vector3D s_outgoing,
                                    Point3D w_reflection_origin, Point3D w_transmission_origin, bint back_face,
                                    AffineMatrix3D world_to_surface, AffineMatrix3D surface_to_world):

        cdef:
            double n, k
            double ci
            double wavelength
            Vector3D s_half
            Spectrum spectrum
            Ray reflected

        # outgoing ray is sampling incident light so s_outgoing = incident

        # material does not transmit
        if s_outgoing.z <= 0:
            return ray.new_spectrum()

        # ignore parallel rays which could cause a divide by zero later
        if s_incoming.z == 0:
            return ray.new_spectrum()

        # calculate half vector
        s_half = new_vector3d(
            s_incoming.x + s_outgoing.x,
            s_incoming.y + s_outgoing.y,
            s_incoming.z + s_outgoing.z
        ).normalise()

        # generate and trace ray
        reflected = ray.spawn_daughter(w_reflection_origin, s_outgoing.transform(surface_to_world))
        spectrum = reflected.trace(world)

        # checking if the wavelength range is narrow enough to consider Fresnel indices as constants
        if spectrum.max_wavelength - spectrum.min_wavelength > 5.:  # 5nm is narrow enough
            raise ValueError("RToptimisedRoughConductor can be used only if wavelength range does not exceed 5 nm.")

        # evaluate lighting with Cook-Torrance bsdf (optimised)
        wavelength = 0.5 * (spectrum.min_wavelength + spectrum.max_wavelength)
        n = self.index.evaluate(wavelength)
        k = self.extinction.evaluate(wavelength)
        ci = s_half.dot(s_outgoing)
        spectrum.mul_scalar(self._d(s_half) * self._g(s_incoming, s_outgoing) * self._fresnel_conductor(ci, n, k) / (4 * s_incoming.z))
        return spectrum
