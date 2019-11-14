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

from numpy cimport ndarray
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

        # evaluate lighting with Cook-Torrance bsdf (optimised)
        wavelength = 0.5 * (spectrum.min_wavelength + spectrum.max_wavelength)
        n = self.index.evaluate(wavelength)
        k = self.extinction.evaluate(wavelength)
        ci = s_half.dot(s_outgoing)
        spectrum.mul_scalar(self._d(s_half) * self._g(s_incoming, s_outgoing) * self._fresnel_conductor(ci, n, k) / (4 * s_incoming.z))
        return spectrum

    @cython.cdivision(True)
    cdef double _d(self, Vector3D s_half):

        cdef double r2, h2, k

        # ggx distribution
        r2 = self._roughness * self._roughness
        h2 = s_half.z * s_half.z
        k = h2 * (r2 - 1) + 1
        return r2 / (M_PI * k * k)

    cdef double _g(self, Vector3D s_incoming, Vector3D s_outgoing):
        # Smith's geometric shadowing model
        return self._g1(s_incoming) * self._g1(s_outgoing)

    @cython.cdivision(True)
    cdef double _g1(self, Vector3D v):
        # Smith's geometric component (G1) for GGX distribution
        cdef double r2 = self._roughness * self._roughness
        return 2 * v.z / (v.z + sqrt(r2 + (1 - r2) * (v.z * v.z)))

    @cython.cdivision(True)
    cdef double _fresnel_conductor(self, double ci, double n, double k) nogil:

        cdef double c12, k0, k1, k2, k3

        ci2 = ci * ci
        k0 = n * n + k * k
        k1 = k0 * ci2 + 1
        k2 = 2 * n * ci
        k3 = k0 + ci2
        return 0.5 * ((k1 - k2) / (k1 + k2) + (k3 - k2) / (k3 + k2))
