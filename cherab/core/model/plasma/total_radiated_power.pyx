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

# TODO: requires reimplementation in future release

# from raysect.optical cimport Spectrum, Point3D, Vector3D
# from cherab.core cimport Plasma, AtomicData
# from cherab.core.utility.constants cimport RECIP_4_PI
#
#
# cdef class TotalRadiatedPower(PlasmaModel):
#
#     def __init__(self, Element element, int ionisation, Plasma plasma=None, AtomicData atomic_data=None):
#
#         super().__init__(plasma, atomic_data)
#
#         self._element = element
#         self._ionisation = ionisation
#
#         # ensure that cache is initialised
#         self._change()
#
#     cpdef Spectrum emission(self, Point3D point, Vector3D direction, Spectrum spectrum):
#
#         cdef:
#             int i
#             double ne, ni, te, plt_radiance, prb_radiance
#
#         # cache data on first run
#         if self._target_species is None:
#             self._populate_cache()
#
#         ne = self._plasma.get_electron_distribution().density(point.x, point.y, point.z)
#         if ne <= 0.0:
#             return spectrum
#
#         te = self._plasma.get_electron_distribution().effective_temperature(point.x, point.y, point.z)
#         if te <= 0.0:
#             return spectrum
#
#         ni = self._target_species.distribution.density(point.x, point.y, point.z)
#         if ni <= 0.0:
#             return spectrum
#
#         # add emission to spectrum
#         plt_radiance = RECIP_4_PI * self._plt_rate.evaluate(ne, te) * ne * ni / (spectrum.max_wavelength - spectrum.min_wavelength)
#         prb_radiance = RECIP_4_PI * self._prb_rate.evaluate(ne, te) * ne * ni / (spectrum.max_wavelength - spectrum.min_wavelength)
#         for i in range(spectrum.bins):
#             spectrum.samples_mv[i] += plt_radiance + prb_radiance
#
#         return spectrum
#
#     cdef int _populate_cache(self) except -1:
#
#         # sanity checks
#         if self._plasma is None or self._atomic_data is None:
#             raise RuntimeError("The emission model is not connected to a plasma object.")
#
#         # locate target species
#         try:
#             self._target_species = self._plasma.composition.get(self._element, self._ionisation)
#         except ValueError:
#             raise RuntimeError("The plasma object does not contain the ion species for the specified line "
#                                "(element={}, ionisation={}).".format(self._element.symbol, self._ionisation))
#
#         # obtain rate function
#         self._plt_rate = self._atomic_data.stage_resolved_line_radiation_rate(self._element, self._ionisation)
#         self._prb_rate = self._atomic_data.stage_resolved_continuum_radiation_rate(self._element, self._ionisation)
#
#     def _change(self):
#
#         # clear cache to force regeneration on first use
#         self._target_species = None
#         self._plt_rate = None
#         self._prb_rate = None
