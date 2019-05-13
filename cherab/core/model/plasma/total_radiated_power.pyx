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


from raysect.optical cimport Spectrum, Point3D, Vector3D
from cherab.core cimport Plasma, AtomicData
from cherab.core.utility.constants cimport RECIP_4_PI


cdef class TotalRadiatedPower(PlasmaModel):

    def __init__(self, Element element, int charge, Plasma plasma=None, AtomicData atomic_data=None):

        if not 0 <= charge < element.atomic_number:
            raise ValueError('TotalRadiatedPower cannot be calculated for charge state (element={}, ionisation={}).'
                             ''.format(element.symbol, charge))

        super().__init__(plasma, atomic_data)

        self._element = element
        self._charge = charge

        # ensure that cache is initialised
        self._change()

    cpdef Spectrum emission(self, Point3D point, Vector3D direction, Spectrum spectrum):

        cdef:
            int i
            double ne, ni, ni_upper, te, plt_radiance, prb_radiance

        # cache data on first run
        if not self._cache_loaded:
            self._populate_cache()

        ne = self._plasma.get_electron_distribution().density(point.x, point.y, point.z)
        if ne <= 0.0:
            return spectrum

        te = self._plasma.get_electron_distribution().effective_temperature(point.x, point.y, point.z)
        if te <= 0.0:
            return spectrum

        ni = self._line_rad_species.distribution.density(point.x, point.y, point.z)

        ni_upper = self._recom_species.distribution.density(point.x, point.y, point.z)

        # add emission to spectrum
        if self._plt_rate and ni > 0:
            plt_radiance = RECIP_4_PI * self._plt_rate.evaluate(ne, te) * ne * ni / (spectrum.max_wavelength - spectrum.min_wavelength)
        else:
            plt_radiance = 0
        if self._prb_rate and ni_upper > 0:
            prb_radiance = RECIP_4_PI * self._prb_rate.evaluate(ne, te) * ne * ni_upper / (spectrum.max_wavelength - spectrum.min_wavelength)
        else:
            prb_radiance = 0
        for i in range(spectrum.bins):
            spectrum.samples_mv[i] += plt_radiance + prb_radiance

        return spectrum

    cdef int _populate_cache(self) except -1:

        # sanity checks
        if self._plasma is None or self._atomic_data is None:
            raise RuntimeError("The emission model is not connected to a plasma object.")

        # cache line radiation species and rate
        self._plt_rate = self._atomic_data.line_radiated_power_rate(self._element, self._charge)
        try:
            self._line_rad_species = self._plasma.composition.get(self._element, self._charge)
        except ValueError:
            raise RuntimeError("The plasma object does not contain the required ion species for calculating"
                               "total line radiaton, (element={}, ionisation={})."
                               "".format(self._element.symbol, self._charge))

        # cache recombination species and radiation rate
        self._prb_rate = self._atomic_data.continuum_radiated_power_rate(self._element, self._charge+1)
        try:
            self._recom_species = self._plasma.composition.get(self._element, self._charge+1)
        except ValueError:
            raise RuntimeError("The plasma object does not contain the required ion species for calculating"
                               "recombination/continuum emission, (element={}, ionisation={})."
                               "".format(self._element.symbol, self._charge+1))

        self._cache_loaded = True

    def _change(self):

        # clear cache to force regeneration on first use
        self._cache_loaded = False
        self._line_rad_species = None
        self._recom_species = None
        self._plt_rate = None
        self._prb_rate = None
