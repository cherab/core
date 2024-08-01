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
from cherab.core.atomic.elements import hydrogen, deuterium, tritium


cdef class TotalRadiatedPower(PlasmaModel):
    r"""
    Emitter that calculates total power radiated by a given ion, which includes:

    - line power due to electron impact excitation,
    - continuum and line power due to recombination and Bremsstrahlung,
    - line power due to charge exchange with thermal neutral hydrogen and its isotopes.

    The emission calculated by this model is spectrally unresolved,
    which means that the total radiated power will be spread of the entire
    observable spectral range.

    .. math::
        \epsilon_{\mathrm{total}} = \frac{1}{4 \pi \Delta\lambda} \left(
        n_{Z_\mathrm{i}} n_\mathrm{e} C_{\mathrm{excit}}(n_\mathrm{e}, T_\mathrm{e}) + 
        n_{Z_\mathrm{i} + 1} n_\mathrm{e} C_{\mathrm{recomb}}(n_\mathrm{e}, T_\mathrm{e}) +
        n_{Z_\mathrm{i} + 1} n_\mathrm{hyd} C_{\mathrm{cx}}(n_\mathrm{e}, T_\mathrm{e}) \right)

    where :math:`n_{Z_\mathrm{i}}` is the target species density;
    :math:`n_{Z_\mathrm{i} + 1}` is the recombining species density;
    :math:`n_{\mathrm{hyd}}` is the total density of all hydrogen isotopes;
    :math:`C_{\mathrm{excit}}, C_{\mathrm{recomb}}, C_{\mathrm{cx}}` are the radiated power
    coefficients in :math:`W m^3` due to electron impact excitation, recombination
    + Bremsstrahlung and charge exchange with thermal neutral hydrogen, respectively;
    :math:`\Delta\lambda` is the observable spectral range.

    :param Element element: The atomic element/isotope.
    :param int charge: The charge state of the element/isotope.
    :param Plasma plasma: The plasma to which this emission model is attached. Default is None.
    :param AtomicData atomic_data: The atomic data provider for this model. Default is None.
    """

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
            double ne, ni, ni_upper, nhyd, te
            double power_density, radiance
            Species hyd_species

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

        nhyd = 0
        for hyd_species in self._hydrogen_species:
            nhyd += hyd_species.distribution.density(point.x, point.y, point.z)

        # add emission to spectrum
        power_density = 0

        if self._plt_rate and ni > 0:  # excitation
            power_density += self._plt_rate.evaluate(ne, te) * ne * ni

        if self._prb_rate and ni_upper > 0:  # recombination + bremsstrahlung
            power_density += self._prb_rate.evaluate(ne, te) * ne * ni_upper

        if self._prc_rate and ni_upper > 0 and nhyd > 0:  # charge exchange
            power_density += self._prc_rate.evaluate(ne, te) * nhyd * ni_upper

        radiance = RECIP_4_PI * power_density / (spectrum.max_wavelength - spectrum.min_wavelength)

        for i in range(spectrum.bins):
            spectrum.samples_mv[i] += radiance

        return spectrum

    cdef int _populate_cache(self) except -1:

        cdef:
            Species hyd_species
            Element hyd_isotope

        # sanity checks
        if self._plasma is None:
            raise RuntimeError("The emission model is not connected to a plasma object.")
        if self._atomic_data is None:
            raise RuntimeError("The emission model is not connected to an atomic data source.")

        # cache line radiation species and rate
        self._plt_rate = self._atomic_data.line_radiated_power_rate(self._element, self._charge)
        try:
            self._line_rad_species = self._plasma.get_composition().get(self._element, self._charge)
        except ValueError:
            raise RuntimeError("The plasma object does not contain the required ion species for calculating"
                               "total line radiaton, (element={}, ionisation={})."
                               "".format(self._element.symbol, self._charge))

        # cache recombination species and radiation rate
        self._prb_rate = self._atomic_data.continuum_radiated_power_rate(self._element, self._charge+1)
        try:
            self._recom_species = self._plasma.get_composition().get(self._element, self._charge+1)
        except ValueError:
            raise RuntimeError("The plasma object does not contain the required ion species for calculating"
                               "recombination/continuum emission, (element={}, ionisation={})."
                               "".format(self._element.symbol, self._charge+1))

        # cache hydrogen species and CX radiation rate
        self._prc_rate = self._atomic_data.cx_radiated_power_rate(self._element, self._charge+1)

        self._hydrogen_species = []
        for hyd_isotope in (hydrogen, deuterium, tritium):
            try:
                hyd_species = self._plasma.get_composition().get(hyd_isotope, 0)
            except ValueError:
                pass
            else:
                self._hydrogen_species.append(hyd_species)

        self._cache_loaded = True

    def _change(self):

        # clear cache to force regeneration on first use
        self._cache_loaded = False
        self._line_rad_species = None
        self._recom_species = None
        self._hydrogen_species = None
        self._plt_rate = None
        self._prb_rate = None
        self._prc_rate = None
