# Copyright 2016-2022 Euratom
# Copyright 2016-2022 United Kingdom Atomic Energy Authority
# Copyright 2016-2022 Centro de Investigaciones Energéticas, Medioambientales y Tecnológicas
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

# cython: language_level=3

import numpy as np
from raysect.optical cimport Spectrum, Point3D, Vector3D
from cherab.core cimport Plasma, AtomicData
from cherab.core.atomic cimport FreeFreeGauntFactor
from cherab.core.species cimport Species
from cherab.core.utility.constants cimport RECIP_4_PI, ELEMENTARY_CHARGE, SPEED_OF_LIGHT, PLANCK_CONSTANT
from libc.math cimport sqrt, log, exp
cimport cython


cdef double PH_TO_J_FACTOR = PLANCK_CONSTANT * SPEED_OF_LIGHT * 1e9

cdef double EXP_FACTOR = PH_TO_J_FACTOR / ELEMENTARY_CHARGE


# todo: doppler shift?
cdef class Bremsstrahlung(PlasmaModel):
    """
    Emitter that calculates bremsstrahlung emission from a plasma object.

    The bremmstrahlung formula implemented is equation 2 from M. Beurskens,
    et. al., 'ITER LIDAR performance analysis', Rev. Sci. Instrum. 79, 10E727 (2008),

    .. math::
        \\epsilon (\\lambda) = \\frac{0.95 \\times 10^{-19}}{\\lambda 4 \\pi} \\sum_{i} \\left(g_{ff}(Z_i, T_e, \\lambda) n_i Z_i^2\\right) n_e T_e^{-1/2} \\times \\exp{\\frac{-hc}{\\lambda T_e}},

    where the emission :math:`\\epsilon (\\lambda)` is in units of radiance (ph/s/sr/m^3/nm).

    :ivar Plasma plasma: The plasma to which this emission model is attached. Default is None.
    :ivar AtomicData atomic_data: The atomic data provider for this model. Default is None.
    :ivar FreeFreeGauntFactor gaunt_factor: Free-free Gaunt factor as a function of Z, Te and
                                            wavelength. If not provided,
                                            the `atomic_data` is used.
    """

    def __init__(self, Plasma plasma=None, AtomicData atomic_data=None, FreeFreeGauntFactor gaunt_factor=None):

        super().__init__(plasma, atomic_data)

        self.gaunt_factor = gaunt_factor

        # ensure that cache is initialised
        self._change()

    @property
    def gaunt_factor(self):

        return self._gaunt_factor

    @gaunt_factor.setter
    def gaunt_factor(self, value):

        self._gaunt_factor = value
        self._user_provided_gaunt_factor = True if value else False

    def __repr__(self):
        return '<PlasmaModel - Bremsstrahlung>'

    @cython.cdivision(True)
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef Spectrum emission(self, Point3D point, Vector3D direction, Spectrum spectrum):

        cdef:
            double ne, te
            double lower_wavelength, upper_wavelength
            double lower_sample, upper_sample
            Species species
            int i

        # cache data on first run
        if self._species_charge is None:
            self._populate_cache()

        ne = self._plasma.get_electron_distribution().density(point.x, point.y, point.z)
        if ne <= 0:
            return spectrum
        te = self._plasma.get_electron_distribution().effective_temperature(point.x, point.y, point.z)
        if te <= 0:
            return spectrum

        # collect densities of charged species
        i = 0
        for species in self._plasma.get_composition():
            if species.charge > 0:
                self._species_density_mv[i] = species.distribution.density(point.x, point.y, point.z)
                i += 1

        # numerically integrate using trapezium rule
        # todo: add sub-sampling to increase numerical accuracy
        lower_wavelength = spectrum.min_wavelength
        lower_sample = self._bremsstrahlung(lower_wavelength, te, ne)
        for i in range(spectrum.bins):

            upper_wavelength = spectrum.min_wavelength + spectrum.delta_wavelength * (i + 1)
            upper_sample = self._bremsstrahlung(upper_wavelength, te, ne)

            spectrum.samples_mv[i] += 0.5 * (lower_sample + upper_sample)

            lower_wavelength = upper_wavelength
            lower_sample = upper_sample

        return spectrum

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double _bremsstrahlung(self, double wvl, double te, double ne):
        """
        :param double wvl: Wavelength in nm.
        :param double te: Electron temperature in eV
        :param double ne: Electron dencity in m^-3
        :return:
        """

        cdef double ni_gff_z2, radiance, pre_factor, ni, z
        cdef int i

        ni_gff_z2 = 0
        for i in range(self._species_charge_mv.shape[0]):
            z = self._species_charge_mv[i]
            ni = self._species_density_mv[i]
            if ni > 0:
                ni_gff_z2 += ni * self._gaunt_factor.evaluate(z, te, wvl) * z * z

        # bremsstrahlung equation W/m^3/str/nm
        pre_factor = 0.95e-19 * RECIP_4_PI * ni_gff_z2 * ne / (sqrt(te) * wvl)
        radiance = pre_factor * exp(- EXP_FACTOR / (te * wvl)) * PH_TO_J_FACTOR

        # convert to W/m^3/str/nm
        return radiance / wvl

    cdef int _populate_cache(self) except -1:

        cdef list species_charge
        cdef Species species

        if self._plasma is None:
            raise RuntimeError("The emission model is not connected to a plasma object.")

        if self._gaunt_factor is None:
            if self._atomic_data is None:
                raise RuntimeError("The emission model is not connected to an atomic data source.")

            # initialise Gaunt factor on first run using the atomic data
            self._gaunt_factor = self._atomic_data.free_free_gaunt_factor()

        species_charge = []
        for species in self._plasma.get_composition():
            if species.charge > 0:
                species_charge.append(species.charge)

        # Gaunt factor takes Z as double to support Zeff, so caching Z as float64
        self._species_charge = np.array(species_charge, dtype=np.float64)
        self._species_charge_mv = self._species_charge

        self._species_density = np.zeros_like(self._species_charge)
        self._species_density_mv = self._species_density

    def _change(self):

        # clear cache to force regeneration on first use
        if not self._user_provided_gaunt_factor:
            self._gaunt_factor = None
        self._species_charge = None
        self._species_charge_mv = None
        self._species_density = None
        self._species_density_mv = None
