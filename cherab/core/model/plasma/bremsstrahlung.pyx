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
from cherab.core.math.integrators cimport GaussianQuadrature
from cherab.core.species cimport Species
from cherab.core.utility.constants cimport RECIP_4_PI, ELEMENTARY_CHARGE, SPEED_OF_LIGHT, PLANCK_CONSTANT, ELECTRON_REST_MASS, VACUUM_PERMITTIVITY
from libc.math cimport sqrt, log, exp, M_PI
cimport cython


cdef double EXP_FACTOR = PLANCK_CONSTANT * SPEED_OF_LIGHT * 1e9 / ELEMENTARY_CHARGE

cdef double BREMS_CONST = (ELEMENTARY_CHARGE**2 * RECIP_4_PI / VACUUM_PERMITTIVITY)**3
BREMS_CONST *= 32 * M_PI**2 / (3 * sqrt(3) * ELECTRON_REST_MASS**2 * SPEED_OF_LIGHT**3)
BREMS_CONST *= sqrt(2 * ELECTRON_REST_MASS / (M_PI * ELEMENTARY_CHARGE))
BREMS_CONST *= SPEED_OF_LIGHT * 1e9 * RECIP_4_PI


cdef class BremsFunction(Function1D):
    """
    Calculates bremsstrahlung spectrum.

    :param FreeFreeGauntFactor gaunt_factor: Free-free Gaunt factor as a function of Z, Te and
                                             wavelength.
    :param object species_density: Array-like object wiyh ions' density in m-3.
    :param object species_charge: Array-like object wiyh ions' charge.
    :param double ne: Electron density in m-3.
    :param double te: Electron temperature in eV.
    """

    def __init__(self, FreeFreeGauntFactor gaunt_factor, object species_density, object species_charge, double ne, double te):

        if ne <= 0:
            raise ValueError("Argument ne must be positive.")
        self.ne = ne

        if te <= 0:
            raise ValueError("Argument te must be positive.")
        self.te = te

        self.gaunt_factor = gaunt_factor
        self.species_density = np.asarray(species_density, dtype=np.float64)  # copied if type does not match
        self.species_density_mv = self.species_density
        self.species_charge = np.asarray(species_charge, dtype=np.float64)  # copied if type does not match
        self.species_charge_mv = self.species_charge

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double evaluate(self, double wvl) except? -1e999:
        """
        :param double wvl: Wavelength in nm.
        :return:
        """

        cdef double ni_gff_z2, radiance, pre_factor, ni, z
        cdef int i

        ni_gff_z2 = 0
        for i in range(self.species_charge_mv.shape[0]):
            z = self.species_charge_mv[i]
            ni = self.species_density_mv[i]
            if ni > 0:
                ni_gff_z2 += ni * self.gaunt_factor.evaluate(z, self.te, wvl) * z * z

        # bremsstrahlung equation W/m^3/str/nm
        pre_factor = BREMS_CONST / (sqrt(self.te) * wvl * wvl) * self.ne * ni_gff_z2
        radiance = pre_factor * exp(- EXP_FACTOR / (self.te * wvl))

        return radiance


# todo: doppler shift?
cdef class Bremsstrahlung(PlasmaModel):
    """
    Emitter that calculates bremsstrahlung emission from a plasma object.

    The bremmstrahlung formula implemented is equation 5.3.40
    from I. H. Hutchinson, 'Principles of Plasma Diagnostics', second edition,
    Cambridge University Press, 2002, ISBN: 9780511613630,
    https://doi.org/10.1017/CBO9780511613630

    Note that in eq. 5.3.40, the emissivity :math:`j(\\nu)` is given in (W/m^3/sr/Hz) with respect
    to frequency, :math:`\\nu`. Here, the emissivity :math:`\\epsilon_{\\mathrm{ff}}(\\lambda)`
    is given in (W/m^3/nm/sr) with respect to wavelength, :math:`\\lambda = \\frac{10^{9} c}{\\nu}`,
    and taking into account that :math:`d\\nu=-\\frac{10^{9} c}{\\lambda^2}d\\lambda`.

    .. math::
        \\epsilon_{\\mathrm{ff}}(\\lambda) = \\left( \\frac{e^2}{4 \\pi \\varepsilon_0} \\right)^3
        \\frac{32 \\pi^2}{3 \\sqrt{3} m_\\mathrm{e}^2 c^3}
        \\sqrt{\\frac{2 m_\\mathrm{e}^3}{\\pi e T_\\mathrm{e}}}
        \\frac{10^{9} c}{4 \\pi \\lambda^2}
        n_\\mathrm{e} \\sum_i \\left( n_\\mathrm{i} g_\\mathrm{ff} (Z_\\mathrm{i}, T_\\mathrm{e}, \\lambda) Z_\\mathrm{i}^2 \\right)
        \\mathrm{e}^{-\\frac{10^9 hc}{e T_\\mathrm{e} \\lambda}}\\,,

    where :math:`T_\\mathrm{e}` is in eV and :math:`\\lambda` is in nm.

    :math:`g_\\mathrm{ff} (Z_\\mathrm{i}, T_\\mathrm{e}, \\lambda)` is the free-free Gaunt factor.

    :ivar Plasma plasma: The plasma to which this emission model is attached. Default is None.
    :ivar AtomicData atomic_data: The atomic data provider for this model. Default is None.
    :ivar FreeFreeGauntFactor gaunt_factor: Free-free Gaunt factor as a function of Z, Te and
                                            wavelength. If not provided,
                                            the `atomic_data` is used.
    :ivar Integrator1D integrator: Integrator1D instance to integrate Bremsstrahlung radiation
                                   over the spectral bin. Default is `GaussianQuadrature`.
    """

    def __init__(self, Plasma plasma=None, AtomicData atomic_data=None, FreeFreeGauntFactor gaunt_factor=None, Integrator1D integrator=None):

        super().__init__(plasma, atomic_data)

        self._brems_func = BremsFunction.__new__(BremsFunction)
        self.gaunt_factor = gaunt_factor
        self.integrator = integrator or GaussianQuadrature()

        # ensure that cache is initialised
        self._change()

    @property
    def gaunt_factor(self):

        return self._brems_func.gaunt_factor

    @gaunt_factor.setter
    def gaunt_factor(self, value):

        self._brems_func.gaunt_factor = value
        self._user_provided_gaunt_factor = True if value else False

    @property
    def integrator(self):

        return self._integrator

    @integrator.setter
    def integrator(self, Integrator1D value not None):

        self._integrator = value
        self._integrator.function = self._brems_func

    def __repr__(self):
        return '<PlasmaModel - Bremsstrahlung>'

    @cython.cdivision(True)
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef Spectrum emission(self, Point3D point, Vector3D direction, Spectrum spectrum):

        cdef:
            double ne, te
            double lower_wavelength, upper_wavelength, bin_integral
            Species species
            int i

        # cache data on first run
        if self._brems_func.species_charge is None:
            self._populate_cache()

        ne = self._plasma.get_electron_distribution().density(point.x, point.y, point.z)
        if ne <= 0:
            return spectrum
        te = self._plasma.get_electron_distribution().effective_temperature(point.x, point.y, point.z)
        if te <= 0:
            return spectrum

        self._brems_func.ne = ne
        self._brems_func.te = te

        # collect densities of charged species
        i = 0
        for species in self._plasma.get_composition():
            if species.charge > 0:
                self._brems_func.species_density_mv[i] = species.distribution.density(point.x, point.y, point.z)
                i += 1

        # add bremsstrahlung to spectrum
        lower_wavelength = spectrum.min_wavelength
        for i in range(spectrum.bins):
            upper_wavelength = spectrum.min_wavelength + spectrum.delta_wavelength * (i + 1)

            bin_integral = self._integrator.evaluate(lower_wavelength, upper_wavelength)
            spectrum.samples_mv[i] += bin_integral / spectrum.delta_wavelength

            lower_wavelength = upper_wavelength

        return spectrum

    cdef int _populate_cache(self) except -1:

        cdef list species_charge
        cdef Species species

        if self._plasma is None:
            raise RuntimeError("The emission model is not connected to a plasma object.")

        if self._brems_func.gaunt_factor is None:
            if self._atomic_data is None:
                raise RuntimeError("The emission model is not connected to an atomic data source.")

            # initialise Gaunt factor on first run using the atomic data
            self._brems_func.gaunt_factor = self._atomic_data.free_free_gaunt_factor()

        species_charge = []
        for species in self._plasma.get_composition():
            if species.charge > 0:
                species_charge.append(species.charge)

        # Gaunt factor takes Z as double to support Zeff, so caching Z as float64
        self._brems_func.species_charge = np.array(species_charge, dtype=np.float64)
        self._brems_func.species_charge_mv = self._brems_func.species_charge

        self._brems_func.species_density = np.zeros_like(self._brems_func.species_charge)
        self._brems_func.species_density_mv = self._brems_func.species_density

    def _change(self):

        # clear cache to force regeneration on first use
        if not self._user_provided_gaunt_factor:
            self._brems_func.gaunt_factor = None
        self._brems_func.species_charge = None
        self._brems_func.species_charge_mv = None
        self._brems_func.species_density = None
        self._brems_func.species_density_mv = None
