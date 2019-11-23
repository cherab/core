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

from cherab.core.utility.conversion import Cm3ToM3, PerCm3ToPerM3, PhotonToJ

cimport cython
from cherab.core.math cimport Interpolate1DCubic


cdef class BeamCXPEC(CoreBeamCXPEC):
    """
    The effective cx rate interpolation class.

    :param donor_metastable: The metastable state of the donor species for which the rate data applies.
    :param wavelength: The natural wavelength of the emission line associated with the rate data in nm.
    :param data: A dictionary holding the rate data.
    :param extrapolate: Set to True to enable extrapolation, False to disable (default).
    """

    @cython.cdivision(True)
    def __init__(self, int donor_metastable, double wavelength, dict data, bint extrapolate=False):

        self.donor_metastable = donor_metastable
        self.wavelength = wavelength
        self.raw_data = data

        # pre-convert data to W m^3 from Photons s^-1 m^3 prior to interpolation
        eb = data["eb"]                                          # eV/amu
        ti = data["ti"]                                          # eV
        ni = data["ni"]                                          # m^-3
        zeff = data["z"]                                         # dimensionless
        bmag = data["b"]                                         # Tesla

        qref = data["qref"]                                      # m^3/s
        qeb = PhotonToJ.to(data["qeb"], wavelength)              # W.m^3
        qti = data["qti"] / qref                                 # dimensionless
        qni = data["qni"] / qref                                 # dimensionless
        qzeff = data["qz"] / qref                                # dimensionless
        qbmag = data["qb"] / qref                                # dimensionless

        # store limits of data
        self.beam_energy_range = eb.min(), eb.max()
        self.density_range = ni.min(), ni.max()
        self.temperature_range = ti.min(), ti.max()
        self.zeff_range = zeff.min(), zeff.max()
        self.b_field_range = bmag.min(), bmag.max()

        # interpolate the rate data
        self._eb = Interpolate1DCubic(eb, qeb, tolerate_single_value=True, extrapolate=extrapolate, extrapolation_type="quadratic")
        self._ti = Interpolate1DCubic(ti, qti, tolerate_single_value=True, extrapolate=extrapolate, extrapolation_type="quadratic")
        self._ni = Interpolate1DCubic(ni, qni, tolerate_single_value=True, extrapolate=extrapolate, extrapolation_type="quadratic")
        self._zeff = Interpolate1DCubic(zeff, qzeff, tolerate_single_value=True, extrapolate=extrapolate, extrapolation_type="quadratic")
        self._b = Interpolate1DCubic(bmag, qbmag, tolerate_single_value=True, extrapolate=extrapolate, extrapolation_type="quadratic")

    cpdef double evaluate(self, double energy, double temperature, double density, double z_effective, double b_field) except? -1e999:
        """
        Interpolates and returns the effective cx rate for the given plasma parameters.

        If the requested data is out-of-range then the call with throw a ValueError exception.

        :param energy: Interaction energy in eV/amu.
        :param temperature: Receiver ion temperature in eV.
        :param density: Receiver ion density in m^-3
        :param z_effective: Plasma Z-effective.
        :param b_field: Magnetic field magnitude in Tesla.
        :return: The effective cx rate in W.m^3
        """

        cdef double rate

        rate = self._eb.evaluate(energy)
        if rate <= 0:
            return 0.0

        rate *= self._ti.evaluate(temperature)
        if rate <= 0:
            return 0.0

        rate *= self._ni.evaluate(density)
        if rate <= 0:
            return 0.0

        rate *= self._zeff.evaluate(z_effective)
        if rate <= 0:
            return 0.0

        rate *= self._b.evaluate(b_field)
        if rate <= 0:
            return 0.0

        return rate


cdef class NullBeamCXPEC(CoreBeamCXPEC):
    """
    A beam CX rate that always returns zero.
    Needed for use cases where the required atomic data is missing.
    """

    cpdef double evaluate(self, double energy, double temperature, double density, double z_effective, double b_field) except? -1e999:
        return 0.0
