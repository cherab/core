# Copyright 2016-2021 Euratom
# Copyright 2016-2021 United Kingdom Atomic Energy Authority
# Copyright 2016-2021 Centro de Investigaciones Energéticas, Medioambientales y Tecnológicas
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

import numpy as np
from cherab.core.utility.conversion import Cm3ToM3, PerCm3ToPerM3, PhotonToJ

cimport cython
from libc.math cimport INFINITY, log10
from raysect.core.math.function.float cimport Interpolator1DArray, Constant1D


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
        qeb = np.log10(PhotonToJ.to(data["qeb"], wavelength))    # W.m^3
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
        extrapolation_type_log = 'quadratic' if extrapolate else 'none'
        extrapolation_type = 'nearest' if extrapolate else 'none'
        self._eb = Interpolator1DArray(np.log10(eb), qeb, 'cubic', extrapolation_type_log, INFINITY) if len(qeb) > 1 else Constant1D(qeb[0])
        self._ti = Interpolator1DArray(ti, qti, 'cubic', extrapolation_type, INFINITY) if len(qti) > 1 else Constant1D(qti[0])
        self._ni = Interpolator1DArray(ni, qni, 'cubic', extrapolation_type, INFINITY) if len(qni) > 1 else Constant1D(qni[0])
        self._zeff = Interpolator1DArray(zeff, qzeff, 'cubic', extrapolation_type, INFINITY) if len(qzeff) > 1 else Constant1D(qzeff[0])
        self._b = Interpolator1DArray(bmag, qbmag, 'cubic', extrapolation_type, INFINITY) if len(qbmag) > 1 else Constant1D(qbmag[0])

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

        # need to handle zeros for log-log interpolation
        if energy < 1.e-300:
            energy = 1.e-300

        rate = 10 ** self._eb.evaluate(log10(energy))

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
