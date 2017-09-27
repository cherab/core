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

from raysect.optical cimport Spectrum, Point3D, Vector3D
from cherab.core cimport Plasma, AtomicData
from cherab.core.model.spectra cimport doppler_shift, thermal_broadening, add_gaussian_line
from libc.math cimport sqrt, log, exp
cimport cython


DEF RECIP_4_PI = 0.07957747154594767

# todo: replace with imports from cherab.utility?
DEF ELEMENTARY_CHARGE = 1.6021766208e-19
DEF SPEED_OF_LIGHT = 299792458.0
DEF PLANCK_CONSTANT = 6.6260700400e-34
DEF PH_TO_J_FACTOR = PLANCK_CONSTANT * SPEED_OF_LIGHT * 1e9


# todo: doppler shift?
cdef class Bremsstrahlung(PlasmaModel):
    """
    Emitter that calculates bremsstrahlung emission from a plasma object.

    The bremmstrahlung formula implemented is equation 2 from M. Beurskens,
    et. al., 'ITER LIDAR performance analysis', Rev. Sci. Instrum. 79, 10E727 (2008),

    .. math::
        \\epsilon (\\lambda) = \\frac{0.95 \\times 10^{-19}}{\\lambda 4 \\pi} g_{ff} n_e^2 Z_{eff} T_e^{1/2} \\times \\exp{\\frac{-hc}{\\lambda T_e}},

    where the emission :math:`\\epsilon (\\lambda)` is in units of radiance (ph/s/sr/m^3/nm).
    """

    def __repr__(self):
        return '<PlasmaModel - Bremsstrahlung>'

    cpdef Spectrum emission(self, Point3D point, Vector3D direction, Spectrum spectrum):

        cdef:
            double ne, te, z_effective
            double lower_wavelength, upper_wavelength
            double lower_sample, upper_sample
            int i

        # TODO - adding this to resolve equilibrium interpolation issues.
        try:
            ne = self._plasma.electron_distribution.density(point.x, point.y, point.z)
            te = self._plasma.electron_distribution.effective_temperature(point.x, point.y, point.z)
            z_effective = self._plasma.z_effective(point.x, point.y, point.z)
        except ValueError:
            return spectrum

        # numerically integrate using trapezium rule
        # todo: add sub-sampling to increase numerical accuracy
        lower_wavelength = spectrum.min_wavelength
        lower_sample = self._bremsstrahlung(lower_wavelength, te, ne, z_effective)
        for i in range(spectrum.bins):

            upper_wavelength = spectrum.min_wavelength + spectrum.delta_wavelength * i
            upper_sample = self._bremsstrahlung(upper_wavelength, te, ne, z_effective)

            spectrum.samples_mv[i] += 0.5 * (lower_sample + upper_sample)

            lower_wavelength = upper_wavelength
            lower_sample = upper_sample

        return spectrum

    @cython.cdivision(True)
    cdef inline double _bremsstrahlung(self, double wvl, double te, double ne, double zeff):
        """
        :param wvl: in nm 
        :param te: in eV
        :param ne: in m^-3
        :param zeff: a.u.
        :return: 
        """

        cdef double gaunt_factor, radiance, prefactor, exp_factor, ph_to_j

        # gaunt factor
        gaunt_factor = 0.6183 * log(te) - 0.0821

        # bremsstrahlung equation W/m^3/str/nm
        pre_factor = 0.95e-19 * RECIP_4_PI * gaunt_factor * ne * ne * zeff / (sqrt(te) * wvl)
        exp_factor = - PLANCK_CONSTANT * SPEED_OF_LIGHT / te
        radiance =  pre_factor * exp(exp_factor / wvl) * PH_TO_J_FACTOR

        # convert to W/m^3/str/nm
        return radiance / wvl
