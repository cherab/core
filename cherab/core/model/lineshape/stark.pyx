# cython: language_level=3

# Copyright 2016-2023 Euratom
# Copyright 2016-2023 United Kingdom Atomic Energy Authority
# Copyright 2016-2023 Centro de Investigaciones Energéticas, Medioambientales y Tecnológicas
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

from scipy.special import hyp2f1

from libc.math cimport sqrt, floor, ceil, fabs, log, exp
from raysect.core.math.function.float cimport Function1D
from raysect.optical cimport Point3D, Vector3D

from cherab.core.atomic cimport Line
from cherab.core.species cimport Species
from cherab.core.plasma cimport Plasma
from cherab.core.atomic.elements import hydrogen, deuterium, tritium
from cherab.core.math.function cimport autowrap_function1d, autowrap_function2d
from cherab.core.math.integrators cimport GaussianQuadrature
from cherab.core.utility.constants cimport BOHR_MAGNETON, HC_EV_NM
from cherab.core.model.lineshape.doppler cimport doppler_shift, thermal_broadening
from cherab.core.model.lineshape.gaussian cimport add_gaussian_line


cimport cython


DEF PI_POLARISATION = 0
DEF SIGMA_POLARISATION = 1
DEF SIGMA_PLUS_POLARISATION = 1
DEF SIGMA_MINUS_POLARISATION = -1
DEF NO_POLARISATION = 2

DEF LORENTZIAN_CUTOFF_GAMMA = 50.0

cdef double _SIGMA2FWHM = 2 * sqrt(2 * log(2))


cdef class StarkFunction(Function1D):
    """
    Normalised Stark function for the StarkBroadenedLine line shape.
    """

    cdef double _a, _x0, _norm

    STARK_NORM_COEFFICIENT = 4 * LORENTZIAN_CUTOFF_GAMMA * hyp2f1(0.4, 1, 1.4, -(2 * LORENTZIAN_CUTOFF_GAMMA)**2.5)

    def __init__(self, double wavelength, double lambda_1_2):

        if wavelength <= 0:
            raise ValueError("Argument 'wavelength' must be positive.")

        if lambda_1_2 <= 0:
            raise ValueError("Argument 'lambda_1_2' must be positive.")

        self._x0 = wavelength
        self._a = (0.5 * lambda_1_2)**2.5
        # normalise, so the integral over x is equal to 1 in the limits
        # (_x0 - LORENTZIAN_CUTOFF_GAMMA * lambda_1_2, _x0 + LORENTZIAN_CUTOFF_GAMMA * lambda_1_2)
        self._norm = (0.5 * lambda_1_2)**1.5 / <double> self.STARK_NORM_COEFFICIENT

    @cython.cdivision(True)
    cdef double evaluate(self, double x) except? -1e999:

        return self._norm / ((fabs(x - self._x0))**2.5 + self._a)


@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef Spectrum add_lorentzian_line(double radiance, double wavelength, double lambda_1_2, Spectrum spectrum, Integrator1D integrator):
    r"""
    Adds a modified Lorentzian line to the given spectrum and returns the new spectrum.

    :param float radiance: Intensity of the line in radiance.
    :param float wavelength: central wavelength of the line in nm.
    :param float sigma: width of the line in nm.
    :param Spectrum spectrum: the current spectrum to which the gaussian line is added.
    :return:
    """

    cdef double cutoff_lower_wavelength, cutoff_upper_wavelength
    cdef double lower_wavelength, upper_wavelength
    cdef double bin_integral
    cdef int start, end, i

    if lambda_1_2 <= 0:
        return spectrum

    integrator.function = StarkFunction(wavelength, lambda_1_2)

    # calculate and check end of limits
    cutoff_lower_wavelength = wavelength - LORENTZIAN_CUTOFF_GAMMA * lambda_1_2
    if spectrum.max_wavelength < cutoff_lower_wavelength:
        return spectrum

    cutoff_upper_wavelength = wavelength + LORENTZIAN_CUTOFF_GAMMA * lambda_1_2
    if spectrum.min_wavelength > cutoff_upper_wavelength:
        return spectrum

    # locate range of bins where there is significant contribution from the gaussian (plus a health margin)
    start = max(0, <int> floor((cutoff_lower_wavelength - spectrum.min_wavelength) / spectrum.delta_wavelength))
    end = min(spectrum.bins, <int> ceil((cutoff_upper_wavelength - spectrum.min_wavelength) / spectrum.delta_wavelength))

    # add line to spectrum
    lower_wavelength = spectrum.min_wavelength + start * spectrum.delta_wavelength

    for i in range(start, end):
        upper_wavelength = spectrum.min_wavelength + spectrum.delta_wavelength * (i + 1)

        bin_integral = integrator.evaluate(lower_wavelength, upper_wavelength)
        spectrum.samples_mv[i] += radiance * bin_integral / spectrum.delta_wavelength

        lower_wavelength = upper_wavelength

    return spectrum


cdef class StarkBroadenedLine(ZeemanLineShapeModel):
    """
    Parametrised Stark broadened line shape based on the Model Microfield Method (MMM).
    Contains embedded atomic data in the form of fits to MMM.
    Only Balmer and Paschen series are supported by default.
    See B. Lomanowski, et al. "Inferring divertor plasma properties from hydrogen Balmer
    and Paschen series spectroscopy in JET-ILW." Nuclear Fusion 55.12 (2015)
    `123028 <https://doi.org/10.1088/0029-5515/55/12/123028>`_.

    Call `show_supported_transitions()` to see the list of supported transitions and
    default model coefficients.

    :param Line line: The emission line object for this line shape.
    :param float wavelength: The rest wavelength for this emission line.
    :param Species target_species: The target plasma species that is emitting.
    :param Plasma plasma: The emitting plasma object.
    :param dict stark_model_coefficients: Alternative model coefficients in the form
                                          {line_ij: (c_ij, a_ij, b_ij), ...}.
                                          If None, the default model parameters will be used.
    :param Integrator1D integrator: Integrator1D instance to integrate the line shape
        over the spectral bin. Default is `GaussianQuadrature()`.

    """

    STARK_MODEL_COEFFICIENTS_DEFAULT = {
        Line(hydrogen, 0, (3, 2)): (3.71e-18, 0.7665, 0.064),
        Line(hydrogen, 0, (4, 2)): (8.425e-18, 0.7803, 0.050),
        Line(hydrogen, 0, (5, 2)): (1.31e-15, 0.6796, 0.030),
        Line(hydrogen, 0, (6, 2)): (3.954e-16, 0.7149, 0.028),
        Line(hydrogen, 0, (7, 2)): (6.258e-16, 0.712, 0.029),
        Line(hydrogen, 0, (8, 2)): (7.378e-16, 0.7159, 0.032),
        Line(hydrogen, 0, (9, 2)): (8.947e-16, 0.7177, 0.033),
        Line(hydrogen, 0, (4, 3)): (1.330e-16, 0.7449, 0.045),
        Line(hydrogen, 0, (5, 3)): (6.64e-16, 0.7356, 0.044),
        Line(hydrogen, 0, (6, 3)): (2.481e-15, 0.7118, 0.016),
        Line(hydrogen, 0, (7, 3)): (3.270e-15, 0.7137, 0.029),
        Line(hydrogen, 0, (8, 3)): (4.343e-15, 0.7133, 0.032),
        Line(hydrogen, 0, (9, 3)): (5.588e-15, 0.7165, 0.033),
        Line(deuterium, 0, (3, 2)): (3.71e-18, 0.7665, 0.064),
        Line(deuterium, 0, (4, 2)): (8.425e-18, 0.7803, 0.050),
        Line(deuterium, 0, (5, 2)): (1.31e-15, 0.6796, 0.030),
        Line(deuterium, 0, (6, 2)): (3.954e-16, 0.7149, 0.028),
        Line(deuterium, 0, (7, 2)): (6.258e-16, 0.712, 0.029),
        Line(deuterium, 0, (8, 2)): (7.378e-16, 0.7159, 0.032),
        Line(deuterium, 0, (9, 2)): (8.947e-16, 0.7177, 0.033),
        Line(deuterium, 0, (4, 3)): (1.330e-16, 0.7449, 0.045),
        Line(deuterium, 0, (5, 3)): (6.64e-16, 0.7356, 0.044),
        Line(deuterium, 0, (6, 3)): (2.481e-15, 0.7118, 0.016),
        Line(deuterium, 0, (7, 3)): (3.270e-15, 0.7137, 0.029),
        Line(deuterium, 0, (8, 3)): (4.343e-15, 0.7133, 0.032),
        Line(deuterium, 0, (9, 3)): (5.588e-15, 0.7165, 0.033),
        Line(tritium, 0, (3, 2)): (3.71e-18, 0.7665, 0.064),
        Line(tritium, 0, (4, 2)): (8.425e-18, 0.7803, 0.050),
        Line(tritium, 0, (5, 2)): (1.31e-15, 0.6796, 0.030),
        Line(tritium, 0, (6, 2)): (3.954e-16, 0.7149, 0.028),
        Line(tritium, 0, (7, 2)): (6.258e-16, 0.712, 0.029),
        Line(tritium, 0, (8, 2)): (7.378e-16, 0.7159, 0.032),
        Line(tritium, 0, (9, 2)): (8.947e-16, 0.7177, 0.033),
        Line(tritium, 0, (4, 3)): (1.330e-16, 0.7449, 0.045),
        Line(tritium, 0, (5, 3)): (6.64e-16, 0.7356, 0.044),
        Line(tritium, 0, (6, 3)): (2.481e-15, 0.7118, 0.016),
        Line(tritium, 0, (7, 3)): (3.270e-15, 0.7137, 0.029),
        Line(tritium, 0, (8, 3)): (4.343e-15, 0.7133, 0.032),
        Line(tritium, 0, (9, 3)): (5.588e-15, 0.7165, 0.033)
    }

    def __init__(self, Line line, double wavelength, Species target_species, Plasma plasma,
                 dict stark_model_coefficients=None, Integrator1D integrator=GaussianQuadrature(), polarisation='no'):

        stark_model_coefficients = stark_model_coefficients or self.STARK_MODEL_COEFFICIENTS_DEFAULT

        try:
            # Fitted Stark Constants
            cij, aij, bij = stark_model_coefficients[line]
            if cij <= 0:
                raise ValueError('Coefficient c_ij must be positive.')
            if aij <= 0:
                raise ValueError('Coefficient a_ij must be positive.')
            if bij <= 0:
                raise ValueError('Coefficient b_ij must be positive.')
            self._aij = aij
            self._bij = bij
            self._cij = cij
        except IndexError:
            raise ValueError('Stark broadening coefficients for {} is not currently available.'.format(line))

        # polynomial coefficients in increasing powers
        self._fwhm_poly_coeff_gauss = [1., 0, 0.57575, 0.37902, -0.42519, -0.31525, 0.31718]
        self._fwhm_poly_coeff_lorentz = [1., 0.15882, 1.04388, -1.38281, 0.46251, 0.82325, -0.58026]

        self._weight_poly_coeff = [5.14820e-04, 1.38821e+00, -9.60424e-02, -3.83995e-02, -7.40042e-03, -5.47626e-04]

        super().__init__(line, wavelength, target_species, plasma, polarisation, integrator)

    @classmethod
    def show_supported_transitions(cls):
        """ Prints all supported transitions."""
        for line, coeff in cls.STARK_MODEL_COEFFICIENTS_DEFAULT.items():
            print('{}: c_ij={}, a_ij={}, b_ij={}'.format(line, coeff[0], coeff[1], coeff[2]))

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @cython.cdivision(True)
    cpdef Spectrum add_line(self, double radiance, Point3D point, Vector3D direction, Spectrum spectrum):

        cdef:
            double ne, te, ts, shifted_wavelength, photon_energy, b_magn, comp_radiance, cos_sqr, sin_sqr
            double sigma, fwhm_lorentz, fwhm_gauss, fwhm_full, fwhm_ratio, fwhm_lorentz_to_total, lorentz_weight, gauss_weight
            cdef Vector3D ion_velocity, b_field
            int i

        ne = self.plasma.get_electron_distribution().density(point.x, point.y, point.z)

        te = self.plasma.get_electron_distribution().effective_temperature(point.x, point.y, point.z)

        fwhm_lorentz = self._cij * ne**self._aij / (te**self._bij) if ne > 0 and te > 0 else 0

        ts = self.target_species.distribution.effective_temperature(point.x, point.y, point.z)

        fwhm_gauss = _SIGMA2FWHM * thermal_broadening(self.wavelength, ts, self.line.element.atomic_weight) if ts > 0 else 0

        if fwhm_lorentz == 0 and fwhm_gauss == 0:
            return spectrum

        # calculating full FWHM
        if fwhm_gauss <= fwhm_lorentz:
            fwhm_ratio = fwhm_gauss / fwhm_lorentz
            fwhm_full = self._fwhm_poly_coeff_gauss[0]
            for i in range(1, 7):
                fwhm_full += self._fwhm_poly_coeff_gauss[i] * fwhm_ratio**i
            fwhm_full *= fwhm_lorentz
        else:
            fwhm_ratio = fwhm_lorentz / fwhm_gauss
            fwhm_full = self._fwhm_poly_coeff_lorentz[0]
            for i in range(1, 7):
                fwhm_full += self._fwhm_poly_coeff_lorentz[i] * fwhm_ratio**i
            fwhm_full *= fwhm_gauss

        sigma = fwhm_full / _SIGMA2FWHM

        fwhm_lorentz_to_total = fwhm_lorentz / fwhm_full

        # calculating Lorentzian weight
        if fwhm_lorentz_to_total < 0.01:
            lorentz_weight = 0
            fwhm_full = 0  # force add_lorentzian_line() to immediately return
        elif fwhm_lorentz_to_total > 0.999:
            lorentz_weight = 1
            sigma = 0  # force add_gaussian_line() to immediately return
        else:
            lorentz_weight = self._weight_poly_coeff[0]
            for i in range(1, 6):
                lorentz_weight += self._weight_poly_coeff[i] * log(fwhm_lorentz_to_total)**i
            lorentz_weight = exp(lorentz_weight)

        gauss_weight = 1 - lorentz_weight

        ion_velocity = self.target_species.distribution.bulk_velocity(point.x, point.y, point.z)

        # calculate emission line central wavelength, doppler shifted along observation direction
        shifted_wavelength = doppler_shift(self.wavelength, direction, ion_velocity)

        # obtain magnetic field
        b_field = self.plasma.get_b_field().evaluate(point.x, point.y, point.z)
        b_magn = b_field.get_length()

        if b_magn == 0:
            # no splitting if magnetic field strength is zero
            if self._polarisation != NO_POLARISATION:
                radiance *= 0.5  # pi or sigma polarisation, collecting only half of intensity

            spectrum = add_gaussian_line(gauss_weight * radiance, shifted_wavelength, sigma, spectrum)
            spectrum = add_lorentzian_line(lorentz_weight * radiance, shifted_wavelength, fwhm_full, spectrum, self.integrator)

            return spectrum

        # coefficients for intensities parallel and perpendicular to magnetic field
        cos_sqr = (b_field.dot(direction.normalise()) / b_magn)**2
        sin_sqr = 1. - cos_sqr

        # adding pi component of the Zeeman triplet in case of NO_POLARISATION or PI_POLARISATION
        if self._polarisation != SIGMA_POLARISATION:
            comp_radiance = 0.5 * sin_sqr * radiance
            spectrum = add_gaussian_line(gauss_weight * comp_radiance, shifted_wavelength, sigma, spectrum)
            spectrum = add_lorentzian_line(lorentz_weight * comp_radiance, shifted_wavelength, fwhm_full, spectrum, self.integrator)

        # adding sigma +/- components of the Zeeman triplet in case of NO_POLARISATION or SIGMA_POLARISATION
        if self._polarisation != PI_POLARISATION:
            comp_radiance = (0.25 * sin_sqr + 0.5 * cos_sqr) * radiance

            photon_energy = HC_EV_NM / self.wavelength

            shifted_wavelength = doppler_shift(HC_EV_NM / (photon_energy - BOHR_MAGNETON * b_magn), direction, ion_velocity)
            spectrum = add_gaussian_line(gauss_weight * comp_radiance, shifted_wavelength, sigma, spectrum)
            spectrum = add_lorentzian_line(lorentz_weight * comp_radiance, shifted_wavelength, fwhm_full, spectrum, self.integrator)

            shifted_wavelength = doppler_shift(HC_EV_NM / (photon_energy + BOHR_MAGNETON * b_magn), direction, ion_velocity)
            spectrum = add_gaussian_line(gauss_weight * comp_radiance, shifted_wavelength, sigma, spectrum)
            spectrum = add_lorentzian_line(lorentz_weight * comp_radiance, shifted_wavelength, fwhm_full, spectrum, self.integrator)

        return spectrum
