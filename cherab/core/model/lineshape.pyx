# cython: language_level=3

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

import numpy as np
from scipy.special import hyp2f1

cimport numpy as np
from libc.math cimport sqrt, erf, M_SQRT2, floor, ceil, fabs
from raysect.optical.spectrum cimport new_spectrum
from raysect.core.math.function.float cimport Function1D

from cherab.core cimport Plasma
from cherab.core.atomic.elements import hydrogen, deuterium, tritium, helium, helium3, beryllium, boron, carbon, nitrogen, oxygen, neon
from cherab.core.math.function cimport autowrap_function1d, autowrap_function2d
from cherab.core.math.integrators cimport GaussianQuadrature
from cherab.core.utility.constants cimport ATOMIC_MASS, ELEMENTARY_CHARGE, SPEED_OF_LIGHT

cimport cython

# required by numpy c-api
np.import_array()


cdef double RECIP_ATOMIC_MASS = 1 / ATOMIC_MASS


cdef double evamu_to_ms(double x):
    return sqrt(2 * x * ELEMENTARY_CHARGE * RECIP_ATOMIC_MASS)


@cython.cdivision(True)
cpdef double doppler_shift(double wavelength, Vector3D observation_direction, Vector3D velocity):
    """
    Calculates the Doppler shifted wavelength for a given velocity and observation direction.

    :param wavelength: The wavelength to Doppler shift in nanometers.
    :param observation_direction: A Vector defining the direction of observation.
    :param velocity: A Vector defining the relative velocity of the emitting source in m/s.
    :return: The Doppler shifted wavelength in nanometers.
    """
    cdef double projected_velocity

    # flow velocity projected on the direction of observation
    observation_direction = observation_direction.normalise()
    projected_velocity = velocity.dot(observation_direction)

    return wavelength * (1 + projected_velocity / SPEED_OF_LIGHT)


@cython.cdivision(True)
cpdef double thermal_broadening(double wavelength, double temperature, double atomic_weight):
    """
    Returns the line width for a gaussian line as a standard deviation.

    :param wavelength: Central wavelength.
    :param temperature: Temperature in eV.
    :param atomic_weight: Atomic weight in AMU.
    :return: Standard deviation of gaussian line.
    """

    # todo: add input sanity checks
    return sqrt(temperature * ELEMENTARY_CHARGE / (atomic_weight * ATOMIC_MASS)) * wavelength / SPEED_OF_LIGHT


# the number of standard deviations outside the rest wavelength the line is considered to add negligible value (including a margin for safety)
DEF GAUSSIAN_CUTOFF_SIGMA = 10.0


@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef Spectrum add_gaussian_line(double radiance, double wavelength, double sigma, Spectrum spectrum):
    r"""
    Adds a Gaussian line to the given spectrum and returns the new spectrum.

    The formula used is based on the following definite integral:
    :math:`\frac{1}{\sigma \sqrt{2 \pi}} \int_{\lambda_0}^{\lambda_1} \exp(-\frac{(x-\mu)^2}{2\sigma^2}) dx = \frac{1}{2} \left[ -Erf(\frac{a-\mu}{\sqrt{2}\sigma}) +Erf(\frac{b-\mu}{\sqrt{2}\sigma}) \right]`

    :param float radiance: Intensity of the line in radiance.
    :param float wavelength: central wavelength of the line in nm.
    :param float sigma: width of the line in nm.
    :param Spectrum spectrum: the current spectrum to which the gaussian line is added.
    :return:
    """

    cdef double temp
    cdef double cutoff_lower_wavelength, cutoff_upper_wavelength
    cdef double lower_wavelength, upper_wavelength
    cdef double lower_integral, upper_integral
    cdef int start, end, i

    if sigma <= 0:
        return spectrum

    # calculate and check end of limits
    cutoff_lower_wavelength = wavelength - GAUSSIAN_CUTOFF_SIGMA * sigma
    if spectrum.max_wavelength < cutoff_lower_wavelength:
        return spectrum

    cutoff_upper_wavelength = wavelength + GAUSSIAN_CUTOFF_SIGMA * sigma
    if spectrum.min_wavelength > cutoff_upper_wavelength:
        return spectrum

    # locate range of bins where there is significant contribution from the gaussian (plus a health margin)
    start = max(0, <int> floor((cutoff_lower_wavelength - spectrum.min_wavelength) / spectrum.delta_wavelength))
    end = min(spectrum.bins, <int> ceil((cutoff_upper_wavelength - spectrum.min_wavelength) / spectrum.delta_wavelength))

    # add line to spectrum
    temp = 1 / (M_SQRT2 * sigma)
    lower_wavelength = spectrum.min_wavelength + start * spectrum.delta_wavelength
    lower_integral = erf((lower_wavelength - wavelength) * temp)
    for i in range(start, end):

        upper_wavelength = spectrum.min_wavelength + spectrum.delta_wavelength * (i + 1)
        upper_integral = erf((upper_wavelength - wavelength) * temp)

        spectrum.samples_mv[i] += radiance * 0.5 * (upper_integral - lower_integral) / spectrum.delta_wavelength

        lower_wavelength = upper_wavelength
        lower_integral = upper_integral

    return spectrum


cdef class LineShapeModel:
    """
    A base class for building line shapes.

    :param Line line: The emission line object for this line shape.
    :param float wavelength: The rest wavelength for this emission line.
    :param Species target_species: The target plasma species that is emitting.
    :param Plasma plasma: The emitting plasma object.
    :param Integrator1D integrator: Integrator1D instance to integrate the line shape
        over the spectral bin. Default is None.
    """

    def __init__(self, Line line, double wavelength, Species target_species, Plasma plasma, Integrator1D integrator=None):

        self.line = line
        self.wavelength = wavelength
        self.target_species = target_species
        self.plasma = plasma
        self.integrator = integrator

    cpdef Spectrum add_line(self, double radiance, Point3D point, Vector3D direction, Spectrum spectrum):
        raise NotImplementedError('Child lineshape class must implement this method.')


cdef class GaussianLine(LineShapeModel):
    """
    Produces Gaussian line shape.

    :param Line line: The emission line object for this line shape.
    :param float wavelength: The rest wavelength for this emission line.
    :param Species target_species: The target plasma species that is emitting.
    :param Plasma plasma: The emitting plasma object.

    .. code-block:: pycon

       >>> from cherab.core.atomic import Line, deuterium
       >>> from cherab.core.model import ExcitationLine, GaussianLine
       >>>
       >>> # Adding Gaussian line to the plasma model.
       >>> d_alpha = Line(deuterium, 0, (3, 2))
       >>> excit = ExcitationLine(d_alpha, lineshape=GaussianLine)
       >>> plasma.models.add(excit)
    """

    def __init__(self, Line line, double wavelength, Species target_species, Plasma plasma):

        super().__init__(line, wavelength, target_species, plasma)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @cython.cdivision(True)
    cpdef Spectrum add_line(self, double radiance, Point3D point, Vector3D direction, Spectrum spectrum):

        cdef double ts, sigma, shifted_wavelength
        cdef Vector3D ion_velocity

        ts = self.target_species.distribution.effective_temperature(point.x, point.y, point.z)
        if ts <= 0.0:
            return spectrum

        ion_velocity = self.target_species.distribution.bulk_velocity(point.x, point.y, point.z)

        # calculate emission line central wavelength, doppler shifted along observation direction
        shifted_wavelength = doppler_shift(self.wavelength, direction, ion_velocity)

        # calculate the line width
        sigma = thermal_broadening(self.wavelength, ts, self.line.element.atomic_weight)

        return add_gaussian_line(radiance, shifted_wavelength, sigma, spectrum)


DEF MULTIPLET_WAVELENGTH = 0
DEF MULTIPLET_RATIO = 1


cdef class MultipletLineShape(LineShapeModel):
    """
    Produces Multiplet line shapes.

    The lineshape radiance is calculated from a base PEC rate that is unresolved. This
    radiance is then divided over a number of components as specified in the multiplet
    argument. The multiplet components are specified with an Nx2 array where N is the
    number of components in the multiplet. The first axis of the array contains the
    wavelengths of each component, the second contains the line ratio for each component.
    The component line ratios must sum to one. For example:

    :param Line line: The emission line object for the base rate radiance calculation.
    :param float wavelength: The rest wavelength of the base emission line.
    :param Species target_species: The target plasma species that is emitting.
    :param Plasma plasma: The emitting plasma object.
    :param multiplet: An Nx2 array that specifies the multiplet wavelengths and line ratios.

    .. code-block:: pycon

       >>> from cherab.core.atomic import Line, nitrogen
       >>> from cherab.core.model import ExcitationLine, MultipletLineShape
       >>>
       >>> # multiplet specification in Nx2 array
       >>> multiplet = [[403.509, 404.132, 404.354, 404.479, 405.692], [0.205, 0.562, 0.175, 0.029, 0.029]]
       >>>
       >>> # Adding the multiplet to the plasma model.
       >>> nitrogen_II_404 = Line(nitrogen, 1, ("2s2 2p1 4f1 3G13.0", "2s2 2p1 3d1 3F10.0"))
       >>> excit = ExcitationLine(nitrogen_II_404, lineshape=MultipletLineShape, lineshape_args=[multiplet])
       >>> plasma.models.add(excit)
    """

    def __init__(self, Line line, double wavelength, Species target_species, Plasma plasma,
                 object multiplet):

        super().__init__(line, wavelength, target_species, plasma)

        multiplet = np.array(multiplet, dtype=np.float64)

        if not (len(multiplet.shape) == 2 and multiplet.shape[0] == 2):
            raise ValueError("The multiplet specification must be an array of shape (Nx2).")

        if not multiplet[1,:].sum() == 1.0:
            raise ValueError("The multiplet line ratios should sum to one.")

        self._number_of_lines = multiplet.shape[1]
        self._multiplet = multiplet
        self._multiplet_mv = self._multiplet

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef Spectrum add_line(self, double radiance, Point3D point, Vector3D direction, Spectrum spectrum):

        cdef double ts, sigma, shifted_wavelength, component_wavelength, component_radiance
        cdef Vector3D ion_velocity

        ts = self.target_species.distribution.effective_temperature(point.x, point.y, point.z)
        if ts <= 0.0:
            return spectrum

        ion_velocity = self.target_species.distribution.bulk_velocity(point.x, point.y, point.z)

        # calculate the line width
        sigma = thermal_broadening(self.wavelength, ts, self.line.element.atomic_weight)

        for i in range(self._number_of_lines):

            component_wavelength = self._multiplet_mv[MULTIPLET_WAVELENGTH, i]
            component_radiance = radiance * self._multiplet_mv[MULTIPLET_RATIO, i]

            # calculate emission line central wavelength, doppler shifted along observation direction
            shifted_wavelength = doppler_shift(component_wavelength, direction, ion_velocity)

            spectrum = add_gaussian_line(component_radiance, shifted_wavelength, sigma, spectrum)

        return spectrum


DEF LORENZIAN_CUTOFF_GAMMA = 50.0


cdef class StarkFunction(Function1D):
    """
    Normalised Stark function for the StarkBroadenedLine line shape.
    """

    cdef double _a, _x0, _norm

    STARK_NORM_COEFFICIENT = 4 * LORENZIAN_CUTOFF_GAMMA * hyp2f1(0.4, 1, 1.4, -(2 * LORENZIAN_CUTOFF_GAMMA)**2.5)

    def __init__(self, double wavelength, double lambda_1_2):

        if wavelength <= 0:
            raise ValueError("Argument 'wavelength' must be positive.")

        if lambda_1_2 <= 0:
            raise ValueError("Argument 'lambda_1_2' must be positive.")

        self._x0 = wavelength
        self._a = (0.5 * lambda_1_2)**2.5
        # normalise, so the integral over x is equal to 1 in the limits
        # (_x0 - LORENZIAN_CUTOFF_GAMMA * lambda_1_2, _x0 + LORENZIAN_CUTOFF_GAMMA * lambda_1_2)
        self._norm = (0.5 * lambda_1_2)**1.5 / <double> self.STARK_NORM_COEFFICIENT

    @cython.cdivision(True)
    cdef double evaluate(self, double x) except? -1e999:

        return self._norm / ((fabs(x - self._x0))**2.5 + self._a)


cdef class StarkBroadenedLine(LineShapeModel):
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
                 dict stark_model_coefficients=None, integrator=GaussianQuadrature()):

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

        super().__init__(line, wavelength, target_species, plasma, integrator)

    def show_supported_transitions(self):
        """ Prints all supported transitions."""
        for line, coeff in self.STARK_MODEL_COEFFICIENTS_DEFAULT.items():
            print('{}: c_ij={}, a_ij={}, b_ij={}'.format(line, coeff[0], coeff[1], coeff[2]))

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @cython.cdivision(True)
    cpdef Spectrum add_line(self, double radiance, Point3D point, Vector3D direction, Spectrum spectrum):

        cdef:
            double ne, te, lambda_1_2, lambda_5_2, wvl
            double cutoff_lower_wavelength, cutoff_upper_wavelength
            double lower_wavelength, upper_wavelength
            double bin_integral
            int start, end, i
            Spectrum raw_lineshape

        ne = self.plasma.get_electron_distribution().density(point.x, point.y, point.z)
        if ne <= 0.0:
            return spectrum

        te = self.plasma.get_electron_distribution().effective_temperature(point.x, point.y, point.z)
        if te <= 0.0:
            return spectrum

        lambda_1_2 = self._cij * ne**self._aij / (te**self._bij)

        self.integrator.function = StarkFunction(self.wavelength, lambda_1_2)

        # calculate and check end of limits
        cutoff_lower_wavelength = self.wavelength - LORENZIAN_CUTOFF_GAMMA * lambda_1_2
        if spectrum.max_wavelength < cutoff_lower_wavelength:
            return spectrum

        cutoff_upper_wavelength = self.wavelength + LORENZIAN_CUTOFF_GAMMA * lambda_1_2
        if spectrum.min_wavelength > cutoff_upper_wavelength:
            return spectrum

        # locate range of bins where there is significant contribution from the gaussian (plus a health margin)
        start = max(0, <int> floor((cutoff_lower_wavelength - spectrum.min_wavelength) / spectrum.delta_wavelength))
        end = min(spectrum.bins, <int> ceil((cutoff_upper_wavelength - spectrum.min_wavelength) / spectrum.delta_wavelength))

        # add line to spectrum
        lower_wavelength = spectrum.min_wavelength + start * spectrum.delta_wavelength

        for i in range(start, end):
            upper_wavelength = spectrum.min_wavelength + spectrum.delta_wavelength * (i + 1)

            bin_integral = self.integrator.evaluate(lower_wavelength, upper_wavelength)
            spectrum.samples_mv[i] += radiance * bin_integral / spectrum.delta_wavelength

            lower_wavelength = upper_wavelength

        return spectrum


DEF BOHR_MAGNETON = 5.78838180123e-5  # in eV/T
DEF HC_EV_NM = 1239.8419738620933  # (Planck constant in eV s) x (speed of light in nm/s)

DEF PI_POLARISATION = 0
DEF SIGMA_POLARISATION = 1
DEF SIGMA_PLUS_POLARISATION = 1
DEF SIGMA_MINUS_POLARISATION = -1
DEF NO_POLARISATION = 2


cdef class ZeemanLineShapeModel(LineShapeModel):
    r"""
    A base class for building Zeeman line shapes.

    :param Line line: The emission line object for this line shape.
    :param float wavelength: The rest wavelength for this emission line.
    :param Species target_species: The target plasma species that is emitting.
    :param Plasma plasma: The emitting plasma object.
    :param str polarisation: Leaves only :math:`\pi`-/:math:`\sigma`-polarised components:
                             "pi" - leave only :math:`\pi`-polarised components,
                             "sigma" - leave only :math:`\sigma`-polarised components,
                             "no" - leave all components (default).
    """

    def __init__(self, Line line, double wavelength, Species target_species, Plasma plasma, polarisation):
        super().__init__(line, wavelength, target_species, plasma)

        self.polarisation = polarisation

    @property
    def polarisation(self):
        if self._polarisation == PI_POLARISATION:
            return 'pi'
        if self._polarisation == SIGMA_POLARISATION:
            return 'sigma'
        if self._polarisation == NO_POLARISATION:
            return 'no'

    @polarisation.setter
    def polarisation(self, value):
        if value.lower() == 'pi':
            self._polarisation = PI_POLARISATION
        elif value.lower() == 'sigma':
            self._polarisation = SIGMA_POLARISATION
        elif value.lower() == 'no':
            self._polarisation = NO_POLARISATION
        else:
            raise ValueError('Select between "pi", "sigma" or "no", {} is unsupported.'.format(value))


cdef class ZeemanTriplet(ZeemanLineShapeModel):
    r"""
    Simple Doppler-Zeeman triplet (Paschen-Back effect).

    :param Line line: The emission line object for this line shape.
    :param float wavelength: The rest wavelength for this emission line.
    :param Species target_species: The target plasma species that is emitting.
    :param Plasma plasma: The emitting plasma object.
    :param str polarisation: Leaves only :math:`\pi`-/:math:`\sigma`-polarised components:
                             "pi" - leave central component,
                             "sigma" - leave side components,
                             "no" - all components (default).
    """

    def __init__(self, Line line, double wavelength, Species target_species, Plasma plasma, polarisation='no'):

        super().__init__(line, wavelength, target_species, plasma, polarisation)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @cython.cdivision(True)
    cpdef Spectrum add_line(self, double radiance, Point3D point, Vector3D direction, Spectrum spectrum):

        cdef double ts, sigma, shifted_wavelength, photon_energy, b_magn, component_radiance, cos_sqr, sin_sqr
        cdef Vector3D ion_velocity, b_field

        ts = self.target_species.distribution.effective_temperature(point.x, point.y, point.z)
        if ts <= 0.0:
            return spectrum

        ion_velocity = self.target_species.distribution.bulk_velocity(point.x, point.y, point.z)

        # calculate emission line central wavelength, doppler shifted along observation direction
        shifted_wavelength = doppler_shift(self.wavelength, direction, ion_velocity)

        # calculate the line width
        sigma = thermal_broadening(self.wavelength, ts, self.line.element.atomic_weight)

        # obtain magnetic field
        b_field = self.plasma.get_b_field().evaluate(point.x, point.y, point.z)
        b_magn = b_field.get_length()

        if b_magn == 0:
            # no splitting if magnetic field strength is zero
            if self._polarisation == NO_POLARISATION:
                return add_gaussian_line(radiance, shifted_wavelength, sigma, spectrum)

            return add_gaussian_line(0.5 * radiance, shifted_wavelength, sigma, spectrum)

        # coefficients for intensities parallel and perpendicular to magnetic field
        cos_sqr = (b_field.dot(direction.normalise()) / b_magn)**2
        sin_sqr = 1. - cos_sqr

        # adding pi component of the Zeeman triplet in case of NO_POLARISATION or PI_POLARISATION
        if self._polarisation != SIGMA_POLARISATION:
            component_radiance = 0.5 * sin_sqr * radiance
            spectrum = add_gaussian_line(component_radiance, shifted_wavelength, sigma, spectrum)

        # adding sigma +/- components of the Zeeman triplet in case of NO_POLARISATION or SIGMA_POLARISATION
        if self._polarisation != PI_POLARISATION:
            component_radiance = (0.25 * sin_sqr + 0.5 * cos_sqr) * radiance

            photon_energy = HC_EV_NM / self.wavelength

            shifted_wavelength = doppler_shift(HC_EV_NM / (photon_energy - BOHR_MAGNETON * b_magn), direction, ion_velocity)
            spectrum = add_gaussian_line(component_radiance, shifted_wavelength, sigma, spectrum)

            shifted_wavelength = doppler_shift(HC_EV_NM / (photon_energy + BOHR_MAGNETON * b_magn), direction, ion_velocity)
            spectrum = add_gaussian_line(component_radiance, shifted_wavelength, sigma, spectrum)

        return spectrum


cdef class ParametrisedZeemanTriplet(ZeemanLineShapeModel):
    r"""
    Parametrised Doppler-Zeeman triplet. It takes into account additional broadening due to
    the line's fine structure without resolving the individual components of the fine
    structure. The model is described with three parameters: :math:`\alpha`,
    :math:`\beta` and :math:`\gamma`.

    The distance between :math:`\sigma^+` and :math:`\sigma^-` peaks: 
    :math:`\Delta \lambda_{\sigma} = \alpha B`, 
    where `B` is the magnetic field strength.
    The ratio between Zeeman and thermal broadening line widths: 
    :math:`\frac{W_{Zeeman}}{W_{Doppler}} = \beta T^{\gamma}`,
    where `T` is the species temperature in eV.

    Call `show_supported_transitions()` to see the list of supported transitions and
    default parameters of the model.

    For details see A. Blom and C. Jupén, Parametrisation of the Zeeman effect
    for hydrogen-like spectra in high-temperature plasmas,
    Plasma Phys. Control. Fusion 44 (2002) `1229-1241
    <https://doi.org/10.1088/0741-3335/44/7/312>`_.

    :param Line line: The emission line object for this line shape.
    :param float wavelength: The rest wavelength for this emission line.
    :param Species target_species: The target plasma species that is emitting.
    :param Plasma plasma: The emitting plasma object.
    :param dict line_parameters: Alternative parameters of the model in the form
                                 {line_i: (alpha_i, beta_i, gamma_i), ...}.
                                 If None, the default model parameters will be used.
    :param str polarisation: Leaves only :math:`\pi`-/:math:`\sigma`-polarised components:
                             "pi" - leave central component,
                             "sigma" - leave side components,
                             "no" - all components (default).
    """

    LINE_PARAMETERS_DEFAULT = {  # alpha, beta, gamma parameters for selected lines
        Line(hydrogen, 0, (3, 2)): (0.0402267, 0.3415, -0.5247),
        Line(hydrogen, 0, (4, 2)): (0.0220724, 0.2837, -0.5346),
        Line(deuterium, 0, (3, 2)): (0.0402068, 0.4384, -0.5015),
        Line(deuterium, 0, (4, 2)): (0.0220610, 0.3702, -0.5132),
        Line(helium3, 1, (4, 3)): (0.0205200, 1.4418, -0.4892),
        Line(helium3, 1, (5, 3)): (0.0095879, 1.2576, -0.5001),
        Line(helium3, 1, (6, 4)): (0.0401980, 0.8976, -0.4971),
        Line(helium3, 1, (7, 4)): (0.0273538, 0.8529, -0.5039),
        Line(helium, 1, (4, 3)): (0.0205206, 1.6118, -0.4838),
        Line(helium, 1, (5, 3)): (0.0095879, 1.4294, -0.4975),
        Line(helium, 1, (6, 4)): (0.0401955, 1.0058, -0.4918),
        Line(helium, 1, (7, 4)): (0.0273521, 0.9563, -0.4981),
        Line(beryllium, 3, (5, 4)): (0.0060354, 2.1245, -0.3190),
        Line(beryllium, 3, (6, 5)): (0.0202754, 1.6538, -0.3192),
        Line(beryllium, 3, (7, 5)): (0.0078966, 1.7017, -0.3348),
        Line(beryllium, 3, (8, 6)): (0.0205025, 1.4581, -0.3450),
        Line(boron, 4, (6, 5)): (0.0083423, 2.0519, -0.2960),
        Line(boron, 4, (7, 6)): (0.0228379, 1.6546, -0.2941),
        Line(boron, 4, (8, 6)): (0.0084065, 1.8041, -0.3177),
        Line(boron, 4, (8, 7)): (0.0541883, 1.4128, -0.2966),
        Line(boron, 4, (9, 7)): (0.0190781, 1.5440, -0.3211),
        Line(boron, 4, (10, 8)): (0.0391914, 1.3569, -0.3252),
        Line(carbon, 5, (6, 5)): (0.0040900, 2.4271, -0.2818),
        Line(carbon, 5, (7, 6)): (0.0110398, 1.9785, -0.2816),
        Line(carbon, 5, (8, 6)): (0.0040747, 2.1776, -0.3035),
        Line(carbon, 5, (8, 7)): (0.0261405, 1.6689, -0.2815),
        Line(carbon, 5, (9, 7)): (0.0092096, 1.8495, -0.3049),
        Line(carbon, 5, (10, 8)): (0.0189020, 1.6191, -0.3078),
        Line(carbon, 5, (11, 8)): (0.0110428, 1.6600, -0.3162),
        Line(carbon, 5, (10, 9)): (0.0359009, 1.4464, -0.3104),
        Line(nitrogen, 6, (7, 6)): (0.0060010, 2.4789, -0.2817),
        Line(nitrogen, 6, (8, 7)): (0.0141271, 2.0249, -0.2762),
        Line(nitrogen, 6, (9, 8)): (0.0300127, 1.7415, -0.2753),
        Line(nitrogen, 6, (10, 8)): (0.0102089, 1.9464, -0.2975),
        Line(nitrogen, 6, (11, 9)): (0.0193799, 1.7133, -0.2973),
        Line(oxygen, 7, (8, 7)): (0.0083081, 2.4263, -0.2747),
        Line(oxygen, 7, (9, 8)): (0.0176049, 2.0652, -0.2721),
        Line(oxygen, 7, (10, 8)): (0.0059933, 2.3445, -0.2944),
        Line(oxygen, 7, (10, 9)): (0.0343805, 1.8122, -0.2718),
        Line(oxygen, 7, (11, 9)): (0.0113640, 2.0268, -0.2911),
        Line(neon, 9, (9, 8)): (0.0072488, 2.8838, -0.2758),
        Line(neon, 9, (10, 9)): (0.0141002, 2.4755, -0.2718),
        Line(neon, 9, (11, 9)): (0.0046673, 2.8410, -0.2917),
        Line(neon, 9, (11, 10)): (0.0257292, 2.1890, -0.2715)
    }

    def __init__(self, Line line, double wavelength, Species target_species, Plasma plasma, dict line_parameters=None, polarisation='no'):

        super().__init__(line, wavelength, target_species, plasma, polarisation)

        line_parameters = line_parameters or self.LINE_PARAMETERS_DEFAULT

        try:
            alpha, beta, gamma = line_parameters[self.line]
            if alpha <= 0:
                raise ValueError('Parameter alpha must be positive.')
            if beta < 0:
                raise ValueError('Parameter beta must be non-negative.')
            self._alpha = alpha
            self._beta = beta
            self._gamma = gamma

        except KeyError:
            raise ValueError('Data for {} is not available.'.format(self.line))

    def show_supported_transitions(self):
        """ Prints all supported transitions."""
        for line, param in self.LINE_PARAMETERS_DEFAULT.items():
            print('{}: alpha={}, beta={}, gamma={}'.format(line, param[0], param[1], param[2]))

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @cython.cdivision(True)
    cpdef Spectrum add_line(self, double radiance, Point3D point, Vector3D direction, Spectrum spectrum):

        cdef double ts, sigma, shifted_wavelength, b_magn, component_radiance, cos_sqr, sin_sqr
        cdef Vector3D ion_velocity, b_field

        ts = self.target_species.distribution.effective_temperature(point.x, point.y, point.z)
        if ts <= 0.0:
            return spectrum

        ion_velocity = self.target_species.distribution.bulk_velocity(point.x, point.y, point.z)

        # calculate emission line central wavelength, doppler shifted along observation direction
        shifted_wavelength = doppler_shift(self.wavelength, direction, ion_velocity)

        # calculate the line width
        sigma = thermal_broadening(self.wavelength, ts, self.line.element.atomic_weight)

        # fine structure broadening correction
        sigma *= sqrt(1. + self._beta * self._beta * ts**(2. * self._gamma))

        # obtain magnetic field
        b_field = self.plasma.get_b_field().evaluate(point.x, point.y, point.z)
        b_magn = b_field.get_length()

        if b_magn == 0:
            # no splitting if magnetic filed strength is zero
            if self._polarisation == NO_POLARISATION:
                return add_gaussian_line(radiance, shifted_wavelength, sigma, spectrum)

            return add_gaussian_line(0.5 * radiance, shifted_wavelength, sigma, spectrum)

        # coefficients for intensities parallel and perpendicular to magnetic field
        cos_sqr = (b_field.dot(direction.normalise()) / b_magn)**2
        sin_sqr = 1. - cos_sqr

        # adding pi component of the Zeeman triplet in case of NO_POLARISATION or PI_POLARISATION
        if self._polarisation != SIGMA_POLARISATION:
            component_radiance = 0.5 * sin_sqr * radiance
            spectrum = add_gaussian_line(component_radiance, shifted_wavelength, sigma, spectrum)

        # adding sigma +/- components of the Zeeman triplet in case of NO_POLARISATION or SIGMA_POLARISATION
        if self._polarisation != PI_POLARISATION:
            component_radiance = (0.25 * sin_sqr + 0.5 * cos_sqr) * radiance

            shifted_wavelength = doppler_shift(self.wavelength + 0.5 * self._alpha * b_magn, direction, ion_velocity)
            spectrum = add_gaussian_line(component_radiance, shifted_wavelength, sigma, spectrum)
            shifted_wavelength = doppler_shift(self.wavelength - 0.5 * self._alpha * b_magn, direction, ion_velocity)
            spectrum = add_gaussian_line(component_radiance, shifted_wavelength, sigma, spectrum)

        return spectrum


cdef class ZeemanMultiplet(ZeemanLineShapeModel):
    r"""
    Doppler-Zeeman Multiplet.

    The lineshape radiance is calculated from a base PEC rate that is unresolved. This
    radiance is then divided over a number of components as specified in the ``zeeman_structure``
    argument. The ``zeeman_structure`` specifies wavelengths and ratios of
    :math:`\pi`-/:math:`\sigma`-polarised components as functions of the magnetic field strength.
    These functions can be obtained using the output of the ADAS603 routines.

    :param Line line: The emission line object for the base rate radiance calculation.
    :param float wavelength: The rest wavelength of the base emission line.
    :param Species target_species: The target plasma species that is emitting.
    :param Plasma plasma: The emitting plasma object.
    :param zeeman_structure: A ``ZeemanStructure`` object that provides wavelengths and ratios
                             of :math:`\pi`-/:math:`\sigma^{+}`-/:math:`\sigma^{-}`-polarised
                             components for any given magnetic field strength.
    :param str polarisation: Leaves only :math:`\pi`-/:math:`\sigma`-polarised components:
                             "pi" - leave only :math:`\pi`-polarised components,
                             "sigma" - leave only :math:`\sigma`-polarised components,
                             "no" - leave all components (default).

    """

    def __init__(self, Line line, double wavelength, Species target_species, Plasma plasma,
                 ZeemanStructure zeeman_structure, polarisation='no'):

        super().__init__(line, wavelength, target_species, plasma, polarisation)

        self._zeeman_structure = zeeman_structure

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @cython.cdivision(True)
    cpdef Spectrum add_line(self, double radiance, Point3D point, Vector3D direction, Spectrum spectrum):

        cdef int i
        cdef double ts, sigma, shifted_wavelength, component_radiance
        cdef Vector3D ion_velocity
        cdef double[:, :] multiplet_pi_mv, multiplet_sigma_mv

        ts = self.target_species.distribution.effective_temperature(point.x, point.y, point.z)
        if ts <= 0.0:
            return spectrum

        ion_velocity = self.target_species.distribution.bulk_velocity(point.x, point.y, point.z)

        # calculate the line width
        sigma = thermal_broadening(self.wavelength, ts, self.line.element.atomic_weight)

        # obtain magnetic field
        b_field = self.plasma.get_b_field().evaluate(point.x, point.y, point.z)
        b_magn = b_field.get_length()

        if b_magn == 0:
            # no splitting if magnetic filed strength is zero
            shifted_wavelength = doppler_shift(self.wavelength, direction, ion_velocity)
            if self._polarisation == NO_POLARISATION:
                return add_gaussian_line(radiance, shifted_wavelength, sigma, spectrum)

            return add_gaussian_line(0.5 * radiance, shifted_wavelength, sigma, spectrum)

        # coefficients for intensities parallel and perpendicular to magnetic field
        cos_sqr = (b_field.dot(direction.normalise()) / b_magn)**2
        sin_sqr = 1. - cos_sqr

        # adding pi components of the Zeeman multiplet in case of NO_POLARISATION or PI_POLARISATION
        if self._polarisation != SIGMA_POLARISATION:
            component_radiance = 0.5 * sin_sqr * radiance
            multiplet_mv = self._zeeman_structure.evaluate(b_magn, PI_POLARISATION)

            for i in range(multiplet_mv.shape[1]):
                shifted_wavelength = doppler_shift(multiplet_mv[MULTIPLET_WAVELENGTH, i], direction, ion_velocity)
                spectrum = add_gaussian_line(component_radiance * multiplet_mv[MULTIPLET_RATIO, i], shifted_wavelength, sigma, spectrum)

        # adding sigma components of the Zeeman multiplet in case of NO_POLARISATION or SIGMA_POLARISATION
        if self._polarisation != PI_POLARISATION:
            component_radiance = (0.25 * sin_sqr + 0.5 * cos_sqr) * radiance

            multiplet_mv = self._zeeman_structure.evaluate(b_magn, SIGMA_PLUS_POLARISATION)

            for i in range(multiplet_mv.shape[1]):
                shifted_wavelength = doppler_shift(multiplet_mv[MULTIPLET_WAVELENGTH, i], direction, ion_velocity)
                spectrum = add_gaussian_line(component_radiance * multiplet_mv[MULTIPLET_RATIO, i], shifted_wavelength, sigma, spectrum)

            multiplet_mv = self._zeeman_structure.evaluate(b_magn, SIGMA_MINUS_POLARISATION)

            for i in range(multiplet_mv.shape[1]):
                shifted_wavelength = doppler_shift(multiplet_mv[MULTIPLET_WAVELENGTH, i], direction, ion_velocity)
                spectrum = add_gaussian_line(component_radiance * multiplet_mv[MULTIPLET_RATIO, i], shifted_wavelength, sigma, spectrum)

        return spectrum


cdef class BeamLineShapeModel:
    """
    A base class for building beam emission line shapes.

    :param Line line: The emission line object for this line shape.
    :param float wavelength: The rest wavelength for this emission line.
    :param Beam beam: The beam class that is emitting.
    """

    def __init__(self, Line line, double wavelength, Beam beam):

        self.line = line
        self.wavelength = wavelength
        self.beam = beam

    cpdef Spectrum add_line(self, double radiance, Point3D beam_point, Point3D plasma_point,
                            Vector3D beam_direction, Vector3D observation_direction, Spectrum spectrum):
        raise NotImplementedError('Child lineshape class must implement this method.')


DEF STARK_SPLITTING_FACTOR = 2.77e-8


cdef class BeamEmissionMultiplet(BeamLineShapeModel):
    """
    Produces Beam Emission Multiplet line shape, also known as the Motional Stark Effect spectrum.
    """

    def __init__(self, Line line, double wavelength, Beam beam, object sigma_to_pi,
                 object sigma1_to_sigma0, object pi2_to_pi3, object pi4_to_pi3):

        super().__init__(line, wavelength, beam)

        self._sigma_to_pi = autowrap_function2d(sigma_to_pi)
        self._sigma1_to_sigma0 = autowrap_function1d(sigma1_to_sigma0)
        self._pi2_to_pi3 = autowrap_function1d(pi2_to_pi3)
        self._pi4_to_pi3 = autowrap_function1d(pi4_to_pi3)

    @cython.cdivision(True)
    cpdef Spectrum add_line(self, double radiance, Point3D beam_point, Point3D plasma_point,
                            Vector3D beam_direction, Vector3D observation_direction, Spectrum spectrum):

        cdef double x, y, z
        cdef Plasma plasma
        cdef double te, ne, beam_energy, sigma, stark_split, beam_ion_mass, beam_temperature
        cdef double natural_wavelength, central_wavelength
        cdef double sigma_to_pi, d, intensity_sig, intensity_pi, e_field
        cdef double s1_to_s0, intensity_s0, intensity_s1
        cdef double pi2_to_pi3, pi4_to_pi3, intensity_pi2, intensity_pi3, intensity_pi4
        cdef Vector3D b_field, beam_velocity

        # extract for more compact code
        x = plasma_point.x
        y = plasma_point.y
        z = plasma_point.z

        plasma = self.beam.get_plasma()

        te = plasma.get_electron_distribution().effective_temperature(x, y, z)
        if te <= 0.0:
            return spectrum

        ne = plasma.get_electron_distribution().density(x, y, z)
        if ne <= 0.0:
            return spectrum

        beam_energy = self.beam.get_energy()

        # calculate Stark splitting
        b_field = plasma.get_b_field().evaluate(x, y, z)
        beam_velocity = beam_direction.normalise().mul(evamu_to_ms(beam_energy))
        e_field = beam_velocity.cross(b_field).get_length()
        stark_split = fabs(STARK_SPLITTING_FACTOR * e_field)  # TODO - calculate splitting factor? Reject other lines?

        # calculate emission line central wavelength, doppler shifted along observation direction
        natural_wavelength = self.wavelength
        central_wavelength = doppler_shift(natural_wavelength, observation_direction, beam_velocity)

        # calculate doppler broadening
        beam_ion_mass = self.beam.get_element().atomic_weight
        beam_temperature = self.beam.get_temperature()
        sigma = thermal_broadening(self.wavelength, beam_temperature, beam_ion_mass)

        # calculate relative intensities of sigma and pi lines
        sigma_to_pi = self._sigma_to_pi.evaluate(ne, beam_energy)
        d = 1 / (1 + sigma_to_pi)
        intensity_sig = sigma_to_pi * d * radiance
        intensity_pi = 0.5 * d * radiance

        # add Sigma lines to output
        s1_to_s0 = self._sigma1_to_sigma0.evaluate(ne)
        intensity_s0 = 1 / (s1_to_s0 + 1)
        intensity_s1 = 0.5 * s1_to_s0 * intensity_s0

        spectrum = add_gaussian_line(intensity_sig * intensity_s0, central_wavelength, sigma, spectrum)
        spectrum = add_gaussian_line(intensity_sig * intensity_s1, central_wavelength + stark_split, sigma, spectrum)
        spectrum = add_gaussian_line(intensity_sig * intensity_s1, central_wavelength - stark_split, sigma, spectrum)

        # add Pi lines to output
        pi2_to_pi3 = self._pi2_to_pi3.evaluate(ne)
        pi4_to_pi3 = self._pi4_to_pi3.evaluate(ne)
        intensity_pi3 = 1 / (1 + pi2_to_pi3 + pi4_to_pi3)
        intensity_pi2 = pi2_to_pi3 * intensity_pi3
        intensity_pi4 = pi4_to_pi3 * intensity_pi3

        spectrum = add_gaussian_line(intensity_pi * intensity_pi2, central_wavelength + 2 * stark_split, sigma, spectrum)
        spectrum = add_gaussian_line(intensity_pi * intensity_pi2, central_wavelength - 2 * stark_split, sigma, spectrum)
        spectrum = add_gaussian_line(intensity_pi * intensity_pi3, central_wavelength + 3 * stark_split, sigma, spectrum)
        spectrum = add_gaussian_line(intensity_pi * intensity_pi3, central_wavelength - 3 * stark_split, sigma, spectrum)
        spectrum = add_gaussian_line(intensity_pi * intensity_pi4, central_wavelength + 4 * stark_split, sigma, spectrum)
        spectrum = add_gaussian_line(intensity_pi * intensity_pi4, central_wavelength - 4 * stark_split, sigma, spectrum)

        return spectrum
