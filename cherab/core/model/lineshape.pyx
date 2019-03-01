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
from libc.math cimport sqrt, erf, M_SQRT2, floor, ceil, fabs
from cherab.core.utility.constants cimport ATOMIC_MASS, ELEMENTARY_CHARGE, SPEED_OF_LIGHT
from raysect.optical.spectrum cimport new_spectrum
cimport cython


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
DEF GAUSSIAN_CUTOFF_SIGMA=10.0


cpdef Spectrum add_gaussian_line(double radiance, double wavelength, double sigma, Spectrum spectrum):
    """
    Adds a Gaussian line to the given spectrum and returns the new spectrum.

    The formula used is based on the following definite integral:
    \frac{1}{\sigma \sqrt{2 \pi}} \int_{\lambda_0}^{\lambda_1} \exp(-\frac{(x-\mu)^2}{2\sigma^2}) dx = \frac{1}{2} \left[ -Erf(\frac{a-\mu}{\sqrt{2}\sigma}) +Erf(\frac{b-\mu}{\sqrt{2}\sigma}) \right]

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
    temp = M_SQRT2 * sigma
    lower_wavelength = spectrum.min_wavelength + start * spectrum.delta_wavelength
    lower_integral = erf((lower_wavelength - wavelength) / temp)
    for i in range(start, end):

        upper_wavelength = spectrum.min_wavelength + spectrum.delta_wavelength * (i + 1)
        upper_integral = erf((upper_wavelength - wavelength) / temp)

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
    """

    def __init__(self, Line line, double wavelength, Species target_species, Plasma plasma):

        self.line = line
        self.wavelength = wavelength
        self.target_species = target_species
        self.plasma = plasma

    cpdef Spectrum add_line(self, double radiance, Point3D point, Vector3D direction, Spectrum spectrum):
        raise NotImplementedError('Child lineshape class must implement this method.')


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
cdef class GaussianLine(LineShapeModel):

    def __init__(self, Line line, double wavelength, Species target_species, Plasma plasma):

        super().__init__(line, wavelength, target_species, plasma)

    cpdef Spectrum add_line(self, double radiance, Point3D point, Vector3D direction, Spectrum spectrum):

        cdef double te, sigma, shifted_wavelength
        cdef Vector3D ion_velocity

        te = self.plasma.get_electron_distribution().effective_temperature(point.x, point.y, point.z)
        if te <= 0.0:
            return spectrum

        ion_velocity = self.target_species.distribution.bulk_velocity(point.x, point.y, point.z)

        # calculate emission line central wavelength, doppler shifted along observation direction
        shifted_wavelength = doppler_shift(self.wavelength, direction, ion_velocity)

        # calculate the line width
        sigma = thermal_broadening(self.wavelength, te, self.line.element.atomic_weight)

        return add_gaussian_line(radiance, shifted_wavelength, sigma, spectrum)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
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

       >>> from cherab.core.atomic import Line, deuterium
       >>> from cherab.core.model import ExcitationLine, MultipletLineShape
       >>>
       >>> # multiplet specification in Nx2 array
       >>> multiplet = [[403.5, 404.1, 404.3], [0.2, 0.5, 0.3]]
       >>>
       >>> # Adding the multiplet to the plasma model.
       >>> d_alpha = Line(deuterium, 0, (3, 2))
       >>> excit = ExcitationLine(d_alpha, lineshape=MultipletLineShape, lineshape_args=[multiplet])
       >>> plasma.models.add(excit)
    """

    def __init__(self, Line line, double wavelength, Species target_species, Plasma plasma,
                 object multiplet):

        super().__init__(line, wavelength, target_species, plasma)

        multiplet = np.array(multiplet)

        if not (len(multiplet.shape) == 2 and multiplet.shape[0] == 2):
            raise ValueError("The multiplet specification must be an array of shape (Nx2).")

        if not multiplet[1,:].sum() == 1.0:
            raise ValueError("The multiplet line ratios should sum to one.")

        self.number_of_lines = multiplet.shape[1]
        self.multiplet = multiplet

    cpdef Spectrum add_line(self, double radiance, Point3D point, Vector3D direction, Spectrum spectrum):

        cdef double te, sigma, shifted_wavelength, component_wavelength, component_radiance
        cdef Vector3D ion_velocity

        te = self.plasma.get_electron_distribution().effective_temperature(point.x, point.y, point.z)
        if te <= 0.0:
            return spectrum

        ion_velocity = self.target_species.distribution.bulk_velocity(point.x, point.y, point.z)

        # calculate the line width
        sigma = thermal_broadening(self.wavelength, te, self.line.element.atomic_weight)

        for i in range(self.number_of_lines):

            component_wavelength = self.multiplet[0, i]
            component_radiance = radiance * self.multiplet[1, i]

            # calculate emission line central wavelength, doppler shifted along observation direction
            shifted_wavelength = doppler_shift(component_wavelength, direction, ion_velocity)

            spectrum = add_gaussian_line(component_radiance, shifted_wavelength, sigma, spectrum)

        return spectrum


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
cdef class StarkBroadenedLine(LineShapeModel):

    # Parametrised Microfield Method Stark profile coefficients.
    # Contains embedded atomic data in the form of fits to numerical models.
    # Only a limited range of lines is supported.
    # See B. Lomanowski, et al. "Inferring divertor plasma properties from hydrogen Balmer
    # and Paschen series spectroscopy in JET-ILW." Nuclear Fusion 55.12 (2015): 123028.
    STARK_MODEL_COEFFICIENTS = {
        (3, 2): (3.71e-18, 0.7665, 0.064),
        (4, 2): (8.425e-18, 0.7803, 0.050),
        (5, 2): (1.31e-15, 0.6796, 0.030),
        (6, 2): (3.954e-16, 0.7149, 0.028),
        (7, 2): (6.258e-16, 0.712, 0.029),
        (8, 2): (7.378e-16, 0.7159, 0.032),
        (9, 2): (8.947e-16, 0.7177, 0.033),
        (4, 3): (1.330e-16, 0.7449, 0.045),
        (5, 3): (6.64e-16, 0.7356, 0.044),
        (6, 3): (2.481e-15, 0.7118, 0.016),
        (7, 3): (3.270e-15, 0.7137, 0.029),
        (8, 3): (4.343e-15, 0.7133, 0.032),
        (9, 3): (5.588e-15, 0.7165, 0.033),
    }

    def __init__(self, Line line, double wavelength, Species target_species, Plasma plasma):

        if not line.element.atomic_number == 1:
            raise ValueError('Stark broadening coefficients only available for hydrogenic species.')
        try:
            # Fitted Stark Constants
            aij, bij, cij = self.STARK_MODEL_COEFFICIENTS[line.transition]
            self._aij = aij
            self._bij = bij
            self._cij = cij
        except IndexError:
            raise ValueError('Stark data for H transition {} is not currently available.'.format(line.transition))

        super().__init__(line, wavelength, target_species, plasma)

    cpdef Spectrum add_line(self, double radiance, Point3D point, Vector3D direction, Spectrum spectrum):

        cdef double ne, te, lambda_1_2, lambda_5_2, wvl
        cdef double cutoff_lower_wavelength, cutoff_upper_wavelength
        cdef double lower_value, lower_wavelength, upper_value, upper_wavelength
        cdef int start, end, i
        cdef Spectrum raw_lineshape

        ne = self.plasma.get_electron_distribution().density(point.x, point.y, point.z)
        if ne <= 0.0:
            return spectrum

        te = self.plasma.get_electron_distribution().effective_temperature(point.x, point.y, point.z)
        if te <= 0.0:
            return spectrum

        lambda_1_2 = self._cij * ne**self._aij / (te**self._bij)

        # calculate and check end of limits
        cutoff_lower_wavelength = self.wavelength - GAUSSIAN_CUTOFF_SIGMA * lambda_1_2
        if spectrum.max_wavelength < cutoff_lower_wavelength:
            return spectrum

        cutoff_upper_wavelength = self.wavelength + GAUSSIAN_CUTOFF_SIGMA * lambda_1_2
        if spectrum.min_wavelength > cutoff_upper_wavelength:
            return spectrum

        # locate range of bins where there is significant contribution from the gaussian (plus a health margin)
        start = max(0, <int> floor((cutoff_lower_wavelength - spectrum.min_wavelength) / spectrum.delta_wavelength))
        end = min(spectrum.bins, <int> ceil((cutoff_upper_wavelength - spectrum.min_wavelength) / spectrum.delta_wavelength))

        # TODO - replace with cumulative integrals
        # add line to spectrum
        raw_lineshape = spectrum.new_spectrum()

        lower_wavelength = raw_lineshape.min_wavelength + start * raw_lineshape.delta_wavelength
        lower_value = 1 / ((fabs(lower_wavelength - self.wavelength))**2.5 + (0.5*lambda_1_2)**2.5)
        for i in range(start, end):

            upper_wavelength = raw_lineshape.min_wavelength + raw_lineshape.delta_wavelength * (i + 1)
            upper_value = 1 / ((fabs(upper_wavelength - self.wavelength))**2.5 + (0.5*lambda_1_2)**2.5)

            raw_lineshape.samples_mv[i] += 0.5 * (upper_value + lower_value)

            lower_wavelength = upper_wavelength
            lower_value = upper_value

        # perform normalisation
        raw_lineshape.div_scalar(raw_lineshape.total())

        for i in range(start, end):
            # Radiance ???
            spectrum.samples_mv[i] += radiance * raw_lineshape.samples_mv[i]

        return spectrum
