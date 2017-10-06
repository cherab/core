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

from libc.math cimport sqrt, erf, M_SQRT2, floor, ceil
from cherab.core.utility.constants cimport ATOMIC_MASS, ELEMENTARY_CHARGE, SPEED_OF_LIGHT
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


cdef class Lineshape:

    cpdef Spectrum add_line(self, double radiance, double wavelength, double sigma, Spectrum spectrum, Point3D point):
        raise NotImplementedError('Child lineshape class must implement this method.')


# the number of standard deviations outside the rest wavelength the line is considered to add negligible value (including a margin for safety)
DEF GAUSSIAN_CUTOFF_SIGMA=10.0


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
cdef class GaussianLine(Lineshape):

    cpdef Spectrum add_line(self, double radiance, double wavelength, double sigma, Spectrum spectrum, Point3D point):
        """

        The formula used is based on the following definite integral:
        \frac{1}{\sigma \sqrt{2 \pi}} \int_{\lambda_0}^{\lambda_1} \exp(-\frac{(x-\mu)^2}{2\sigma^2}) dx = \frac{1}{2} \left[ -Erf(\frac{a-\mu}{\sqrt{2}\sigma}) +Erf(\frac{b-\mu}{\sqrt{2}\sigma}) \right]

        :param radiance:
        :param wavelength:
        :param sigma:
        :param spectrum:
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


# Parametrised Microfield Method Stark profile coefficients.
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


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
cdef class StarkBroadenedLine(Lineshape):

    def __init__(self, Line line, Plasma plasma):
        self._plasma = plasma
        self._line = line

        if not line.element.atomic_number == 1:
            raise ValueError('Stark broadening coefficients only available for hydrogenic species.')
        try:
            # Fitted Stark Constants
            aij, bij, cij = STARK_MODEL_COEFFICIENTS[line.transition]
            self._aij = aij
            self._bij = bij
            self._cij = cij
        except IndexError:
            raise ValueError('Stark data for H transition {} is not currently available.'.format(line.transition))

    cpdef Spectrum add_line(self, double radiance, double wavelength, double sigma, Spectrum spectrum, Point3D point):

        cdef double ne, te, lambda_1_2, lambda_5_2, wvl
        cdef double cutoff_lower_wavelength, cutoff_upper_wavelength
        cdef int start, end, i

        ne = self._plasma.get_electron_distribution().density(point.x, point.y, point.z)
        if ne <= 0.0:
            return spectrum

        te = self._plasma.get_electron_distribution().effective_temperature(point.x, point.y, point.z)
        if te <= 0.0:
            return spectrum

        if sigma <= 0:
            return spectrum

        lambda_1_2 = self._cij * ne**self._aij / (te**self._bij)
        lambda_5_2 = wavelength**2.5  # lambda^(5/2)

        # calculate and check end of limits
        cutoff_lower_wavelength = wavelength - GAUSSIAN_CUTOFF_SIGMA * lambda_1_2
        if spectrum.max_wavelength < cutoff_lower_wavelength:
            return spectrum

        cutoff_upper_wavelength = wavelength + GAUSSIAN_CUTOFF_SIGMA * lambda_1_2
        if spectrum.min_wavelength > cutoff_upper_wavelength:
            return spectrum

        # locate range of bins where there is significant contribution from the gaussian (plus a health margin)
        start = max(0, <int> floor((cutoff_lower_wavelength - spectrum.min_wavelength) / spectrum.delta_wavelength))
        end = min(spectrum.bins, <int> ceil((cutoff_upper_wavelength - spectrum.min_wavelength) / spectrum.delta_wavelength))

        # TODO - trapezium rule integration
        # add line to spectrum
        for i in range(start, end):
            wvl = spectrum.min_wavelength + spectrum.delta_wavelength * (i + 0.5)
            spectrum.samples_mv[i] += radiance / (lambda_5_2 + (0.5*lambda_1_2)**2/5) / spectrum.delta_wavelength

        return spectrum

