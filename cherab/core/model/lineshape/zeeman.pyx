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

from libc.math cimport sqrt
from raysect.optical cimport Spectrum, Point3D, Vector3D

from cherab.core.atomic cimport Line
from cherab.core.species cimport Species
from cherab.core.plasma cimport Plasma
from cherab.core.atomic.elements import hydrogen, deuterium, tritium, helium, helium3, beryllium, boron, carbon, nitrogen, oxygen, neon
from cherab.core.math.integrators cimport Integrator1D
from cherab.core.utility.constants cimport BOHR_MAGNETON, HC_EV_NM
from cherab.core.model.lineshape.doppler cimport doppler_shift, thermal_broadening
from cherab.core.model.lineshape.gaussian cimport add_gaussian_line

cimport cython


DEF MULTIPLET_WAVELENGTH = 0
DEF MULTIPLET_RATIO = 1

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
    :param Integrator1D integrator: Integrator1D instance to integrate the line shape
                                    over the spectral bin. Default is None.
    """

    def __init__(self, Line line, double wavelength, Species target_species, Plasma plasma, polarisation,
                 Integrator1D integrator=None):
        super().__init__(line, wavelength, target_species, plasma, integrator)

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

    @classmethod
    def show_supported_transitions(cls):
        """ Prints all supported transitions."""
        for line, param in cls.LINE_PARAMETERS_DEFAULT.items():
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
