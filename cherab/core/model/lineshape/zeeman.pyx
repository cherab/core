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

from cherab.core.atomic cimport Line, AtomicData
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
    :param AtomicData atomic_data: The atomic data provider.
    :param str polarisation: Leaves only :math:`\pi`-/:math:`\sigma`-polarised components:
                             "pi" - leave only :math:`\pi`-polarised components,
                             "sigma" - leave only :math:`\sigma`-polarised components,
                             "no" - leave all components (default).
    :param Integrator1D integrator: Integrator1D instance to integrate the line shape
                                    over the spectral bin. Default is None.
    """

    def __init__(self, Line line, double wavelength, Species target_species, Plasma plasma, AtomicData atomic_data,
                 polarisation='no', Integrator1D integrator=None):
        super().__init__(line, wavelength, target_species, plasma, atomic_data, integrator)

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
    :param AtomicData atomic_data: The atomic data provider.
    :param str polarisation: Leaves only :math:`\pi`-/:math:`\sigma`-polarised components:
                             "pi" - leave central component,
                             "sigma" - leave side components,
                             "no" - all components (default).
    """

    def __init__(self, Line line, double wavelength, Species target_species, Plasma plasma, AtomicData atomic_data, polarisation='no'):

        super().__init__(line, wavelength, target_species, plasma, atomic_data, polarisation)

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

    For details see A. Blom and C. Jupén, Parametrisation of the Zeeman effect
    for hydrogen-like spectra in high-temperature plasmas,
    Plasma Phys. Control. Fusion 44 (2002) `1229-1241
    <https://doi.org/10.1088/0741-3335/44/7/312>`_.

    :param Line line: The emission line object for this line shape.
    :param float wavelength: The rest wavelength for this emission line.
    :param Species target_species: The target plasma species that is emitting.
    :param Plasma plasma: The emitting plasma object.
    :param AtomicData atomic_data: The atomic data provider.
    :param tuple line_parameters: Parameters of the model in the form (alpha, beta, gamma).
                                  Default is None (will use `atomic_data.zeeman_triplet_parameters`).
    :param str polarisation: Leaves only :math:`\pi`-/:math:`\sigma`-polarised components:
                             "pi" - leave central component,
                             "sigma" - leave side components,
                             "no" - all components (default).
    """

    def __init__(self, Line line, double wavelength, Species target_species, Plasma plasma, AtomicData atomic_data,
                 tuple line_parameters=None, polarisation='no'):

        super().__init__(line, wavelength, target_species, plasma, atomic_data, polarisation)

        try:
            alpha, beta, gamma = line_parameters or self.atomic_data.zeeman_triplet_parameters(line)
            if alpha <= 0:
                raise ValueError('Parameter alpha must be positive.')
            if beta < 0:
                raise ValueError('Parameter beta must be non-negative.')
            self._alpha = alpha
            self._beta = beta
            self._gamma = gamma

        except KeyError:
            raise ValueError('Data for {} is not available.'.format(self.line))

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
    :param AtomicData atomic_data: The atomic data provider.
    :param zeeman_structure: A ``ZeemanStructure`` object that provides wavelengths and ratios
                             of :math:`\pi`-/:math:`\sigma^{+}`-/:math:`\sigma^{-}`-polarised
                             components for any given magnetic field strength.
                             Default is None (will use atomic_data.zeeman_structure).
    :param str polarisation: Leaves only :math:`\pi`-/:math:`\sigma`-polarised components:
                             "pi" - leave only :math:`\pi`-polarised components,
                             "sigma" - leave only :math:`\sigma`-polarised components,
                             "no" - leave all components (default).

    """

    def __init__(self, Line line, double wavelength, Species target_species, Plasma plasma, AtomicData atomic_data,
                 ZeemanStructure zeeman_structure=None, polarisation='no'):

        super().__init__(line, wavelength, target_species, plasma, atomic_data, polarisation)

        self._zeeman_structure = zeeman_structure or self.atomic_data.zeeman_structure(line)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @cython.cdivision(True)
    cpdef Spectrum add_line(self, double radiance, Point3D point, Vector3D direction, Spectrum spectrum):

        cdef int i
        cdef double ts, sigma, shifted_wavelength, component_radiance, b_magn, cos_sqr, sin_sqr
        cdef Vector3D ion_velocity, b_field
        cdef double[:, :] multiplet_mv

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
