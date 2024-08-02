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

import unittest

import numpy as np
from scipy.special import erf, hyp2f1
from scipy.integrate import quad

from raysect.core import Point3D, Vector3D
from raysect.core.math.function.float import Arg1D, Constant1D
from raysect.optical import Spectrum

from cherab.core import Beam, Line, AtomicData
from cherab.core.math.integrators import GaussianQuadrature
from cherab.core.atomic import deuterium, nitrogen, ZeemanStructure
from cherab.tools.plasmas.slab import build_constant_slab_plasma
from cherab.core.model import GaussianLine, MultipletLineShape, StarkBroadenedLine, ZeemanTriplet, ParametrisedZeemanTriplet, ZeemanMultiplet
from cherab.core.model import BeamEmissionMultiplet


ATOMIC_MASS = 1.66053906660e-27
ELEMENTARY_CHARGE = 1.602176634e-19
SPEED_OF_LIGHT = 299792458.0
BOHR_MAGNETON = 5.78838180123e-5  # in eV/T
HC_EV_NM = 1239.8419738620933  # (Planck constant in eV s) x (speed of light in nm/s)


class TestLineShapes(unittest.TestCase):

    def setUp(self):

        plasma_species = [(deuterium, 0, 1.e18, 5., Vector3D(2.e4, 0, 0)),
                          (nitrogen, 1, 1.e17, 10., Vector3D(1.e4, 5.e4, 0))]
        self.plasma = build_constant_slab_plasma(length=1, width=1, height=1,
                                                 electron_density=1e19,
                                                 electron_temperature=20.,
                                                 plasma_species=plasma_species,
                                                 b_field=Vector3D(0, 5., 0))
        self.atomic_data = AtomicData()
        self.beam = Beam()
        self.beam.plasma = self.plasma
        self.beam.energy = 60000
        self.beam.temperature = 10
        self.beam.element = deuterium

    def test_gaussian_line(self):
        # setting up a line shape model
        line = Line(deuterium, 0, (3, 2))  # D-alpha line
        target_species = self.plasma.composition.get(line.element, line.charge)
        wavelength = 656.104
        gaussian_line = GaussianLine(line, wavelength, target_species, self.plasma, self.atomic_data)

        # spectrum parameters
        min_wavelength = wavelength - 0.5
        max_wavelength = wavelength + 0.5
        bins = 256
        point = Point3D(0.5, 0.5, 0.5)
        direction = Vector3D(-1, 0, 0)

        # obtaining spectrum
        radiance = 1.0
        spectrum = Spectrum(min_wavelength, max_wavelength, bins)
        spectrum = gaussian_line.add_line(radiance, point, direction, spectrum)

        # validating
        temperature = target_species.distribution.effective_temperature(point.x, point.y, point.z)
        velocity = target_species.distribution.bulk_velocity(point.x, point.y, point.z)
        shifted_wavelength = wavelength * (1 + velocity.dot(direction.normalise()) / SPEED_OF_LIGHT)
        sigma = np.sqrt(temperature * ELEMENTARY_CHARGE / (line.element.atomic_weight * ATOMIC_MASS)) * wavelength / SPEED_OF_LIGHT
        temp = 1. / (np.sqrt(2.) * sigma)

        wavelengths, delta = np.linspace(min_wavelength, max_wavelength, bins + 1, retstep=True)
        erfs = erf((wavelengths - shifted_wavelength) * temp)
        gaussian = 0.5 * radiance * (erfs[1:] - erfs[:-1]) / delta

        for i in range(bins):
            self.assertAlmostEqual(gaussian[i], spectrum.samples[i], delta=1e-10,
                                   msg='GaussianLine.add_line() method gives a wrong value at {} nm.'.format(wavelengths[i]))

    def test_multiplet_line_shape(self):
        # setting up a line shape model
        line = Line(nitrogen, 1, ("2s2 2p1 4f1 3G13.0", "2s2 2p1 3d1 3F10.0"))
        target_species = self.plasma.composition.get(line.element, line.charge)
        multiplet = [[403.509, 404.132, 404.354, 404.479, 405.692], [0.205, 0.562, 0.175, 0.029, 0.029]]
        wavelength = 404.21
        multiplet_line = MultipletLineShape(line, wavelength, target_species, self.plasma, self.atomic_data, multiplet)

        # spectrum parameters
        min_wavelength = min(multiplet[0]) - 0.5
        max_wavelength = max(multiplet[0]) + 0.5
        bins = 512
        point = Point3D(0.5, 0.5, 0.5)
        direction = Vector3D(-1, 0, 0)

        # obtaining spectrum
        radiance = 1.0
        spectrum = Spectrum(min_wavelength, max_wavelength, bins)
        spectrum = multiplet_line.add_line(radiance, point, direction, spectrum)

        # validating
        temperature = target_species.distribution.effective_temperature(point.x, point.y, point.z)
        velocity = target_species.distribution.bulk_velocity(point.x, point.y, point.z)
        sigma = np.sqrt(temperature * ELEMENTARY_CHARGE / (line.element.atomic_weight * ATOMIC_MASS)) * wavelength / SPEED_OF_LIGHT
        doppler_factor = (1 + velocity.dot(direction.normalise()) / SPEED_OF_LIGHT)
        temp = 1. / (np.sqrt(2.) * sigma)

        wavelengths, delta = np.linspace(min_wavelength, max_wavelength, bins + 1, retstep=True)
        multi_gaussian = 0
        for wvl, ratio in zip(multiplet[0], multiplet[1]):
            erfs = erf((wavelengths - wvl * doppler_factor) * temp)
            multi_gaussian += 0.5 * radiance * ratio * (erfs[1:] - erfs[:-1]) / delta

        for i in range(bins):
            self.assertAlmostEqual(multi_gaussian[i], spectrum.samples[i], delta=1e-10,
                                   msg='MultipletLineShape.add_line() method gives a wrong value at {} nm.'.format(wavelengths[i]))

    def test_zeeman_triplet(self):
        # setting up a line shape model
        line = Line(deuterium, 0, (3, 2))  # D-alpha line
        target_species = self.plasma.composition.get(line.element, line.charge)
        wavelength = 656.104
        triplet = ZeemanTriplet(line, wavelength, target_species, self.plasma, self.atomic_data)

        # spectrum parameters
        min_wavelength = wavelength - 0.5
        max_wavelength = wavelength + 0.5
        bins = 256
        point = Point3D(0.5, 0.5, 0.5)
        direction = Vector3D(-1, 1, 0) / np.sqrt(2)

        # obtaining spectrum
        radiance = 1.0
        spectrum = {}
        for pol in ('no', 'pi', 'sigma'):
            spectrum[pol] = Spectrum(min_wavelength, max_wavelength, bins)
            triplet.polarisation = pol
            spectrum[pol] = triplet.add_line(radiance, point, direction, spectrum[pol])

        # validating
        temperature = target_species.distribution.effective_temperature(point.x, point.y, point.z)
        velocity = target_species.distribution.bulk_velocity(point.x, point.y, point.z)
        sigma = np.sqrt(temperature * ELEMENTARY_CHARGE / (line.element.atomic_weight * ATOMIC_MASS)) * wavelength / SPEED_OF_LIGHT
        doppler_factor = (1 + velocity.dot(direction.normalise()) / SPEED_OF_LIGHT)
        temp = 1. / (np.sqrt(2.) * sigma)
        b_field = self.plasma.b_field(point.x, point.y, point.z)
        b_magn = b_field.length
        photon_energy = HC_EV_NM / wavelength
        wl_sigma_plus = HC_EV_NM / (photon_energy - BOHR_MAGNETON * b_magn)
        wl_sigma_minus = HC_EV_NM / (photon_energy + BOHR_MAGNETON * b_magn)
        cos_sqr = (b_field.dot(direction.normalise()) / b_magn)**2
        sin_sqr = 1. - cos_sqr

        wavelengths, delta = np.linspace(min_wavelength, max_wavelength, bins + 1, retstep=True)
        erfs = erf((wavelengths - wavelength * doppler_factor) * temp)
        gaussian_pi = 0.5 * radiance * (erfs[1:] - erfs[:-1]) / delta
        erfs = erf((wavelengths - wl_sigma_plus * doppler_factor) * temp)
        gaussian_sigma = 0.5 * radiance * (erfs[1:] - erfs[:-1]) / delta
        erfs = erf((wavelengths - wl_sigma_minus * doppler_factor) * temp)
        gaussian_sigma += 0.5 * radiance * (erfs[1:] - erfs[:-1]) / delta
        tri_gaussian = {'pi': 0.5 * sin_sqr * radiance * gaussian_pi, 'sigma': (0.25 * sin_sqr + 0.5 * cos_sqr) * radiance * gaussian_sigma}
        tri_gaussian['no'] = tri_gaussian['pi'] + tri_gaussian['sigma']

        for pol in ('no', 'pi', 'sigma'):
            for i in range(bins):
                self.assertAlmostEqual(tri_gaussian[pol][i], spectrum[pol].samples[i], delta=1e-10,
                                       msg='ZeemanTriplet.add_line() method gives a wrong value at {} nm.'.format(wavelengths[i]))

    def test_parametrised_zeeman_triplet(self):
        # setting up a line shape model
        line = Line(deuterium, 0, (3, 2))  # D-alpha line
        target_species = self.plasma.composition.get(line.element, line.charge)
        wavelength = 656.104
        triplet = ParametrisedZeemanTriplet(line, wavelength, target_species, self.plasma, self.atomic_data)

        # spectrum parameters
        min_wavelength = wavelength - 0.5
        max_wavelength = wavelength + 0.5
        bins = 256
        point = Point3D(0.5, 0.5, 0.5)
        direction = Vector3D(-1, 1, 0) / np.sqrt(2)

        # obtaining spectrum
        radiance = 1.0
        spectrum = {}
        for pol in ('no', 'pi', 'sigma'):
            spectrum[pol] = Spectrum(min_wavelength, max_wavelength, bins)
            triplet.polarisation = pol
            spectrum[pol] = triplet.add_line(radiance, point, direction, spectrum[pol])

        # validating
        alpha, beta, gamma = self.atomic_data.zeeman_triplet_parameters(line)
        temperature = target_species.distribution.effective_temperature(point.x, point.y, point.z)
        velocity = target_species.distribution.bulk_velocity(point.x, point.y, point.z)
        sigma = np.sqrt(temperature * ELEMENTARY_CHARGE / (line.element.atomic_weight * ATOMIC_MASS)) * wavelength / SPEED_OF_LIGHT
        sigma *= np.sqrt(1. + beta * beta * temperature**(2. * gamma))
        doppler_factor = (1 + velocity.dot(direction.normalise()) / SPEED_OF_LIGHT)
        temp = 1. / (np.sqrt(2.) * sigma)
        b_field = self.plasma.b_field(point.x, point.y, point.z)
        b_magn = b_field.length
        wl_sigma_plus = wavelength + 0.5 * alpha * b_magn
        wl_sigma_minus = wavelength - 0.5 * alpha * b_magn
        cos_sqr = (b_field.dot(direction.normalise()) / b_magn)**2
        sin_sqr = 1. - cos_sqr

        wavelengths, delta = np.linspace(min_wavelength, max_wavelength, bins + 1, retstep=True)
        erfs = erf((wavelengths - wavelength * doppler_factor) * temp)
        gaussian_pi = 0.5 * radiance * (erfs[1:] - erfs[:-1]) / delta
        erfs = erf((wavelengths - wl_sigma_plus * doppler_factor) * temp)
        gaussian_sigma = 0.5 * radiance * (erfs[1:] - erfs[:-1]) / delta
        erfs = erf((wavelengths - wl_sigma_minus * doppler_factor) * temp)
        gaussian_sigma += 0.5 * radiance * (erfs[1:] - erfs[:-1]) / delta
        tri_gaussian = {'pi': 0.5 * sin_sqr * radiance * gaussian_pi, 'sigma': (0.25 * sin_sqr + 0.5 * cos_sqr) * radiance * gaussian_sigma}
        tri_gaussian['no'] = tri_gaussian['pi'] + tri_gaussian['sigma']

        for pol in ('no', 'pi', 'sigma'):
            for i in range(bins):
                self.assertAlmostEqual(tri_gaussian[pol][i], spectrum[pol].samples[i], delta=1e-10,
                                       msg='ParametrisedZeemanTriplet.add_line() method gives a wrong value at {} nm.'.format(wavelengths[i]))

    def test_zeeman_multiplet(self):
        # setting up a line shape model
        line = Line(deuterium, 0, (3, 2))  # D-alpha line
        target_species = self.plasma.composition.get(line.element, line.charge)
        wavelength = 656.104
        photon_energy = HC_EV_NM / wavelength

        pi_components = [(Constant1D(wavelength), Constant1D(1.0))]
        sigma_plus_components = [(HC_EV_NM / (photon_energy - BOHR_MAGNETON * Arg1D()), Constant1D(0.5))]
        sigma_minus_components = [(HC_EV_NM / (photon_energy + BOHR_MAGNETON * Arg1D()), Constant1D(0.5))]

        zeeman_structure = ZeemanStructure(pi_components, sigma_plus_components, sigma_minus_components)
        multiplet = ZeemanMultiplet(line, wavelength, target_species, self.plasma, self.atomic_data, zeeman_structure)

        # spectrum parameters
        min_wavelength = wavelength - 0.5
        max_wavelength = wavelength + 0.5
        bins = 256
        point = Point3D(0.5, 0.5, 0.5)
        direction = Vector3D(-1, 1, 0) / np.sqrt(2)

        # obtaining spectrum
        radiance = 1.0
        spectrum = {}
        for pol in ('no', 'pi', 'sigma'):
            spectrum[pol] = Spectrum(min_wavelength, max_wavelength, bins)
            multiplet.polarisation = pol
            spectrum[pol] = multiplet.add_line(radiance, point, direction, spectrum[pol])

        # validating
        temperature = target_species.distribution.effective_temperature(point.x, point.y, point.z)
        velocity = target_species.distribution.bulk_velocity(point.x, point.y, point.z)
        sigma = np.sqrt(temperature * ELEMENTARY_CHARGE / (line.element.atomic_weight * ATOMIC_MASS)) * wavelength / SPEED_OF_LIGHT
        doppler_factor = (1 + velocity.dot(direction.normalise()) / SPEED_OF_LIGHT)
        temp = 1. / (np.sqrt(2.) * sigma)
        b_field = self.plasma.b_field(point.x, point.y, point.z)
        b_magn = b_field.length
        photon_energy = HC_EV_NM / wavelength
        wl_sigma_minus = HC_EV_NM / (photon_energy - BOHR_MAGNETON * b_magn)
        wl_sigma_plus = HC_EV_NM / (photon_energy + BOHR_MAGNETON * b_magn)
        cos_sqr = (b_field.dot(direction.normalise()) / b_magn)**2
        sin_sqr = 1. - cos_sqr

        wavelengths, delta = np.linspace(min_wavelength, max_wavelength, bins + 1, retstep=True)
        erfs = erf((wavelengths - wavelength * doppler_factor) * temp)
        gaussian_pi = 0.5 * radiance * (erfs[1:] - erfs[:-1]) / delta
        erfs = erf((wavelengths - wl_sigma_plus * doppler_factor) * temp)
        gaussian_sigma = 0.5 * radiance * (erfs[1:] - erfs[:-1]) / delta
        erfs = erf((wavelengths - wl_sigma_minus * doppler_factor) * temp)
        gaussian_sigma += 0.5 * radiance * (erfs[1:] - erfs[:-1]) / delta
        tri_gaussian = {'pi': 0.5 * sin_sqr * radiance * gaussian_pi, 'sigma': (0.25 * sin_sqr + 0.5 * cos_sqr) * radiance * gaussian_sigma}
        tri_gaussian['no'] = tri_gaussian['pi'] + tri_gaussian['sigma']

        for pol in ('no', 'pi', 'sigma'):
            for i in range(bins):
                self.assertAlmostEqual(tri_gaussian[pol][i], spectrum[pol].samples[i], delta=1e-10,
                                       msg='ZeemanMultiplet.add_line() method gives a wrong value at {} nm.'.format(wavelengths[i]))

    def test_stark_broadened_line(self):
        # setting up a line shape model
        line = Line(deuterium, 0, (6, 2))  # D-delta line
        target_species = self.plasma.composition.get(line.element, line.charge)
        wavelength = 656.104
        relative_tolerance = 1.e-8
        integrator = GaussianQuadrature(relative_tolerance=relative_tolerance)
        stark_line = StarkBroadenedLine(line, wavelength, target_species, self.plasma, self.atomic_data, integrator=integrator)

        # spectrum parameters
        min_wavelength = wavelength - 0.2
        max_wavelength = wavelength + 0.2
        bins = 512
        point = Point3D(0.5, 0.5, 0.5)
        direction = Vector3D(-1, 1, 0) / np.sqrt(2)

        # obtaining spectrum
        radiance = 1.0
        spectrum = Spectrum(min_wavelength, max_wavelength, bins)
        spectrum = stark_line.add_line(radiance, point, direction, spectrum)

        # validating
        velocity = target_species.distribution.bulk_velocity(point.x, point.y, point.z)
        doppler_factor = (1 + velocity.dot(direction.normalise()) / SPEED_OF_LIGHT)

        b_field = self.plasma.b_field(point.x, point.y, point.z)
        b_magn = b_field.length
        photon_energy = HC_EV_NM / wavelength
        wl_sigma_plus = HC_EV_NM / (photon_energy - BOHR_MAGNETON * b_magn)
        wl_sigma_minus = HC_EV_NM / (photon_energy + BOHR_MAGNETON * b_magn)
        cos_sqr = (b_field.dot(direction.normalise()) / b_magn)**2
        sin_sqr = 1. - cos_sqr

        # Gaussian parameters
        temperature = target_species.distribution.effective_temperature(point.x, point.y, point.z)
        sigma = np.sqrt(temperature * ELEMENTARY_CHARGE / (line.element.atomic_weight * ATOMIC_MASS)) * wavelength / SPEED_OF_LIGHT
        fwhm_gauss = 2 * np.sqrt(2 * np.log(2)) * sigma

        # Lorentzian parameters
        cij, aij, bij = self.atomic_data.stark_model_coefficients(line)
        ne = self.plasma.electron_distribution.density(point.x, point.y, point.z)
        te = self.plasma.electron_distribution.effective_temperature(point.x, point.y, point.z)
        fwhm_lorentz = cij * ne**aij / (te**bij)

        # Total FWHM
        if fwhm_gauss <= fwhm_lorentz:
            fwhm_poly_coeff = [1., 0, 0.57575, 0.37902, -0.42519, -0.31525, 0.31718]
            fwhm_ratio = fwhm_gauss / fwhm_lorentz
            fwhm_full = fwhm_lorentz * np.poly1d(fwhm_poly_coeff[::-1])(fwhm_ratio)
        else:
            fwhm_poly_coeff = [1., 0.15882, 1.04388, -1.38281, 0.46251, 0.82325, -0.58026]
            fwhm_ratio = fwhm_lorentz / fwhm_gauss
            fwhm_full = fwhm_gauss * np.poly1d(fwhm_poly_coeff[::-1])(fwhm_ratio)

        wavelengths, delta = np.linspace(min_wavelength, max_wavelength, bins + 1, retstep=True)

        # Gaussian part
        temp = 2 * np.sqrt(np.log(2)) / fwhm_full
        erfs = erf((wavelengths - wavelength * doppler_factor) * temp)
        gaussian = 0.25 * sin_sqr * (erfs[1:] - erfs[:-1]) / delta
        erfs = erf((wavelengths - wl_sigma_plus * doppler_factor) * temp)
        gaussian += 0.5 * (0.25 * sin_sqr + 0.5 * cos_sqr) * (erfs[1:] - erfs[:-1]) / delta
        erfs = erf((wavelengths - wl_sigma_minus * doppler_factor) * temp)
        gaussian += 0.5 * (0.25 * sin_sqr + 0.5 * cos_sqr) * (erfs[1:] - erfs[:-1]) / delta

        # Lorentzian part
        lorenzian_cutoff_gamma = 50
        stark_norm_coeff = 4 * lorenzian_cutoff_gamma * hyp2f1(0.4, 1, 1.4, -(2 * lorenzian_cutoff_gamma)**2.5)
        norm = (0.5 * fwhm_full)**1.5 / stark_norm_coeff

        def stark_lineshape_pi(x):
            return norm / ((np.abs(x - wavelength * doppler_factor))**2.5 + (0.5 * fwhm_full)**2.5)

        def stark_lineshape_sigma_plus(x):
            return norm / ((np.abs(x - wl_sigma_plus * doppler_factor))**2.5 + (0.5 * fwhm_full)**2.5)

        def stark_lineshape_sigma_minus(x):
            return norm / ((np.abs(x - wl_sigma_minus * doppler_factor))**2.5 + (0.5 * fwhm_full)**2.5)

        weight_poly_coeff = [5.14820e-04, 1.38821e+00, -9.60424e-02, -3.83995e-02, -7.40042e-03, -5.47626e-04]
        lorentz_weight = np.exp(np.poly1d(weight_poly_coeff[::-1])(np.log(fwhm_lorentz / fwhm_full)))

        for i in range(bins):
            lorentz_bin = 0.5 * sin_sqr * quad(stark_lineshape_pi, wavelengths[i], wavelengths[i + 1],
                                               epsrel=relative_tolerance)[0]
            lorentz_bin += (0.25 * sin_sqr + 0.5 * cos_sqr) * quad(stark_lineshape_sigma_plus, wavelengths[i], wavelengths[i + 1],
                                                                   epsrel=relative_tolerance)[0]
            lorentz_bin += (0.25 * sin_sqr + 0.5 * cos_sqr) * quad(stark_lineshape_sigma_minus, wavelengths[i], wavelengths[i + 1],
                                                                   epsrel=relative_tolerance)[0]
            ref_value = lorentz_bin / delta * lorentz_weight + gaussian[i] * (1. - lorentz_weight)
            if ref_value:
                self.assertAlmostEqual(spectrum.samples[i] / ref_value, 1., delta=relative_tolerance,
                                       msg='StarkBroadenedLine.add_line() method gives a wrong value at {} nm.'.format(wavelengths[i]))
            else:
                self.assertAlmostEqual(ref_value, spectrum.samples[i], delta=relative_tolerance,
                                       msg='StarkBroadenedLine.add_line() method gives a wrong value at {} nm.'.format(wavelengths[i]))

    def test_beam_emission_multiplet(self):
        # Test MSE line shape
        # setting up a line shape model
        line = Line(deuterium, 0, (3, 2))  # D-alpha line
        wavelength = 656.104
        sigma_to_pi = 0.56
        sigma1_to_sigma0 = 0.7060001671878492
        pi2_to_pi3 = 0.3140003593919741
        pi4_to_pi3 = 0.7279994935840365
        mse_line = BeamEmissionMultiplet(line, wavelength, self.beam, self.atomic_data,
                                         sigma_to_pi, sigma1_to_sigma0, pi2_to_pi3, pi4_to_pi3)

        # spectrum parameters
        min_wavelength = wavelength - 3
        max_wavelength = wavelength + 3
        bins = 512
        point = Point3D(0.5, 0.5, 0.5)
        direction = Vector3D(-1, 1, 0) / np.sqrt(2)
        beam_direction = self.beam.direction(point.x, point.y, point.z)

        # obtaining spectrum
        radiance = 1.0
        spectrum = Spectrum(min_wavelength, max_wavelength, bins)
        spectrum = mse_line.add_line(radiance, point, point, beam_direction, direction, spectrum)

        # validating

        # calculate Stark splitting
        b_field = self.plasma.b_field(point.x, point.y, point.z)
        beam_velocity = beam_direction.normalise() * np.sqrt(2 * self.beam.energy * ELEMENTARY_CHARGE / ATOMIC_MASS)
        e_field = beam_velocity.cross(b_field).length
        STARK_SPLITTING_FACTOR = 2.77e-8
        stark_split = np.abs(STARK_SPLITTING_FACTOR * e_field)

        # calculate emission line central wavelength, doppler shifted along observation direction
        central_wavelength = wavelength * (1 + beam_velocity.dot(direction.normalise()) / SPEED_OF_LIGHT)

        # calculate doppler broadening
        beam_ion_mass = self.beam.element.atomic_weight
        beam_temperature = self.beam.temperature
        sigma = np.sqrt(beam_temperature * ELEMENTARY_CHARGE / (beam_ion_mass * ATOMIC_MASS)) * wavelength / SPEED_OF_LIGHT
        temp = 1. / (np.sqrt(2.) * sigma)

        # calculate relative intensities of sigma and pi lines
        d = 1 / (1 + sigma_to_pi)
        intensity_sig = sigma_to_pi * d * radiance
        intensity_pi = 0.5 * d * radiance

        wavelengths, delta = np.linspace(min_wavelength, max_wavelength, bins + 1, retstep=True)

        # add Sigma lines to output
        intensity_s0 = 1 / (sigma1_to_sigma0 + 1)
        intensity_s1 = 0.5 * sigma1_to_sigma0 * intensity_s0

        erfs = erf((wavelengths - central_wavelength) * temp)
        test_spectrum = 0.5 * intensity_sig * intensity_s0 * (erfs[1:] - erfs[:-1]) / delta
        erfs = erf((wavelengths - central_wavelength - stark_split) * temp)
        test_spectrum += 0.5 * intensity_sig * intensity_s1 * (erfs[1:] - erfs[:-1]) / delta
        erfs = erf((wavelengths - central_wavelength + stark_split) * temp)
        test_spectrum += 0.5 * intensity_sig * intensity_s1 * (erfs[1:] - erfs[:-1]) / delta

        # add Pi lines to output
        intensity_pi3 = 1 / (1 + pi2_to_pi3 + pi4_to_pi3)
        intensity_pi2 = pi2_to_pi3 * intensity_pi3
        intensity_pi4 = pi4_to_pi3 * intensity_pi3

        erfs = erf((wavelengths - central_wavelength - 2 * stark_split) * temp)
        test_spectrum += 0.5 * intensity_pi * intensity_pi2 * (erfs[1:] - erfs[:-1]) / delta
        erfs = erf((wavelengths - central_wavelength + 2 * stark_split) * temp)
        test_spectrum += 0.5 * intensity_pi * intensity_pi2 * (erfs[1:] - erfs[:-1]) / delta
        erfs = erf((wavelengths - central_wavelength - 3 * stark_split) * temp)
        test_spectrum += 0.5 * intensity_pi * intensity_pi3 * (erfs[1:] - erfs[:-1]) / delta
        erfs = erf((wavelengths - central_wavelength + 3 * stark_split) * temp)
        test_spectrum += 0.5 * intensity_pi * intensity_pi3 * (erfs[1:] - erfs[:-1]) / delta
        erfs = erf((wavelengths - central_wavelength - 4 * stark_split) * temp)
        test_spectrum += 0.5 * intensity_pi * intensity_pi4 * (erfs[1:] - erfs[:-1]) / delta
        erfs = erf((wavelengths - central_wavelength + 4 * stark_split) * temp)
        test_spectrum += 0.5 * intensity_pi * intensity_pi4 * (erfs[1:] - erfs[:-1]) / delta

        for i in range(bins):
            self.assertAlmostEqual(test_spectrum[i], spectrum.samples[i], delta=1e-10,
                                   msg='BeamEmissionMultiplet.add_line() method gives a wrong value at {} nm.'.format(wavelengths[i]))


if __name__ == '__main__':
    unittest.main()
