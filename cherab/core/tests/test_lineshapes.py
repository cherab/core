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

import unittest

import numpy as np
from scipy.special import erf, hyp2f1
from scipy.integrate import quadrature

from raysect.core import Point3D, Vector3D
from raysect.core.math.function.float import Arg1D, Constant1D
from raysect.optical import Spectrum

from cherab.core import Line
from cherab.core.math.integrators import GaussianQuadrature
from cherab.core.atomic import deuterium, nitrogen, ZeemanStructure
from cherab.tools.plasmas.slab import build_constant_slab_plasma
from cherab.core.model import GaussianLine, MultipletLineShape, StarkBroadenedLine, ZeemanTriplet, ParametrisedZeemanTriplet, ZeemanMultiplet


ATOMIC_MASS = 1.66053906660e-27
ELEMENTARY_CHARGE = 1.602176634e-19
SPEED_OF_LIGHT = 299792458.0
BOHR_MAGNETON = 5.78838180123e-5  # in eV/T
HC_EV_NM = 1239.8419738620933  # (Planck constant in eV s) x (speed of light in nm/s)


class TestLineShapes(unittest.TestCase):

    plasma_species = [(deuterium, 0, 1.e18, 5., Vector3D(2.e4, 0, 0)),
                      (nitrogen, 1, 1.e17, 10., Vector3D(1.e4, 5.e4, 0))]
    plasma = build_constant_slab_plasma(length=1, width=1, height=1, electron_density=1e19, electron_temperature=20.,
                                        plasma_species=plasma_species, b_field=Vector3D(0, 5., 0))

    def test_gaussian_line(self):
        # setting up a line shape model
        line = Line(deuterium, 0, (3, 2))  # D-alpha line
        target_species = self.plasma.composition.get(line.element, line.charge)
        wavelength = 656.104
        gaussian_line = GaussianLine(line, wavelength, target_species, self.plasma)

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
        multiplet_line = MultipletLineShape(line, wavelength, target_species, self.plasma, multiplet)

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
        triplet = ZeemanTriplet(line, wavelength, target_species, self.plasma)

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
        triplet = ParametrisedZeemanTriplet(line, wavelength, target_species, self.plasma)

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
        alpha, beta, gamma = triplet.LINE_PARAMETERS_DEFAULT[line]
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
        multiplet = ZeemanMultiplet(line, wavelength, target_species, self.plasma, zeeman_structure)

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
        integrator = GaussianQuadrature(relative_tolerance=1.e-5)
        stark_line = StarkBroadenedLine(line, wavelength, target_species, self.plasma, integrator=integrator)

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
        cij, aij, bij = stark_line.STARK_MODEL_COEFFICIENTS_DEFAULT[line]
        ne = self.plasma.electron_distribution.density(point.x, point.y, point.z)
        te = self.plasma.electron_distribution.effective_temperature(point.x, point.y, point.z)
        lambda_1_2 = cij * ne**aij / (te**bij)

        lorenzian_cutoff_gamma = 50
        stark_norm_coeff = 4 * lorenzian_cutoff_gamma * hyp2f1(0.4, 1, 1.4, -(2 * lorenzian_cutoff_gamma)**2.5)
        norm = (0.5 * lambda_1_2)**1.5 / stark_norm_coeff

        wavelengths, delta = np.linspace(min_wavelength, max_wavelength, bins + 1, retstep=True)

        def stark_lineshape(x):
            return norm / ((np.abs(x - wavelength))**2.5 + (0.5 * lambda_1_2)**2.5)

        for i in range(bins):
            stark_bin = quadrature(stark_lineshape, wavelengths[i], wavelengths[i + 1], rtol=integrator.relative_tolerance)[0] / delta
            self.assertAlmostEqual(stark_bin, spectrum.samples[i], delta=1e-9,
                                   msg='StarkBroadenedLine.add_line() method gives a wrong value at {} nm.'.format(wavelengths[i]))


if __name__ == '__main__':
    unittest.main()
