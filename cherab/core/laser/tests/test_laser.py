import unittest
import numpy as np

from raysect.core import Vector3D, Point3D, translate
from raysect.optical import World, Ray
from raysect.optical.spectrum import Spectrum
from raysect.primitive import Box

from cherab.core import Plasma, Maxwellian
from cherab.core.laser.node import Laser
from cherab.core.laser.models.laserspectrum import ConstantSpectrum, GaussianSpectrum
from cherab.core.laser.scattering import SeldenMatobaThomsonSpectrum
from cherab.core.laser.models.model import UniformPowerDensity, ConstantAxisymmetricGaussian, ConstantBivariateGaussian
from cherab.core.laser.models.model import TrivariateGaussian, GaussianBeamAxisymmetric
from cherab.core.laser.models.laserspectrum_base import LaserSpectrum

from cherab.tools.plasmas.slab import build_slab_plasma

from math import exp, sqrt, cos, sin
from scipy.constants import pi, c, e, m_e, epsilon_0
from scipy.integrate import nquad


class TestLaser(unittest.TestCase):
    world = World()

    plasma = build_slab_plasma()
    plasma.parent = world

    spectrum = ConstantSpectrum(1059.5, 1060.5, 5)
    scattering = SeldenMatobaThomsonSpectrum()
    model = UniformPowerDensity()

    laser_radius = 3e-3
    laser_length = 1

    def test_laser_init(self):
        """
        Test correct initialisation of a laser instance.
        """
        laser = Laser()
        laser.parent = self.world

        self._connect_plasma(laser)

        laser.laser_spectrum = self.spectrum

        laser.scattering_model = self.scattering
        self._connect_plasma(laser)

        laser.laser_model = self.model

        laser.plasma = self.plasma

    def _connect_plasma(self, laser):
        """
        Laser can be connected to plasma only when connected to laser, scattering and spectrum models
        """

        with self.assertRaises(ValueError,
                               msg="Laser has to be connected to spectrum, scattering and laser models"
                                   " before connecting to plasma."):
            laser.plasma = self.plasma

    def test_laser_geometry(self):
        """

        :return:
        """

    def _build_laser(self):

        laser = Laser()
        laser.parent = self.world

        laser.radius = self.laser_radius
        laser.length = self.laser_length

        laser.laser_spectrum = self.spectrum
        laser.scattering_model = self.scattering
        laser.laser_model = self.model
        laser.plasma = self.plasma

        return laser

    def test_laser_segment_position(self):
        laser = self._build_laser()

        geometry = laser.get_geometry()

        # segments should have length same as 2 * radius. Test the segment boundaries
        # precalculate boundaries
        z_lower = np.zeros((len(geometry)))
        z_upper = np.zeros((len(geometry)))

        for index in range(len(geometry)):
            z_lower[index] = 2 * self.laser_radius * index
            z_upper[index] = 2 * self.laser_radius * (index + 1)

        z_upper[-1] = laser.length
        x_lower = -1 * self.laser_radius
        x_upper = self.laser_radius
        y_lower = x_lower
        y_upper = x_upper

        transform = laser.to_root()
        for index, segment in enumerate(geometry):
            bounding_box = segment.bounding_box()

            lower_boundary = bounding_box.lower.transform(transform)
            upper_boundary = bounding_box.upper.transform(transform)

            self.assertAlmostEqual(z_lower[index], lower_boundary.z, delta=1e-8,
                                   msg="Lower z boundary of the segment {0} is displaced"
                                       " from expected position.".format(index))

            self.assertAlmostEqual(z_upper[index], upper_boundary.z, delta=1e-8,
                                   msg="Upper z boundary of the segment {0} is displaced"
                                       " from expected position.".format(index))

            self.assertAlmostEqual(x_upper, upper_boundary.x, delta=1e-8,
                                   msg="Lower x boundary of the segment {0} is displaced"
                                       " from expected position.".format(index))

            self.assertAlmostEqual(x_lower, lower_boundary.x, delta=1e-8,
                                   msg="Upper x boundary of the segment {0} is displaced"
                                       " from expected position.".format(index))

            self.assertAlmostEqual(y_upper, upper_boundary.y, delta=1e-8,
                                   msg="Lower y boundary of the segment {0} is displaced"
                                       " from expected position.".format(index))

            self.assertAlmostEqual(y_lower, lower_boundary.y, delta=1e-8,
                                   msg="Upper y boundary of the segment {0} is displaced"
                                       " from expected position.".format(index))

    def test_laser_segmentmaterial_notifications(self):
        """
        Initialisation of Laser.laser_model, Laser.laser_spectrum and Laser.scattering_model
        has to propagate to all laser segments.
        """
        laser = self._build_laser()
        geometry = laser.get_geometry()

        for segment in geometry:
            self.assertIs(segment.material.laser_spectrum, self.spectrum,
                          msg="Segment reference to laser spectrum is incorrectly initialised.")
            self.assertIs(segment.material.scattering_model, self.scattering,
                          msg="Segment reference to scattering model is incorrectly initialised.")
            self.assertIs(segment.material.laser_model, self.model,
                          msg="Segment reference to laser model is incorrectly initialised.")

    def test_model_changes(self):
        """
        Changing Laser.laser_model, Laser.laser_spectrum and Laser.scattering model
        has to propagate to all laser segments.
        """
        laser = self._build_laser()
        model = UniformPowerDensity()
        spectrum = ConstantSpectrum(1200, 1230, 30)
        scattering = SeldenMatobaThomsonSpectrum()

        laser.laser_model = model
        laser.laser_spectrum = spectrum
        laser.scattering_model = scattering

        geometry = laser.get_geometry()

        for segment in geometry:
            self.assertIs(segment.material.laser_spectrum, spectrum,
                          msg="Segment reference to laser spectrum is incorrectly initialised.")
            self.assertIs(segment.material.scattering_model, scattering,
                          msg="Segment reference to scattering model is incorrectly initialised.")
            self.assertIs(segment.material.laser_model, model,
                          msg="Segment reference to laser model is incorrectly initialised.")


class TestLaserSpectrum(unittest.TestCase):

    def test_init(self):
        # init zero min_wavelength error
        with self.assertRaises(ValueError,
                               msg="LaserSpectrum did not raise a ValueError with min_wavelength being zero."):
            LaserSpectrum(0., 100, 50)

        # init negative min_wavelength error
        with self.assertRaises(ValueError,
                               msg="LaserSpectrum did not raise a ValueError with min_wavelength being negative."):
            LaserSpectrum(-1, 100, 50)

        # init zero max_wavelength error
        with self.assertRaises(ValueError,
                               msg="LaserSpectrum did not raise a ValueError with max_wavelength being zero."):
            LaserSpectrum(10, 0, 50)

        # init negative max_wavelength error
        with self.assertRaises(ValueError,
                               msg="LaserSpectrum did not raise a ValueError with max_wavelength being negative."):
            LaserSpectrum(10, -1, 50)

        # init min_wavelength being larger than max_wavelength
        with self.assertRaises(ValueError,
                               msg="LaserSpectrum did not raise a ValueError with min_wavelength < max_wavelength."):
            LaserSpectrum(30, 20, 50)

    def test_wavelength_chages(self):
        # LaserSpectrum for tests of change of properties, caching and etc.
        laser_spectrum = ConstantSpectrum(100, 200, 100)

        # change min_wavelength to be larger than max_wavelength
        with self.assertRaises(ValueError,
                               msg="LaserSpectrum did not raise ValueError for min_wavelength change "
                                   "with min_wavelength > max_wavelength."):
            laser_spectrum.min_wavelength = 300

        # change min_wavelength to be larger than max_wavelength
        with self.assertRaises(ValueError,
                               msg="LaserSpectrum did not raise ValueError for max_wavelength change "
                                   "with min_wavelength > max_wavelength."):
            laser_spectrum.max_wavelength = 50

        # laser spectrum should have same behaviour as Spectrum from raysect.optical
        spectrum = Spectrum(laser_spectrum.min_wavelength, laser_spectrum.max_wavelength, laser_spectrum.bins)

        # test caching of spectrum data, behaviour should be consistent with raysect.optical.spectrum.Spectrum
        self.assertTrue(np.array_equal(laser_spectrum.wavelengths, spectrum.wavelengths),
                        "LaserSpectrum.wavelengths values are not equal to Spectrum.wavelengths "
                        "with same boundaries and number of bins")

    def test_constantspectrum(self):
        """
        Laser spectrum should be normalized, i.e. integral from minuns inf. to inf. should be one.
        :return:
        """
        # spectrum properties
        min_wavelength = 1039.9
        max_wavelength = 1040.1
        bins = 10

        # initialise spectrum
        spectrum = ConstantSpectrum(min_wavelength, max_wavelength, bins)

        # check if the power_spectral density is normalized

        integral = spectrum.power_spectral_density.sum() * spectrum.delta_wavelength
        self.assertTrue(integral == 1, msg="Power spectral density is not normalised.")

    def test_gaussian_spectrum(self):
        """
        Laser spectrum should be normalized, i.e. integral from minuns inf. to inf. should be one.
        :return:
        """
        min_wavelength = 1035
        max_wavelength = 1045
        bins = 100
        mean = 1040
        stddev = 0.5

        spectrum = GaussianSpectrum(min_wavelength, max_wavelength, bins, mean, stddev)
        integral = nquad(spectrum, [(min_wavelength, max_wavelength)])[0]

        self.assertAlmostEqual(integral, 1., 8, msg="Power spectral density function is not normalised.")

        psd = spectrum.power_spectral_density
        integral = 0
        for index in range(0, spectrum.bins - 1):
            integral += (psd[index] + psd[index + 1]) / 2 * spectrum.delta_wavelength

        self.assertAlmostEqual(integral, 1, 8, msg="Power spectral density is not normalised.")


class TestLaserModels(unittest.TestCase):

    def test_uniform_power_density(self):
        polarisation = Vector3D(1, 3, 8).normalise()
        power_density = 2
        model = UniformPowerDensity(power_density=power_density, polarization=polarisation)

        # test polarisation
        pol_model = model.get_polarization(1, 1, 1)
        self.assertEqual(pol_model.x, polarisation.x,
                         msg="Model polarization x vector component does not agreee with input value.")
        self.assertEqual(pol_model.y, polarisation.y,
                         msg="Model polarization y vector component does not agreee with input value.")
        self.assertEqual(pol_model.z, polarisation.z,
                         msg="Model polarization z vector component does not agreee with input value.")

        # test power
        self.assertEqual(model.get_power_density(3, 4, 1), power_density,
                         msg="Model power density distribution does not agree with input.")

    def test_bivariate_gaussian(self):

        pulse_energy = 2
        pulse_length = 1e-8
        pulse_power = pulse_energy / pulse_length
        stddev_x = 0.03
        stddev_y = 0.06
        polarisation = Vector3D(2, 3, 4).normalise()
        model = ConstantBivariateGaussian(pulse_energy=pulse_energy, pulse_length=pulse_length, stddev_x=stddev_x,
                                          stddev_y=stddev_y, polarization=polarisation)

        # test polarisation
        pol_model = model.get_polarization(1, 1, 1)
        self.assertEqual(pol_model.x, polarisation.x,
                         msg="Model polarization x vector component does not agreee with input value.")
        self.assertEqual(pol_model.y, polarisation.y,
                         msg="Model polarization y vector component does not agreee with input value.")
        self.assertEqual(pol_model.z, polarisation.z,
                         msg="Model polarization z vector component does not agreee with input value.")

        # Integrate over laser volume to check energy
        xlim = [-5 * stddev_x, 5 * stddev_x]
        ylim = [-5 * stddev_y, 5 * stddev_y]
        zlim = [0, pulse_length * c]
        energy_integrated = nquad(model.get_power_density, [xlim, ylim, zlim])[0]
        self.assertTrue(np.isclose(energy_integrated / pulse_power, 1, 1e-3),
                        msg="Integrated laser energy of the model does not give results close to input energy")

        # Check laser power density profile
        x = np.linspace(-3 * stddev_x, 3 * stddev_x, 30)
        y = np.linspace(-3 * stddev_y, 3 * stddev_y, 30)
        for ix, vx in enumerate(x):
            for iy, vy in enumerate(y):
                tmp = _constant_bivariate_gaussian2d(vx, vy, pulse_energy, pulse_length, stddev_x, stddev_y)
                tmp2 = model.get_power_density(vx, vy, 0)
                self.assertTrue(np.isclose(tmp, tmp2, 1e-9),
                                msg="Model power density distribution for ({},{},{}) does not agree with input.".format(
                                    vx, vy, 0))

    def test_trivariate_gaussian(self):

        pulse_energy = 2
        pulse_length = 1e-8
        one_stddev = 0.682689492137
        pulse_power = one_stddev * pulse_energy / pulse_length
        mean_z = 0
        stddev_x = 0.03
        stddev_y = 0.06
        polarisation = Vector3D(2, 3, 4).normalise()
        model = TrivariateGaussian(pulse_energy=pulse_energy, pulse_length=pulse_length, mean_z=mean_z,
                                   stddev_x=stddev_x, stddev_y=stddev_y, polarization=polarisation)

        # test polarisation
        pol_model = model.get_polarization(1, 1, 1)
        self.assertEqual(pol_model.x, polarisation.x,
                         msg="Model polarization x vector component does not agreee with input value.")
        self.assertEqual(pol_model.y, polarisation.y,
                         msg="Model polarization y vector component does not agreee with input value.")
        self.assertEqual(pol_model.z, polarisation.z,
                         msg="Model polarization z vector component does not agreee with input value.")

        # Integrate over laser volume to check energy
        xlim = [-5 * stddev_x, 5 * stddev_x]
        ylim = [-5 * stddev_y, 5 * stddev_y]
        zlim = [mean_z - 5 * pulse_length * c, mean_z + 5 * pulse_length * c]
        energy_integrated = nquad(model.get_power_density, [xlim, ylim, zlim])[0]

        self.assertTrue(np.isclose(energy_integrated / pulse_power, 1, 1e-3),
                        msg="Integrated laser energy of the model does not give results close to input energy")

        # Check laser power density profile
        x = np.linspace(-3 * stddev_x, 3 * stddev_x, 30)
        y = np.linspace(-3 * stddev_y, 3 * stddev_y, 30)
        z = np.linspace(-3 * pulse_length * c, 3 * pulse_length * c, 30)

        for ix, vx in enumerate(x):
            for iy, vy in enumerate(y):
                for iz, vz in enumerate(z):
                    tmp = _constant_trivariate_gaussian3d(vx, vy, vz, pulse_energy, pulse_length, mean_z, stddev_x,
                                                          stddev_y)
                    tmp2 = model.get_power_density(vx, vy, vz)
                    self.assertTrue(np.isclose(tmp, tmp2, 1e-9),
                                    msg="Model power density distribution for ({},{},{})"
                                        " does not agree with input.".format(vx, vy, vz))

    def test_gaussianbeam(self):

        pulse_energy = 2
        pulse_length = 1e-9
        pulse_power = pulse_energy / pulse_length
        waist_z = 0
        stddev_waist = 0.003
        laser_wavelength = 1040
        polarisation = Vector3D(2, 3, 4).normalise()

        model = GaussianBeamAxisymmetric(pulse_energy=pulse_energy, pulse_length=pulse_length, waist_z=waist_z,
                                         stddev_waist=stddev_waist, laser_wavelength=laser_wavelength,
                                         polarization=polarisation)

        # test polarisation
        pol_model = model.get_polarization(1, 1, 1)
        self.assertEqual(pol_model.x, polarisation.x,
                         msg="Model polarization x vector component does not agreee with input value.")
        self.assertEqual(pol_model.y, polarisation.y,
                         msg="Model polarization y vector component does not agreee with input value.")
        self.assertEqual(pol_model.z, polarisation.z,
                         msg="Model polarization z vector component does not agreee with input value.")

        # Integrate over laser volume to check energy
        xlim = [-20 * stddev_waist, 20 * stddev_waist]
        zlim = [-1 * pulse_length / 2 * c, pulse_length / 2 * c]
        energy_integrated = nquad(model.get_power_density, [xlim, xlim, zlim])[0]

        self.assertTrue(np.isclose(energy_integrated / pulse_power, 1, 1e-3),
                        msg="Integrated laser energy of the model does not give results close to input energy")

        # Check laser power density profile
        x = np.linspace(-3 * stddev_waist, 3 * stddev_waist, 30)
        y = np.linspace(-3 * stddev_waist, 3 * stddev_waist, 30)
        z = np.linspace(-3 * pulse_length * c, 3 * pulse_length * c, 30)

        for ix, vx in enumerate(x):
            for iy, vy in enumerate(y):
                for iz, vz in enumerate(z):
                    tmp = _gaussian_beam_model(vx, vy, vz, pulse_energy, pulse_length, waist_z, stddev_waist,
                                               laser_wavelength * 1e-9)
                    tmp2 = model.get_power_density(vx, vy, vz)
                    self.assertTrue(np.isclose(tmp, tmp2, 1e-9),
                                    msg="Model power density distribution for ({},{},{}) "
                                        "does not agree with input.".format(vx, vy, vz))


class TestScatteringModel(unittest.TestCase):
    laser_wavelength = 1040
    wavelength = np.linspace(600, 1200, 800)
    scatangle = 45

    def test_selden_matoba_scattered_spectrum(self):

        # calculate TS cross section and constants
        r_e = e ** 2 / (4 * pi * epsilon_0 * m_e * c ** 2)  # classical electron radius
        ts_cx = 8 * pi / 3 * r_e ** 2  # Thomson scattering cross section
        scat_const = ts_cx / 4 / pi  # constant for scattered power per unit solid angle
        ray_origin = Point3D(0, 0, 0)

        # angle of scattering
        scattering_angle = [45, 90, 135]
        for scatangle in scattering_angle:

            # pointing vector is in +z direction, angle of observation is scatangle - 90
            x = cos((scatangle - 90) / 180 * pi)
            z = sin((scatangle - 90) / 180 * pi)
            ray_direction = Vector3D(x, 0, z).normalise()

            # ray spectrum properties
            min_wavelength = 600
            max_wavelength = 1200
            bins = 800

            # plasma properties
            e_density = 8e19
            zero_vector = Vector3D(0, 0, 0)
            t_e = [100, 1e3, 1e4]

            for vte in t_e:
                # setup scene
                world = World()
                plasma = self._make_plasma(e_density, vte, zero_vector, world)

                # setup laser
                laser_spectrum = ConstantSpectrum(1039.8, 1040.2, 2)
                laser_model = UniformPowerDensity()
                scattering_model = SeldenMatobaThomsonSpectrum()
                laser = Laser()
                laser.parent = world
                laser.length = 1
                laser.radius = 0.015
                laser.transform = translate(0.05, 0, -0.5)
                laser.laser_model = laser_model
                laser.scattering_model = scattering_model
                laser.laser_spectrum = laser_spectrum
                laser.plasma = plasma

                # trace a single ray through the laser
                ray = Ray(origin=ray_origin, direction=ray_direction, min_wavelength=min_wavelength,
                          max_wavelength=max_wavelength, bins=bins)
                traced_spectrum = ray.trace(world)

                # calculate spectrum ray-tracing should deliver
                intensity_test = np.zeros_like(traced_spectrum.wavelengths)

                dl = 2 * laser.radius / cos((scatangle - 90) / 180 * pi)  # ray-laser cross section length
                for il, vl in enumerate(laser.laser_spectrum.wavelengths):
                    for iwvl, vwvl in enumerate(traced_spectrum.wavelengths):
                        # normalized scattered spectrum shape
                        intensity_test[iwvl] += _selden_matoba_shape(vwvl, vte, scatangle, vl)

                        # multiply by scattered power along the ray path
                        intensity_test[iwvl] *= (1 / laser_spectrum.bins * laser_model.power_density * dl *
                                                 e_density * scat_const / traced_spectrum.delta_wavelength)

                for index, (vtest, vray) in enumerate(zip(intensity_test, traced_spectrum.samples)):
                    self.assertAlmostEqual(vtest, vray, 8,
                                           msg="Traced and test spectrum value do not match: "
                                               "scattering angle = {0} deg, Te = {1} eV, wavelength = {2} nm."
                                           .format(scattering_angle, vte, traced_spectrum.wavelengths[index]))

    def _make_plasma(self, e_density, e_temperature, velocity, parent=None):

        plasma_box = Box(Point3D(0, -0.5, -0.5), Point3D(1, 0.5, 0.5))
        plasma = Plasma(parent=parent)
        plasma.geometry = plasma_box
        e_distribution = Maxwellian(e_density, e_temperature, velocity, m_e)

        plasma.electron_distribution = e_distribution
        plasma.b_field = Vector3D(0, 0, 1)

        return plasma


def _selden_matoba_shape(wavelength, te, scatangle, laser_wavelength):
    epsilon = wavelength / laser_wavelength - 1

    alpha = m_e * c ** 2 / (2 * e * te)

    scatangle = scatangle / 180 * pi
    cos_scat = cos(scatangle)

    const_c = sqrt(alpha / pi) * (1 - 15 / 16 / alpha + 345 / 512 / alpha ** 2)
    const_a = (1 + epsilon) ** 3 * sqrt(2 * (1 - cos_scat) * (1 + epsilon) + epsilon ** 2)
    const_b = sqrt(1 + epsilon ** 2 / (2 * (1 - cos_scat) * (1 + epsilon))) - 1

    return const_c / const_a * exp(-2 * alpha * const_b)


def _gaussian_beam_model(x, y, z, pulse_energy, pulse_length, waist_z, stddev_waist, wavelength):
    laser_power_axis = pulse_energy / (pulse_length ** 2 * c)

    n = 1  # refractive index
    rayleigh_distance = 2 * pi * stddev_waist ** 2 * n / wavelength

    z_prime = z - waist_z

    stddev_z2 = stddev_waist ** 2 * (1 + (z_prime / rayleigh_distance) ** 2)

    r2 = x ** 2 + y ** 2

    return laser_power_axis / (2 * pi * stddev_z2) * exp(r2 / (-2 * stddev_z2))


def _constant_trivariate_gaussian3d(x, y, z, pulse_energy, pulse_length, mean_z, stddev_x, stddev_y):
    stddev_z = pulse_length * c
    one_stddev = 0.682689492137
    normalisation = one_stddev * pulse_energy / pulse_length
    return (normalisation / (sqrt((2 * pi) ** 3) * stddev_x * stddev_y * stddev_z) *
            exp(-1 / 2 * ((x / stddev_x) ** 2 + (y / stddev_y) ** 2 + ((z - mean_z) / stddev_z) ** 2)))


def _constant_axisymmetric_gaussian(x, y, pulse_energy, pulse_length, stddev):
    length = pulse_length * c
    normalisation = pulse_energy / length / pulse_length
    return normalisation / (stddev ** 2 * 2 * pi) * exp(-1 / 2 * ((x ** 2 + y ** 2) / stddev ** 2))


def _constant_bivariate_gaussian2d(x, y, pulse_energy, pulse_length, stddev_x, stddev_y):
    length = pulse_length * c
    normalisation = pulse_energy / length / pulse_length
    return normalisation / (stddev_x * stddev_y * 2 * pi) * exp(-1 / 2 * ((x / stddev_x) ** 2 + (y / stddev_y) ** 2))
