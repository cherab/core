import unittest
import numpy as np
from math import exp, sqrt

from raysect.core import Vector3D

from cherab.core.model.laser.profile import UniformEnergyDensity, ConstantBivariateGaussian
from cherab.core.model.laser.profile import TrivariateGaussian, GaussianBeamAxisymmetric, generate_segmented_cylinder

from scipy.integrate import nquad
from scipy.constants import c, pi

class TestSegmentedCylinder(unittest.TestCase):

    def test_number_of_primitives(self):

        # for r > l there should be 1 cylinder segment only
        primitives = generate_segmented_cylinder(radius=1, length=0.5)
        self.assertEqual(len(primitives), 1, msg="Wrong nuber of laser segments, expected 1.")

        # for r < l tehre should be length // (2 * radius) segments
        primitives = generate_segmented_cylinder(radius=0.5, length=10)
        self.assertEqual(len(primitives), 10, msg="Wrong nuber of laser segments, expected 20.")


class TestLaserProfile(unittest.TestCase):

    def test_uniform_energy_density(self):
        polarisation = Vector3D(1, 3, 8).normalise()
        energy_density = 2
        model = UniformEnergyDensity(energy_density=energy_density, polarization=polarisation)

        # test polarisation
        pol_model = model.get_polarization(1, 1, 1)
        self.assertEqual(pol_model.x, polarisation.x,
                         msg="Model polarization x vector component does not agreee with input value.")
        self.assertEqual(pol_model.y, polarisation.y,
                         msg="Model polarization y vector component does not agreee with input value.")
        self.assertEqual(pol_model.z, polarisation.z,
                         msg="Model polarization z vector component does not agreee with input value.")

        # test power
        self.assertEqual(model.get_energy_density(3, 4, 1), energy_density,
                         msg="Model power density distribution does not agree with input.")

    def test_bivariate_gaussian(self):

        pulse_energy = 2
        pulse_length = 1e-8
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
        energy_integrated = nquad(model.get_energy_density, [xlim, ylim, zlim])[0]
        self.assertTrue(np.isclose(energy_integrated / pulse_energy, 1, 1e-3),
                        msg="Integrated laser energy of the model does not give results close to input energy")

        # Check laser power density profile
        x = np.linspace(-3 * stddev_x, 3 * stddev_x, 30)
        y = np.linspace(-3 * stddev_y, 3 * stddev_y, 30)
        for vx in x:
            for vy in y:
                tmp = _constant_bivariate_gaussian2d(vx, vy, pulse_energy, pulse_length, stddev_x, stddev_y)
                tmp2 = model.get_energy_density(vx, vy, 0)
                self.assertTrue(np.isclose(tmp, tmp2, 1e-9),
                                msg="Model power density distribution for ({},{},{}) does not agree with input.".format(
                                    vx, vy, 0))

    def test_trivariate_gaussian(self):

        pulse_energy = 2
        pulse_length = 1e-8
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
        energy_integrated = nquad(model.get_energy_density, [xlim, ylim, zlim])[0]

        self.assertTrue(np.isclose(energy_integrated / pulse_energy, 1, 1e-3),
                        msg="Integrated laser energy of the model does not give results close to input energy")

        # Check laser power density profile
        x = np.linspace(-3 * stddev_x, 3 * stddev_x, 30)
        y = np.linspace(-3 * stddev_y, 3 * stddev_y, 30)
        z = np.linspace(-3 * pulse_length * c, 3 * pulse_length * c, 30)

        for vx in x:
            for vy in y:
                for vz in z:
                    tmp = _constant_trivariate_gaussian3d(vx, vy, vz, pulse_energy, pulse_length, mean_z, stddev_x,
                                                          stddev_y)
                    tmp2 = model.get_energy_density(vx, vy, vz)
                    self.assertTrue(np.isclose(tmp, tmp2, 1e-9),
                                    msg="Model power density distribution for ({},{},{})"
                                        " does not agree with input.".format(vx, vy, vz))

    def test_gaussianbeam(self):

        pulse_energy = 2
        pulse_length = 1e-9
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
        energy_integrated = nquad(model.get_energy_density, [xlim, xlim, zlim])[0]

        self.assertTrue(np.isclose(energy_integrated / pulse_energy, 1, 1e-3),
                        msg="Integrated laser energy of the model does not give results close to input energy")

        # Check laser power density profile
        x = np.linspace(-3 * stddev_waist, 3 * stddev_waist, 30)
        y = np.linspace(-3 * stddev_waist, 3 * stddev_waist, 30)
        z = np.linspace(-3 * pulse_length * c, 3 * pulse_length * c, 30)

        for vx in x:
            for vy in y:
                for vz in z:
                    tmp = _gaussian_beam_model(vx, vy, vz, pulse_energy, pulse_length, waist_z, stddev_waist,
                                               laser_wavelength * 1e-9)
                    tmp2 = model.get_energy_density(vx, vy, vz)
                    self.assertTrue(np.isclose(tmp, tmp2, 1e-9),
                                    msg="Model power density distribution for ({},{},{}) "
                                        "does not agree with input.".format(vx, vy, vz))


def _gaussian_beam_model(x, y, z, pulse_energy, pulse_length, waist_z, stddev_waist, wavelength):
    laser_power_axis = pulse_energy / (pulse_length * c)

    n = 1  # refractive index
    rayleigh_distance = 2 * pi * stddev_waist ** 2 * n / wavelength

    z_prime = z - waist_z

    stddev_z2 = stddev_waist ** 2 * (1 + (z_prime / rayleigh_distance) ** 2)

    r2 = x ** 2 + y ** 2

    return laser_power_axis / (2 * pi * stddev_z2) * exp(r2 / (-2 * stddev_z2))


def _constant_trivariate_gaussian3d(x, y, z, pulse_energy, pulse_length, mean_z, stddev_x, stddev_y):
    stddev_z = pulse_length * c
    return (pulse_energy / (sqrt((2 * pi) ** 3) * stddev_x * stddev_y * stddev_z) *
            exp(-1 / 2 * ((x / stddev_x) ** 2 + (y / stddev_y) ** 2 + ((z - mean_z) / stddev_z) ** 2)))


def _constant_bivariate_gaussian2d(x, y, pulse_energy, pulse_length, stddev_x, stddev_y):
    length = pulse_length * c
    normalisation = pulse_energy / length
    return normalisation / (stddev_x * stddev_y * 2 * pi) * exp(-1 / 2 * ((x / stddev_x) ** 2 + (y / stddev_y) ** 2))
