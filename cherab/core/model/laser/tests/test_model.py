import unittest
import numpy as np
from math import cos, sin, sqrt, radians, exp
from scipy.constants import pi, c, e, m_e, epsilon_0

from raysect.optical import World, Point3D, Vector3D, translate, Ray

from cherab.core.laser import Laser
from cherab.core.model.laser import ConstantSpectrum, SeldenMatobaThomsonSpectrum, UniformEnergyDensity

from cherab.tools.plasmas.slab import build_constant_slab_plasma

class TestScatteringModel(unittest.TestCase):
    laser_wavelength = 1040
    wavelength = np.linspace(600, 1200, 800)
    scatangle = 45

    def test_selden_matoba_scattered_spectrum(self):

        # calculate TS cross section and constants
        r_e = e ** 2 / (4 * pi * epsilon_0 * m_e * c ** 2)  # classical electron radius

        # re ** 2 is the cross section, c transforms xsection into rate constant
        scat_const = r_e ** 2 * c  

        ray_origin = Point3D(0, 0, 0)

        # angle of scattering
        observation_angle = [45, 90, 135]
        for obsangle in observation_angle:

            # pointing vector is in +z direction, angle of observation is 180 - obsangle
            z = cos((obsangle) / 180 * pi)
            x = sin((obsangle) / 180 * pi)
            ray_direction = Vector3D(x, 0, z).normalise()

            # ray spectrum properties
            min_wavelength = 600
            max_wavelength = 1200
            bins = 800

            # plasma properties
            e_density = 8e19
            t_e = [100, 1e3, 1e4]

            for vte in t_e:
                # setup scene
                world = World()
                plasma = build_constant_slab_plasma(length=1, width=1, height=1, electron_density=e_density,
                                            electron_temperature=vte, plasma_species=[], parent=world)

                # setup laser
                laser_spectrum = ConstantSpectrum(1059, 1061, 1)
                laser_profile = UniformEnergyDensity(laser_length=1, laser_radius=0.015)
                scattering_model = SeldenMatobaThomsonSpectrum()
                laser = Laser()
                laser.parent = world
                laser.transform = translate(0.05, 0, -0.5)
                laser.laser_profile = laser_profile
                laser.laser_spectrum = laser_spectrum
                laser.plasma = plasma
                laser.models = [scattering_model]

                # trace a single ray through the laser
                ray = Ray(origin=ray_origin, direction=ray_direction, min_wavelength=min_wavelength,
                          max_wavelength=max_wavelength, bins=bins)
                traced_spectrum = ray.trace(world)


                # calculate spectrum ray-tracing should deliver
                dl = 2 * laser_profile.laser_radius / sin((obsangle) / 180 * pi)  # ray-laser cross section length
                intensity_test = np.zeros_like(traced_spectrum.wavelengths)
                
                for vl in laser.laser_spectrum.wavelengths:
                    intensity_const = scat_const * e_density / vl * dl
                    for iwvl, vwvl in enumerate(traced_spectrum.wavelengths):
                        intensity_test[iwvl] += _selden_matoba_shape(vwvl, vte, obsangle, vl) * intensity_const
                
                for index, (vtest, vray) in enumerate(zip(intensity_test, traced_spectrum.samples)):
                    # skip the test for too low values of power spectral density, max is approx. 3e-5
                    if vray > 1e-30: 
                        rel_error = np.abs((vtest - vray) / vray)
                        self.assertLess(rel_error, 1e-7,
                                            msg="Traced and test spectrum value do not match: "
                                                "scattering angle = {0} deg, Te = {1} eV, wavelength = {2} nm."
                                            .format(180 - obsangle, vte, traced_spectrum.wavelengths[index]))


def _selden_matoba_shape(wavelength, te, obsangle, laser_wavelength):
    """
    Returns Selden-Matoba Spectral shape
    """
    epsilon = wavelength / laser_wavelength - 1

    alpha = m_e * c ** 2 / (2 * e * te)

    scat_angle = 180 - obsangle
    cos_scat = cos(radians(scat_angle))

    const_c = sqrt(alpha / pi) * (1 - 15 / 16 / alpha + 345 / 512 / alpha ** 2)
    const_a = (1 + epsilon) ** 3 * sqrt(2 * (1 - cos_scat) * (1 + epsilon) + epsilon ** 2)
    const_b = sqrt(1 + epsilon ** 2 / (2 * (1 - cos_scat) * (1 + epsilon))) - 1

    return const_c / const_a * exp(-2 * alpha * const_b)
