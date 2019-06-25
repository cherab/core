import unittest
from cherab.core.atomic import neon, hydrogen, helium
from cherab.core.math import Interpolate1DCubic
from cherab.openadas import OpenADAS
from cherab.tools.plasmas.ionisationbalance import (_fractional_abundance, fractional_abundance,
                                                    interpolators1d_fractional,_from_elementdensity, from_elementdensity,
                                                    _match_plasma_neutrality, interpolators1d_from_elementdensity,
                                                    interpolators1d_match_plasma_neutrality)

import matplotlib.pyplot as plt
import matplotlib._color_data as mcd
import numpy as np


def doubleparabola(r, Centre, Edge, p, q):
    return (Centre - Edge) * np.power((1 - np.power((r - r.min()) / (r.max() - r.min()), p)), q) + Edge


def normal(x, mu, sd, height=1, offset=0):
    return height * np.exp(-1 * np.power(x - mu, 2) / (2 * sd ** 2)) + offset


def get_electron_density_profile(abundances):
    n_e = np.zeros((abundances[0].shape[1]))
    for abundance in abundances:
        for rownumber, row in enumerate(abundance.T):
            n_e[rownumber] += np.sum(row * np.arange(row.shape[0]))

    return n_e


def get_electron_density_spot(densities):
    n_e = 0
    for spec in densities:
        for index, value in enumerate(spec):
            n_e += index * value

    return n_e


class TestIonizationBalance1D(unittest.TestCase):

    # create plasma profiles and interpolators
    psin_1d = np.linspace(0, 1.1, 50, endpoint=True)
    psin_1d_detailed = np.linspace(0, 1.1, 450, endpoint=True)

    t_e_profile = doubleparabola(psin_1d, 5000, 10, 2, 2)
    n_e_profile = doubleparabola(psin_1d, 6e19, 5e18, 2, 2)

    t_element_profile = doubleparabola(psin_1d, 1500, 40, 2, 2)
    n_element_profile = doubleparabola(psin_1d, 1e17, 1e17, 2, 2) + normal(psin_1d, 0.9, 0.1, 5e17)
    n_element2_profile = doubleparabola(psin_1d, 5e17, 1e17, 2, 2)

    t_e = Interpolate1DCubic(psin_1d, t_e_profile)
    n_e = Interpolate1DCubic(psin_1d, n_e_profile)

    t_element = Interpolate1DCubic(psin_1d, t_element_profile)
    n_element = Interpolate1DCubic(psin_1d, n_element_profile)
    n_element2 = Interpolate1DCubic(psin_1d, n_element2_profile)

    # denser psi array to test interpolators
    n_e_profile_detailed = np.zeros_like(psin_1d_detailed)
    for index, value in enumerate(psin_1d_detailed):
        n_e_profile_detailed[index] = n_e(value)

    # define psi for single-point tests
    psi_value = 0.9

    # load adas atomic database and define elements
    adas = OpenADAS(permit_extrapolation=True)

    element = neon
    element2 = helium
    element_bulk = hydrogen

    tcx_donor = hydrogen

    n_tcx_donor = 3e16

    TOLERANCE = 1e-3

    def sumup_fractions(self, fractions):

        total = 0
        if isinstance(fractions, dict):
            iterator =  fractions.items()
        elif isinstance(fractions, np.ndarray):
            iterator = enumerate(fractions)

        for index, value in iterator:
            total += value

        return total

    def test_0d_from_0d(self):
        """
        test fractional abundance calculation with float numbers as inputs
        :return:
        """
        abundance_fractional = fractional_abundance(self.adas, self.element, self.n_e(self.psi_value), self.t_e(self.psi_value))

        fraction_sum = self.sumup_fractions(abundance_fractional)
        self.assertTrue(1 - fraction_sum < self.TOLERANCE)

    def test_0d_from_0d_tcx(self):
        """
        test fractional abundance calculation with thermal cx and float numbers as inputs
        :return:
        """
        abundance_fractional = fractional_abundance(self.adas, self.element, self.n_e(self.psi_value), self.t_e(self.psi_value),
                                                    self.tcx_donor, self.n_tcx_donor, 0)

        fraction_sum = self.sumup_fractions(abundance_fractional)
        self.assertTrue(1 - fraction_sum < self.TOLERANCE)

