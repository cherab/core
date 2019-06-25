import unittest
from collections.abc import Iterable

import numpy as np
from cherab.core.atomic import neon, hydrogen, helium
from cherab.core.math import Interpolate1DCubic
from cherab.openadas import OpenADAS
from cherab.tools.plasmas.ionisationbalance import (fractional_abundance, from_elementdensity, match_plasma_neutrality,
                                                    interpolators1d_fractional, interpolators1d_from_elementdensity,
                                                    interpolators1d_match_plasma_neutrality)


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


def exp_decay(r, lamb, max_val):
    return max_val * np.exp((r - r.max()) * lamb)


class TestIonizationBalance1D(unittest.TestCase):
    # create plasma profiles and interpolators
    psin_1d = np.linspace(0, 1.1, 50, endpoint=True)
    psin_1d_detailed = np.linspace(0, 1.1, 450, endpoint=True)

    t_e_profile = doubleparabola(psin_1d, 5000, 10, 2, 2)
    n_e_profile = doubleparabola(psin_1d, 6e19, 5e18, 2, 2)

    t_element_profile = doubleparabola(psin_1d, 1500, 40, 2, 2)
    n_element_profile = doubleparabola(psin_1d, 1e17, 1e17, 2, 2) + normal(psin_1d, 0.9, 0.1, 5e17)
    n_element2_profile = doubleparabola(psin_1d, 5e17, 1e17, 2, 2)

    n_tcx_donor_profile = exp_decay(psin_1d, 10, 3e16)

    t_e = Interpolate1DCubic(psin_1d, t_e_profile)
    n_e = Interpolate1DCubic(psin_1d, n_e_profile)

    t_element = Interpolate1DCubic(psin_1d, t_element_profile)
    n_element = Interpolate1DCubic(psin_1d, n_element_profile)
    n_element2 = Interpolate1DCubic(psin_1d, n_element2_profile)

    n_tcx_donor = Interpolate1DCubic(psin_1d, n_tcx_donor_profile)

    # denser psi array to test interpolators
    n_e_profile_detailed = np.zeros_like(psin_1d_detailed)
    for index, value in enumerate(psin_1d_detailed):
        n_e_profile_detailed[index] = n_e(value)

    # define psi for single-point tests
    psi_value = 0.9

    # load adas atomic database and define elements
    atomic_data = OpenADAS(permit_extrapolation=True)

    element = neon
    element2 = helium
    element_bulk = hydrogen

    tcx_donor = hydrogen

    TOLERANCE = 1e-3

    def sumup_fractions(self, fractions):

        if isinstance(fractions, dict):
            if isinstance(fractions[0], Iterable):
                total = np.zeros_like(fractions[0])
            else:
                total = 0

            for index, values in fractions.items():
                total += values

        elif isinstance(fractions, np.ndarray):
            total = np.zeros_like(fractions[0, ...])

            for index in np.ndindex(fractions.shape):
                total[index] += fractions[index]

        return total

    def sumup_electrons(self, densities):
        if isinstance(densities, dict):
            if isinstance(densities[0], Iterable):
                total = np.zeros_like(densities[0])
            else:
                total = 0

            for index, values in densities.items():
                total += values * index

        elif isinstance(densities, np.ndarray):
            total = np.zeros_like(densities[0, ...])

            for index in np.ndindex(densities.shape):
                total[index] += densities[index] * index[0]

        return total

    def evaluate_interpolators(self, interpolators, free_variable):

        profiles = {}
        for key, item in interpolators.items():
            profiles[key] = np.zeros_like(free_variable)
            for index in np.ndindex(*free_variable.shape):
                profiles[key][index] = item(free_variable[index])

        return profiles

    def test_fractional_0d_from_0d(self):
        """
        test fractional abundance calculation with float numbers as inputs
        :return:
        """
        abundance_fractional = fractional_abundance(self.atomic_data, self.element, self.n_e(self.psi_value),
                                                    self.t_e(self.psi_value))

        fraction_sum = self.sumup_fractions(abundance_fractional)
        self.assertTrue(1 - fraction_sum < self.TOLERANCE)

    def test_fractional_0d_from_0d_tcx(self):
        """
        test fractional abundance calculation with thermal cx and float numbers as inputs
        :return:
        """
        abundance_fractional = fractional_abundance(self.atomic_data, self.element, self.n_e(self.psi_value),
                                                    self.t_e(self.psi_value),
                                                    self.tcx_donor, self.n_tcx_donor(self.psi_value), 0)

        fraction_sum = self.sumup_fractions(abundance_fractional)
        self.assertTrue(1 - fraction_sum < self.TOLERANCE)

    def test_fractional_0d_from_interpolators(self):
        """
        test interpolators and free_variable as inputs
        :return:
        """

        abundance_fractional = fractional_abundance(self.atomic_data, self.element, self.n_e, self.t_e,
                                                    free_variable=self.psi_value)

        fraction_sum = self.sumup_fractions(abundance_fractional)
        self.assertTrue(1 - fraction_sum < self.TOLERANCE)

    def test_fractional_0d_from_interpolators_tcx(self):
        """
        test interpolators and free_variable as inputs
        :return:
        """

        abundance_fractional = fractional_abundance(self.atomic_data, self.element, self.n_e, self.t_e,
                                                    tcx_donor=self.tcx_donor, tcx_donor_n=self.n_tcx_donor,
                                                    tcx_donor_charge=0, free_variable=self.psi_value)

        fraction_sum = self.sumup_fractions(abundance_fractional)
        self.assertTrue(1 - fraction_sum < self.TOLERANCE)

    def test_fractional_0d_from_mixed(self):
        """
        test mixed types of inputs
        :return:
        """

        abundance_fractional = fractional_abundance(self.atomic_data, self.element, self.n_e(self.psi_value), self.t_e,
                                                    free_variable=self.psi_value)

        fraction_sum = self.sumup_fractions(abundance_fractional)
        self.assertTrue(1 - fraction_sum < self.TOLERANCE)

        abundance_fractional = fractional_abundance(self.atomic_data, self.element, self.n_e, self.t_e(self.psi_value),
                                                    free_variable=self.psi_value)

        fraction_sum = self.sumup_fractions(abundance_fractional)
        self.assertTrue(1 - fraction_sum < self.TOLERANCE)

    def test_fractional_1d_from_1d(self):
        """
        test calculation of 1d fractional profiles with 1d iterables as inputs
        :return:
        """
        abundance_fractional = fractional_abundance(self.atomic_data, self.element, self.n_e_profile, self.t_e_profile)

        fraction_sum = self.sumup_fractions(abundance_fractional)
        self.assertTrue(np.allclose(fraction_sum, 1, atol=self.TOLERANCE))

    def test_fractional_1d_from_1d_tcx(self):
        """
        test calculation of 1d fractional profiles with 1d iterables as inputs
        :return:
        """

        abundance_fractional = fractional_abundance(self.atomic_data, self.element, self.n_e_profile, self.t_e_profile,
                                                    tcx_donor=self.tcx_donor, tcx_donor_n=self.n_tcx_donor_profile,
                                                    tcx_donor_charge=0)

        fraction_sum = self.sumup_fractions(abundance_fractional)
        self.assertTrue(np.allclose(fraction_sum, 1, atol=self.TOLERANCE))

    def test_fractional_1d_from_interpolators(self):
        """
        test calculation of 1d fractional profiles with 1d interpolators as inputs
        :return:
        """
        abundance_fractional = fractional_abundance(self.atomic_data, self.element, self.n_e, self.t_e,
                                                    free_variable=self.psin_1d)

        fraction_sum = self.sumup_fractions(abundance_fractional)
        self.assertTrue(np.allclose(fraction_sum, 1, atol=self.TOLERANCE))

    def test_fractional_1d_from_interpolators_tcx(self):
        """
        test calculation of 1d fractional profiles with 1d iterables as inputs
        :return:
        """

        abundance_fractional = fractional_abundance(self.atomic_data, self.element, self.n_e, self.t_e,
                                                    tcx_donor=self.tcx_donor, tcx_donor_n=self.n_tcx_donor,
                                                    tcx_donor_charge=0, free_variable=self.psin_1d)

        fraction_sum = self.sumup_fractions(abundance_fractional)
        self.assertTrue(np.allclose(fraction_sum, 1, atol=self.TOLERANCE))

    def test_fractional_from_mixed(self):
        """
        test calculation of 1d fractional profiles with mixed types as inputs
        :return:
        """

        abundance_fractional = fractional_abundance(self.atomic_data, self.element, self.n_e_profile, self.t_e,
                                                    free_variable=self.psin_1d)

        fraction_sum = self.sumup_fractions(abundance_fractional)
        self.assertTrue(np.allclose(fraction_sum, 1, atol=self.TOLERANCE))

        abundance_fractional = fractional_abundance(self.atomic_data, self.element, self.n_e, self.t_e_profile,
                                                    free_variable=self.psin_1d)

        fraction_sum = self.sumup_fractions(abundance_fractional)
        self.assertTrue(np.allclose(fraction_sum, 1, atol=self.TOLERANCE))

    def test_fractional_from_mixed_tcx(self):
        """
        test calculation of 1d fractional profiles with mixed types as inputs
        :return:
        """

        abundance_fractional = fractional_abundance(self.atomic_data, self.element, self.n_e_profile, self.t_e,
                                                    tcx_donor=self.tcx_donor, tcx_donor_n=self.n_tcx_donor,
                                                    tcx_donor_charge=0, free_variable=self.psin_1d)

        fraction_sum = self.sumup_fractions(abundance_fractional)
        self.assertTrue(np.allclose(fraction_sum, 1, atol=self.TOLERANCE))

        abundance_fractional = fractional_abundance(self.atomic_data, self.element, self.n_e, self.t_e_profile,
                                                    tcx_donor=self.tcx_donor, tcx_donor_n=self.n_tcx_donor,
                                                    tcx_donor_charge=0, free_variable=self.psin_1d)

        fraction_sum = self.sumup_fractions(abundance_fractional)
        self.assertTrue(np.allclose(fraction_sum, 1, atol=self.TOLERANCE))

        abundance_fractional = fractional_abundance(self.atomic_data, self.element, self.n_e, self.t_e_profile,
                                                    tcx_donor=self.tcx_donor, tcx_donor_n=self.n_tcx_donor_profile,
                                                    tcx_donor_charge=0, free_variable=self.psin_1d)

        fraction_sum = self.sumup_fractions(abundance_fractional)
        self.assertTrue(np.allclose(fraction_sum, 1, atol=self.TOLERANCE))

    def test_fractional_inetrpolators_1d(self):
        """
        test calculation of 1d fractional interpolators
        :return:
        """

        interpolators_fractional = interpolators1d_fractional(self.atomic_data, self.element, self.psin_1d,
                                                              self.n_e_profile, self.t_e)

        profiles = self.evaluate_interpolators(interpolators_fractional, self.psin_1d)
        fraction_sum = self.sumup_fractions(profiles)

        self.assertTrue(np.allclose(fraction_sum, 1, atol=self.TOLERANCE))

    def test_fractional_inetrpolators_tcx(self):
        """
        test calculation of 1d fractional interpolators with thermal cx
        :return:
        """

        interpolators_fractional = interpolators1d_fractional(self.atomic_data, self.element, self.psin_1d,
                                                              self.n_e_profile, self.t_e, tcx_donor=self.tcx_donor,
                                                              tcx_donor_n=self.n_tcx_donor_profile,
                                                              tcx_donor_charge=0)

        profiles = self.evaluate_interpolators(interpolators_fractional, self.psin_1d)
        fraction_sum = self.sumup_fractions(profiles)

        self.assertTrue(np.allclose(fraction_sum, 1, atol=self.TOLERANCE))

    def test_balance_0d_elementdensity(self):
        """
        test calculation of ionization balance
        :return:
        """
        # test with floats as input
        densities = from_elementdensity(self.atomic_data, self.element, self.n_element(self.psi_value),
                                        self.n_e(self.psi_value), self.t_e(self.psi_value))
        total = self.sumup_fractions(densities)
        self.assertTrue(np.isclose(total, self.n_element(self.psi_value), rtol=self.TOLERANCE))

        # test with interpolators
        densities = from_elementdensity(self.atomic_data, self.element, self.n_element(self.psi_value),
                                        self.n_e(self.psi_value), self.t_e(self.psi_value))
        total = self.sumup_fractions(densities)
        self.assertTrue(np.isclose(total, self.n_element(self.psi_value), rtol=self.TOLERANCE))

        # test with mixed parameters
        densities = from_elementdensity(self.atomic_data, self.element, self.n_element,
                                        self.n_e(self.psi_value), self.t_e(self.psi_value),
                                        free_variable=self.psi_value)
        total = self.sumup_fractions(densities)
        self.assertTrue(np.isclose(total, self.n_element(self.psi_value), rtol=self.TOLERANCE))

    def test_balance_0d_plasma_neutrality(self):
        """test matching of plasma neutrality"""

        densities_1 = from_elementdensity(self.atomic_data, self.element, self.n_element,
                                          self.n_e(self.psi_value), self.t_e(self.psi_value),
                                          free_variable=self.psi_value)

        densities_2 = from_elementdensity(self.atomic_data, self.element2, self.n_element2,
                                          self.n_e(self.psi_value), self.t_e(self.psi_value),
                                          free_variable=self.psi_value)

        densities_3 = match_plasma_neutrality(self.atomic_data, self.element_bulk, [densities_1, densities_2],
                                              self.n_e(self.psi_value), self.t_e(self.psi_value))

        total = self.sumup_electrons(densities_1)
        total += self.sumup_electrons(densities_2)
        total += self.sumup_electrons(densities_3)

        self.assertTrue(np.isclose(total, self.n_e(self.psi_value), rtol=self.TOLERANCE))

    def test_balance_0d_plasma_neutrality_tcx(self):
        """test matching of plasma neutrality"""

        densities_1 = from_elementdensity(self.atomic_data, self.element, self.n_element,
                                          self.n_e(self.psi_value), self.t_e(self.psi_value),
                                          free_variable=self.psi_value,
                                          tcx_donor=self.tcx_donor, tcx_donor_n=self.n_tcx_donor, tcx_donor_charge=0,
                                          )

        densities_2 = from_elementdensity(self.atomic_data, self.element2, self.n_element2,
                                          self.n_e(self.psi_value), self.t_e(self.psi_value),
                                          free_variable=self.psi_value,
                                          tcx_donor=self.tcx_donor, tcx_donor_n=self.n_tcx_donor, tcx_donor_charge=0,
                                          )

        densities_3 = match_plasma_neutrality(self.atomic_data, self.element_bulk, [densities_1, densities_2],
                                              self.n_e(self.psi_value), self.t_e(self.psi_value))

        total = self.sumup_electrons(densities_1)
        total += self.sumup_electrons(densities_2)
        total += self.sumup_electrons(densities_3)

        self.assertTrue(np.isclose(total, self.n_e(self.psi_value), rtol=self.TOLERANCE))

    def test_balance_1d_plasma_neutrality(self):
        """test matching of plasma neutrality for 1d profiles"""

        densities_1 = from_elementdensity(self.atomic_data, self.element, self.n_element,
                                          self.n_e, self.t_e_profile,
                                          free_variable=self.psin_1d)

        densities_2 = from_elementdensity(self.atomic_data, self.element2, self.n_element2,
                                          self.n_e_profile, self.t_e,
                                          free_variable=self.psin_1d)

        densities_3 = match_plasma_neutrality(self.atomic_data, self.element_bulk, [densities_1, densities_2],
                                              self.n_e, self.t_e_profile,
                                              free_variable=self.psin_1d)

        total = self.sumup_electrons(densities_1)
        total += self.sumup_electrons(densities_2)
        total += self.sumup_electrons(densities_3)

        self.assertTrue(np.allclose(total, self.n_e_profile, rtol=self.TOLERANCE))

    def test_balance_1d_plasma_neutrality_tcx(self):
        """test matching of plasma neutrality for 1d profiles with thermal cx"""

        densities_1 = from_elementdensity(self.atomic_data, self.element, self.n_element,
                                          self.n_e, self.t_e_profile,
                                          free_variable=self.psin_1d,
                                          tcx_donor=self.tcx_donor, tcx_donor_n=self.n_tcx_donor, tcx_donor_charge=0
                                          )

        densities_2 = from_elementdensity(self.atomic_data, self.element2, self.n_element2,
                                          self.n_e_profile, self.t_e,
                                          free_variable=self.psin_1d,
                                          tcx_donor=self.tcx_donor, tcx_donor_n=self.n_tcx_donor, tcx_donor_charge=0)

        densities_3 = match_plasma_neutrality(self.atomic_data, self.element_bulk, [densities_1, densities_2],
                                              self.n_e, self.t_e_profile,
                                              free_variable=self.psin_1d,
                                              tcx_donor=self.tcx_donor, tcx_donor_n=self.n_tcx_donor,
                                              tcx_donor_charge=0)

        total = self.sumup_electrons(densities_1)
        total += self.sumup_electrons(densities_2)
        total += self.sumup_electrons(densities_3)

        self.assertTrue(np.allclose(total, self.n_e_profile, rtol=self.TOLERANCE))

    def test_balance_1d_interpolators_from_element_density(self):
        """
        test calculation of 1d interpolators of charge stage densities
        :return:
        """

        interpolators_abundance = interpolators1d_from_elementdensity(self.atomic_data, self.element, self.psin_1d,
                                                                      self.n_element,
                                                                      self.n_e_profile, self.t_e)

        profiles = self.evaluate_interpolators(interpolators_abundance, self.psin_1d)
        fraction_sum = self.sumup_fractions(profiles)

        self.assertTrue(np.allclose(fraction_sum, self.n_element_profile, rtol=self.TOLERANCE))

    def test_balance_1d_interpolators_from_element_density_tcx(self):
        """
        test calculation of 1d interpolators of ion charge state densities
        :return:
        """

        interpolators_abundance = interpolators1d_from_elementdensity(self.atomic_data, self.element, self.psin_1d,
                                                                      self.n_element,
                                                                      self.n_e_profile, self.t_e,
                                                                      tcx_donor=self.tcx_donor,
                                                                      tcx_donor_n=self.n_tcx_donor, tcx_donor_charge=0)

        profiles = self.evaluate_interpolators(interpolators_abundance, self.psin_1d)
        fraction_sum = self.sumup_fractions(profiles)

        self.assertTrue(np.allclose(fraction_sum, self.n_element_profile, rtol=self.TOLERANCE))

    def test_balance_1d_interpolators_plasma_neutrality(self):
        """
        test calulation of 1d interpolators for ion charge state densities using plasma neutrality condition.
        :return:
        """

        interpolators_abundance_1 = interpolators1d_from_elementdensity(self.atomic_data, self.element, self.psin_1d,
                                                                        self.n_element,
                                                                        self.n_e_profile, self.t_e)

        interpolators_abundance_2 = interpolators1d_from_elementdensity(self.atomic_data, self.element2, self.psin_1d,
                                                                        self.n_element2,
                                                                        self.n_e_profile, self.t_e)

        interpolators_abundance_3 = interpolators1d_match_plasma_neutrality(self.atomic_data, self.element,
                                                                            self.psin_1d,
                                                                            [interpolators_abundance_1,
                                                                             interpolators_abundance_2],
                                                                            self.n_e_profile, self.t_e)

        profiles1 = self.evaluate_interpolators(interpolators_abundance_1, self.psin_1d)
        profiles2 = self.evaluate_interpolators(interpolators_abundance_2, self.psin_1d)
        profiles3 = self.evaluate_interpolators(interpolators_abundance_3, self.psin_1d)

        total = self.sumup_electrons(profiles1)
        total += self.sumup_electrons(profiles2)
        total += self.sumup_electrons(profiles3)

        self.assertTrue(np.allclose(total, self.n_e_profile, rtol=self.TOLERANCE))

    def test_balance_1d_interpolators_plasma_neutrality_tcx(self):
        """
        test calulation of 1d interpolators for ion charge state densities using plasma neutrality condition.
        :return:
        """

        interpolators_abundance_1 = interpolators1d_from_elementdensity(self.atomic_data, self.element, self.psin_1d,
                                                                        self.n_element,
                                                                        self.n_e_profile, self.t_e,
                                                                        tcx_donor=self.tcx_donor,
                                                                        tcx_donor_n=self.n_tcx_donor,
                                                                        tcx_donor_charge=0)

        interpolators_abundance_2 = interpolators1d_from_elementdensity(self.atomic_data, self.element2, self.psin_1d,
                                                                        self.n_element2,
                                                                        self.n_e_profile, self.t_e,
                                                                        tcx_donor=self.tcx_donor,
                                                                        tcx_donor_n=self.n_tcx_donor,
                                                                        tcx_donor_charge=0)

        interpolators_abundance_3 = interpolators1d_match_plasma_neutrality(self.atomic_data, self.element,
                                                                            self.psin_1d,
                                                                            [interpolators_abundance_1,
                                                                             interpolators_abundance_2],
                                                                            self.n_e_profile, self.t_e,
                                                                            tcx_donor=self.tcx_donor,
                                                                            tcx_donor_n=self.n_tcx_donor,
                                                                            tcx_donor_charge=0)

        profiles1 = self.evaluate_interpolators(interpolators_abundance_1, self.psin_1d)
        profiles2 = self.evaluate_interpolators(interpolators_abundance_2, self.psin_1d)
        profiles3 = self.evaluate_interpolators(interpolators_abundance_3, self.psin_1d)

        total = self.sumup_electrons(profiles1)
        total += self.sumup_electrons(profiles2)
        total += self.sumup_electrons(profiles3)

        self.assertTrue(np.allclose(total, self.n_e_profile, rtol=self.TOLERANCE))
