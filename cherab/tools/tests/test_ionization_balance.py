#
# # Copyright 2016-2018 Euratom
# # Copyright 2016-2018 United Kingdom Atomic Energy Authority
# # Copyright 2016-2018 Centro de Investigaciones Energéticas, Medioambientales y Tecnológicas
# #
# # Licensed under the EUPL, Version 1.1 or – as soon they will be approved by the
# # European Commission - subsequent versions of the EUPL (the "Licence");
# # You may not use this work except in compliance with the Licence.
# # You may obtain a copy of the Licence at:
# #
# # https://joinup.ec.europa.eu/software/page/eupl5
# #
# # Unless required by applicable law or agreed to in writing, software distributed
# # under the Licence is distributed on an "AS IS" basis, WITHOUT WARRANTIES OR
# # CONDITIONS OF ANY KIND, either express or implied.
# #
# # See the Licence for the specific language governing permissions and limitations
# # under the Licence.
#
#
# import unittest
# from collections.abc import Iterable
# import numpy as np
#
# from cherab.core.atomic import neon, hydrogen, helium
# from cherab.core.math import Interpolate1DCubic, Interpolate2DCubic, Function1D, Function2D, AxisymmetricMapper
# from cherab.openadas import OpenADAS
# from cherab.tools.plasmas.ionisation_balance import (fractional_abundance, from_elementdensity, match_plasma_neutrality,
#                                                      interpolators1d_fractional, interpolators1d_from_elementdensity,
#                                                      interpolators1d_match_plasma_neutrality,
#                                                      interpolators2d_fractional, interpolators2d_from_elementdensity,
#                                                      interpolators2d_match_plasma_neutrality,
#                                                      abundance_axisymmetric_mapper,
#                                                      equilibrium_map3d_fractional, equilibrium_map3d_from_elementdensity,
#                                                      equilibrium_map3d_match_plasma_neutrality)
#
# from cherab.tools.equilibrium import example_equilibrium
#
#
# def double_parabola(r, centre, edge, p, q):
#     return (centre - edge) * np.power((1 - np.power((r - r.min()) / (r.max() - r.min()), p)), q) + edge
#
#
# def normal(x, mu, sd, height=1, offset=0):
#     return height * np.exp(-1 * np.power(x - mu, 2) / (2 * sd ** 2)) + offset
#
#
# def get_electron_density_profile(abundances):
#     n_e = np.zeros((abundances[0].shape[1]))
#     for abundance in abundances:
#         for rownumber, row in enumerate(abundance.T):
#             n_e[rownumber] += np.sum(row * np.arange(row.shape[0]))
#
#     return n_e
#
#
# def get_electron_density_spot(densities):
#     n_e = 0
#     for spec in densities:
#         for index, value in enumerate(spec):
#             n_e += index * value
#
#     return n_e
#
#
# def exp_decay(r, lamb, max_val):
#     return max_val * np.exp((r - r.max()) * lamb)
#
#
# class TestIonizationBalance1D(unittest.TestCase):
#
#     # create plasma profiles and interpolators
#     # 1d profiles
#     psin_1d = np.linspace(0, 1.1, 15, endpoint=True)
#     psin_1d_detailed = np.linspace(0, 1.1, 50, endpoint=True)
#
#     t_e_profile_1d = double_parabola(psin_1d, 5000, 10, 2, 2)
#     n_e_profile_1d = double_parabola(psin_1d, 6e19, 5e18, 2, 2)
#
#     t_element_profile_1d = double_parabola(psin_1d, 1500, 40, 2, 2)
#     n_element_profile_1d = double_parabola(psin_1d, 1e17, 1e17, 2, 2) + normal(psin_1d, 0.9, 0.1, 5e17)
#     n_element2_profile_1d = double_parabola(psin_1d, 5e17, 1e17, 2, 2)
#
#     n_tcx_donor_profile_1d = exp_decay(psin_1d, 10, 3e16)
#
#     t_e_1d = Interpolate1DCubic(psin_1d, t_e_profile_1d)
#     n_e_1d = Interpolate1DCubic(psin_1d, n_e_profile_1d)
#
#     t_element_1d = Interpolate1DCubic(psin_1d, t_element_profile_1d)
#     n_element_1d = Interpolate1DCubic(psin_1d, n_element_profile_1d)
#     n_element2_1d = Interpolate1DCubic(psin_1d, n_element2_profile_1d)
#
#     n_tcx_donor_1d = Interpolate1DCubic(psin_1d, n_tcx_donor_profile_1d)
#
#     # denser psi array to test interpolators
#     n_e_profile_detailed_1d = np.zeros_like(psin_1d_detailed)
#     for index, value in enumerate(psin_1d_detailed):
#         n_e_profile_detailed_1d[index] = n_e_1d(value)
#
#     # 2d profiles
#
#     r = np.linspace(-1, 1, 6)
#     z = np.linspace(-1, 1, 8)
#
#     psin_2d = np.zeros((*r.shape, *z.shape))
#
#     for index0, value0 in enumerate(r):
#         for index1, value1 in enumerate(z):
#             if np.sqrt(value0 ** 2 + value1 ** 2) < psin_1d.max():
#                 psin_2d[index0, index1] = np.sqrt(value0 ** 2 + value1 ** 2)
#             else:
#                 psin_2d[index0, index1] = psin_1d.max()
#
#     t_e_profile_2d = np.zeros_like(psin_2d)
#     for index in np.ndindex(*t_e_profile_2d.shape):
#         t_e_profile_2d[index] = t_e_1d(psin_2d[index])
#
#     t_e_2d = Interpolate2DCubic(r, z, t_e_profile_2d)
#
#     n_e_profile_2d = np.zeros_like(psin_2d)
#     for index in np.ndindex(*n_e_profile_2d.shape):
#         n_e_profile_2d[index] = n_e_1d(psin_2d[index])
#
#     n_e_2d = Interpolate2DCubic(r, z, n_e_profile_2d)
#
#     t_element_profile_2d = np.zeros_like(psin_2d)
#     for index in np.ndindex(*t_element_profile_2d.shape):
#         t_element_profile_2d[index] = t_element_1d(psin_2d[index])
#
#     t_element_2d = Interpolate2DCubic(r, z, t_element_profile_2d)
#
#     n_element_profile_2d = np.zeros_like(psin_2d)
#     for index in np.ndindex(*n_element_profile_2d.shape):
#         n_element_profile_2d[index] = n_element_1d(psin_2d[index])
#
#     n_element_2d = Interpolate2DCubic(r, z, n_element_profile_2d)
#
#     n_element2_profile_2d = np.zeros_like(psin_2d)
#     for index in np.ndindex(*n_element2_profile_2d.shape):
#         n_element2_profile_2d[index] = n_element2_1d(psin_2d[index])
#
#     n_element2_2d = Interpolate2DCubic(r, z, n_element2_profile_2d)
#
#     n_tcx_donor_profile_2d = np.zeros_like(psin_2d)
#     for index in np.ndindex(*n_tcx_donor_profile_2d.shape):
#         n_tcx_donor_profile_2d[index] = n_tcx_donor_1d(psin_2d[index])
#
#     n_tcx_donor_2d = Interpolate2DCubic(r, z, n_element_profile_2d)
#
#     # define psi for single-point tests
#     psi_value = 0.9
#
#     # load adas atomic database and define elements
#     atomic_data = OpenADAS(permit_extrapolation=True)
#
#     element = neon
#     element2 = helium
#     element_bulk = hydrogen
#
#     tcx_donor = hydrogen
#
#     TOLERANCE = 1e-3
#
#     def sumup_fractions(self, fractions):
#
#         if isinstance(fractions, dict):
#             if isinstance(fractions[0], Iterable):
#                 total = np.zeros_like(fractions[0])
#             else:
#                 total = 0
#
#             for index, values in fractions.items():
#                 total += values
#
#         elif isinstance(fractions, np.ndarray):
#             total = np.zeros_like(fractions[0, ...])
#
#             for index in np.ndindex(fractions.shape):
#                 total[index] += fractions[index]
#
#         return total
#
#     def sumup_electrons(self, densities):
#         if isinstance(densities, dict):
#             if isinstance(densities[0], Iterable):
#                 total = np.zeros_like(densities[0])
#             else:
#                 total = 0
#
#             for index, values in densities.items():
#                 total += values * index
#
#         elif isinstance(densities, np.ndarray):
#             total = np.zeros_like(densities[0, ...])
#
#             for index in np.ndindex(densities.shape):
#                 total[index] += densities[index] * index[0]
#
#         return total
#
#     def evaluate_interpolators(self, interpolators, free_variable):
#
#         profiles = {}
#         for key, item in interpolators.items():
#             if isinstance(item, Function1D):
#                 profiles[key] = np.zeros_like(free_variable)
#                 for index in np.ndindex(*free_variable.shape):
#                     profiles[key][index] = item(free_variable[index])
#             elif isinstance(item, Function2D):
#                 profiles[key] = np.zeros((*free_variable[0].shape, *free_variable[1].shape))
#                 for index0, value0 in enumerate(free_variable[0]):
#                     for index1, value1 in enumerate(free_variable[1]):
#                         profiles[key][index0, index1] = item(value0, value1)
#             elif isinstance(item, AxisymmetricMapper):
#                 profiles[key] = np.zeros_like(free_variable)
#                 for index, value in enumerate(free_variable):
#                         profiles[key][index] = item(value, 0, 0)
#
#         return profiles
#
#     def test_fractional_0d_from_0d(self):
#         """
#         test fractional abundance calculation with float numbers as inputs
#         """
#         abundance_fractional = fractional_abundance(self.atomic_data, self.element, self.n_e_1d(self.psi_value),
#                                                     self.t_e_1d(self.psi_value))
#
#         fraction_sum = self.sumup_fractions(abundance_fractional)
#         self.assertTrue(1 - fraction_sum < self.TOLERANCE)
#
#     def test_fractional_0d_from_0d_tcx(self):
#         """
#         test fractional abundance calculation with thermal cx and float numbers as inputs
#         """
#         abundance_fractional = fractional_abundance(self.atomic_data, self.element, self.n_e_1d(self.psi_value),
#                                                     self.t_e_1d(self.psi_value),
#                                                     self.tcx_donor, self.n_tcx_donor_1d(self.psi_value), 0)
#
#         fraction_sum = self.sumup_fractions(abundance_fractional)
#         self.assertTrue(1 - fraction_sum < self.TOLERANCE)
#
#     def test_fractional_0d_from_interpolators(self):
#         """
#         test interpolators and free_variable as inputs
#         """
#
#         abundance_fractional = fractional_abundance(self.atomic_data, self.element, self.n_e_1d, self.t_e_1d,
#                                                     free_variable=self.psi_value)
#
#         fraction_sum = self.sumup_fractions(abundance_fractional)
#         self.assertTrue(1 - fraction_sum < self.TOLERANCE)
#
#     def test_fractional_0d_from_interpolators_tcx(self):
#         """
#         test interpolators and free_variable as inputs
#         """
#
#         abundance_fractional = fractional_abundance(self.atomic_data, self.element, self.n_e_1d, self.t_e_1d,
#                                                     tcx_donor=self.tcx_donor, tcx_donor_n=self.n_tcx_donor_1d,
#                                                     tcx_donor_charge=0, free_variable=self.psi_value)
#
#         fraction_sum = self.sumup_fractions(abundance_fractional)
#         self.assertTrue(1 - fraction_sum < self.TOLERANCE)
#
#     def test_fractional_0d_from_mixed(self):
#         """
#         test mixed types of inputs
#         """
#
#         abundance_fractional = fractional_abundance(self.atomic_data, self.element, self.n_e_1d(self.psi_value),
#                                                     self.t_e_1d,
#                                                     free_variable=self.psi_value)
#
#         fraction_sum = self.sumup_fractions(abundance_fractional)
#         self.assertTrue(1 - fraction_sum < self.TOLERANCE)
#
#         abundance_fractional = fractional_abundance(self.atomic_data, self.element, self.n_e_1d,
#                                                     self.t_e_1d(self.psi_value),
#                                                     free_variable=self.psi_value)
#
#         fraction_sum = self.sumup_fractions(abundance_fractional)
#         self.assertTrue(1 - fraction_sum < self.TOLERANCE)
#
#     def test_fractional_1d_from_1d(self):
#         """
#         test calculation of 1d fractional profiles with 1d iterables as inputs
#         """
#         abundance_fractional = fractional_abundance(self.atomic_data, self.element, self.n_e_profile_1d,
#                                                     self.t_e_profile_1d)
#
#         fraction_sum = self.sumup_fractions(abundance_fractional)
#         self.assertTrue(np.allclose(fraction_sum, 1, atol=self.TOLERANCE))
#
#     def test_fractional_1d_from_1d_tcx(self):
#         """
#         test calculation of 1d fractional profiles with 1d iterables as inputs
#         """
#
#         abundance_fractional = fractional_abundance(self.atomic_data, self.element, self.n_e_profile_1d,
#                                                     self.t_e_profile_1d,
#                                                     tcx_donor=self.tcx_donor, tcx_donor_n=self.n_tcx_donor_profile_1d,
#                                                     tcx_donor_charge=0)
#
#         fraction_sum = self.sumup_fractions(abundance_fractional)
#         self.assertTrue(np.allclose(fraction_sum, 1, atol=self.TOLERANCE))
#
#     def test_fractional_1d_from_interpolators(self):
#         """
#         test calculation of 1d fractional profiles with 1d interpolators as inputs
#         """
#         abundance_fractional = fractional_abundance(self.atomic_data, self.element, self.n_e_1d, self.t_e_1d,
#                                                     free_variable=self.psin_1d)
#
#         fraction_sum = self.sumup_fractions(abundance_fractional)
#         self.assertTrue(np.allclose(fraction_sum, 1, atol=self.TOLERANCE))
#
#     def test_fractional_1d_from_interpolators_tcx(self):
#         """
#         test calculation of 1d fractional profiles with 1d iterables as inputs
#         """
#
#         abundance_fractional = fractional_abundance(self.atomic_data, self.element, self.n_e_1d, self.t_e_1d,
#                                                     tcx_donor=self.tcx_donor, tcx_donor_n=self.n_tcx_donor_1d,
#                                                     tcx_donor_charge=0, free_variable=self.psin_1d)
#
#         fraction_sum = self.sumup_fractions(abundance_fractional)
#         self.assertTrue(np.allclose(fraction_sum, 1, atol=self.TOLERANCE))
#
#     def test_fractional_from_mixed(self):
#         """
#         test calculation of 1d fractional profiles with mixed types as inputs
#         """
#
#         abundance_fractional = fractional_abundance(self.atomic_data, self.element, self.n_e_profile_1d, self.t_e_1d,
#                                                     free_variable=self.psin_1d)
#
#         fraction_sum = self.sumup_fractions(abundance_fractional)
#         self.assertTrue(np.allclose(fraction_sum, 1, atol=self.TOLERANCE))
#
#         abundance_fractional = fractional_abundance(self.atomic_data, self.element, self.n_e_1d, self.t_e_profile_1d,
#                                                     free_variable=self.psin_1d)
#
#         fraction_sum = self.sumup_fractions(abundance_fractional)
#         self.assertTrue(np.allclose(fraction_sum, 1, atol=self.TOLERANCE))
#
#     def test_fractional_from_mixed_tcx(self):
#         """
#         test calculation of 1d fractional profiles with mixed types as inputs
#         """
#
#         abundance_fractional = fractional_abundance(self.atomic_data, self.element, self.n_e_profile_1d, self.t_e_1d,
#                                                     tcx_donor=self.tcx_donor, tcx_donor_n=self.n_tcx_donor_1d,
#                                                     tcx_donor_charge=0, free_variable=self.psin_1d)
#
#         fraction_sum = self.sumup_fractions(abundance_fractional)
#         self.assertTrue(np.allclose(fraction_sum, 1, atol=self.TOLERANCE))
#
#         abundance_fractional = fractional_abundance(self.atomic_data, self.element, self.n_e_1d, self.t_e_profile_1d,
#                                                     tcx_donor=self.tcx_donor, tcx_donor_n=self.n_tcx_donor_1d,
#                                                     tcx_donor_charge=0, free_variable=self.psin_1d)
#
#         fraction_sum = self.sumup_fractions(abundance_fractional)
#         self.assertTrue(np.allclose(fraction_sum, 1, atol=self.TOLERANCE))
#
#         abundance_fractional = fractional_abundance(self.atomic_data, self.element, self.n_e_1d, self.t_e_profile_1d,
#                                                     tcx_donor=self.tcx_donor, tcx_donor_n=self.n_tcx_donor_profile_1d,
#                                                     tcx_donor_charge=0, free_variable=self.psin_1d)
#
#         fraction_sum = self.sumup_fractions(abundance_fractional)
#         self.assertTrue(np.allclose(fraction_sum, 1, atol=self.TOLERANCE))
#
#     def test_fractional_inetrpolators_1d(self):
#         """
#         test calculation of 1d fractional interpolators
#         """
#
#         interpolators_fractional = interpolators1d_fractional(self.atomic_data, self.element, self.psin_1d,
#                                                               self.n_e_profile_1d, self.t_e_1d)
#
#         profiles = self.evaluate_interpolators(interpolators_fractional, self.psin_1d)
#         fraction_sum = self.sumup_fractions(profiles)
#
#         self.assertTrue(np.allclose(fraction_sum, 1, atol=self.TOLERANCE))
#
#     def test_fractional_inetrpolators_1d_tcx(self):
#         """
#         test calculation of 1d fractional interpolators with thermal cx
#         """
#
#         interpolators_fractional = interpolators1d_fractional(self.atomic_data, self.element, self.psin_1d,
#                                                               self.n_e_profile_1d, self.t_e_1d,
#                                                               tcx_donor=self.tcx_donor,
#                                                               tcx_donor_n=self.n_tcx_donor_profile_1d,
#                                                               tcx_donor_charge=0)
#
#         profiles = self.evaluate_interpolators(interpolators_fractional, self.psin_1d)
#         fraction_sum = self.sumup_fractions(profiles)
#
#         self.assertTrue(np.allclose(fraction_sum, 1, atol=self.TOLERANCE))
#
#     def test_balance_0d_elementdensity(self):
#         """
#         test calculation of ionization balance
#         """
#         # test with floats as input
#         densities = from_elementdensity(self.atomic_data, self.element, self.n_element_1d(self.psi_value),
#                                         self.n_e_1d(self.psi_value), self.t_e_1d(self.psi_value))
#         total = self.sumup_fractions(densities)
#         self.assertTrue(np.isclose(total, self.n_element_1d(self.psi_value), rtol=self.TOLERANCE))
#
#         # test with interpolators
#         densities = from_elementdensity(self.atomic_data, self.element, self.n_element_1d(self.psi_value),
#                                         self.n_e_1d(self.psi_value), self.t_e_1d(self.psi_value))
#         total = self.sumup_fractions(densities)
#         self.assertTrue(np.isclose(total, self.n_element_1d(self.psi_value), rtol=self.TOLERANCE))
#
#         # test with mixed parameters
#         densities = from_elementdensity(self.atomic_data, self.element, self.n_element_1d,
#                                         self.n_e_1d(self.psi_value), self.t_e_1d(self.psi_value),
#                                         free_variable=self.psi_value)
#         total = self.sumup_fractions(densities)
#         self.assertTrue(np.isclose(total, self.n_element_1d(self.psi_value), rtol=self.TOLERANCE))
#
#     def test_balance_0d_plasma_neutrality(self):
#         """test matching of plasma neutrality"""
#
#         densities_1 = from_elementdensity(self.atomic_data, self.element, self.n_element_1d,
#                                           self.n_e_1d(self.psi_value), self.t_e_1d(self.psi_value),
#                                           free_variable=self.psi_value)
#
#         densities_2 = from_elementdensity(self.atomic_data, self.element2, self.n_element2_1d,
#                                           self.n_e_1d(self.psi_value), self.t_e_1d(self.psi_value),
#                                           free_variable=self.psi_value)
#
#         densities_3 = match_plasma_neutrality(self.atomic_data, self.element_bulk, [densities_1, densities_2],
#                                               self.n_e_1d(self.psi_value), self.t_e_1d(self.psi_value))
#
#         total = self.sumup_electrons(densities_1)
#         total += self.sumup_electrons(densities_2)
#         total += self.sumup_electrons(densities_3)
#
#         self.assertTrue(np.isclose(total, self.n_e_1d(self.psi_value), rtol=self.TOLERANCE))
#
#     def test_balance_0d_plasma_neutrality_tcx(self):
#         """test matching of plasma neutrality"""
#
#         densities_1 = from_elementdensity(self.atomic_data, self.element, self.n_element_1d,
#                                           self.n_e_1d(self.psi_value), self.t_e_1d(self.psi_value),
#                                           free_variable=self.psi_value,
#                                           tcx_donor=self.tcx_donor, tcx_donor_n=self.n_tcx_donor_1d, tcx_donor_charge=0,
#                                           )
#
#         densities_2 = from_elementdensity(self.atomic_data, self.element2, self.n_element2_1d,
#                                           self.n_e_1d(self.psi_value), self.t_e_1d(self.psi_value),
#                                           free_variable=self.psi_value,
#                                           tcx_donor=self.tcx_donor, tcx_donor_n=self.n_tcx_donor_1d, tcx_donor_charge=0,
#                                           )
#
#         densities_3 = match_plasma_neutrality(self.atomic_data, self.element_bulk, [densities_1, densities_2],
#                                               self.n_e_1d(self.psi_value), self.t_e_1d(self.psi_value))
#
#         total = self.sumup_electrons(densities_1)
#         total += self.sumup_electrons(densities_2)
#         total += self.sumup_electrons(densities_3)
#
#         self.assertTrue(np.isclose(total, self.n_e_1d(self.psi_value), rtol=self.TOLERANCE))
#
#     def test_balance_1d_plasma_neutrality(self):
#         """test matching of plasma neutrality for 1d profiles"""
#
#         densities_1 = from_elementdensity(self.atomic_data, self.element, self.n_element_1d,
#                                           self.n_e_1d, self.t_e_profile_1d,
#                                           free_variable=self.psin_1d)
#
#         densities_2 = from_elementdensity(self.atomic_data, self.element2, self.n_element2_1d,
#                                           self.n_e_profile_1d, self.t_e_1d,
#                                           free_variable=self.psin_1d)
#
#         densities_3 = match_plasma_neutrality(self.atomic_data, self.element_bulk, [densities_1, densities_2],
#                                               self.n_e_1d, self.t_e_profile_1d,
#                                               free_variable=self.psin_1d)
#
#         total = self.sumup_electrons(densities_1)
#         total += self.sumup_electrons(densities_2)
#         total += self.sumup_electrons(densities_3)
#
#         self.assertTrue(np.allclose(total, self.n_e_profile_1d, rtol=self.TOLERANCE))
#
#     def test_balance_1d_plasma_neutrality_tcx(self):
#         """test matching of plasma neutrality for 1d profiles with thermal cx"""
#
#         densities_1 = from_elementdensity(self.atomic_data, self.element, self.n_element_1d,
#                                           self.n_e_1d, self.t_e_profile_1d,
#                                           free_variable=self.psin_1d,
#                                           tcx_donor=self.tcx_donor, tcx_donor_n=self.n_tcx_donor_1d, tcx_donor_charge=0
#                                           )
#
#         densities_2 = from_elementdensity(self.atomic_data, self.element2, self.n_element2_1d,
#                                           self.n_e_profile_1d, self.t_e_1d,
#                                           free_variable=self.psin_1d,
#                                           tcx_donor=self.tcx_donor, tcx_donor_n=self.n_tcx_donor_1d, tcx_donor_charge=0)
#
#         densities_3 = match_plasma_neutrality(self.atomic_data, self.element_bulk, [densities_1, densities_2],
#                                               self.n_e_1d, self.t_e_profile_1d,
#                                               free_variable=self.psin_1d,
#                                               tcx_donor=self.tcx_donor, tcx_donor_n=self.n_tcx_donor_1d,
#                                               tcx_donor_charge=0)
#
#         total = self.sumup_electrons(densities_1)
#         total += self.sumup_electrons(densities_2)
#         total += self.sumup_electrons(densities_3)
#
#         self.assertTrue(np.allclose(total, self.n_e_profile_1d, rtol=self.TOLERANCE))
#
#     def test_balance_1d_interpolators_from_element_density(self):
#         """
#         test calculation of 1d interpolators of charge stage densities
#         """
#
#         interpolators_abundance = interpolators1d_from_elementdensity(self.atomic_data, self.element, self.psin_1d,
#                                                                       self.n_element_1d,
#                                                                       self.n_e_profile_1d, self.t_e_1d)
#
#         profiles = self.evaluate_interpolators(interpolators_abundance, self.psin_1d)
#         fraction_sum = self.sumup_fractions(profiles)
#
#         self.assertTrue(np.allclose(fraction_sum, self.n_element_profile_1d, rtol=self.TOLERANCE))
#
#     def test_balance_1d_interpolators_from_element_density_tcx(self):
#         """
#         test calculation of 1d interpolators of ion charge state densities
#         """
#
#         interpolators_abundance = interpolators1d_from_elementdensity(self.atomic_data, self.element, self.psin_1d,
#                                                                       self.n_element_1d,
#                                                                       self.n_e_profile_1d, self.t_e_1d,
#                                                                       tcx_donor=self.tcx_donor,
#                                                                       tcx_donor_n=self.n_tcx_donor_1d,
#                                                                       tcx_donor_charge=0)
#
#         profiles = self.evaluate_interpolators(interpolators_abundance, self.psin_1d)
#         fraction_sum = self.sumup_fractions(profiles)
#
#         self.assertTrue(np.allclose(fraction_sum, self.n_element_profile_1d, rtol=self.TOLERANCE))
#
#     def test_balance_1d_interpolators_plasma_neutrality(self):
#         """
#         test calulation of 1d interpolators for ion charge state densities using plasma neutrality condition.
#         """
#
#         interpolators_abundance_1 = interpolators1d_from_elementdensity(self.atomic_data, self.element, self.psin_1d,
#                                                                         self.n_element_1d,
#                                                                         self.n_e_profile_1d, self.t_e_1d)
#
#         interpolators_abundance_2 = interpolators1d_from_elementdensity(self.atomic_data, self.element2, self.psin_1d,
#                                                                         self.n_element2_1d,
#                                                                         self.n_e_profile_1d, self.t_e_1d)
#
#         interpolators_abundance_3 = interpolators1d_match_plasma_neutrality(self.atomic_data, self.element,
#                                                                             self.psin_1d,
#                                                                             [interpolators_abundance_1,
#                                                                              interpolators_abundance_2],
#                                                                             self.n_e_profile_1d, self.t_e_1d)
#
#         profiles1 = self.evaluate_interpolators(interpolators_abundance_1, self.psin_1d)
#         profiles2 = self.evaluate_interpolators(interpolators_abundance_2, self.psin_1d)
#         profiles3 = self.evaluate_interpolators(interpolators_abundance_3, self.psin_1d)
#
#         total = self.sumup_electrons(profiles1)
#         total += self.sumup_electrons(profiles2)
#         total += self.sumup_electrons(profiles3)
#
#         self.assertTrue(np.allclose(total, self.n_e_profile_1d, rtol=self.TOLERANCE))
#
#     def test_balance_1d_interpolators_plasma_neutrality_tcx(self):
#         """
#         test calulation of 1d interpolators for ion charge state densities using plasma neutrality condition.
#         """
#
#         interpolators_abundance_1 = interpolators1d_from_elementdensity(self.atomic_data, self.element, self.psin_1d,
#                                                                         self.n_element_1d,
#                                                                         self.n_e_profile_1d, self.t_e_1d,
#                                                                         tcx_donor=self.tcx_donor,
#                                                                         tcx_donor_n=self.n_tcx_donor_1d,
#                                                                         tcx_donor_charge=0)
#
#         interpolators_abundance_2 = interpolators1d_from_elementdensity(self.atomic_data, self.element2, self.psin_1d,
#                                                                         self.n_element2_1d,
#                                                                         self.n_e_profile_1d, self.t_e_1d,
#                                                                         tcx_donor=self.tcx_donor,
#                                                                         tcx_donor_n=self.n_tcx_donor_1d,
#                                                                         tcx_donor_charge=0)
#
#         interpolators_abundance_3 = interpolators1d_match_plasma_neutrality(self.atomic_data, self.element,
#                                                                             self.psin_1d,
#                                                                             [interpolators_abundance_1,
#                                                                              interpolators_abundance_2],
#                                                                             self.n_e_profile_1d, self.t_e_1d,
#                                                                             tcx_donor=self.tcx_donor,
#                                                                             tcx_donor_n=self.n_tcx_donor_1d,
#                                                                             tcx_donor_charge=0)
#
#         profiles1 = self.evaluate_interpolators(interpolators_abundance_1, self.psin_1d)
#         profiles2 = self.evaluate_interpolators(interpolators_abundance_2, self.psin_1d)
#         profiles3 = self.evaluate_interpolators(interpolators_abundance_3, self.psin_1d)
#
#         total = self.sumup_electrons(profiles1)
#         total += self.sumup_electrons(profiles2)
#         total += self.sumup_electrons(profiles3)
#
#         self.assertTrue(np.allclose(total, self.n_e_profile_1d, rtol=self.TOLERANCE))
#
#     def test_fractional_2d_from_2d(self):
#         """
#         test fractional abundance 2d profile calculation with arrays as inputs
#         """
#         abundance_fractional = fractional_abundance(self.atomic_data, self.element, self.n_e_profile_2d,
#                                                     self.t_e_profile_2d)
#
#         fraction_sum = self.sumup_fractions(abundance_fractional)
#         self.assertTrue(np.allclose(fraction_sum, np.ones_like(self.n_e_profile_2d), rtol=self.TOLERANCE))
#
#     def test_fractional_2d_from_2d_tcx(self):
#         """
#         test fractional abundance 2d profile calculation with arrays as inputs with thermal cx
#         """
#         abundance_fractional = fractional_abundance(self.atomic_data, self.element, self.n_e_profile_2d,
#                                                     self.t_e_profile_2d, tcx_donor=self.tcx_donor,
#                                                     tcx_donor_n=self.n_tcx_donor_profile_2d, tcx_donor_charge=0)
#
#         fraction_sum = self.sumup_fractions(abundance_fractional)
#         self.assertTrue(np.allclose(fraction_sum, np.ones_like(self.n_e_profile_2d), rtol=self.TOLERANCE))
#
#     def test_fractional_2d_from_interpolators(self):
#         """
#         test fractional abundance 2d profile calculation with interpolators as inputs
#         """
#         abundance_fractional = fractional_abundance(self.atomic_data, self.element, self.n_e_2d,
#                                                     self.t_e_2d, free_variable=(self.r, self.z))
#
#         fraction_sum = self.sumup_fractions(abundance_fractional)
#         self.assertTrue(np.allclose(fraction_sum, np.ones_like(self.n_e_profile_2d), rtol=self.TOLERANCE))
#
#     def test_fractional_2d_from_interpolators_tcx(self):
#         """
#         test fractional abundance 2d profile calculation with interpolators as inputs with thermal cx
#         """
#         abundance_fractional = fractional_abundance(self.atomic_data, self.element, self.n_e_2d,
#                                                     self.t_e_2d, tcx_donor=self.tcx_donor,
#                                                     tcx_donor_n=self.n_tcx_donor_2d, tcx_donor_charge=0,
#                                                     free_variable=(self.r, self.z))
#
#         fraction_sum = self.sumup_fractions(abundance_fractional)
#         self.assertTrue(np.allclose(fraction_sum, np.ones_like(self.n_e_profile_2d), rtol=self.TOLERANCE))
#
#     def test_fractional_2d_from_mixed(self):
#         """
#         test fractional abundance 2d profile calculation with mixed inputs
#         """
#         abundance_fractional = fractional_abundance(self.atomic_data, self.element, self.n_e_profile_2d,
#                                                     self.t_e_2d, free_variable=(self.r, self.z))
#
#         fraction_sum = self.sumup_fractions(abundance_fractional)
#         self.assertTrue(np.allclose(fraction_sum, np.ones_like(self.n_e_profile_2d), rtol=self.TOLERANCE))
#
#     def test_fractional_2d_from_mixed_tcx(self):
#         """
#         test fractional abundance 2d profile calculation with mixed inputs with thermal cx
#         """
#         abundance_fractional = fractional_abundance(self.atomic_data, self.element, self.n_e_profile_2d,
#                                                     self.t_e_2d, tcx_donor=self.tcx_donor,
#                                                     tcx_donor_n=self.n_tcx_donor_2d, tcx_donor_charge=0,
#                                                     free_variable=(self.r, self.z))
#
#         fraction_sum = self.sumup_fractions(abundance_fractional)
#         self.assertTrue(np.allclose(fraction_sum, np.ones_like(self.n_e_profile_2d), rtol=self.TOLERANCE))
#
#     def test_balance_2d_elementdensity(self):
#         """
#         test abundance 2d profile calculation from element density
#         """
#         abundance = from_elementdensity(self.atomic_data, self.element, self.n_element_profile_2d, self.n_e_2d,
#                                         self.t_e_profile_2d, free_variable=(self.r, self.z))
#
#         fraction_sum = self.sumup_fractions(abundance)
#         self.assertTrue(np.allclose(fraction_sum, self.n_element_profile_2d, rtol=self.TOLERANCE))
#
#     def test_balance_2d_elementdensity_tcx(self):
#         """
#         test abundance 2d profile calculation from element density
#         """
#         abundance = from_elementdensity(self.atomic_data, self.element, self.n_element_profile_2d, self.n_e_2d,
#                                         self.t_e_profile_2d, tcx_donor=self.tcx_donor,
#                                         tcx_donor_n=self.n_tcx_donor_2d, tcx_donor_charge=0,
#                                         free_variable=(self.r, self.z))
#
#         fraction_sum = self.sumup_fractions(abundance)
#         self.assertTrue(np.allclose(fraction_sum, self.n_element_profile_2d, rtol=self.TOLERANCE))
#
#     def test_balance_2d_plasma_neutrality(self):
#         """test matching of plasma neutrality for 2d profiles"""
#
#         densities_1 = from_elementdensity(self.atomic_data, self.element, self.n_element_2d,
#                                           self.n_e_2d, self.t_e_profile_2d,
#                                           free_variable=(self.r, self.z))
#
#         densities_2 = from_elementdensity(self.atomic_data, self.element2, self.n_element2_2d,
#                                           self.n_e_profile_2d, self.t_e_2d,
#                                           free_variable=(self.r, self.z))
#
#         densities_3 = match_plasma_neutrality(self.atomic_data, self.element_bulk, [densities_1, densities_2],
#                                               self.n_e_2d, self.t_e_profile_2d,
#                                               free_variable=(self.r, self.z))
#
#         total = self.sumup_electrons(densities_1)
#         total += self.sumup_electrons(densities_2)
#         total += self.sumup_electrons(densities_3)
#
#         self.assertTrue(np.allclose(total, self.n_e_profile_2d, rtol=self.TOLERANCE))
#
#     def test_balance_2d_plasma_neutrality_tcx(self):
#         """test matching of plasma neutrality for 2d profiles"""
#
#         densities_1 = from_elementdensity(self.atomic_data, self.element, self.n_element_2d,
#                                           self.n_e_2d, self.t_e_profile_2d, tcx_donor=self.tcx_donor,
#                                           tcx_donor_n=self.n_tcx_donor_2d, tcx_donor_charge=0,
#                                           free_variable=(self.r, self.z))
#
#         densities_2 = from_elementdensity(self.atomic_data, self.element2, self.n_element2_2d,
#                                           self.n_e_profile_2d, self.t_e_2d, tcx_donor=self.tcx_donor,
#                                           tcx_donor_n=self.n_tcx_donor_2d, tcx_donor_charge=0,
#                                           free_variable=(self.r, self.z))
#
#         densities_3 = match_plasma_neutrality(self.atomic_data, self.element_bulk, [densities_1, densities_2],
#                                               self.n_e_2d, self.t_e_profile_2d, tcx_donor=self.tcx_donor,
#                                               tcx_donor_n=self.n_tcx_donor_2d, tcx_donor_charge=0,
#                                               free_variable=(self.r, self.z))
#
#         total = self.sumup_electrons(densities_1)
#         total += self.sumup_electrons(densities_2)
#         total += self.sumup_electrons(densities_3)
#
#         self.assertTrue(np.allclose(total, self.n_e_profile_2d, rtol=self.TOLERANCE))
#
#     def test_fractional_inetrpolators_2d(self):
#         """
#         test calculation of 1d fractional interpolators
#         """
#
#         interpolators_fractional = interpolators2d_fractional(self.atomic_data, self.element, (self.r, self.z),
#                                                               self.n_e_profile_2d, self.t_e_2d)
#
#         profiles = self.evaluate_interpolators(interpolators_fractional, (self.r, self.z))
#         fraction_sum = self.sumup_fractions(profiles)
#
#         self.assertTrue(np.allclose(fraction_sum, 1, atol=self.TOLERANCE))
#
#     def test_fractional_inetrpolators_2d_tcx(self):
#         """
#         test calculation of 1d fractional interpolators with thermal cx
#         """
#
#         interpolators_fractional = interpolators2d_fractional(self.atomic_data, self.element, (self.r, self.z),
#                                                               self.n_e_profile_2d, self.t_e_2d,
#                                                               tcx_donor=self.tcx_donor,
#                                                               tcx_donor_n=self.n_tcx_donor_profile_2d,
#                                                               tcx_donor_charge=0)
#
#         profiles = self.evaluate_interpolators(interpolators_fractional, (self.r, self.z))
#         fraction_sum = self.sumup_fractions(profiles)
#
#         self.assertTrue(np.allclose(fraction_sum, 1, atol=self.TOLERANCE))
#
#     def test_balance_2d_interpolators_from_element_density(self):
#         """
#         test calculation of 2d interpolators of charge stage densities
#         """
#
#         interpolators_abundance = interpolators2d_from_elementdensity(self.atomic_data, self.element, (self.r, self.z),
#                                                                       self.n_element_2d,
#                                                                       self.n_e_profile_2d, self.t_e_2d)
#
#         profiles = self.evaluate_interpolators(interpolators_abundance, (self.r, self.z))
#         fraction_sum = self.sumup_fractions(profiles)
#
#         self.assertTrue(np.allclose(fraction_sum, self.n_element_profile_2d, rtol=self.TOLERANCE))
#
#     def test_balance_2d_interpolators_from_element_density_tcx(self):
#         """
#         test calculation of 2d interpolators of ion charge state densities
#         """
#
#         interpolators_abundance = interpolators2d_from_elementdensity(self.atomic_data, self.element, (self.r, self.z),
#                                                                       self.n_element_2d,
#                                                                       self.n_e_profile_2d, self.t_e_2d,
#                                                                       tcx_donor=self.tcx_donor,
#                                                                       tcx_donor_n=self.n_tcx_donor_2d,
#                                                                       tcx_donor_charge=0)
#
#         profiles = self.evaluate_interpolators(interpolators_abundance, (self.r, self.z))
#         fraction_sum = self.sumup_fractions(profiles)
#
#         self.assertTrue(np.allclose(fraction_sum, self.n_element_profile_2d, rtol=self.TOLERANCE))
#
#     def test_balance_2d_interpolators_plasma_neutrality(self):
#         """
#         test calulation of 2d interpolators for ion charge state densities using plasma neutrality condition.
#         """
#
#         interpolators_abundance_1 = interpolators2d_from_elementdensity(self.atomic_data, self.element,
#                                                                         (self.r, self.z),
#                                                                         self.n_element_2d,
#                                                                         self.n_e_profile_2d, self.t_e_2d)
#
#         interpolators_abundance_2 = interpolators2d_from_elementdensity(self.atomic_data, self.element2,
#                                                                         (self.r, self.z),
#                                                                         self.n_element2_2d,
#                                                                         self.n_e_profile_2d, self.t_e_2d)
#
#         interpolators_abundance_3 = interpolators2d_match_plasma_neutrality(self.atomic_data, self.element,
#                                                                             (self.r, self.z),
#                                                                             [interpolators_abundance_1,
#                                                                              interpolators_abundance_2],
#                                                                             self.n_e_profile_2d, self.t_e_2d)
#
#         profiles1 = self.evaluate_interpolators(interpolators_abundance_1, (self.r, self.z))
#         profiles2 = self.evaluate_interpolators(interpolators_abundance_2, (self.r, self.z))
#         profiles3 = self.evaluate_interpolators(interpolators_abundance_3, (self.r, self.z))
#
#         total = self.sumup_electrons(profiles1)
#         total += self.sumup_electrons(profiles2)
#         total += self.sumup_electrons(profiles3)
#
#         self.assertTrue(np.allclose(total, self.n_e_profile_2d, rtol=self.TOLERANCE))
#
#     def test_balance_2d_interpolators_plasma_neutrality_tcx(self):
#         """
#         test calulation of 2d interpolators for ion charge state densities using plasma neutrality condition.
#         """
#
#         interpolators_abundance_1 = interpolators2d_from_elementdensity(self.atomic_data, self.element,
#                                                                         (self.r, self.z),
#                                                                         self.n_element_2d,
#                                                                         self.n_e_profile_2d, self.t_e_2d,
#                                                                         tcx_donor=self.tcx_donor,
#                                                                         tcx_donor_n=self.n_tcx_donor_2d,
#                                                                         tcx_donor_charge=0)
#
#         interpolators_abundance_2 = interpolators2d_from_elementdensity(self.atomic_data, self.element2,
#                                                                         (self.r, self.z),
#                                                                         self.n_element2_2d,
#                                                                         self.n_e_profile_2d, self.t_e_2d,
#                                                                         tcx_donor=self.tcx_donor,
#                                                                         tcx_donor_n=self.n_tcx_donor_2d,
#                                                                         tcx_donor_charge=0)
#
#         interpolators_abundance_3 = interpolators2d_match_plasma_neutrality(self.atomic_data, self.element,
#                                                                             (self.r, self.z),
#                                                                             [interpolators_abundance_1,
#                                                                              interpolators_abundance_2],
#                                                                             self.n_e_profile_2d, self.t_e_2d,
#                                                                             tcx_donor=self.tcx_donor,
#                                                                             tcx_donor_n=self.n_tcx_donor_2d,
#                                                                             tcx_donor_charge=0)
#
#         profiles1 = self.evaluate_interpolators(interpolators_abundance_1, (self.r, self.z))
#         profiles2 = self.evaluate_interpolators(interpolators_abundance_2, (self.r, self.z))
#         profiles3 = self.evaluate_interpolators(interpolators_abundance_3, (self.r, self.z))
#
#         total = self.sumup_electrons(profiles1)
#         total += self.sumup_electrons(profiles2)
#         total += self.sumup_electrons(profiles3)
#
#         self.assertTrue(np.allclose(total, self.n_e_profile_2d, rtol=self.TOLERANCE))
#
#     def test_axisymmetric_mapper(self):
#         "test axisymmetric mapping"
#
#         #fractional abundance
#         interpolators_fractional = interpolators2d_fractional(self.atomic_data, self.element, (self.r, self.z),
#                                                               self.n_e_profile_2d, self.t_e_2d,
#                                                               tcx_donor=self.tcx_donor,
#                                                               tcx_donor_n=self.n_tcx_donor_profile_2d,
#                                                               tcx_donor_charge=0)
#
#         mappers = abundance_axisymmetric_mapper(interpolators_fractional)
#
#
#
#         profile = self.evaluate_interpolators(mappers, self.r)
#
#         total = self.sumup_fractions(profile)
#
#
#         self.assertTrue(np.allclose(total, 1, rtol=self.TOLERANCE))
#
#     def test_equilibrium_map3d_fractional(self):
#         """
#         test calculation of fractional abundance and application of map3d functionality of equilibrium
#         """
#
#         equilibrium = example_equilibrium()
#
#         mapper = equilibrium_map3d_fractional(self.atomic_data, self.element, equilibrium, self.psin_1d,
#                                                                       self.n_e_profile_1d, self.t_e_profile_1d, self.tcx_donor,
#                                                                       self.n_tcx_donor_profile_1d, tcx_donor_charge=0)
#         psin_1d = np.linspace(0, 0.99, 10)
#         r = np.zeros_like(psin_1d)
#         for index, value in enumerate(psin_1d):
#             r[index] = equilibrium.psin_to_r(value)
#
#         profile = self.evaluate_interpolators(mapper, r)
#
#         total = self.sumup_fractions(profile)
#
#         self.assertTrue(np.allclose(total, 1, rtol=self.TOLERANCE))
#
#     def test_equilibrium_map3d_from_elementdensity(self):
#         """
#         test calculation of abundance and application of map3d functionality of equilibrium
#         """
#
#         equilibrium = example_equilibrium()
#
#         mapper = equilibrium_map3d_from_elementdensity(self.atomic_data, self.element, equilibrium, self.psin_1d,
#                                                        self.n_element_1d, self.n_e_1d,
#                                                        self.t_e_1d, self.tcx_donor, self.n_tcx_donor_1d,
#                                                        tcx_donor_charge=0)
#
#         inlcfs = np.where(self.psin_1d < 1)[0]
#         psin_1d = self.psin_1d[inlcfs]
#         n = np.zeros_like(psin_1d)
#         r = np.zeros_like(psin_1d)
#         for index, value in enumerate(psin_1d):
#             n[index] = self.n_element_1d(value)
#             r[index] = equilibrium.psin_to_r(value)
#
#         profile = self.evaluate_interpolators(mapper, r)
#         total = self.sumup_fractions(profile)
#
#         self.assertTrue(np.allclose(total, n, rtol=self.TOLERANCE))
#
#     def test_equilibrium_map3d_plasma_neutrality(self):
#         """
#         test calculation of abundance and application of map3d functionality of equilibrium
#         """
#
#         equilibrium = example_equilibrium()
#
#         interpolators_element1 = interpolators1d_from_elementdensity(self.atomic_data, self.element, self.psin_1d,
#                                                                       self.n_element_1d,
#                                                                       self.n_e_1d, self.t_e_1d,
#                                                                       tcx_donor=self.tcx_donor,
#                                                                       tcx_donor_n=self.n_tcx_donor_1d,
#                                                                       tcx_donor_charge=0)
#
#         interpolators_element2 = interpolators1d_from_elementdensity(self.atomic_data, self.element2, self.psin_1d,
#                                                                       self.n_element2_1d,
#                                                                       self.n_e_1d, self.t_e_1d,
#                                                                       tcx_donor=self.tcx_donor,
#                                                                       tcx_donor_n=self.n_tcx_donor_1d,
#                                                                       tcx_donor_charge=0)
#
#         mapper = equilibrium_map3d_match_plasma_neutrality(self.atomic_data, self.element_bulk, equilibrium, self.psin_1d,
#                                                   [interpolators_element1, interpolators_element2],
#                                                   self.n_e_1d, self.t_e_1d, self.tcx_donor,
#                                                   self.n_tcx_donor_1d, 0)
#
#         inlcfs = np.where(self.psin_1d < 1)[0]
#         psin_1d = self.psin_1d[inlcfs]
#         n = np.zeros_like(psin_1d)
#         r = np.zeros_like(psin_1d)
#         for index, value in enumerate(psin_1d):
#             n[index] = self.n_e_1d(value)
#             r[index] = equilibrium.psin_to_r(value)
#
#         profile1 = self.evaluate_interpolators(interpolators_element1, psin_1d)
#         profile2 = self.evaluate_interpolators(interpolators_element2, psin_1d)
#         profile3 = self.evaluate_interpolators(mapper, r)
#
#         total = self.sumup_electrons(profile1)
#         total += self.sumup_electrons(profile2)
#         total += self.sumup_electrons(profile3)
#
#         self.assertTrue(np.allclose(total, n, rtol=self.TOLERANCE))
