import unittest

import numpy as np
from raysect.core.math.function.float import Arg1D, Constant1D

from cherab.core.atomic import ZeemanStructure


class TestZeemanStructure(unittest.TestCase):
    def test_initialisation_rejects_invalid_component_shape(self):
        with self.assertRaises(ValueError):
            ZeemanStructure([(Constant1D(656.1),)], [], [])

        with self.assertRaises(ValueError):
            ZeemanStructure([], [(Constant1D(656.1),)], [])

        with self.assertRaises(ValueError):
            ZeemanStructure([], [], [(Constant1D(656.1),)])

    def test_call_returns_expected_components_and_normalised_ratios(self):
        pi_components = [
            (Constant1D(656.1), Constant1D(2.0)),
            (Constant1D(656.2), Constant1D(6.0)),
        ]
        sigma_plus_components = [
            (656.0 + 0.01 * Arg1D(), Constant1D(3.0)),
            (656.3 + 0.02 * Arg1D(), Constant1D(1.0)),
        ]
        sigma_minus_components = [
            (Constant1D(655.9), Constant1D(1.0)),
            (Constant1D(656.4), Constant1D(1.0)),
        ]
        zeeman = ZeemanStructure(pi_components, sigma_plus_components, sigma_minus_components)

        b = 2.0

        pi = zeeman(b, 'PI')
        np.testing.assert_allclose(pi[0], np.array([656.1, 656.2]))
        np.testing.assert_allclose(pi[1], np.array([0.25, 0.75]))

        sigma_plus = zeeman(b, 'SIGMA_PLUS')
        np.testing.assert_allclose(sigma_plus[0], np.array([656.02, 656.34]))
        np.testing.assert_allclose(sigma_plus[1], np.array([0.75, 0.25]))

        sigma_minus = zeeman(b, 'sigma_minus')
        np.testing.assert_allclose(sigma_minus[0], np.array([655.9, 656.4]))
        np.testing.assert_allclose(sigma_minus[1], np.array([0.5, 0.5]))

    def test_call_keeps_zero_ratios_when_sum_is_zero(self):
        zeeman = ZeemanStructure(
            pi_components=[
                (Constant1D(656.1), Constant1D(0.0)),
                (Constant1D(656.2), Constant1D(0.0)),
            ],
            sigma_plus_components=[],
            sigma_minus_components=[],
        )

        pi = zeeman(0.0, 'pi')
        np.testing.assert_allclose(pi[0], np.array([656.1, 656.2]))
        np.testing.assert_allclose(pi[1], np.array([0.0, 0.0]))

    def test_call_raises_for_invalid_arguments(self):
        zeeman = ZeemanStructure([], [], [])

        with self.assertRaises(ValueError):
            zeeman(-1.0, 'pi')

        with self.assertRaises(ValueError):
            zeeman(1.0, 'sigma')
