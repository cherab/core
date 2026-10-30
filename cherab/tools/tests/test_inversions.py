# Copyright 2026 Oak Ridge National Laboratory
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

from cherab.tools.inversions import invert_regularised_lstsq
from cherab.tools.inversions import invert_regularised_nnls, invert_sparse_regularised_nnls
from cherab.tools.inversions import invert_svd
from cherab.tools.inversions import invert_sart, invert_constrained_sart


class TestRegularisedInversionAlgorithms(unittest.TestCase):
    """
    For each of the inversion algorithms, we test:

      - The routine runs with default arguments without errors.
      - Increasing the regularisation has the expected impact.
      - Changing the regularisation operator has the expected impact.
      - The inversion results are "sensible". In practice, this means running
        the test case by hand and checking the result, then simply regression
        testing against this known-sensible result.

    We assume that the underlying Numpy and Scipy routines for performing
    matrix inverision are correct, so these tests are really just to ensure that
    the augmentations to these for regularisation are functional. It is overkill
    to mathematically prove the correctness of the whole algorithm.

    That said, different regularisation schemes are appropriate for different
    solutions. So we use 2 different solutions: one which makes sense for
    ridge-regression (minimal outliers) and one which makes sense for maximal
    smoothness (small second derivative, goes to zero at the edges).
    """

    def setUp(self):
        # Geometry matrix. We test under-determined problems, where the number
        # of measurement points is less than the number of solution points.
        # Inversion problems typically have somewhat sparse geometry matrices.
        self.n_meas, self.n_true = 4, 5
        self.geom = np.array(
            [
                [1, 0, 3, 4, 0],
                [6, 0, 8, 9, 0],
                [5, 4.5, 0, 3.5, 3],
                [0, 2, 0, 0, 0.5],
            ],
            dtype=np.float64,
        )
        # Some suitable solutions which roughly conform to the type of solution
        # preferred by the regularisation scheme.
        # The identity operator prefers a solution with small outliers.
        # The second derivative operator prefers smooth solutions which go to
        # zero at the boundaries.
        self.true_solution_ridge = np.array([0.1, 0.5, 0.3, 0.4, 0.2])
        self.true_solution_smooth = np.array([0, 0.3, 0.1, 0.2, 0])
        # Ridge regression operator (default Tikhonov regularisation)
        self.ridge = np.eye(self.n_true)
        # Second-derivative operator for 1D measurements.
        self.penalty = (
            + 1 * np.eye(self.n_true, k=-1)
            - 2 * np.eye(self.n_true)
            + 1 * np.eye(self.n_true, k=1)
        )

    def test_lstsq(self):
        """
        Test lstsq with default tikhonov matrix: identiy operator.

        Increasing the alpha parameter should increase the residual and
        decrease the solution magnitude.

        Default alpha (at the time these tests were written) is 0.01.
        """
        solution = self.true_solution_ridge
        measurements = self.geom @ solution
        x1, res1 = invert_regularised_lstsq(self.geom, measurements)
        x2, res2 = invert_regularised_lstsq(self.geom, measurements, alpha=0.1)
        self.assertGreater(res2, res1)
        self.assertLess(np.linalg.norm(x2), np.linalg.norm(x1))
        x3, _ = invert_regularised_lstsq(self.geom, measurements, alpha=1e-3)
        # Ridge solution is [0.1, 0.5, 0.3, 0.4, 0.2]. Damped solution is:
        expected = [0.108345, 0.513352, 0.274965, 0.41669, 0.146593]
        np.testing.assert_allclose(x3, expected, atol=1e-6)

    def test_lstsq_smoothness(self):
        """
        Test lstsq with a second derivative operator as the tikhonov matrix.

        Increasing the alpha parameter should therefore decrease the
        second derivative of the solution, at the expense of larger
        residuals.

        Default alpha (at the time these tests were written) is 0.01.
        """
        solution = self.true_solution_smooth
        measurements = self.geom @ solution
        x1, res1 = invert_regularised_lstsq(self.geom, measurements,
                                            tikhonov_matrix=self.penalty)
        x2, res2 = invert_regularised_lstsq(self.geom, measurements,
                                            tikhonov_matrix=self.penalty, alpha=0.1)
        self.assertGreater(res2, res1)
        deriv1 = np.diff(np.diff(x1))
        deriv2 = np.diff(np.diff(x2))
        self.assertLess(np.linalg.norm(deriv2), np.linalg.norm(deriv1))
        x3, _ = invert_regularised_lstsq(self.geom, measurements,
                                         tikhonov_matrix=self.penalty, alpha=1e-3)
        # Smooth solution is [0, 0.3, 0.1, 0.2, 0]. Damped solution is:
        expected = [-0.023824,  0.261881,  0.171473,  0.152351,  0.152475]
        np.testing.assert_allclose(x3, expected, atol=1e-6)

    def test_nnls(self):
        """
        Test nnls with default tikhonov matrix: identiy operator.

        Increasing the alpha parameter should increase the residual and
        decrease the solution magnitude.

        Default alpha (at the time these tests were written) is 0.01.
        """
        solution = self.true_solution_ridge
        measurements = self.geom @ solution
        x1, res1 = invert_regularised_nnls(self.geom, measurements)
        x2, res2 = invert_regularised_nnls(self.geom, measurements, alpha=0.1)
        self.assertGreater(res2, res1)
        self.assertLess(np.linalg.norm(x2), np.linalg.norm(x1))
        # Ridge solution is [0.1, 0.5, 0.3, 0.4, 0.2]. Damped solution is:
        expected = [0.108345, 0.513352, 0.274965, 0.41669, 0.146593]
        x3, _ = invert_regularised_nnls(self.geom, measurements, alpha=1e-3)
        np.testing.assert_allclose(x3, expected, atol=1e-6)

    def test_nnls_smoothness(self):
        """
        Test nnls with a second derivative operator as the tikhonov matrix.

        Increasing the alpha parameter should therefore decrease the
        second derivative of the solution, at the expense of larger
        residuals.

        Default alpha (at the time these tests were written) is 0.01.
        """
        solution = self.true_solution_smooth
        measurements = self.geom @ solution
        x1, res1 = invert_regularised_nnls(self.geom, measurements,
                                           tikhonov_matrix=self.penalty)
        x2, res2 = invert_regularised_nnls(self.geom, measurements,
                                           tikhonov_matrix=self.penalty, alpha=0.1)
        self.assertGreater(res2, res1)
        deriv1 = np.diff(np.diff(x1))
        deriv2 = np.diff(np.diff(x2))
        self.assertLess(np.linalg.norm(deriv2), np.linalg.norm(deriv1))
        x3, _ = invert_regularised_nnls(self.geom, measurements,
                                        tikhonov_matrix=self.penalty, alpha=1e-3)
        # Smooth solution is [0, 0.3, 0.1, 0.2, 0]. Damped solution is:
        expected = [0, 0.299992, 0.100016, 0.199986, 0.000029]
        np.testing.assert_allclose(x3, expected, atol=1e-6)

    def test_sparse_nnls(self):
        """
        Test nnls with default tikhonov matrix: identiy operator.

        Increasing the alpha parameter should increase the residual and
        decrease the solution magnitude.

        Default alpha (at the time these tests were written) is 0.01.
        """
        solution = self.true_solution_ridge
        measurements = self.geom @ solution
        x1, res1 = invert_sparse_regularised_nnls(self.geom, measurements)
        x2, res2 = invert_sparse_regularised_nnls(self.geom, measurements, alpha=0.1)
        self.assertGreater(res2, res1)
        self.assertLess(np.linalg.norm(x2), np.linalg.norm(x1))
        # Ridge solution is [0.1, 0.5, 0.3, 0.4, 0.2]. Damped solution is:
        expected = [0.108345, 0.513352, 0.274965, 0.41669, 0.146593]
        x3, _ = invert_sparse_regularised_nnls(self.geom, measurements, alpha=1e-3)
        np.testing.assert_allclose(x3, expected, atol=1e-6)

    def test_sparse_nnls_smoothness(self):
        """
        Test nnls with a second derivative operator as the tikhonov matrix.

        Increasing the alpha parameter should therefore decrease the
        second derivative of the solution, at the expense of larger
        residuals.

        Default alpha (at the time these tests were written) is 0.01.
        """
        solution = self.true_solution_smooth
        measurements = self.geom @ solution
        x1, res1 = invert_sparse_regularised_nnls(self.geom, measurements,
                                                  tikhonov_matrix=self.penalty)
        x2, res2 = invert_sparse_regularised_nnls(self.geom, measurements,
                                                  tikhonov_matrix=self.penalty, alpha=0.1)
        self.assertGreater(res2, res1)
        deriv1 = np.diff(np.diff(x1))
        deriv2 = np.diff(np.diff(x2))
        self.assertLess(np.linalg.norm(deriv2), np.linalg.norm(deriv1))
        x3, _ = invert_sparse_regularised_nnls(self.geom, measurements,
                                               tikhonov_matrix=self.penalty, alpha=1e-3)
        # Smooth solution is [0, 0.3, 0.1, 0.2, 0]. Damped solution is:
        expected = [0.000008, 0.299993, 0.100013, 0.199984, 0.000016]
        np.testing.assert_allclose(x3, expected, atol=1e-6)


class TestSVDInversion(unittest.TestCase):
    def test_svd(self):
        geom = np.array(
            [
                [1, 0, 3, 4, 0],
                [6, 0, 8, 9, 0],
                [5, 4.5, 0, 3.5, 3],
                [0, 2, 0, 0, 0.5],
            ],
            dtype=np.float64,
        )
        true_solution = np.array([0.1, 0.5, 0.3, 0.4, 0.2])
        measurements = geom @ true_solution
        inverted = invert_svd(geom, measurements)
        # SVD inversion is not exact, but is close.
        expected = [0.108345, 0.513352, 0.274965, 0.41669, 0.146592]
        np.testing.assert_allclose(inverted, expected, atol=1e-6)


class TestSART(unittest.TestCase):
    """
    These tests largely match those of the TestRegularisedInversionAlgorithms
    class, but for the locally-implemented SART algorithm. So we still test
    that the function runs without error and the changes in regularisation
    have the expected effect. But the mathematical correctness of the
    algorithm is not tested any more strenuously here than it was for the
    numpy- and scipy-derived inversion routines.

    *Important note:* unlike the least-squares regularisation schemes
    (where the norm of the regularisation matrix is used), the SART
    algorithm is sensitive to the sign of the operator. The docstring
    for ``invert_constrained_sart`` specifies the sign convention to use:
    getting this wrong produces the opposite effect (de-regularisation of
    the result).
    """
    def setUp(self):
        # Geometry matrix. We test under-determined problems, where the number
        # of measurement points is less than the number of solution points.
        # Inversion problems typically have somewhat sparse geometry matrices.
        self.n_meas, self.n_true = 4, 5
        self.geom = np.array(
            [
                [1, 0, 3, 4, 0],
                [6, 0, 8, 9, 0],
                [5, 4.5, 0, 3.5, 3],
                [0, 2, 0, 0, 0.5],
            ],
            dtype=np.float64,
        )
        # Some suitable solutions which roughly conform to the type of solution
        # preferred by the regularisation scheme.
        # The identity operator prefers a solution with small outliers.
        # The second derivative operator prefers smooth solutions which go to
        # zero at the boundaries.
        self.true_solution_ridge = np.array([0.1, 0.5, 0.3, 0.4, 0.2])
        self.true_solution_smooth = np.array([0, 0.3, 0.1, 0.2, 0])
        # Second-derivative operator for 1D measurements.
        # Note that the sign of the operator matters for this SART impementation:
        # the docstring for invert_constrained_sart specifes that the
        # smoothing operator index Li,j should be positive for i=j and negative
        # for the neighbours j of cell i. This is actually the opposite of what
        # one would derive from a difference-of-differences operator.
        self.penalty = (
            - 1 * np.eye(self.n_true, k=-1)
            + 2 * np.eye(self.n_true)
            - 1 * np.eye(self.n_true, k=1)
        )

    def test_invert_sart(self):
        solution = self.true_solution_smooth
        measurements = self.geom @ solution
        conv_tol = 1e-4
        x, conv = invert_sart(self.geom, measurements, conv_tol=conv_tol)
        # Check convergence is calculated properly. It's defined in the docstring
        # as the normalised squared difference between the measurement and the
        # back-calculated measurement from the solution.
        self.assertLess(conv[-1], conv_tol)
        meas2 = np.dot(measurements, measurements)
        backcalc = self.geom @ x
        back2 = np.dot(backcalc, backcalc)
        conv_expected = (meas2 - back2) / meas2
        self.assertAlmostEqual(conv[-1], conv_expected)
        # Check the solution is sensible.
        # True solution is [0, 0.3, 0.1, 0.2, 0]. SART solution is:
        expected = [0.00646, 0.279187, 0.168596, 0.138839, 0.089245]
        np.testing.assert_allclose(x, expected, atol=1e-6)

    def test_invert_sart_scalar_initial_guess(self):
        solution = self.true_solution_smooth
        measurements = self.geom @ solution
        x, conv = invert_sart(self.geom, measurements, initial_guess=0.1)
        # Slightly different expected with different initial guess.
        expected = [0.00646, 0.279187, 0.168596, 0.138839, 0.089245]
        np.testing.assert_allclose(x, expected, atol=1e-6)

    def test_invert_sart_vector_initial_guess(self):
        solution = self.true_solution_smooth
        measurements = self.geom @ solution
        guess = [0.001, 0.33, 0.15, 0.25, 0]
        x, conv = invert_sart(self.geom, measurements, initial_guess=guess)
        # Slightly different expected with different initial guess.
        expected = [0, 0.299338, 0.097487, 0.202028, 0]
        np.testing.assert_allclose(x, expected, atol=1e-6)


    def test_invert_constrained_sart(self):
        """
        Test invert_constrained_sart with a second derivative operator.

        Increasing the beta_laplace parameter should therefore decrease
        the second derivative of the solution.

        Default beta_laplace (at the time these tests were written) is 0.01.
        """
        solution = self.true_solution_smooth
        measurements = self.geom @ solution
        conv_tol = 1e-4
        x1, conv1 = invert_constrained_sart(self.geom, self.penalty, measurements,
                                            conv_tol=conv_tol)
        x2, conv2 = invert_constrained_sart(self.geom, self.penalty, measurements,
                                            conv_tol=conv_tol, beta_laplace=0.1)
        # Check convergence is calculated properly. It's defined in the docstring
        # as the normalised squared difference between the measurement and the
        # back-calculated measurement from the solution.
        self.assertLess(conv1[-1], conv_tol)
        meas2 = np.dot(measurements, measurements)
        backcalc = self.geom @ x1
        back2 = np.dot(backcalc, backcalc)
        conv_expected = (meas2 - back2) / meas2
        self.assertAlmostEqual(conv1[-1], conv_expected)
        # Check the regularisation works properly: higher beta_laplace should
        # give a smaller second derivative.
        deriv1 = np.diff(np.diff(x1))
        deriv2 = np.diff(np.diff(x2))
        self.assertLess(np.linalg.norm(deriv2), np.linalg.norm(deriv1))
        x3, _ = invert_constrained_sart(self.geom, self.penalty, measurements,
                                        beta_laplace=1e-3)
        # Smooth solution is [0, 0.3, 0.1, 0.2, 0]. Damped solution is:
        expected = [0.013176, 0.276939, 0.169838, 0.134259, 0.088389]
        np.testing.assert_allclose(x3, expected, atol=1e-6)

    def test_invert_constrained_sart_scalar_initial_guess(self):
        """
        Test invert_constrained_sart with a scalar value initial guess.

        Uses the same second-derivative operator as test_constrained_sart.
        """
        solution = self.true_solution_smooth
        measurements = self.geom @ solution
        x, conv = invert_constrained_sart(self.geom, self.penalty, measurements,
                                          initial_guess=0.1, beta_laplace=1e-3)
        # Slightly different expected with different initial guess.
        expected = [0.013197, 0.276938, 0.169868, 0.134222, 0.088396]
        np.testing.assert_allclose(x, expected, atol=1e-6)

    def test_invert_constrained_sart_vector_initial_guess(self):
        """
        Test invert_constrained_sart with an initial vector guess.

        Uses the same second-derivative operator as test_constrained_sart.
        """
        solution = self.true_solution_smooth
        measurements = self.geom @ solution
        guess = [0.001, 0.33, 0.15, 0.25, 0]
        x, conv = invert_constrained_sart(self.geom, self.penalty, measurements,
                                          initial_guess=guess, beta_laplace=1e-3)
        # Slightly different expected with different initial guess.
        expected = [0.00248, 0.297245, 0.101434, 0.197541, 0.003531]
        np.testing.assert_allclose(x, expected, atol=1e-6)
