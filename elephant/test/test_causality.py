# -*- coding: utf-8 -*-
"""
Unit tests for the causality module.

:copyright: Copyright 2014-2020 by the Elephant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
"""
from __future__ import division, print_function

import unittest

import numpy as np
import quantities as pq
from neo.core import AnalogSignal
from numpy.testing import assert_array_almost_equal

import elephant.causality.granger


class PairwiseGrangerTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        np.random.seed(1)
        cls.ground_truth = cls._generate_ground_truth()

    @staticmethod
    def _generate_ground_truth(length_2d=30000):
        order = 2
        signal = np.zeros((2, length_2d + order))

        weights_1 = np.array([[0.9, 0], [0.9, -0.8]])
        weights_2 = np.array([[-0.5, 0], [-0.2, -0.5]])

        weights = np.stack((weights_1, weights_2))

        noise_covariance = np.array([[1., 0.0], [0.0, 1.]])

        for i in range(length_2d):
            for lag in range(order):
                signal[:, i + order] += np.dot(weights[lag],
                                               signal[:, i + 1 - lag])
            rnd_var = np.random.multivariate_normal([0, 0],
                                                    noise_covariance)
            signal[:, i+order] += rnd_var

        signal = signal[:, 2:]

        # Return signals as Nx2
        return signal.T

    def setUp(self):
        # Generate a smaller random dataset for tests other than ground truth,
        # using a different seed than in the ground truth - the convergence
        # should not depend on the seed.
        np.random.seed(10)
        self.signal = self._generate_ground_truth(length_2d=1000)

        # Estimate Granger causality
        self.causality = elephant.causality.granger.pairwise_granger(
            self.signal, max_order=10,
            information_criterion='bic')

    def test_analog_signal_input(self):
        """
        Check if analog signal input result matches an otherwise identical 2D
        numpy array input result.
        """
        analog_signal = AnalogSignal(self.signal, units='V',
                                     sampling_rate=1*pq.Hz)
        analog_signal_causality = \
            elephant.causality.granger.pairwise_granger(
                analog_signal, max_order=10,
                information_criterion='bic')
        self.assertEqual(analog_signal_causality.directional_causality_x_y,
                         self.causality.directional_causality_x_y)
        self.assertEqual(analog_signal_causality.directional_causality_y_x,
                         self.causality.directional_causality_y_x)
        self.assertEqual(analog_signal_causality.instantaneous_causality,
                         self.causality.instantaneous_causality)
        self.assertEqual(analog_signal_causality.total_interdependence,
                         self.causality.total_interdependence)

    def test_aic(self):
        identity_matrix = np.eye(2, 2)
        self.assertEqual(elephant.causality.granger._aic(
            identity_matrix, order=2, dimension=2, length=2
        ), 8.0)

    def test_bic(self):
        identity_matrix = np.eye(2, 2)
        assert_array_almost_equal(elephant.causality.granger._bic(
            identity_matrix, order=2, dimension=2, length=2
        ), 5.54517744, decimal=8)

    def test_lag_covariances_error(self):
        """
        Check that if a signal length is shorter than the set max_lag, a
        ValueError is raised.
        """
        short_signals = np.array([[1, 2], [3, 4]])
        self.assertRaises(ValueError,
                          elephant.causality.granger._lag_covariances,
                          short_signals, dimension=2, max_lag=3)

    def test_pairwise_granger_error_null_signals(self):
        null_signals = np.array([[0, 0], [0, 0]])
        self.assertRaises(ValueError,
                          elephant.causality.granger.pairwise_granger,
                          null_signals, max_order=2)

    def test_pairwise_granger_identical_signal(self):
        same_signal = np.hstack([self.signal[:, 0, np.newaxis],
                                 self.signal[:, 0, np.newaxis]])
        self.assertRaises(ValueError,
                          elephant.causality.granger.pairwise_granger,
                          signals=same_signal, max_order=2)

    def test_pairwise_granger_error_1d_array(self):
        array_1d = np.ones(10, dtype=np.float32)
        self.assertRaises(ValueError,
                          elephant.causality.granger.pairwise_granger,
                          array_1d, max_order=2)

    def test_result_namedtuple(self):
        """
        Check if the result of pairwise_granger is in the form of namedtuple.
        """
        # Import the namedtuple class for the result formatting
        from elephant.causality.granger import Causality

        # Check that the output matches the class
        self.assertIsInstance(self.causality, Causality)

    def test_result_directional_causalities_not_negative(self):
        """
        The directional causalities should never be negative.
        """
        self.assertTrue(self.causality.directional_causality_x_y >= 0)
        self.assertTrue(self.causality.directional_causality_y_x >= 0)

    def test_result_instantaneous_causality_not_negative(self):
        """
        The time-series granger instantaneous causality should never assume
        negative values.
        """
        self.assertTrue(self.causality.instantaneous_causality >= 0)

    def test_total_channel_interdependence_equals_sum_of_other_three(self):
        """
        Test if total interdependence is equal to the sum of the other three
        measures. It should be equal. In this test, however, almost equality
        is asserted due to a loss of significance with larger datasets.
        """
        causality_sum = self.causality.directional_causality_x_y \
            + self.causality.directional_causality_y_x \
            + self.causality.instantaneous_causality
        assert_array_almost_equal(self.causality.total_interdependence,
                                  causality_sum, decimal=2)

    def test_all_four_result_values_are_floats(self):
        self.assertIsInstance(self.causality.directional_causality_x_y,
                              float)
        self.assertIsInstance(self.causality.directional_causality_y_x,
                              float)
        self.assertIsInstance(self.causality.instantaneous_causality,
                              float)
        self.assertIsInstance(self.causality.total_interdependence, float)

    def test_ground_truth_vector_autoregressive_model(self):
        """
        Test the output of _optimal_vector_arm against the output of R vars
        generated using VAR(t(signal), lag.max=10, ic='AIC').
        """
        # First equation coefficients from R vars
        first_y1_l1 = 0.8947573989
        first_y2_l1 = -0.0003449514
        first_y1_l2 = -0.4934377020
        first_y2_l2 = -0.0018548490

        # Second equation coefficients from R vars
        second_y1_l1 = 9.009503e-01
        second_y2_l1 = -8.124731e-01
        second_y1_l2 = -1.871460e-01
        second_y2_l2 = -5.012730e-01

        coefficients, _, _ = elephant.causality.granger._optimal_vector_arm(
            self.ground_truth.T, dimension=2, max_order=10,
            information_criterion='aic')

        # Arrange the ground truth values in the same shape as coefficients
        ground_truth_coefficients = np.asarray(
            [[[first_y1_l1, first_y2_l1],
              [second_y1_l1, second_y2_l1]],
             [[first_y1_l2, first_y2_l2],
              [second_y1_l2, second_y2_l2]]]
        )

        assert_array_almost_equal(coefficients, ground_truth_coefficients,
                                  decimal=4)


class ConditionalGrangerTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        np.random.seed(1)
        cls.ground_truth = cls._generate_ground_truth()

    @staticmethod
    def _generate_ground_truth(length_2d=30000, causality_type="indirect"):
        """
        Recreated from Example 2 section 5.2 of :cite:'granger-Ding06-0608035'.
        The following should generate three signals in one of the two ways:
         1. "indirect" would generate data which contains no direct
        causal influence from Y to X, but mediated through Z
        (i.e. Y -> Z -> X).
        2. "both" would generate data which contains both direct and indirect
        causal influences from Y to X.

        """
        if causality_type == "indirect":
            y_t_lag_2 = 0
        elif causality_type == "both":
            y_t_lag_2 = 0.2
        else:
            raise ValueError("causality_type should be either 'indirect' or "
                             "'both'")

        order = 2
        signal = np.zeros((3, length_2d + order))

        weights_1 = np.array([[0.8, 0, 0.4],
                              [0, 0.9, 0],
                              [0., 0.5, 0.5]])

        weights_2 = np.array([[-0.5, y_t_lag_2, 0.],
                              [0., -0.8, 0],
                              [0, 0, -0.2]])

        weights = np.stack((weights_1, weights_2))

        noise_covariance = np.array([[0.3, 0.0, 0.0],
                                     [0.0, 1., 0.0],
                                     [0.0, 0.0, 0.2]])

        for i in range(length_2d):
            for lag in range(order):
                signal[:, i + order] += np.dot(weights[lag],
                                               signal[:, i + 1 - lag])
            rnd_var = np.random.multivariate_normal([0, 0, 0],
                                                    noise_covariance)
            signal[:, i + order] += rnd_var

        signal = signal[:, 2:]

        # Return signals as Nx3
        return signal.T

    def setUp(self):
        # Generate a smaller random dataset for tests other than ground truth,
        # using a different seed than in the ground truth - the convergence
        # should not depend on the seed.
        np.random.seed(10)
        self.signal = self._generate_ground_truth(length_2d=1000)

        # Generate a small dataset for containing both direct and indirect
        # causality.
        self.non_zero_signal = self._generate_ground_truth(
            length_2d=1000, causality_type="both")
        # Estimate Granger causality
        self.conditional_causality = elephant.causality.granger.\
            conditional_granger(self.signal, max_order=10,
                                information_criterion='bic')

    def test_result_is_float(self):
        self.assertIsInstance(self.conditional_causality, float)

    def test_ground_truth_zero_value_conditional_causality(self):
        self.assertEqual(elephant.causality.granger.conditional_granger(
            self.ground_truth, 10, 'bic'), 0.0)

    def test_non_zero_conditional_causality(self):
        self.assertGreater(elephant.causality.granger.conditional_granger(
            self.non_zero_signal, 10, 'bic'), 0.0)


if __name__ == '__main__':
    unittest.main()
