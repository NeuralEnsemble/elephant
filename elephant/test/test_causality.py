# -*- coding: utf-8 -*-
"""
Unit tests for the causality module.

:copyright: Copyright 2014-2020 by the Elephant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
"""
from __future__ import division, print_function

import unittest
import elephant.causality.granger

import sys
import numpy as np
from numpy.testing.utils import assert_array_almost_equal
from neo.core import AnalogSignal
import quantities as pq


class PairwiseGrangerTestCase(unittest.TestCase):
    def setUp(self):
        """
        The ground truth dataset was generated using the following script:
        >>>np.random.seed(1)
        >>>length_2d = 30000
        >>>signal = np.zeros((2, length_2d))

        >>>order = 2
        >>>weights_1 = np.array([[0.9, 0], [0.9, -0.8]])
        >>>weights_2 = np.array([[-0.5, 0], [-0.2, -0.5]])

        >>>weights = np.stack((weights_1, weights_2))

        >>>noise_covariance = np.array([[1., 0.0], [0.0, 1.]])

        >>>for i in range(length_2d):
        >>>    for lag in range(order):
        >>>        signal[:, i] += np.dot(weights[lag],
        >>>                               signal[:, i - lag - 1])
        >>>    rnd_var = np.random.multivariate_normal([0, 0],
        >>>    noise_covariance)
        >>>    signal[0, i] += rnd_var[0]
        >>>    signal[1, i] += rnd_var[1]

        >>>np.save('/home/jurkus/granger_timeseries_groundtruth_data', signal)
        """
        # Load ground truth
        self.ground_truth = \
            np.load('/home/jurkus/granger_timeseries_groundtruth_data.npy')

        # Generate a smaller random dataset for tests other than ground truth
        length_2d = 1000
        self.signal = np.zeros((2, length_2d))

        order = 2
        weights_1 = np.array([[0.9, 0], [0.9, -0.8]])
        weights_2 = np.array([[-0.5, 0], [-0.2, -0.5]])

        weights = np.stack((weights_1, weights_2))

        noise_covariance = np.array([[1., 0.0], [0.0, 1.]])

        for i in range(length_2d):
            for lag in range(order):
                self.signal[:, i] += np.dot(weights[lag],
                                            self.signal[:, i - lag - 1])
            rnd_var = np.random.multivariate_normal([0, 0], noise_covariance)
            self.signal[0, i] += rnd_var[0]
            self.signal[1, i] += rnd_var[1]

        # Estimate Granger causality
        self.causality = elephant.causality.granger.pairwise_granger(
            self.signal, max_order=10,
            information_criterion=elephant.causality.granger.bic)

    def test_analog_signal_input(self):
        """
        Check if analog signal input result matches an otherwise identical 2D
        numpy array input result.
        """
        analog_signal = AnalogSignal(self.signal.T, units='V',
                                     sampling_rate=1*pq.Hz)
        analog_signal_causality = \
            elephant.causality.granger.pairwise_granger(
                analog_signal, max_order=10,
                information_criterion=elephant.causality.granger.bic)
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
        self.assertEqual(elephant.causality.granger.aic(
            identity_matrix, order=2, dimension=2, length=2
        ), 8.0)

    def test_bic(self):
        identity_matrix = np.eye(2, 2)
        assert_array_almost_equal(elephant.causality.granger.bic(
            identity_matrix, order=2, dimension=2, length=2
        ), 5.545177444479562, decimal=15)

    @unittest.skipUnless(sys.version_info >= (2, 7),
                         "requires Python 2.7 or above")
    def test_lag_covariances_error(self):
        """
        Check that if a signal length is shorter than the set max_lag, a
        ValueError is raised.
        """
        short_signals = np.array([[1, 2], [3, 4]])
        with self.assertRaises(ValueError):
            elephant.causality.granger._lag_covariances(
                short_signals, dimension=2, max_lag=3)

    @unittest.skipUnless(sys.version_info >= (2, 7),
                         "requires Python 2.7 or above")
    def test_pairwise_granger_error(self):
        null_signals = np.array([[0, 0], [0, 0]])
        with self.assertRaises(ValueError):
            elephant.causality.granger.pairwise_granger(
                null_signals, max_order=2)

    @unittest.skipUnless(sys.version_info >= (3, 1),
                         "requires Python 3.1 or above")
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
        self.assertFalse(self.causality.directional_causality_x_y < 0)
        self.assertFalse(self.causality.directional_causality_y_x < 0)

    def test_result_instantaneous_causality_not_negative(self):
        """
        The time-series granger instantaneous causality should never assume
        negative values.
        """
        self.assertFalse(self.causality.instantaneous_causality < 0)

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
            self.ground_truth, dimension=2, max_order=10,
            information_criterion=elephant.causality.granger.aic)

        # Arrange the ground truth values in the same shape as coefficients
        ground_truth_coefficients = np.asarray(
            [[[first_y1_l1, first_y2_l1],
              [second_y1_l1, second_y2_l1]],
             [[first_y1_l2, first_y2_l2],
              [second_y1_l2, second_y2_l2]]]
        )

        assert_array_almost_equal(coefficients, ground_truth_coefficients,
                                  decimal=4)

    def tearDown(self) -> None:
        pass


if __name__ == '__main__':
    unittest.main()