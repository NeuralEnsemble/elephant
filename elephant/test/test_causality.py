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
import neo
import numpy as np
import scipy.signal as spsig
import scipy.stats
from numpy.testing.utils import assert_array_almost_equal
import quantities as pq
from numpy.ma.testutils import assert_array_equal, assert_allclose


class PairwiseGrangerTestCase(unittest.TestCase):
    def setUp(self):
        # Set up that is equivalent to the one in POC granger repository
        np.random.seed(1)
        length_2d = 10000
        self.signal = np.zeros((2, length_2d))

        order = 2
        weights_1 = np.array([[0.9, 0], [0.9, -0.8]])
        weights_2 = np.array([[-0.5, 0], [-0.2, -0.5]])

        weights = np.stack((weights_1, weights_2))

        noise_covariance = np.array([[1., 0.0], [0.0, 1.]])

        for i in range(length_2d):
            for lag in range(order):
                self.signal[:, i] += np.dot(weights[lag], self.signal[:, i - lag - 1])
            rnd_var = np.random.multivariate_normal([0, 0], noise_covariance)
            self.signal[0, i] += rnd_var[0]
            self.signal[1, i] += rnd_var[1]

    def test_analog_signal_input(self):
        pass

    def test_numpy_array(self):
        pass

    def test_lag_covariances(self):
        # Not essential
        # Passing a signal with variance of 0, should equal 0
        pass

    def test_vector_arm(self):
        # Not essential
        # Test a static signal that could not possibly predict itself
        # Test white noise?
        pass

    def test_yule_walker(self):
        # Not essential
        # Some unit tests from statsmodels for inspiration
        # https://github.com/statsmodels/statsmodels/blob/master/statsmodels/tsa/tests/test_tsa_tools.py
        # https://github.com/statsmodels/statsmodels/blob/master/statsmodels/tsa/tests/test_stattools.py
        pass

    def test_basic_pairwise_granger(self):
        """
        Test the results of pairwise granger against hardcoded values produced
        by the Granger proof-of-concept script.
        """
        causality = elephant.causality.granger.pairwise_granger(self.signal, 2)
        hc_x_y = -0.13913054
        hc_y_x = -1.3364389
        hc_instantaneous_causality = 0.8167648618784381
        hc_total_interdependence = -0.65880457
        self.assertEqual(causality.directional_causality_x_y, hc_x_y)
        self.assertEqual(causality.directional_causality_y_x, hc_y_x)
        self.assertEqual(causality.instantaneous_causality,
                         hc_instantaneous_causality)
        self.assertEqual(causality.total_interdependence,
                         hc_total_interdependence)

    def same_signal_pairwise_granger(self):
        """
        Pass two identical signals to pairwise granger. This should yield zero
        causality for the directional causality metrics.
        Here the (almost) equality is asserted to 15 decimal places.
        """
        same_signal = np.vstack([self.signal[0], self.signal[0]])
        causality = elephant.causality.granger.pairwise_granger(same_signal, 2)
        self.assertAlmostEqual(causality.directional_causality_y_x, 0, places=15)
        self.assertAlmostEqual(causality.directional_causality_x_y, 0, places=15)

    @unittest.skipUnless(sys.version_info >= (3, 1))  # Python 3.1 or above
    def test_negative_order_parameter(self):
        """
        Use assertRaises as a context manager to catch the ValueError.
        Order parameter should always be a positive integer.

        """
        with self.assertRaises(ValueError):
            causality = elephant.causality.granger.pairwise_granger(self.signal, -1)

    @unittest.skipUnless(sys.version_info >= (3, 1))  # Python 3.1 or above
    def test_result_namedtuple(self):
        """
        Check if the result of pairwise_granger is in the form of namedtuple.
        """
        # Import the namedtuple class for the result formatting
        from elephant.causality.granger import Causality

        # Calculate the pairwise_granger result
        causality = elephant.causality.granger.pairwise_granger(self.signal, 2)

        # Check that the output matches the class
        self.assertIsInstance(causality, Causality)

    def test_result_directional_causalities_not_negative(self):
        """
        The directional causalities should never be negative.
        """
        causality = elephant.causality.granger.pairwise_granger(self.signal, 2)
        self.assertFalse(causality.directional_causality_x_y < 0)
        self.assertFalse(causality.directional_causality_y_x < 0)

    def test_result_instantaneous_causality_not_negative(self):
        """
        The time-series granger instantaneous causality should never assume
        negative values.
        """
        causality = elephant.causality.granger.pairwise_granger(self.signal, 2)
        self.assertFalse(causality.instantaneous_causality < 0)

    def test_total_channel_interdependence_equals_sum_of_other_three(self):
        """
        Test if total interdependence is equal to the sum of the other three
        measures. It should be equal.
        """
        causality = elephant.causality.granger.pairwise_granger(self.signal, 2)
        causality_sum = causality.directional_causality_x_y \
                        + causality.directional_causality_y_x \
                        + causality.instantaneous_causality
        self.assertEqual(causality.total_interdependence, causality_sum)

    def tearDown(self) -> None:
        pass


if __name__ == '__main__':
    unittest.main()
