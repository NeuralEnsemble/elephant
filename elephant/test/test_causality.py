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
from neo.core import AnalogSignal
import quantities as pq
from numpy.ma.testutils import assert_array_equal, assert_allclose


class PairwiseGrangerTestCase(unittest.TestCase):
    def setUp(self):
        # Set up that is equivalent to the one in POC granger repository
        np.random.seed(1)
        # length_2d = 10000
        length_2d = 100
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

        self.causality = elephant.causality.granger.pairwise_granger(self.signal, 2)

    def test_analog_signal_input(self):
        """
        Check if analog signal input result matches an otherwise identical 2D
        numpy array input result.
        """
        analog_signal = AnalogSignal(self.signal.T, units='V',
                                     sampling_rate=1*pq.Hz)
        analog_signal_causality = elephant.causality.granger.pairwise_granger(analog_signal, 2)
        self.assertEqual(analog_signal_causality.directional_causality_x_y,
                         self.causality.directional_causality_x_y)
        self.assertEqual(analog_signal_causality.directional_causality_y_x,
                         self.causality.directional_causality_y_x)
        self.assertEqual(analog_signal_causality.instantaneous_causality,
                         self.causality.instantaneous_causality)
        self.assertEqual(analog_signal_causality.total_interdependence,
                         self.causality.total_interdependence)

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
        hc_x_y = np.asarray([-0.5461382])
        hc_y_x = np.asarray([-0.25516382])
        hc_instantaneous_causality = 0.11647517002016418
        hc_total_interdependence = np.asarray([-0.68482685])
        assert_array_almost_equal(self.causality.directional_causality_x_y, hc_x_y, decimal=8)
        assert_array_almost_equal(self.causality.directional_causality_y_x, hc_y_x, decimal=8)
        assert_array_almost_equal(self.causality.instantaneous_causality,
                         hc_instantaneous_causality, decimal=8)
        assert_array_almost_equal(self.causality.total_interdependence,
                         hc_total_interdependence, decimal=8)

    def same_signal_pairwise_granger(self):
        """
        Pass two identical signals to pairwise granger. This should yield zero
        causality for the directional causality metrics.
        Here the (almost) equality is asserted to 15 decimal places.
        """
        same_signal = np.vstack([self.signal[0], self.signal[0]])
        assert_array_almost_equal(self.causality.directional_causality_y_x, 0, decimal=15)
        assert_array_almost_equal(self.causality.directional_causality_x_y, 0, decimal=15)

    @unittest.skipUnless(sys.version_info >= (3, 1), "requires Python 3.1 or above")
    def test_negative_order_parameter(self):
        """
        Use assertRaises as a context manager to catch the ValueError.
        Order parameter should always be a positive integer.

        """
        with self.assertRaises(ValueError):
            causality = elephant.causality.granger.pairwise_granger(self.signal, -1)

    @unittest.skipUnless(sys.version_info >= (3, 1), "requires Python 3.1 or above")
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
        assert_array_almost_equal(self.causality.total_interdependence, causality_sum, decimal=8)

    def test_all_four_result_values_are_numpy_arrays(self):
        self.assertIsInstance(self.causality.directional_causality_x_y,
                              np.ndarray)
        self.assertIsInstance(self.causality.directional_causality_y_x,
                              np.ndarray)
        self.assertIsInstance(self.causality.instantaneous_causality,
                              np.ndarray)
        self.assertIsInstance(self.causality.total_interdependence, np.ndarray)

    def tearDown(self) -> None:
        pass


if __name__ == '__main__':
    unittest.main()
