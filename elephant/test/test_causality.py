# -*- coding: utf-8 -*-
"""
Unit tests for the causality module.

:copyright: Copyright 2014-2020 by the Elephant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
"""
from __future__ import division, print_function

import unittest
import elephant.causality.granger

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
        # Passing a signal with variance of 0, should equal 0
        pass

    def test_vector_arm(self):
        # Test a static signal that could not possibly predict itself
        # Test white noise?
        pass

    def test_yule_walker(self):
        # Some unit tests from statsmodels for inspiration
        # https://github.com/statsmodels/statsmodels/blob/master/statsmodels/tsa/tests/test_tsa_tools.py
        # https://github.com/statsmodels/statsmodels/blob/master/statsmodels/tsa/tests/test_stattools.py
        pass

    def test_basic_pairwise_granger(self):
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

    def test_total_interdependence_relates_to_coherence(self):
        pass

    def same_signal_pairwise_granger(self):
        # Pass two instances of the same signal, should yield 0
        pass

    def test_negative_order_parameter(self):
        # Orders or lags should always be positive
        pass

    def test_namedTuple_result_output(self):
        pass

    def test_total_channel_interdependence(self):
        # Total interdependence should be equal to the sum of the other three values
        pass

    def tearDown(self) -> None:
        pass


if __name__ == '__main__':
    unittest.main()
