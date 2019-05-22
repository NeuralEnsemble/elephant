# -*- coding: utf-8 -*-
"""
Unit tests for the GPFA module.

:copyright: Copyright 2014-2019 by the Elephant team, see AUTHORS.txt.
:license: Modified BSD, see LICENSE.txt for details.
"""
from __future__ import division

import unittest

import numpy as np
import quantities as pq
from numpy.testing.utils import assert_array_equal

from elephant.gpfa.neural_trajectory import neural_trajectory
from elephant.spike_train_generation import homogeneous_poisson_process


class GPFATest(unittest.TestCase):

    def setUp(self):
        np.random.seed(27)
        self.data = []
        self.n_iters = 10
        self.bin_size = 20 * pq.ms
        self.n_trials = 10
        for trial in range(self.n_trials):
            n_channels = 20
            firing_rates = np.random.randint(low=1, high=100,
                                             size=n_channels) * pq.Hz
            spike_times = [homogeneous_poisson_process(rate=rate)
                           for rate in firing_rates]
            self.data.append((trial, spike_times))

    def test_neural_trajectory(self):
        method = 'gpfa'
        params_est, seqs_train, seqs_test, fit_info = neural_trajectory(
            self.data, method=method, bin_size=self.bin_size, x_dim=8,
            em_max_iters=self.n_iters)
        self.assertEqual(fit_info['method'], method,
                         "Input and output methods don't match")
        self.assertEqual(fit_info['bin_size'], self.bin_size,
                         "Input and output bin_size don't match")
        n_bins = 50
        assert_array_equal(seqs_train['T'], np.repeat(n_bins,
                                                      repeats=self.n_trials))
        assert_array_equal(seqs_train['trialId'], np.arange(self.n_trials))
        for key in ['y', 'xsm', 'Vsm', 'VsmGP', 'xorth']:
            self.assertEqual(len(seqs_train[key]), self.n_trials,
                             msg="Failed ndarray field {0}".format(key))
        self.assertEqual(len(seqs_train), self.n_trials)

    def test_invalid_input_type(self):
        self.assertRaises(ValueError, neural_trajectory, data=self.data,
                          bin_size=10)
