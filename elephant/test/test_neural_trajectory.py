# -*- coding: utf-8 -*-
"""
Unit tests for the neural_trajectory analysis.

:copyright: Copyright 2014-2016 by the Elephant team, see AUTHORS.txt.
:license: Modified BSD, see LICENSE.txt for details.
"""

import sys
import unittest

import neo
import numpy as np
import quantities as pq
from numpy.testing import assert_array_equal, assert_array_almost_equal

from elephant.spike_train_generation import homogeneous_poisson_process

try:
    import sklearn
except ImportError:
    HAVE_SKLEARN = False
else:
    HAVE_SKLEARN = True
    from elephant.gpfa_src import gpfa_util
    from elephant.gpfa import gpfa

python_version_major = sys.version_info.major


@unittest.skipUnless(HAVE_SKLEARN, 'requires sklearn')
class NeuralTrajectoryTestCase(unittest.TestCase):
    def setUp(self):
        def gamma_train(k, tetha, t_max):
            x = []

            for i in range(int(t_max * (k * tetha) ** (-1) * 3)):
                x.append(np.random.gamma(k, tetha))

            s = np.cumsum(x)
            idx = np.where(s < t_max)
            s = s[idx]  # Poisson process

            return s

        def h_alt(rate1, rate2, rate3, rate4, c1, c2, c3, T, k1=1, k2=1, k3=1,
                  k4=1):
            teta1 = rate1 ** -1
            teta2 = rate2 ** -1
            teta3 = rate3 ** -1
            teta4 = rate4 ** -1
            s1 = gamma_train(k1, teta1, c1)
            s2 = gamma_train(k2, teta2, c2) + c1
            s3 = gamma_train(k3, teta3, c3) + c1 + c2
            s4 = gamma_train(k4, teta4, T) + c1 + c2 + c3

            return np.concatenate((s1, s2, s3, s4))

        self.data1 = []
        for tr in range(1):
            np.random.seed(tr)
            n1 = neo.SpikeTrain(h_alt(2, 10, 2, 2, 2.5, 2.5, 2.5, 2.5),
                                t_start=0, t_stop=10, units=1*pq.s)
            n2 = neo.SpikeTrain(h_alt(2, 10, 2, 2, 2.5, 2.5, 2.5, 2.5),
                                t_start=0, t_stop=10, units=1*pq.s)
            n3 = neo.SpikeTrain(h_alt(2, 2, 10, 2, 2.5, 2.5, 2.5, 2.5),
                                t_start=0, t_stop=10, units=1*pq.s)
            n4 = neo.SpikeTrain(h_alt(2, 2, 10, 2, 2.5, 2.5, 2.5, 2.5),
                                t_start=0, t_stop=10, units=1*pq.s)
            self.data1.append((tr, [n1, n2, n3, n4]))
        self.x_dim = 4

        # data2 setup
        np.random.seed(27)
        self.data2 = []
        self.n_iters = 10
        self.bin_size = 20 * pq.ms
        self.n_trials = 10
        for trial in range(self.n_trials):
            n_channels = 20
            firing_rates = np.random.randint(low=1, high=100,
                                             size=n_channels)*pq.Hz
            spike_times = [homogeneous_poisson_process(rate=rate)
                           for rate in firing_rates]
            self.data2.append((trial, spike_times))

    def test_data1(self):
        params_est, seqs_train, seqs_test, fit_info = gpfa(
            self.data1, x_dim=self.x_dim, em_max_iters=self.n_iters)
        self.assertEqual(fit_info['bin_size'], 20*pq.ms)
        assert_array_equal(fit_info['has_spikes_bool'], np.array([True] * 4))
        self.assertAlmostEqual(fit_info['log_likelihood'], -27.222600197474762)
        self.assertEqual(fit_info['min_var_frac'], 0.01)

    def test_invalid_bin_size_type(self):
        invalid_bin_size = 10
        self.assertRaises(ValueError, gpfa, data=self.data2,
                          bin_size=invalid_bin_size)

    def test_invalid_input_data(self):
        invalid_data = [(0, [0, 1, 2])]
        self.assertRaises(ValueError, gpfa, data=invalid_data)

    def test_data2(self):
        params_est, seqs_train, seqs_test, fit_info = gpfa(
            self.data2, bin_size=self.bin_size, x_dim=8,
            em_max_iters=self.n_iters)
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

    def test_get_seq_sqrt(self):
        data = [self.data2[0]]
        seqs = gpfa_util.get_seq(data, bin_size=self.bin_size)
        seqs_not_sqrt = gpfa_util.get_seq(data, bin_size=self.bin_size,
                                          use_sqrt=False)
        for common_key in ('trialId', 'T'):
            self.assertEqual(seqs[common_key], seqs_not_sqrt[common_key])
        self.assertEqual(seqs['y'].shape, seqs_not_sqrt['y'].shape)

    def test_cut_trials_inf(self):
        same_data = gpfa_util.cut_trials(self.data2, seg_length=np.Inf)
        assert same_data is self.data2

    def test_cut_trials_zero_length(self):
        seqs = gpfa_util.get_seq(self.data2, bin_size=self.bin_size)
        with self.assertRaises(AssertionError):
            gpfa_util.cut_trials(seqs, seg_length=0)

    def test_cut_trials_same_length(self):
        data = [self.data2[0]]
        seqs = gpfa_util.get_seq(data, bin_size=self.bin_size)
        seg_length = seqs[0]['T']
        seqs_cut = gpfa_util.cut_trials(seqs, seg_length=seg_length)
        assert_array_almost_equal(seqs[0]['y'], seqs_cut[0]['y'])

    @unittest.skipUnless(python_version_major == 3, "assertWarns requires 3.2")
    def test_cut_trials_larger_length(self):
        data = [self.data2[0]]
        seqs = gpfa_util.get_seq(data, bin_size=self.bin_size)
        seg_length = seqs[0]['T'] + 1
        with self.assertWarns(UserWarning):
            gpfa_util.cut_trials(seqs, seg_length=seg_length)

    @unittest.skip("Test are broken. Consider removing. Or fix it.")
    def test_make_k_big(self):
        seqs = gpfa_util.get_seq(self.data2, bin_size=self.bin_size)
        parameter_estimates, _, _ = gpfa.gpfa_engine(seqs,
                                                     seq_test=[],
                                                     em_max_iters=self.n_iters)
        timesteps = len(parameter_estimates['eps'])
        parameter_estimates['a'] = np.random.random(timesteps)
        for covType in ('rbf', 'tri', 'logexp'):
            parameter_estimates['covType'] = covType
            # see if there is no errors
            gpfa_util.make_k_big(params=parameter_estimates,
                                 n_timesteps=timesteps)


def suite():
    suite = unittest.makeSuite(NeuralTrajectoryTestCase, 'test')
    return suite


def run():
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())


if __name__ == "__main__":
    unittest.main()
