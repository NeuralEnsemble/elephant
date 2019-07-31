# -*- coding: utf-8 -*-
"""
Unit tests for the GPFA analysis.

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
class GPFATestCase(unittest.TestCase):
    def setUp(self):
        def gen_gamma_spike_train(k, theta, t_max):
            x = []
            for i in range(int(3 * t_max / (k*theta))):
                x.append(np.random.gamma(k, theta))
            s = np.cumsum(x)
            return s[s < t_max]

        def gen_test_data(rates, durs, shapes=(1, 1, 1, 1)):
            s = gen_gamma_spike_train(shapes[0], 1./rates[0], durs[0])
            for i in range(1, 4):
                s_i = gen_gamma_spike_train(shapes[i], 1./rates[i], durs[i])
                s = np.concatenate([s, s_i + np.sum(durs[:i])])
            return s

        self.n_iters = 10
        self.bin_size = 20 * pq.ms

        # generate data1
        rates_a = (2, 10, 2, 2)
        rates_b = (2, 2, 10, 2)
        durs = (2.5, 2.5, 2.5, 2.5)
        self.data1 = []
        for tr in range(1):
            np.random.seed(tr)
            n1 = neo.SpikeTrain(gen_test_data(rates_a, durs), units=1*pq.s,
                                t_start=0*pq.s, t_stop=10*pq.s)
            n2 = neo.SpikeTrain(gen_test_data(rates_a, durs), units=1*pq.s,
                                t_start=0*pq.s, t_stop=10*pq.s)
            n3 = neo.SpikeTrain(gen_test_data(rates_b, durs), units=1*pq.s,
                                t_start=0*pq.s, t_stop=10*pq.s)
            n4 = neo.SpikeTrain(gen_test_data(rates_b, durs), units=1*pq.s,
                                t_start=0*pq.s, t_stop=10*pq.s)
            self.data1.append((tr, [n1, n2, n3, n4]))
        self.x_dim = 4

        # generate data2
        np.random.seed(27)
        self.data2 = []
        n_trials = 10
        n_channels = 20
        for trial in range(n_trials):
            rates = np.random.randint(low=1, high=100, size=n_channels)
            spike_times = [homogeneous_poisson_process(rate=rate*pq.Hz)
                           for rate in rates]
            self.data2.append((trial, spike_times))

    def test_data1(self):
        params_est, seqs_train, fit_info = gpfa(
            self.data1, x_dim=self.x_dim, em_max_iters=self.n_iters)
        self.assertEqual(fit_info['bin_size'], 20*pq.ms)
        self.assertEqual(fit_info['min_var_frac'], 0.01)
        self.assertTrue(np.all(fit_info['has_spikes_bool']))
        self.assertAlmostEqual(fit_info['log_likelihood'], -27.222600197474762)
        # Since data1 is inherently 2 dimensional, only the first two
        # dimensions of xorth should have finite power.
        for i in [0, 1]:
            self.assertNotEqual(seqs_train['xorth'][0][i].mean(), 0)
            self.assertNotEqual(seqs_train['xorth'][0][i].var(), 0)
        for i in [2, 3]:
            self.assertEqual(seqs_train['xorth'][0][i].mean(), 0)
            self.assertEqual(seqs_train['xorth'][0][i].var(), 0)

    def test_invalid_input_data(self):
        invalid_data = [(0, [0, 1, 2])]
        invalid_bin_size = 10
        with self.assertRaises(ValueError):
            gpfa(data=invalid_data)
        with self.assertRaises(ValueError):
            gpfa(data=[])
        with self.assertRaises(ValueError):
            gpfa(data=self.data2, bin_size=invalid_bin_size)

    def test_data2(self):
        params_est, seqs_train, fit_info = gpfa(
            self.data2, bin_size=self.bin_size, x_dim=8,
            em_max_iters=self.n_iters)
        self.assertEqual(fit_info['bin_size'], self.bin_size,
                         "Input and output bin_size don't match")
        n_trials = len(self.data2)
        t_start = self.data2[0][1][0].t_stop
        t_stop = self.data2[0][1][0].t_start
        n_bins = int(((t_start - t_stop) / self.bin_size).magnitude)
        assert_array_equal(seqs_train['T'], [n_bins,] * n_trials)
        assert_array_equal(seqs_train['trialId'], np.arange(n_trials))
        for key in ['y', 'xsm', 'Vsm', 'VsmGP', 'xorth']:
            self.assertEqual(len(seqs_train[key]), n_trials,
                             msg="Failed ndarray field {0}".format(key))
        self.assertEqual(len(seqs_train), n_trials)

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
        with self.assertRaises(ValueError):
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

    def test_logdet(self):
        np.random.seed(27)
        # generate a positive definite matrix
        matrix = np.random.randn(20, 20)
        matrix = matrix.dot(matrix.T)
        logdet_fast = gpfa_util.logdet(matrix)
        logdet_ground_truth = np.log(np.linalg.det(matrix))
        assert_array_almost_equal(logdet_fast, logdet_ground_truth)


def suite():
    suite = unittest.makeSuite(GPFATestCase, 'test')
    return suite


def run():
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())


if __name__ == "__main__":
    unittest.main()
