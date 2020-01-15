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
    from elephant.gpfa import gpfa_util
    from elephant.gpfa import GPFA
    from sklearn.model_selection import cross_val_score

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
        np.random.seed(0)
        n_trials = 100
        self.data0 = []
        for trial in range(n_trials):
            n1 = neo.SpikeTrain(gen_test_data(rates_a, durs), units=1*pq.s,
                                t_start=0*pq.s, t_stop=10*pq.s)
            n2 = neo.SpikeTrain(gen_test_data(rates_a, durs), units=1*pq.s,
                                t_start=0*pq.s, t_stop=10*pq.s)
            n3 = neo.SpikeTrain(gen_test_data(rates_b, durs), units=1*pq.s,
                                t_start=0*pq.s, t_stop=10*pq.s)
            n4 = neo.SpikeTrain(gen_test_data(rates_b, durs), units=1*pq.s,
                                t_start=0*pq.s, t_stop=10*pq.s)
            n5 = neo.SpikeTrain(gen_test_data(rates_a, durs), units=1*pq.s,
                                t_start=0*pq.s, t_stop=10*pq.s)
            n6 = neo.SpikeTrain(gen_test_data(rates_a, durs), units=1*pq.s,
                                t_start=0*pq.s, t_stop=10*pq.s)
            n7 = neo.SpikeTrain(gen_test_data(rates_b, durs), units=1*pq.s,
                                t_start=0*pq.s, t_stop=10*pq.s)
            n8 = neo.SpikeTrain(gen_test_data(rates_b, durs), units=1*pq.s,
                                t_start=0*pq.s, t_stop=10*pq.s)
            self.data0.append([n1, n2, n3, n4, n5, n6, n7, n8])
        self.x_dim = 4

        self.data1 = self.data0[:20]

        # generate data2
        np.random.seed(27)
        self.data2 = []
        n_trials = 10
        n_channels = 20
        for trial in range(n_trials):
            rates = np.random.randint(low=1, high=100, size=n_channels)
            spike_times = [homogeneous_poisson_process(rate=rate*pq.Hz)
                           for rate in rates]
            self.data2.append(spike_times)

    def test_data1(self):
        gpfa = GPFA(x_dim=self.x_dim, em_max_iters=self.n_iters)
        gpfa.fit(self.data1)
        xorth = gpfa.transform(self.data1)
        self.assertEqual(gpfa.fit_info['bin_size'], 20*pq.ms)
        self.assertEqual(gpfa.fit_info['min_var_frac'], 0.01)
        self.assertTrue(np.all(gpfa.fit_info['has_spikes_bool']))
        self.assertAlmostEqual(gpfa.fit_info['log_likelihood'], -8172.004695554373)
        # Since data1 is inherently 2 dimensional, only the first two
        # dimensions of xorth should have finite power.
        for i in [0, 1]:
            self.assertNotEqual(xorth[0][i].mean(), 0)
            self.assertNotEqual(xorth[0][i].var(), 0)
        for i in [2, 3]:
            self.assertAlmostEqual(xorth[0][i].mean(), 0, places=2)
            self.assertAlmostEqual(xorth[0][i].var(), 0, places=2)

    def test_transform_testing_data(self):
        gpfa1 = GPFA(x_dim=self.x_dim, em_max_iters=self.n_iters)
        gpfa1.fit(self.data1)
        with self.assertRaises(ValueError):
            gpfa1.transform(self.data2)

        gpfa1 = GPFA(x_dim=self.x_dim, em_max_iters=self.n_iters)
        gpfa1.fit(self.data0)
        xorth1 = gpfa1.transform(self.data0)

        gpfa2 = GPFA(x_dim=self.x_dim, em_max_iters=self.n_iters)
        gpfa2.fit(self.data0[:-2])
        xorth2 = gpfa2.transform(self.data0[-2:])

        # we expect better consistency for the first 2 dimensions then for
        # the rest, as the data is inherently 2 dimensional
        # TODO: is this a good test?
        assert_array_almost_equal(xorth1[-1][0], xorth2[-1][0], decimal=2)
        assert_array_almost_equal(xorth1[-1][1], xorth2[-1][1], decimal=2)
        assert_array_almost_equal(xorth1[-1][2], xorth2[-1][2], decimal=1)
        assert_array_almost_equal(xorth1[-1][3], xorth2[-1][3], decimal=1)

    def test_cross_validation(self):
        lls = []
        for x_dim in range(1, self.x_dim+1):
            gpfa = GPFA(x_dim=x_dim, em_max_iters=self.n_iters)
            lls.append(np.mean(cross_val_score(gpfa, self.data1, cv=5)))
        self.assertTrue(np.argmax(lls)==1)

    def test_invalid_input_data(self):
        invalid_data = [(0, [0, 1, 2])]
        invalid_bin_size = 10
        invalid_tau_init = 100
        with self.assertRaises(ValueError):
            _ = GPFA(bin_size=invalid_bin_size)
        with self.assertRaises(ValueError):
            _ = GPFA(tau_init=invalid_tau_init)
        gpfa = GPFA()
        with self.assertRaises(ValueError):
            gpfa.fit(data=invalid_data)
        with self.assertRaises(ValueError):
            gpfa.fit(data=[])

    def test_data2(self):
        gpfa = GPFA(bin_size=self.bin_size, x_dim=8, em_max_iters=self.n_iters)
        gpfa.fit(self.data2)
        returned_data = ['y', 'xsm', 'Vsm', 'VsmGP', 'xorth']
        seqs = gpfa.transform(self.data2, returned_data=returned_data)
        self.assertEqual(gpfa.fit_info['bin_size'], self.bin_size,
                         "Input and output bin_size don't match")
        n_trials = len(self.data2)
        t_start = self.data2[0][0].t_stop
        t_stop = self.data2[0][0].t_start
        n_bins = int(((t_start - t_stop) / self.bin_size).magnitude)
        assert_array_equal(gpfa.T, [n_bins,] * n_trials)
        for key, data in seqs.items():
            self.assertEqual(len(data), n_trials, msg="Failed ndarray field {0}".format(key))

    def test_fit_transform(self):
        gpfa1 = GPFA(bin_size=self.bin_size, x_dim=self.x_dim, em_max_iters=self.n_iters)
        gpfa1.fit(self.data1)
        xorth1 = gpfa1.transform(self.data1)
        xorth2 = GPFA(bin_size=self.bin_size, x_dim=self.x_dim, em_max_iters=self.n_iters).fit_transform(self.data1)
        for i in range(len(self.data1)):
            for j in range(self.x_dim):
                assert_array_almost_equal(xorth1[i][j], xorth2[i][j])

    def test_get_seq_sqrt(self):
        data = [self.data2[0]]
        seqs = gpfa_util.get_seq(data, bin_size=self.bin_size)
        seqs_not_sqrt = gpfa_util.get_seq(data, bin_size=self.bin_size,
                                          use_sqrt=False)
        self.assertEqual(seqs['T'], seqs_not_sqrt['T'])
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
