# -*- coding: utf-8 -*-
"""
Unit tests for the GPFA analysis.

:copyright: Copyright 2014-2020 by the Elephant team, see AUTHORS.txt.
:license: Modified BSD, see LICENSE.txt for details.
"""

import unittest

import neo
import numpy as np
import quantities as pq
from numpy.testing import assert_array_equal, assert_array_almost_equal

from elephant.spike_train_generation import homogeneous_poisson_process

try:
    import sklearn
    from elephant.gpfa import gpfa_util
    from elephant.gpfa import GPFA
    from sklearn.model_selection import cross_val_score
    HAVE_SKLEARN = True
except ImportError:
    HAVE_SKLEARN = False


@unittest.skipUnless(HAVE_SKLEARN, 'requires sklearn')
class GPFATestCase(unittest.TestCase):
    def setUp(self):
        def gen_gamma_spike_train(k, theta, t_max):
            x = []
            for i in range(int(3 * t_max / (k * theta))):
                x.append(np.random.gamma(k, theta))
            s = np.cumsum(x)
            return s[s < t_max]

        def gen_test_data(rates, durs, shapes=(1, 1, 1, 1)):
            s = gen_gamma_spike_train(shapes[0], 1. / rates[0], durs[0])
            for i in range(1, 4):
                s_i = gen_gamma_spike_train(shapes[i], 1. / rates[i], durs[i])
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
            n1 = neo.SpikeTrain(gen_test_data(rates_a, durs), units=1 * pq.s,
                                t_start=0 * pq.s, t_stop=10 * pq.s)
            n2 = neo.SpikeTrain(gen_test_data(rates_a, durs), units=1 * pq.s,
                                t_start=0 * pq.s, t_stop=10 * pq.s)
            n3 = neo.SpikeTrain(gen_test_data(rates_b, durs), units=1 * pq.s,
                                t_start=0 * pq.s, t_stop=10 * pq.s)
            n4 = neo.SpikeTrain(gen_test_data(rates_b, durs), units=1 * pq.s,
                                t_start=0 * pq.s, t_stop=10 * pq.s)
            n5 = neo.SpikeTrain(gen_test_data(rates_a, durs), units=1 * pq.s,
                                t_start=0 * pq.s, t_stop=10 * pq.s)
            n6 = neo.SpikeTrain(gen_test_data(rates_a, durs), units=1 * pq.s,
                                t_start=0 * pq.s, t_stop=10 * pq.s)
            n7 = neo.SpikeTrain(gen_test_data(rates_b, durs), units=1 * pq.s,
                                t_start=0 * pq.s, t_stop=10 * pq.s)
            n8 = neo.SpikeTrain(gen_test_data(rates_b, durs), units=1 * pq.s,
                                t_start=0 * pq.s, t_stop=10 * pq.s)
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
            spike_times = [homogeneous_poisson_process(rate=rate * pq.Hz)
                           for rate in rates]
            self.data2.append(spike_times)

    def test_data1(self):
        gpfa = GPFA(x_dim=self.x_dim, em_max_iters=self.n_iters)
        gpfa.fit(self.data1)
        latent_variable_orth = gpfa.transform(self.data1)
        self.assertAlmostEqual(gpfa.transform_info['log_likelihood'],
                               -8172.004695554373, places=5)
        # Since data1 is inherently 2 dimensional, only the first two
        # dimensions of latent_variable_orth should have finite power.
        for i in [0, 1]:
            self.assertNotEqual(latent_variable_orth[0][i].mean(), 0)
            self.assertNotEqual(latent_variable_orth[0][i].var(), 0)
        for i in [2, 3]:
            self.assertAlmostEqual(latent_variable_orth[0][i].mean(), 0,
                                   places=2)
            self.assertAlmostEqual(latent_variable_orth[0][i].var(), 0,
                                   places=2)

    def test_transform_testing_data(self):
        # check if the num. of neurons in the test data matches the
        # num. of neurons in the training data
        gpfa1 = GPFA(x_dim=self.x_dim, em_max_iters=self.n_iters)
        gpfa1.fit(self.data1)
        with self.assertRaises(ValueError):
            gpfa1.transform(self.data2)

    def test_cross_validation(self):
        # If GPFA.__init__ is decorated, sklearn signature function parsing
        # magic throws the error
        # __init__() got an unexpected keyword argument 'args'
        lls = []
        for x_dim in range(1, self.x_dim + 1):
            gpfa = GPFA(x_dim=x_dim, em_max_iters=self.n_iters)
            lls.append(np.mean(cross_val_score(gpfa, self.data1, cv=5)))
        self.assertTrue(np.argmax(lls) == 1)

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
            gpfa.fit(spiketrains=invalid_data)
        with self.assertRaises(ValueError):
            gpfa.fit(spiketrains=[])

    def test_data2(self):
        gpfa = GPFA(bin_size=self.bin_size, x_dim=8, em_max_iters=self.n_iters)
        gpfa.fit(self.data2)
        n_trials = len(self.data2)
        returned_data = gpfa.valid_data_names
        seqs = gpfa.transform(self.data2, returned_data=returned_data)
        for key, data in seqs.items():
            self.assertEqual(len(data), n_trials,
                             msg="Failed ndarray field {0}".format(key))
        t_start = self.data2[0][0].t_stop
        t_stop = self.data2[0][0].t_start
        n_bins = int(((t_start - t_stop) / self.bin_size).magnitude)
        assert_array_equal(gpfa.transform_info['num_bins'],
                           [n_bins, ] * n_trials)

    def test_returned_data(self):
        gpfa = GPFA(bin_size=self.bin_size, x_dim=8, em_max_iters=self.n_iters)
        gpfa.fit(self.data2)
        seqs = gpfa.transform(self.data2)
        self.assertTrue(isinstance(seqs, np.ndarray))

        returned_data = gpfa.valid_data_names
        seqs = gpfa.transform(self.data2, returned_data=returned_data)
        self.assertTrue(len(returned_data) == len(seqs))
        self.assertTrue(isinstance(seqs, dict))
        with self.assertRaises(ValueError):
            seqs = gpfa.transform(self.data2, returned_data=['invalid_name'])

    def test_fit_transform(self):
        gpfa1 = GPFA(bin_size=self.bin_size, x_dim=self.x_dim,
                     em_max_iters=self.n_iters)
        gpfa1.fit(self.data1)
        latent_variable_orth1 = gpfa1.transform(self.data1)
        latent_variable_orth2 = GPFA(
            bin_size=self.bin_size, x_dim=self.x_dim,
            em_max_iters=self.n_iters).fit_transform(self.data1)
        for i in range(len(self.data1)):
            for j in range(self.x_dim):
                assert_array_almost_equal(latent_variable_orth1[i][j],
                                          latent_variable_orth2[i][j])

    def test_get_seq_sqrt(self):
        data = [self.data2[0]]
        seqs = gpfa_util.get_seqs(data, bin_size=self.bin_size)
        seqs_not_sqrt = gpfa_util.get_seqs(data, bin_size=self.bin_size,
                                           use_sqrt=False)
        self.assertEqual(seqs['T'], seqs_not_sqrt['T'])
        self.assertEqual(seqs['y'].shape, seqs_not_sqrt['y'].shape)

    def test_cut_trials_inf(self):
        same_data = gpfa_util.cut_trials(self.data2, seg_length=np.Inf)
        assert same_data is self.data2

    def test_cut_trials_zero_length(self):
        seqs = gpfa_util.get_seqs(self.data2, bin_size=self.bin_size)
        with self.assertRaises(ValueError):
            gpfa_util.cut_trials(seqs, seg_length=0)

    def test_cut_trials_same_length(self):
        data = [self.data2[0]]
        seqs = gpfa_util.get_seqs(data, bin_size=self.bin_size)
        seg_length = seqs[0]['T']
        seqs_cut = gpfa_util.cut_trials(seqs, seg_length=seg_length)
        assert_array_almost_equal(seqs[0]['y'], seqs_cut[0]['y'])

    def test_cut_trials_larger_length(self):
        data = [self.data2[0]]
        seqs = gpfa_util.get_seqs(data, bin_size=self.bin_size)
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
