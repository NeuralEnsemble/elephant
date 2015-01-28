# -*- coding: utf-8 -*-
"""
docstring goes here.

:copyright: Copyright 2014 by the Elephant team, see AUTHORS.txt.
:license: Modified BSD, see LICENSE.txt for details.
"""

import unittest

import neo
import numpy as np
from numpy.testing.utils import assert_array_almost_equal, assert_array_equal
import quantities as pq

import elephant.statistics as es


class isi_TestCase(unittest.TestCase):
    def setUp(self):
        self.test_array_2d = np.array([[0.3, 0.56, 0.87, 1.23],
                                       [0.02, 0.71, 1.82, 8.46],
                                       [0.03, 0.14, 0.15, 0.92]])
        self.targ_array_2d_0 = np.array([[-0.28,  0.15,  0.95,  7.23],
                                         [0.01, -0.57, -1.67, -7.54]])
        self.targ_array_2d_1 = np.array([[0.26, 0.31, 0.36],
                                         [0.69, 1.11, 6.64],
                                         [0.11, 0.01, 0.77]])
        self.targ_array_2d_default = self.targ_array_2d_1

        self.test_array_1d = self.test_array_2d[0, :]
        self.targ_array_1d = self.targ_array_2d_1[0, :]

    def test_isi_with_spiketrain(self):
        st = neo.SpikeTrain(self.test_array_1d, units='ms', t_stop=10.0)
        target = pq.Quantity(self.targ_array_1d, 'ms')
        res = es.isi(st)
        assert_array_almost_equal(res, target, decimal=9)

    def test_isi_with_quantities_1d(self):
        st = pq.Quantity(self.test_array_1d, units='ms')
        target = pq.Quantity(self.targ_array_1d, 'ms')
        res = es.isi(st)
        assert_array_almost_equal(res, target, decimal=9)

    def test_isi_with_plain_array_1d(self):
        st = self.test_array_1d
        target = self.targ_array_1d
        res = es.isi(st)
        assert not isinstance(res, pq.Quantity)
        assert_array_almost_equal(res, target, decimal=9)

    def test_isi_with_plain_array_2d_default(self):
        st = self.test_array_2d
        target = self.targ_array_2d_default
        res = es.isi(st)
        assert not isinstance(res, pq.Quantity)
        assert_array_almost_equal(res, target, decimal=9)

    def test_isi_with_plain_array_2d_0(self):
        st = self.test_array_2d
        target = self.targ_array_2d_0
        res = es.isi(st, axis=0)
        assert not isinstance(res, pq.Quantity)
        assert_array_almost_equal(res, target, decimal=9)

    def test_isi_with_plain_array_2d_1(self):
        st = self.test_array_2d
        target = self.targ_array_2d_1
        res = es.isi(st, axis=1)
        assert not isinstance(res, pq.Quantity)
        assert_array_almost_equal(res, target, decimal=9)


class isi_cv_TestCase(unittest.TestCase):
    def setUp(self):
        self.test_array_regular = np.arange(1, 6)

    def test_cv_isi_regular_spiketrain_is_zero(self):
        st = neo.SpikeTrain(self.test_array_regular,  units='ms', t_stop=10.0)
        targ = 0.0
        res = es.cv(es.isi(st))
        self.assertEqual(res, targ)

    def test_cv_isi_regular_array_is_zero(self):
        st = self.test_array_regular
        targ = 0.0
        res = es.cv(es.isi(st))
        self.assertEqual(res, targ)


class mean_firing_rate_TestCase(unittest.TestCase):
    def setUp(self):
        self.test_array_3d = np.ones([5, 7, 13])
        self.test_array_2d = np.array([[0.3, 0.56, 0.87, 1.23],
                                       [0.02, 0.71, 1.82, 8.46],
                                       [0.03, 0.14, 0.15, 0.92]])

        self.targ_array_2d_0 = np.array([3, 3, 3, 3])
        self.targ_array_2d_1 = np.array([4, 4, 4])
        self.targ_array_2d_None = 12
        self.targ_array_2d_default = self.targ_array_2d_None

        self.max_array_2d_0 = np.array([0.3, 0.71, 1.82, 8.46])
        self.max_array_2d_1 = np.array([1.23, 8.46, 0.92])
        self.max_array_2d_None = 8.46
        self.max_array_2d_default = self.max_array_2d_None

        self.test_array_1d = self.test_array_2d[0, :]
        self.targ_array_1d = self.targ_array_2d_1[0]
        self.max_array_1d = self.max_array_2d_1[0]

    def test_mean_firing_rate_with_spiketrain(self):
        st = neo.SpikeTrain(self.test_array_1d, units='ms', t_stop=10.0)
        target = pq.Quantity(self.targ_array_1d/10., '1/ms')
        res = es.mean_firing_rate(st)
        assert_array_almost_equal(res, target, decimal=9)

    def test_mean_firing_rate_with_spiketrain_set_ends(self):
        st = neo.SpikeTrain(self.test_array_1d, units='ms', t_stop=10.0)
        target = pq.Quantity(2/0.5, '1/ms')
        res = es.mean_firing_rate(st, t_start=0.4, t_stop=0.9)
        assert_array_almost_equal(res, target, decimal=9)

    def test_mean_firing_rate_with_quantities_1d(self):
        st = pq.Quantity(self.test_array_1d, units='ms')
        target = pq.Quantity(self.targ_array_1d/self.max_array_1d, '1/ms')
        res = es.mean_firing_rate(st)
        assert_array_almost_equal(res, target, decimal=9)

    def test_mean_firing_rate_with_quantities_1d_set_ends(self):
        st = pq.Quantity(self.test_array_1d, units='ms')
        target = pq.Quantity(2/0.6, '1/ms')
        res = es.mean_firing_rate(st, t_start=400*pq.us, t_stop=1.)
        assert_array_almost_equal(res, target, decimal=9)

    def test_mean_firing_rate_with_plain_array_1d(self):
        st = self.test_array_1d
        target = self.targ_array_1d/self.max_array_1d
        res = es.mean_firing_rate(st)
        assert not isinstance(res, pq.Quantity)
        assert_array_almost_equal(res, target, decimal=9)

    def test_mean_firing_rate_with_plain_array_1d_set_ends(self):
        st = self.test_array_1d
        target = self.targ_array_1d/(1.23-0.3)
        res = es.mean_firing_rate(st, t_start=0.3, t_stop=1.23)
        assert not isinstance(res, pq.Quantity)
        assert_array_almost_equal(res, target, decimal=9)

    def test_mean_firing_rate_with_plain_array_2d_default(self):
        st = self.test_array_2d
        target = self.targ_array_2d_default/self.max_array_2d_default
        res = es.mean_firing_rate(st)
        assert not isinstance(res, pq.Quantity)
        assert_array_almost_equal(res, target, decimal=9)

    def test_mean_firing_rate_with_plain_array_2d_0(self):
        st = self.test_array_2d
        target = self.targ_array_2d_0/self.max_array_2d_0
        res = es.mean_firing_rate(st, axis=0)
        assert not isinstance(res, pq.Quantity)
        assert_array_almost_equal(res, target, decimal=9)

    def test_mean_firing_rate_with_plain_array_2d_1(self):
        st = self.test_array_2d
        target = self.targ_array_2d_1/self.max_array_2d_1
        res = es.mean_firing_rate(st, axis=1)
        assert not isinstance(res, pq.Quantity)
        assert_array_almost_equal(res, target, decimal=9)

    def test_mean_firing_rate_with_plain_array_3d_None(self):
        st = self.test_array_3d
        target = np.sum(self.test_array_3d, None)/5.
        res = es.mean_firing_rate(st, axis=None, t_stop=5.)
        assert not isinstance(res, pq.Quantity)
        assert_array_almost_equal(res, target, decimal=9)

    def test_mean_firing_rate_with_plain_array_3d_0(self):
        st = self.test_array_3d
        target = np.sum(self.test_array_3d, 0)/5.
        res = es.mean_firing_rate(st, axis=0, t_stop=5.)
        assert not isinstance(res, pq.Quantity)
        assert_array_almost_equal(res, target, decimal=9)

    def test_mean_firing_rate_with_plain_array_3d_1(self):
        st = self.test_array_3d
        target = np.sum(self.test_array_3d, 1)/5.
        res = es.mean_firing_rate(st, axis=1, t_stop=5.)
        assert not isinstance(res, pq.Quantity)
        assert_array_almost_equal(res, target, decimal=9)

    def test_mean_firing_rate_with_plain_array_3d_2(self):
        st = self.test_array_3d
        target = np.sum(self.test_array_3d, 2)/5.
        res = es.mean_firing_rate(st, axis=2, t_stop=5.)
        assert not isinstance(res, pq.Quantity)
        assert_array_almost_equal(res, target, decimal=9)

    def test_mean_firing_rate_with_plain_array_2d_1_set_ends(self):
        st = self.test_array_2d
        target = np.array([4, 1, 3])/(1.23-0.14)
        res = es.mean_firing_rate(st, axis=1, t_start=0.14, t_stop=1.23)
        assert not isinstance(res, pq.Quantity)
        assert_array_almost_equal(res, target, decimal=9)

    def test_mean_firing_rate_with_plain_array_2d_None(self):
        st = self.test_array_2d
        target = self.targ_array_2d_None/self.max_array_2d_None
        res = es.mean_firing_rate(st, axis=None)
        assert not isinstance(res, pq.Quantity)
        assert_array_almost_equal(res, target, decimal=9)

    def test_mean_firing_rate_with_plain_array_and_units_start_stop_typeerror(self):
        st = self.test_array_2d
        self.assertRaises(TypeError, es.mean_firing_rate, st,
                          t_start=pq.Quantity(0, 'ms'))
        self.assertRaises(TypeError, es.mean_firing_rate, st,
                          t_stop=pq.Quantity(10, 'ms'))
        self.assertRaises(TypeError, es.mean_firing_rate, st,
                          t_start=pq.Quantity(0, 'ms'),
                          t_stop=pq.Quantity(10, 'ms'))
        self.assertRaises(TypeError, es.mean_firing_rate, st,
                          t_start=pq.Quantity(0, 'ms'),
                          t_stop=10.)
        self.assertRaises(TypeError, es.mean_firing_rate, st,
                          t_start=0.,
                          t_stop=pq.Quantity(10, 'ms'))


class FanoFactorTestCase(unittest.TestCase):
    def setUp(self):
        np.random.seed(100)
        num_st = 300
        self.test_spiketrains = []
        self.test_array = []
        self.test_quantity = []
        self.test_list = []
        self.sp_counts = np.zeros(num_st)
        for i in range(num_st):
            r = np.random.rand(np.random.randint(20) + 1)
            st = neo.core.SpikeTrain(r * pq.ms,
                                     t_start=0.0 * pq.ms,
                                     t_stop=20.0 * pq.ms)
            self.test_spiketrains.append(st)
            self.test_array.append(r)
            self.test_quantity.append(r * pq.ms)
            self.test_list.append(list(r))
            # for cross-validation
            self.sp_counts[i] = len(st)

    def test_fanofactor_spiketrains(self):
        # Test with list of spiketrains
        self.assertEqual(
            np.var(self.sp_counts) / np.mean(self.sp_counts),
            es.fanofactor(self.test_spiketrains))

        # One spiketrain in list
        st = self.test_spiketrains[0]
        self.assertEqual(es.fanofactor([st]), 0.0)

    def test_fanofactor_empty(self):
        # Test with empty list
        self.assertTrue(np.isnan(es.fanofactor([])))
        self.assertTrue(np.isnan(es.fanofactor([[]])))

        # Test with empty quantity
        self.assertTrue(np.isnan(es.fanofactor([] * pq.ms)))

        # Empty spiketrain
        st = neo.core.SpikeTrain([] * pq.ms, t_start=0 * pq.ms,
                                 t_stop=1.5 * pq.ms)
        self.assertTrue(np.isnan(es.fanofactor(st)))

    def test_fanofactor_spiketrains_same(self):
        # Test with same spiketrains in list
        sts = [self.test_spiketrains[0]] * 3
        self.assertEqual(es.fanofactor(sts), 0.0)

    def test_fanofactor_array(self):
        self.assertEqual(es.fanofactor(self.test_array),
                         np.var(self.sp_counts) / np.mean(self.sp_counts))

    def test_fanofactor_array_same(self):
        lst = [self.test_array[0]] * 3
        self.assertEqual(es.fanofactor(lst), 0.0)

    def test_fanofactor_quantity(self):
        self.assertEqual(es.fanofactor(self.test_quantity),
                         np.var(self.sp_counts) / np.mean(self.sp_counts))

    def test_fanofactor_quantity_same(self):
        lst = [self.test_quantity[0]] * 3
        self.assertEqual(es.fanofactor(lst), 0.0)

    def test_fanofactor_list(self):
        self.assertEqual(es.fanofactor(self.test_list),
                         np.var(self.sp_counts) / np.mean(self.sp_counts))

    def test_fanofactor_list_same(self):
        lst = [self.test_list[0]] * 3
        self.assertEqual(es.fanofactor(lst), 0.0)


class LVTestCase(unittest.TestCase):
    def setUp(self):
        self.test_seq = [1, 28,  4, 47,  5, 16,  2,  5, 21, 12,
                         4, 12, 59,  2,  4, 18, 33, 25,  2, 34,
                         4,  1,  1, 14,  8,  1, 10,  1,  8, 20,
                         5,  1,  6,  5, 12,  2,  8,  8,  2,  8,
                         2, 10,  2,  1,  1,  2, 15,  3, 20,  6,
                         11, 6, 18,  2,  5, 17,  4,  3, 13,  6,
                         1, 18,  1, 16, 12,  2, 52,  2,  5,  7,
                         6, 25,  6,  5,  3, 15,  4,  3, 16,  3,
                         6,  5, 24, 21,  3,  3,  4,  8,  4, 11,
                         5,  7,  5,  6,  8, 11, 33, 10,  7,  4]

        self.target = 0.971826029994

    def test_lv_with_quantities(self):
        seq = pq.Quantity(self.test_seq, units='ms')
        assert_array_almost_equal(es.lv(seq), self.target, decimal=9)

    def test_lv_with_plain_array(self):
        seq = np.array(self.test_seq)
        assert_array_almost_equal(es.lv(seq), self.target, decimal=9)

    def test_lv_with_list(self):
        seq = self.test_seq
        assert_array_almost_equal(es.lv(seq), self.target, decimal=9)

    def test_lv_raise_error(self):
        seq = self.test_seq
        self.assertRaises(AttributeError, es.lv, [])
        self.assertRaises(AttributeError, es.lv, 1)
        self.assertRaises(ValueError, es.lv, np.array([seq, seq]))


class PSTHTestCase(unittest.TestCase):
    def setUp(self):
        self.spiketrain_a = neo.SpikeTrain(
            [0.5, 0.7, 1.2, 3.1, 4.3, 5.5, 6.7] * pq.s, t_stop=10.0 * pq.s)
        self.spiketrain_b = neo.SpikeTrain(
            [0.1, 0.7, 1.2, 2.2, 4.3, 5.5, 8.0] * pq.s, t_stop=10.0 * pq.s)
        self.spiketrains = [self.spiketrain_a, self.spiketrain_b]

    def tearDown(self):
        del self.spiketrain_a
        self.spiketrain_a = None
        del self.spiketrain_b
        self.spiketrain_b = None

    def test_psth_unclipped(self):
        targ = [4, 2, 1, 1, 2, 2, 1, 0, 1, 0]
        psth = es.psth(self.spiketrains, w=pq.s)
        assert_array_equal(targ, psth.magnitude)

    def test_psth_clipped(self):
        targ = [2, 2, 1, 1, 2, 2, 1, 0, 1, 0]
        psth = es.psth(self.spiketrains, w=pq.s, clip=True)
        assert_array_equal(targ, psth.magnitude)

    def test_psth_tstart_tstop(self):
        # Start, stop short range
        targ = [2, 1]
        psth = es.psth(self.spiketrains, w=pq.s, t_start=5*pq.s, t_stop=7*pq.s)
        assert_array_equal(targ, psth.magnitude)

    def test_psth_output(self):
        # Normalization mean
        psth = es.psth(self.spiketrains, w=pq.s, output='mean')
        targ = np.array([4, 2, 1, 1, 2, 2, 1, 0, 1, 0], dtype=float) / 2
        assert_array_equal(targ, psth.magnitude)

        # Normalization rate
        psth = es.psth(self.spiketrains, w=pq.s, output='rate')
        assert_array_equal(psth.view(pq.Quantity), targ * 1/pq.s)

        # Normalization unspecified, raises error
        self.assertRaises(ValueError, es.psth, self.spiketrains, w=pq.s,
                          output=' ')

if __name__ == '__main__':
    unittest.main()