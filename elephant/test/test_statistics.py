# -*- coding: utf-8 -*-
"""
docstring goes here.

:copyright: Copyright 2014 by the Elephant team, see AUTHORS.txt.
:license: Modified BSD, see LICENSE.txt for details.
"""

import unittest

import neo.core
import numpy as np
from numpy.testing.utils import assert_array_almost_equal
import quantities as pq

import elephant.statistics as es


class isi_TestCase(unittest.TestCase):
    def setUp(self):
        self.test_array_2d = np.array([[0.3, 0.56, 0.87, 1.23],
                                       [0.02, 0.71, 1.82, 8.46],
                                       [0.03, 0.14, 0.15, 0.92]])
        self.targ_array_2d_0 = np.array([[-0.28, 0.15, 0.95, 7.23],
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
        st = neo.SpikeTrain(self.test_array_regular, units='ms', t_stop=10.0)
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
        target = pq.Quantity(self.targ_array_1d / 10., '1/ms')
        res = es.mean_firing_rate(st)
        assert_array_almost_equal(res, target, decimal=9)

    def test_mean_firing_rate_with_spiketrain_set_ends(self):
        st = neo.SpikeTrain(self.test_array_1d, units='ms', t_stop=10.0)
        target = pq.Quantity(2 / 0.5, '1/ms')
        res = es.mean_firing_rate(st, t_start=0.4, t_stop=0.9)
        assert_array_almost_equal(res, target, decimal=9)

    def test_mean_firing_rate_with_quantities_1d(self):
        st = pq.Quantity(self.test_array_1d, units='ms')
        target = pq.Quantity(self.targ_array_1d / self.max_array_1d, '1/ms')
        res = es.mean_firing_rate(st)
        assert_array_almost_equal(res, target, decimal=9)

    def test_mean_firing_rate_with_quantities_1d_set_ends(self):
        st = pq.Quantity(self.test_array_1d, units='ms')
        target = pq.Quantity(2 / 0.6, '1/ms')
        res = es.mean_firing_rate(st, t_start=400 * pq.us, t_stop=1.)
        assert_array_almost_equal(res, target, decimal=9)

    def test_mean_firing_rate_with_plain_array_1d(self):
        st = self.test_array_1d
        target = self.targ_array_1d / self.max_array_1d
        res = es.mean_firing_rate(st)
        assert not isinstance(res, pq.Quantity)
        assert_array_almost_equal(res, target, decimal=9)

    def test_mean_firing_rate_with_plain_array_1d_set_ends(self):
        st = self.test_array_1d
        target = self.targ_array_1d / (1.23 - 0.3)
        res = es.mean_firing_rate(st, t_start=0.3, t_stop=1.23)
        assert not isinstance(res, pq.Quantity)
        assert_array_almost_equal(res, target, decimal=9)

    def test_mean_firing_rate_with_plain_array_2d_default(self):
        st = self.test_array_2d
        target = self.targ_array_2d_default / self.max_array_2d_default
        res = es.mean_firing_rate(st)
        assert not isinstance(res, pq.Quantity)
        assert_array_almost_equal(res, target, decimal=9)

    def test_mean_firing_rate_with_plain_array_2d_0(self):
        st = self.test_array_2d
        target = self.targ_array_2d_0 / self.max_array_2d_0
        res = es.mean_firing_rate(st, axis=0)
        assert not isinstance(res, pq.Quantity)
        assert_array_almost_equal(res, target, decimal=9)

    def test_mean_firing_rate_with_plain_array_2d_1(self):
        st = self.test_array_2d
        target = self.targ_array_2d_1 / self.max_array_2d_1
        res = es.mean_firing_rate(st, axis=1)
        assert not isinstance(res, pq.Quantity)
        assert_array_almost_equal(res, target, decimal=9)

    def test_mean_firing_rate_with_plain_array_3d_None(self):
        st = self.test_array_3d
        target = np.sum(self.test_array_3d, None) / 5.
        res = es.mean_firing_rate(st, axis=None, t_stop=5.)
        assert not isinstance(res, pq.Quantity)
        assert_array_almost_equal(res, target, decimal=9)

    def test_mean_firing_rate_with_plain_array_3d_0(self):
        st = self.test_array_3d
        target = np.sum(self.test_array_3d, 0) / 5.
        res = es.mean_firing_rate(st, axis=0, t_stop=5.)
        assert not isinstance(res, pq.Quantity)
        assert_array_almost_equal(res, target, decimal=9)

    def test_mean_firing_rate_with_plain_array_3d_1(self):
        st = self.test_array_3d
        target = np.sum(self.test_array_3d, 1) / 5.
        res = es.mean_firing_rate(st, axis=1, t_stop=5.)
        assert not isinstance(res, pq.Quantity)
        assert_array_almost_equal(res, target, decimal=9)

    def test_mean_firing_rate_with_plain_array_3d_2(self):
        st = self.test_array_3d
        target = np.sum(self.test_array_3d, 2) / 5.
        res = es.mean_firing_rate(st, axis=2, t_stop=5.)
        assert not isinstance(res, pq.Quantity)
        assert_array_almost_equal(res, target, decimal=9)

    def test_mean_firing_rate_with_plain_array_2d_1_set_ends(self):
        st = self.test_array_2d
        target = np.array([4, 1, 3]) / (1.23 - 0.14)
        res = es.mean_firing_rate(st, axis=1, t_start=0.14, t_stop=1.23)
        assert not isinstance(res, pq.Quantity)
        assert_array_almost_equal(res, target, decimal=9)

    def test_mean_firing_rate_with_plain_array_2d_None(self):
        st = self.test_array_2d
        target = self.targ_array_2d_None / self.max_array_2d_None
        res = es.mean_firing_rate(st, axis=None)
        assert not isinstance(res, pq.Quantity)
        assert_array_almost_equal(res, target, decimal=9)

    def test_mean_firing_rate_with_plain_array_and_units_start_stop_typeerror(
            self):
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


class fano_TestCase(unittest.TestCase):
    def setUp(self):
        np.random.seed(100)
        num_st = 300
        self.sp_counts = np.zeros(num_st)

        self.blk = neo.core.Block()
        for i in range(num_st):
            self.seg = neo.core.Segment(name='segment %d' % i, index=i)

            st = neo.core.SpikeTrain(
                np.random.rand(np.random.randint(20) + 1) * pq.s,
                t_start=0 * pq.s,
                t_stop=(10.0) * pq.s)

            self.seg.spiketrains.append(st)
            self.blk.segments.append(self.seg)

            # for cross-validation
            self.sp_counts[i] = len(st)

    def test_fano_with_block(self):
        self.assertEqual(
            np.var(self.sp_counts) / np.mean(self.sp_counts),
            es.fano(self.blk))


if __name__ == '__main__':
    unittest.main()
