# -*- coding: utf-8 -*-
"""
Unit tests for the statistics module.

:copyright: Copyright 2014-2020 by the Elephant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
"""
from __future__ import division

import itertools
import math
import unittest

import neo
import numpy as np
import quantities as pq
import scipy.integrate as spint
from numpy.testing import assert_array_almost_equal, assert_array_equal, \
    assert_array_less

import elephant.kernels as kernels
from elephant import statistics
from elephant.spike_train_generation import homogeneous_poisson_process


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
        st = neo.SpikeTrain(
            self.test_array_1d, units='ms', t_stop=10.0, t_start=0.29)
        target = pq.Quantity(self.targ_array_1d, 'ms')
        res = statistics.isi(st)
        assert_array_almost_equal(res, target, decimal=9)

    def test_isi_with_quantities_1d(self):
        st = pq.Quantity(self.test_array_1d, units='ms')
        target = pq.Quantity(self.targ_array_1d, 'ms')
        res = statistics.isi(st)
        assert_array_almost_equal(res, target, decimal=9)

    def test_isi_with_plain_array_1d(self):
        st = self.test_array_1d
        target = self.targ_array_1d
        res = statistics.isi(st)
        assert not isinstance(res, pq.Quantity)
        assert_array_almost_equal(res, target, decimal=9)

    def test_isi_with_plain_array_2d_default(self):
        st = self.test_array_2d
        target = self.targ_array_2d_default
        res = statistics.isi(st)
        assert not isinstance(res, pq.Quantity)
        assert_array_almost_equal(res, target, decimal=9)

    def test_isi_with_plain_array_2d_0(self):
        st = self.test_array_2d
        target = self.targ_array_2d_0
        res = statistics.isi(st, axis=0)
        assert not isinstance(res, pq.Quantity)
        assert_array_almost_equal(res, target, decimal=9)

    def test_isi_with_plain_array_2d_1(self):
        st = self.test_array_2d
        target = self.targ_array_2d_1
        res = statistics.isi(st, axis=1)
        assert not isinstance(res, pq.Quantity)
        assert_array_almost_equal(res, target, decimal=9)

    def test_unsorted_array(self):
        np.random.seed(0)
        array = np.random.rand(100)
        with self.assertWarns(UserWarning):
            isi = statistics.isi(array)


class isi_cv_TestCase(unittest.TestCase):
    def setUp(self):
        self.test_array_regular = np.arange(1, 6)

    def test_cv_isi_regular_spiketrain_is_zero(self):
        st = neo.SpikeTrain(self.test_array_regular, units='ms', t_stop=10.0)
        targ = 0.0
        res = statistics.cv(statistics.isi(st))
        self.assertEqual(res, targ)

    def test_cv_isi_regular_array_is_zero(self):
        st = self.test_array_regular
        targ = 0.0
        res = statistics.cv(statistics.isi(st))
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

    def test_invalid_input_spiketrain(self):
        # empty spiketrain
        self.assertRaises(ValueError, statistics.mean_firing_rate, [])
        for st_invalid in (None, 0.1):
            self.assertRaises(TypeError, statistics.mean_firing_rate,
                              st_invalid)

    def test_mean_firing_rate_with_spiketrain(self):
        st = neo.SpikeTrain(self.test_array_1d, units='ms', t_stop=10.0)
        target = pq.Quantity(self.targ_array_1d / 10., '1/ms')
        res = statistics.mean_firing_rate(st)
        assert_array_almost_equal(res, target, decimal=9)

    def test_mean_firing_rate_typical_use_case(self):
        np.random.seed(92)
        st = homogeneous_poisson_process(rate=100 * pq.Hz, t_stop=100 * pq.s)
        rate1 = statistics.mean_firing_rate(st)
        rate2 = statistics.mean_firing_rate(st, t_start=st.t_start,
                                            t_stop=st.t_stop)
        self.assertEqual(rate1.units, rate2.units)
        self.assertAlmostEqual(rate1.item(), rate2.item())

    def test_mean_firing_rate_with_spiketrain_set_ends(self):
        st = neo.SpikeTrain(self.test_array_1d, units='ms', t_stop=10.0)
        target = pq.Quantity(2 / 0.5, '1/ms')
        res = statistics.mean_firing_rate(st, t_start=0.4 * pq.ms,
                                          t_stop=0.9 * pq.ms)
        assert_array_almost_equal(res, target, decimal=9)

    def test_mean_firing_rate_with_quantities_1d(self):
        st = pq.Quantity(self.test_array_1d, units='ms')
        target = pq.Quantity(self.targ_array_1d / self.max_array_1d, '1/ms')
        res = statistics.mean_firing_rate(st)
        assert_array_almost_equal(res, target, decimal=9)

    def test_mean_firing_rate_with_quantities_1d_set_ends(self):
        st = pq.Quantity(self.test_array_1d, units='ms')

        # t_stop is not a Quantity
        self.assertRaises(TypeError, statistics.mean_firing_rate, st,
                          t_start=400 * pq.us, t_stop=1.)

        # t_start is not a Quantity
        self.assertRaises(TypeError, statistics.mean_firing_rate, st,
                          t_start=0.4, t_stop=1. * pq.ms)

    def test_mean_firing_rate_with_plain_array_1d(self):
        st = self.test_array_1d
        target = self.targ_array_1d / self.max_array_1d
        res = statistics.mean_firing_rate(st)
        assert not isinstance(res, pq.Quantity)
        assert_array_almost_equal(res, target, decimal=9)

    def test_mean_firing_rate_with_plain_array_1d_set_ends(self):
        st = self.test_array_1d
        target = self.targ_array_1d / (1.23 - 0.3)
        res = statistics.mean_firing_rate(st, t_start=0.3, t_stop=1.23)
        assert not isinstance(res, pq.Quantity)
        assert_array_almost_equal(res, target, decimal=9)

    def test_mean_firing_rate_with_plain_array_2d_default(self):
        st = self.test_array_2d
        target = self.targ_array_2d_default / self.max_array_2d_default
        res = statistics.mean_firing_rate(st)
        assert not isinstance(res, pq.Quantity)
        assert_array_almost_equal(res, target, decimal=9)

    def test_mean_firing_rate_with_plain_array_2d_0(self):
        st = self.test_array_2d
        target = self.targ_array_2d_0 / self.max_array_2d_0
        res = statistics.mean_firing_rate(st, axis=0)
        assert not isinstance(res, pq.Quantity)
        assert_array_almost_equal(res, target, decimal=9)

    def test_mean_firing_rate_with_plain_array_2d_1(self):
        st = self.test_array_2d
        target = self.targ_array_2d_1 / self.max_array_2d_1
        res = statistics.mean_firing_rate(st, axis=1)
        assert not isinstance(res, pq.Quantity)
        assert_array_almost_equal(res, target, decimal=9)

    def test_mean_firing_rate_with_plain_array_3d_None(self):
        st = self.test_array_3d
        target = np.sum(self.test_array_3d, None) / 5.
        res = statistics.mean_firing_rate(st, axis=None, t_stop=5.)
        assert not isinstance(res, pq.Quantity)
        assert_array_almost_equal(res, target, decimal=9)

    def test_mean_firing_rate_with_plain_array_3d_0(self):
        st = self.test_array_3d
        target = np.sum(self.test_array_3d, 0) / 5.
        res = statistics.mean_firing_rate(st, axis=0, t_stop=5.)
        assert not isinstance(res, pq.Quantity)
        assert_array_almost_equal(res, target, decimal=9)

    def test_mean_firing_rate_with_plain_array_3d_1(self):
        st = self.test_array_3d
        target = np.sum(self.test_array_3d, 1) / 5.
        res = statistics.mean_firing_rate(st, axis=1, t_stop=5.)
        assert not isinstance(res, pq.Quantity)
        assert_array_almost_equal(res, target, decimal=9)

    def test_mean_firing_rate_with_plain_array_3d_2(self):
        st = self.test_array_3d
        target = np.sum(self.test_array_3d, 2) / 5.
        res = statistics.mean_firing_rate(st, axis=2, t_stop=5.)
        assert not isinstance(res, pq.Quantity)
        assert_array_almost_equal(res, target, decimal=9)

    def test_mean_firing_rate_with_plain_array_2d_1_set_ends(self):
        st = self.test_array_2d
        target = np.array([4, 1, 3]) / (1.23 - 0.14)
        res = statistics.mean_firing_rate(st, axis=1, t_start=0.14,
                                          t_stop=1.23)
        assert not isinstance(res, pq.Quantity)
        assert_array_almost_equal(res, target, decimal=9)

    def test_mean_firing_rate_with_plain_array_2d_None(self):
        st = self.test_array_2d
        target = self.targ_array_2d_None / self.max_array_2d_None
        res = statistics.mean_firing_rate(st, axis=None)
        assert not isinstance(res, pq.Quantity)
        assert_array_almost_equal(res, target, decimal=9)

    def test_mean_firing_rate_with_plain_array_and_units_start_stop_typeerror(
            self):
        st = self.test_array_2d
        self.assertRaises(TypeError, statistics.mean_firing_rate, st,
                          t_start=pq.Quantity(0, 'ms'))
        self.assertRaises(TypeError, statistics.mean_firing_rate, st,
                          t_stop=pq.Quantity(10, 'ms'))
        self.assertRaises(TypeError, statistics.mean_firing_rate, st,
                          t_start=pq.Quantity(0, 'ms'),
                          t_stop=pq.Quantity(10, 'ms'))
        self.assertRaises(TypeError, statistics.mean_firing_rate, st,
                          t_start=pq.Quantity(0, 'ms'),
                          t_stop=10.)
        self.assertRaises(TypeError, statistics.mean_firing_rate, st,
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
            statistics.fanofactor(self.test_spiketrains))

        # One spiketrain in list
        st = self.test_spiketrains[0]
        self.assertEqual(statistics.fanofactor([st]), 0.0)

    def test_fanofactor_empty(self):
        # Test with empty list
        self.assertTrue(np.isnan(statistics.fanofactor([])))
        self.assertTrue(np.isnan(statistics.fanofactor([[]])))

        # Test with empty quantity
        self.assertTrue(np.isnan(statistics.fanofactor([] * pq.ms)))

        # Empty spiketrain
        st = neo.core.SpikeTrain([] * pq.ms, t_start=0 * pq.ms,
                                 t_stop=1.5 * pq.ms)
        self.assertTrue(np.isnan(statistics.fanofactor(st)))

    def test_fanofactor_spiketrains_same(self):
        # Test with same spiketrains in list
        sts = [self.test_spiketrains[0]] * 3
        self.assertEqual(statistics.fanofactor(sts), 0.0)

    def test_fanofactor_array(self):
        self.assertEqual(statistics.fanofactor(self.test_array),
                         np.var(self.sp_counts) / np.mean(self.sp_counts))

    def test_fanofactor_array_same(self):
        lst = [self.test_array[0]] * 3
        self.assertEqual(statistics.fanofactor(lst), 0.0)

    def test_fanofactor_quantity(self):
        self.assertEqual(statistics.fanofactor(self.test_quantity),
                         np.var(self.sp_counts) / np.mean(self.sp_counts))

    def test_fanofactor_quantity_same(self):
        lst = [self.test_quantity[0]] * 3
        self.assertEqual(statistics.fanofactor(lst), 0.0)

    def test_fanofactor_list(self):
        self.assertEqual(statistics.fanofactor(self.test_list),
                         np.var(self.sp_counts) / np.mean(self.sp_counts))

    def test_fanofactor_list_same(self):
        lst = [self.test_list[0]] * 3
        self.assertEqual(statistics.fanofactor(lst), 0.0)

    def test_fanofactor_different_durations(self):
        st1 = neo.SpikeTrain([1, 2, 3] * pq.s, t_stop=4 * pq.s)
        st2 = neo.SpikeTrain([1, 2, 3] * pq.s, t_stop=4.5 * pq.s)
        self.assertWarns(UserWarning, statistics.fanofactor, (st1, st2))

    def test_fanofactor_wrong_type(self):
        # warn_tolerance is not a quantity
        st1 = neo.SpikeTrain([1, 2, 3] * pq.s, t_stop=4 * pq.s)
        self.assertRaises(TypeError, statistics.fanofactor, [st1],
                          warn_tolerance=1e-4)


class LVTestCase(unittest.TestCase):
    def setUp(self):
        self.test_seq = [1, 28, 4, 47, 5, 16, 2, 5, 21, 12,
                         4, 12, 59, 2, 4, 18, 33, 25, 2, 34,
                         4, 1, 1, 14, 8, 1, 10, 1, 8, 20,
                         5, 1, 6, 5, 12, 2, 8, 8, 2, 8,
                         2, 10, 2, 1, 1, 2, 15, 3, 20, 6,
                         11, 6, 18, 2, 5, 17, 4, 3, 13, 6,
                         1, 18, 1, 16, 12, 2, 52, 2, 5, 7,
                         6, 25, 6, 5, 3, 15, 4, 3, 16, 3,
                         6, 5, 24, 21, 3, 3, 4, 8, 4, 11,
                         5, 7, 5, 6, 8, 11, 33, 10, 7, 4]

        self.target = 0.971826029994

    def test_lv_with_quantities(self):
        seq = pq.Quantity(self.test_seq, units='ms')
        assert_array_almost_equal(statistics.lv(seq), self.target, decimal=9)

    def test_lv_with_plain_array(self):
        seq = np.array(self.test_seq)
        assert_array_almost_equal(statistics.lv(seq), self.target, decimal=9)

    def test_lv_with_list(self):
        seq = self.test_seq
        assert_array_almost_equal(statistics.lv(seq), self.target, decimal=9)

    def test_lv_raise_error(self):
        seq = self.test_seq
        self.assertRaises(ValueError, statistics.lv, [])
        self.assertRaises(ValueError, statistics.lv, 1)
        self.assertRaises(ValueError, statistics.lv, np.array([seq, seq]))

    def test_2short_spike_train(self):
        seq = [1]
        with self.assertWarns(UserWarning):
            # Catches UserWarning: Input size is too small. Please provide
            # an input with more than 1 entry.
            self.assertTrue(math.isnan(statistics.lv(seq, with_nan=True)))


class LVRTestCase(unittest.TestCase):
    def setUp(self):
        self.test_seq = [1, 28, 4, 47, 5, 16, 2, 5, 21, 12,
                         4, 12, 59, 2, 4, 18, 33, 25, 2, 34,
                         4, 1, 1, 14, 8, 1, 10, 1, 8, 20,
                         5, 1, 6, 5, 12, 2, 8, 8, 2, 8,
                         2, 10, 2, 1, 1, 2, 15, 3, 20, 6,
                         11, 6, 18, 2, 5, 17, 4, 3, 13, 6,
                         1, 18, 1, 16, 12, 2, 52, 2, 5, 7,
                         6, 25, 6, 5, 3, 15, 4, 3, 16, 3,
                         6, 5, 24, 21, 3, 3, 4, 8, 4, 11,
                         5, 7, 5, 6, 8, 11, 33, 10, 7, 4]

        self.target = 2.1845363464753134

    def test_lvr_with_quantities(self):
        seq = pq.Quantity(self.test_seq, units='ms')
        assert_array_almost_equal(statistics.lvr(seq), self.target, decimal=9)
        seq = pq.Quantity(self.test_seq, units='ms').rescale('s')
        assert_array_almost_equal(statistics.lvr(seq), self.target, decimal=9)

    def test_lvr_with_plain_array(self):
        seq = np.array(self.test_seq)
        with self.assertWarns(UserWarning):
            assert_array_almost_equal(statistics.lvr(seq),
                                      self.target, decimal=9)

    def test_lvr_with_list(self):
        seq = self.test_seq
        with self.assertWarns(UserWarning):
            assert_array_almost_equal(statistics.lvr(seq),
                                      self.target, decimal=9)

    def test_lvr_raise_error(self):
        seq = self.test_seq
        self.assertRaises(ValueError, statistics.lvr, [])
        self.assertRaises(ValueError, statistics.lvr, 1)
        self.assertRaises(ValueError, statistics.lvr, np.array([seq, seq]))
        self.assertRaises(ValueError, statistics.lvr, seq, -1 * pq.ms)

    def test_lvr_refractoriness_kwarg(self):
        seq = np.array(self.test_seq)
        with self.assertWarns(UserWarning):
            assert_array_almost_equal(statistics.lvr(seq, R=5),
                                      self.target, decimal=9)

    def test_2short_spike_train(self):
        seq = [1]
        with self.assertWarns(UserWarning):
            # Catches UserWarning: Input size is too small. Please provide
            # an input with more than 1 entry.
            self.assertTrue(math.isnan(statistics.lvr(seq, with_nan=True)))


class CV2TestCase(unittest.TestCase):
    def setUp(self):
        self.test_seq = [1, 28, 4, 47, 5, 16, 2, 5, 21, 12,
                         4, 12, 59, 2, 4, 18, 33, 25, 2, 34,
                         4, 1, 1, 14, 8, 1, 10, 1, 8, 20,
                         5, 1, 6, 5, 12, 2, 8, 8, 2, 8,
                         2, 10, 2, 1, 1, 2, 15, 3, 20, 6,
                         11, 6, 18, 2, 5, 17, 4, 3, 13, 6,
                         1, 18, 1, 16, 12, 2, 52, 2, 5, 7,
                         6, 25, 6, 5, 3, 15, 4, 3, 16, 3,
                         6, 5, 24, 21, 3, 3, 4, 8, 4, 11,
                         5, 7, 5, 6, 8, 11, 33, 10, 7, 4]

        self.target = 1.0022235296529176

    def test_cv2_with_quantities(self):
        seq = pq.Quantity(self.test_seq, units='ms')
        assert_array_almost_equal(statistics.cv2(seq), self.target, decimal=9)

    def test_cv2_with_plain_array(self):
        seq = np.array(self.test_seq)
        assert_array_almost_equal(statistics.cv2(seq), self.target, decimal=9)

    def test_cv2_with_list(self):
        seq = self.test_seq
        assert_array_almost_equal(statistics.cv2(seq), self.target, decimal=9)

    def test_cv2_raise_error(self):
        seq = self.test_seq
        self.assertRaises(ValueError, statistics.cv2, [])
        self.assertRaises(ValueError, statistics.cv2, 1)
        self.assertRaises(ValueError, statistics.cv2, np.array([seq, seq]))


class InstantaneousRateTest(unittest.TestCase):

    def setUp(self):
        # create a poisson spike train:
        self.st_tr = (0, 20.0)  # seconds
        self.st_dur = self.st_tr[1] - self.st_tr[0]  # seconds
        self.st_margin = 5.0  # seconds
        self.st_rate = 10.0  # Hertz

        np.random.seed(19)
        duration_effective = self.st_dur - 2 * self.st_margin
        st_num_spikes = np.random.poisson(self.st_rate * duration_effective)
        spike_train = sorted(
            np.random.rand(st_num_spikes) *
            duration_effective +
            self.st_margin)

        # convert spike train into neo objects
        self.spike_train = neo.SpikeTrain(spike_train * pq.s,
                                          t_start=self.st_tr[0] * pq.s,
                                          t_stop=self.st_tr[1] * pq.s)

        # generation of a multiply used specific kernel
        self.kernel = kernels.TriangularKernel(sigma=0.03 * pq.s)

    def test_instantaneous_rate_and_warnings(self):
        st = self.spike_train
        sampling_period = 0.01 * pq.s
        with self.assertWarns(UserWarning):
            # Catches warning: The width of the kernel was adjusted to a
            # minimally allowed width.
            inst_rate = statistics.instantaneous_rate(
                st, sampling_period, self.kernel, cutoff=0)
        self.assertIsInstance(inst_rate, neo.core.AnalogSignal)
        self.assertEqual(
            inst_rate.sampling_period.simplified, sampling_period.simplified)
        self.assertEqual(inst_rate.simplified.units, pq.Hz)
        self.assertEqual(inst_rate.t_stop.simplified, st.t_stop.simplified)
        self.assertEqual(inst_rate.t_start.simplified, st.t_start.simplified)

    def test_error_instantaneous_rate(self):
        self.assertRaises(
            TypeError, statistics.instantaneous_rate,
            spiketrains=[1, 2, 3] * pq.s,
            sampling_period=0.01 * pq.ms, kernel=self.kernel)
        self.assertRaises(
            TypeError, statistics.instantaneous_rate, spiketrains=[1, 2, 3],
            sampling_period=0.01 * pq.ms, kernel=self.kernel)
        st = self.spike_train
        self.assertRaises(
            TypeError, statistics.instantaneous_rate, spiketrains=st,
            sampling_period=0.01, kernel=self.kernel)
        self.assertRaises(
            ValueError, statistics.instantaneous_rate, spiketrains=st,
            sampling_period=-0.01 * pq.ms, kernel=self.kernel)
        self.assertRaises(
            TypeError, statistics.instantaneous_rate, spiketrains=st,
            sampling_period=0.01 * pq.ms, kernel='NONE')
        self.assertRaises(TypeError, statistics.instantaneous_rate,
                          self.spike_train,
                          sampling_period=0.01 * pq.s, kernel='wrong_string',
                          t_start=self.st_tr[0] * pq.s,
                          t_stop=self.st_tr[1] * pq.s,
                          trim=False)
        self.assertRaises(
            TypeError, statistics.instantaneous_rate, spiketrains=st,
            sampling_period=0.01 * pq.ms, kernel=self.kernel,
            cutoff=20 * pq.ms)
        self.assertRaises(
            TypeError, statistics.instantaneous_rate, spiketrains=st,
            sampling_period=0.01 * pq.ms, kernel=self.kernel, t_start=2)
        self.assertRaises(
            TypeError, statistics.instantaneous_rate, spiketrains=st,
            sampling_period=0.01 * pq.ms, kernel=self.kernel,
            t_stop=20 * pq.mV)
        self.assertRaises(
            TypeError, statistics.instantaneous_rate, spiketrains=st,
            sampling_period=0.01 * pq.ms, kernel=self.kernel, trim=1)

        # cannot estimate a kernel for a list of spiketrains
        self.assertRaises(ValueError, statistics.instantaneous_rate,
                          spiketrains=[st, st], sampling_period=10 * pq.ms,
                          kernel='auto')

    def test_rate_estimation_consistency(self):
        """
        Test, whether the integral of the rate estimation curve is (almost)
        equal to the number of spikes of the spike train.
        """
        kernel_types = tuple(
            kern_cls for kern_cls in kernels.__dict__.values()
            if isinstance(kern_cls, type) and
            issubclass(kern_cls, kernels.Kernel) and
            kern_cls is not kernels.Kernel and
            kern_cls is not kernels.SymmetricKernel)
        kernels_available = [kern_cls(sigma=0.5 * pq.s, invert=False)
                             for kern_cls in kernel_types]
        kernels_available.append('auto')
        kernel_resolution = 0.01 * pq.s
        for kernel in kernels_available:
            for center_kernel in (False, True):
                rate_estimate = statistics.instantaneous_rate(
                    self.spike_train,
                    sampling_period=kernel_resolution,
                    kernel=kernel,
                    t_start=self.st_tr[0] * pq.s,
                    t_stop=self.st_tr[1] * pq.s,
                    trim=False,
                    center_kernel=center_kernel)
                num_spikes = len(self.spike_train)
                auc = spint.cumtrapz(
                    y=rate_estimate.magnitude[:, 0],
                    x=rate_estimate.times.rescale('s').magnitude)[-1]
                self.assertAlmostEqual(num_spikes, auc,
                                       delta=0.01 * num_spikes)

    def test_not_center_kernel(self):
        # issue 107
        t_spike = 1 * pq.s
        st = neo.SpikeTrain([t_spike], t_start=0 * pq.s, t_stop=2 * pq.s,
                            units=pq.s)
        kernel = kernels.AlphaKernel(200 * pq.ms)
        fs = 0.1 * pq.ms
        rate = statistics.instantaneous_rate(st,
                                             sampling_period=fs,
                                             kernel=kernel,
                                             center_kernel=False)
        rate_nonzero_index = np.nonzero(rate > 1e-6)[0]
        # where the mass is concentrated
        rate_mass = rate.times.rescale(t_spike.units)[rate_nonzero_index]
        all_after_response_onset = (rate_mass >= t_spike).all()
        self.assertTrue(all_after_response_onset)

    def test_regression_288(self):
        np.random.seed(9)
        sampling_period = 200 * pq.ms
        spiketrain = homogeneous_poisson_process(10 * pq.Hz,
                                                 t_start=0 * pq.s,
                                                 t_stop=10 * pq.s)
        kernel = kernels.AlphaKernel(sigma=5 * pq.ms, invert=True)
        # check that instantaneous_rate "works" for kernels with small sigma
        # without triggering an incomprehensible error
        rate = statistics.instantaneous_rate(spiketrain,
                                             sampling_period=sampling_period,
                                             kernel=kernel)
        self.assertEqual(
            len(rate), (spiketrain.t_stop / sampling_period).simplified.item())

    def test_small_kernel_sigma(self):
        # Test that the instantaneous rate is overestimated when
        # kernel.sigma << sampling_period and center_kernel is True.
        # The setup is set to match the issue 288.
        np.random.seed(9)
        sampling_period = 200 * pq.ms
        sigma = 5 * pq.ms
        rate_expected = 10 * pq.Hz
        spiketrain = homogeneous_poisson_process(rate_expected,
                                                 t_start=0 * pq.s,
                                                 t_stop=10 * pq.s)
        kernel_types = tuple(
            kern_cls for kern_cls in kernels.__dict__.values()
            if isinstance(kern_cls, type) and
            issubclass(kern_cls, kernels.Kernel) and
            kern_cls is not kernels.Kernel and
            kern_cls is not kernels.SymmetricKernel)
        for kern_cls, invert in itertools.product(kernel_types, (False, True)):
            kernel = kern_cls(sigma=sigma, invert=invert)
            with self.subTest(kernel=kernel):
                rate = statistics.instantaneous_rate(
                    spiketrain,
                    sampling_period=sampling_period,
                    kernel=kernel, center_kernel=True)
                self.assertGreater(rate.mean(), rate_expected)

    def test_spikes_on_edges(self):
        # this test demonstrates that the trimming (convolve valid mode)
        # removes the edge spikes, underestimating the true firing rate and
        # thus is not able to reconstruct the number of spikes in a
        # spiketrain (see test_rate_estimation_consistency)
        cutoff = 5
        sampling_period = 0.01 * pq.s
        t_spikes = np.array([-cutoff, cutoff]) * pq.s
        spiketrain = neo.SpikeTrain(t_spikes, t_start=t_spikes[0],
                                    t_stop=t_spikes[-1])
        kernel_types = tuple(
            kern_cls for kern_cls in kernels.__dict__.values()
            if isinstance(kern_cls, type) and
            issubclass(kern_cls, kernels.Kernel) and
            kern_cls is not kernels.Kernel and
            kern_cls is not kernels.SymmetricKernel)
        kernels_available = [kern_cls(sigma=1 * pq.s, invert=False)
                             for kern_cls in kernel_types]
        for kernel in kernels_available:
            for center_kernel in (False, True):
                rate = statistics.instantaneous_rate(
                    spiketrain,
                    sampling_period=sampling_period,
                    kernel=kernel,
                    cutoff=cutoff, trim=True,
                    center_kernel=center_kernel)
                assert_array_almost_equal(rate.magnitude, 0, decimal=3)

    def test_trim_as_convolve_mode(self):
        cutoff = 5
        sampling_period = 0.01 * pq.s
        t_spikes = np.linspace(-cutoff, cutoff, num=(2 * cutoff + 1)) * pq.s
        spiketrain = neo.SpikeTrain(t_spikes, t_start=t_spikes[0],
                                    t_stop=t_spikes[-1])
        kernel = kernels.RectangularKernel(sigma=1 * pq.s)
        assert cutoff > kernel.min_cutoff, "Choose larger cutoff"
        kernel_types = tuple(
            kern_cls for kern_cls in kernels.__dict__.values()
            if isinstance(kern_cls, type) and
            issubclass(kern_cls, kernels.SymmetricKernel) and
            kern_cls is not kernels.SymmetricKernel)
        kernels_symmetric = [kern_cls(sigma=1 * pq.s, invert=False)
                             for kern_cls in kernel_types]
        for kernel in kernels_symmetric:
            for trim in (False, True):
                rate_centered = statistics.instantaneous_rate(
                    spiketrain, sampling_period=sampling_period,
                    kernel=kernel, cutoff=cutoff, trim=trim)

                rate_convolve = statistics.instantaneous_rate(
                    spiketrain, sampling_period=sampling_period,
                    kernel=kernel, cutoff=cutoff, trim=trim,
                    center_kernel=False)
                assert_array_almost_equal(rate_centered, rate_convolve)

    def test_instantaneous_rate_spiketrainlist(self):
        np.random.seed(19)
        duration_effective = self.st_dur - 2 * self.st_margin
        st_num_spikes = np.random.poisson(self.st_rate * duration_effective)
        spike_train2 = sorted(
            np.random.rand(st_num_spikes) *
            duration_effective +
            self.st_margin)
        spike_train2 = neo.SpikeTrain(spike_train2 * pq.s,
                                      t_start=self.st_tr[0] * pq.s,
                                      t_stop=self.st_tr[1] * pq.s)
        st_rate_1 = statistics.instantaneous_rate(self.spike_train,
                                                  sampling_period=0.01 * pq.s,
                                                  kernel=self.kernel)
        st_rate_2 = statistics.instantaneous_rate(spike_train2,
                                                  sampling_period=0.01 * pq.s,
                                                  kernel=self.kernel)
        combined_rate = statistics.instantaneous_rate(
            [self.spike_train, spike_train2],
            sampling_period=0.01 * pq.s,
            kernel=self.kernel)
        rate_concat = np.c_[st_rate_1, st_rate_2]
        # 'time_vector.dtype' in instantaneous_rate() is changed from float64
        # to float32 which results in 3e-6 abs difference
        assert_array_almost_equal(combined_rate.magnitude,
                                  rate_concat.magnitude, decimal=5)

    # Regression test for #144
    def test_instantaneous_rate_regression_144(self):
        # The following spike train contains spikes that are so close to each
        # other, that the optimal kernel cannot be detected. Therefore, the
        # function should react with a ValueError.
        st = neo.SpikeTrain([2.12, 2.13, 2.15] * pq.s, t_stop=10 * pq.s)
        self.assertRaises(ValueError, statistics.instantaneous_rate, st,
                          1 * pq.ms)

    # Regression test for #245
    def test_instantaneous_rate_regression_245(self):
        # This test makes sure that the correct kernel width is chosen when
        # selecting 'auto' as kernel
        spiketrain = neo.SpikeTrain(
            range(1, 30) * pq.ms, t_start=0 * pq.ms, t_stop=30 * pq.ms)

        # This is the correct procedure to attain the kernel: first, the result
        # of sskernel retrieves the kernel bandwidth of an optimal Gaussian
        # kernel in terms of its standard deviation sigma, then uses this value
        # directly in the function for creating the Gaussian kernel
        kernel_width_sigma = statistics.optimal_kernel_bandwidth(
            spiketrain.magnitude, times=None, bootstrap=False)['optw']
        kernel = kernels.GaussianKernel(kernel_width_sigma * spiketrain.units)
        result_target = statistics.instantaneous_rate(
            spiketrain, 10 * pq.ms, kernel=kernel)

        # Here, we check if the 'auto' argument leads to the same operation. In
        # the regression, it was incorrectly assumed that the sskernel()
        # function returns the actual bandwidth of the kernel, which is defined
        # as approximately bandwidth = sigma * 5.5 = sigma * (2 * 2.75).
        # factor 2.0 connects kernel width with its half width,
        # factor 2.7 connects half width of Gaussian distribution with
        #            99% probability mass with its standard deviation.
        result_automatic = statistics.instantaneous_rate(
            spiketrain, 10 * pq.ms, kernel='auto')

        assert_array_almost_equal(result_target, result_automatic)

    def test_instantaneous_rate_grows_with_sampling_period(self):
        np.random.seed(0)
        rate_expected = 10 * pq.Hz
        spiketrain = homogeneous_poisson_process(rate=rate_expected,
                                                 t_stop=10 * pq.s)
        kernel = kernels.GaussianKernel(sigma=100 * pq.ms)
        rates_mean = []
        for sampling_period in np.linspace(1, 1000, num=10) * pq.ms:
            with self.subTest(sampling_period=sampling_period):
                rate = statistics.instantaneous_rate(
                    spiketrain,
                    sampling_period=sampling_period,
                    kernel=kernel)
                rates_mean.append(rate.mean())
        # rate means are greater or equal the expected rate
        assert_array_less(rate_expected, rates_mean)
        # check sorted
        self.assertTrue(np.all(rates_mean[:-1] < rates_mean[1:]))

    # Regression test for #360
    def test_centered_at_origin(self):
        # Skip RectangularKernel because it doesn't have a strong peak.
        kernel_types = tuple(
            kern_cls for kern_cls in kernels.__dict__.values()
            if isinstance(kern_cls, type) and
            issubclass(kern_cls, kernels.SymmetricKernel) and
            kern_cls not in (kernels.SymmetricKernel,
                             kernels.RectangularKernel))
        kernels_symmetric = [kern_cls(sigma=50 * pq.ms, invert=False)
                             for kern_cls in kernel_types]

        # first part: a symmetric spiketrain with a symmetric kernel
        spiketrain = neo.SpikeTrain(np.array([-0.0001, 0, 0.0001]) * pq.s,
                                    t_start=-1,
                                    t_stop=1)
        for kernel in kernels_symmetric:
            rate = statistics.instantaneous_rate(spiketrain,
                                                 sampling_period=20 * pq.ms,
                                                 kernel=kernel)
            # the peak time must be centered at origin
            self.assertEqual(rate.times[np.argmax(rate)], 0)

        # second part: a single spike at t=0
        periods = [2 ** c for c in range(-3, 6)]
        for period in periods:
            with self.subTest(period=period):
                spiketrain = neo.SpikeTrain(np.array([0]) * pq.s,
                                            t_start=-period * 10 * pq.ms,
                                            t_stop=period * 10 * pq.ms)
                for kernel in kernels_symmetric:
                    rate = statistics.instantaneous_rate(
                        spiketrain,
                        sampling_period=period * pq.ms,
                        kernel=kernel)
                    self.assertEqual(rate.times[np.argmax(rate)], 0)

    def test_annotations(self):
        spiketrain = neo.SpikeTrain([1, 2], t_stop=2 * pq.s, units=pq.s)
        kernel = kernels.AlphaKernel(sigma=100 * pq.ms)
        rate = statistics.instantaneous_rate(spiketrain,
                                             sampling_period=10 * pq.ms,
                                             kernel=kernel)
        kernel_annotation = dict(type=type(kernel).__name__,
                                 sigma=str(kernel.sigma),
                                 invert=kernel.invert)
        self.assertIn('kernel', rate.annotations)
        self.assertEqual(rate.annotations['kernel'], kernel_annotation)


class TimeHistogramTestCase(unittest.TestCase):
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

    def test_time_histogram(self):
        targ = np.array([4, 2, 1, 1, 2, 2, 1, 0, 1, 0])
        histogram = statistics.time_histogram(self.spiketrains, bin_size=pq.s)
        assert_array_equal(targ, histogram.magnitude[:, 0])

    def test_time_histogram_binary(self):
        targ = np.array([2, 2, 1, 1, 2, 2, 1, 0, 1, 0])
        histogram = statistics.time_histogram(self.spiketrains, bin_size=pq.s,
                                              binary=True)
        assert_array_equal(targ, histogram.magnitude[:, 0])

    def test_time_histogram_tstart_tstop(self):
        # Start, stop short range
        targ = np.array([2, 1])
        histogram = statistics.time_histogram(self.spiketrains, bin_size=pq.s,
                                              t_start=5 * pq.s,
                                              t_stop=7 * pq.s)
        assert_array_equal(targ, histogram.magnitude[:, 0])

        # Test without t_stop
        targ = np.array([4, 2, 1, 1, 2, 2, 1, 0, 1, 0])
        histogram = statistics.time_histogram(self.spiketrains,
                                              bin_size=1 * pq.s,
                                              t_start=0 * pq.s)
        assert_array_equal(targ, histogram.magnitude[:, 0])

        # Test without t_start
        histogram = statistics.time_histogram(self.spiketrains,
                                              bin_size=1 * pq.s,
                                              t_stop=10 * pq.s)
        assert_array_equal(targ, histogram.magnitude[:, 0])

    def test_time_histogram_output(self):
        # Normalization mean
        histogram = statistics.time_histogram(self.spiketrains, bin_size=pq.s,
                                              output='mean')
        targ = np.array([4, 2, 1, 1, 2, 2, 1, 0, 1, 0], dtype=float) / 2
        assert_array_equal(targ.reshape(targ.size, 1), histogram.magnitude)

        # Normalization rate
        histogram = statistics.time_histogram(self.spiketrains, bin_size=pq.s,
                                              output='rate')
        assert_array_equal(histogram.view(pq.Quantity),
                           targ.reshape(targ.size, 1) * 1 / pq.s)

        # Normalization unspecified, raises error
        self.assertRaises(ValueError, statistics.time_histogram,
                          self.spiketrains,
                          bin_size=pq.s, output=' ')

    def test_annotations(self):
        np.random.seed(1)
        spiketrains = [homogeneous_poisson_process(
            rate=10 * pq.Hz, t_stop=10 * pq.s) for _ in range(10)]
        for output in ("counts", "mean", "rate"):
            histogram = statistics.time_histogram(spiketrains,
                                                  bin_size=3 * pq.ms,
                                                  output=output)
            self.assertIn('normalization', histogram.annotations)
            self.assertEqual(histogram.annotations['normalization'], output)


class ComplexityPdfTestCase(unittest.TestCase):
    def test_complexity_pdf_deprecated(self):
        spiketrain_a = neo.SpikeTrain(
            [0.5, 0.7, 1.2, 2.3, 4.3, 5.5, 6.7] * pq.s, t_stop=10.0 * pq.s)
        spiketrain_b = neo.SpikeTrain(
            [0.5, 0.7, 1.2, 2.3, 4.3, 5.5, 8.0] * pq.s, t_stop=10.0 * pq.s)
        spiketrain_c = neo.SpikeTrain(
            [0.5, 0.7, 1.2, 2.3, 4.3, 5.5, 8.0] * pq.s, t_stop=10.0 * pq.s)
        spiketrains = [
            spiketrain_a, spiketrain_b, spiketrain_c]
        # runs the previous function which will be deprecated
        targ = np.array([0.92, 0.01, 0.01, 0.06])
        complexity = statistics.complexity_pdf(spiketrains, binsize=0.1*pq.s)
        assert_array_equal(targ, complexity.magnitude[:, 0])
        self.assertEqual(1, complexity.magnitude[:, 0].sum())
        self.assertEqual(len(spiketrains)+1, len(complexity))
        self.assertIsInstance(complexity, neo.AnalogSignal)
        self.assertEqual(complexity.units, 1 * pq.dimensionless)

    def test_complexity_pdf(self):
        spiketrain_a = neo.SpikeTrain(
            [0.5, 0.7, 1.2, 2.3, 4.3, 5.5, 6.7] * pq.s, t_stop=10.0 * pq.s)
        spiketrain_b = neo.SpikeTrain(
            [0.5, 0.7, 1.2, 2.3, 4.3, 5.5, 8.0] * pq.s, t_stop=10.0 * pq.s)
        spiketrain_c = neo.SpikeTrain(
            [0.5, 0.7, 1.2, 2.3, 4.3, 5.5, 8.0] * pq.s, t_stop=10.0 * pq.s)
        spiketrains = [
            spiketrain_a, spiketrain_b, spiketrain_c]
        # runs the previous function which will be deprecated
        targ = np.array([0.92, 0.01, 0.01, 0.06])
        complexity_obj = statistics.Complexity(spiketrains,
                                               bin_size=0.1 * pq.s)
        pdf = complexity_obj.pdf()
        assert_array_equal(targ, complexity_obj.pdf().magnitude[:, 0])
        self.assertEqual(1, pdf.magnitude[:, 0].sum())
        self.assertEqual(len(spiketrains)+1, len(pdf))
        self.assertIsInstance(pdf, neo.AnalogSignal)
        self.assertEqual(pdf.units, 1*pq.dimensionless)

    def test_complexity_histogram_spread_0(self):

        sampling_rate = 1 / pq.s

        spiketrains = [neo.SpikeTrain([1, 5, 9, 11, 16, 19] * pq.s,
                                      t_stop=20*pq.s),
                       neo.SpikeTrain([1, 4, 8, 12, 16, 18] * pq.s,
                                      t_stop=20*pq.s)]

        correct_histogram = np.array([10, 8, 2])

        correct_time_histogram = np.array([0, 2, 0, 0, 1, 1, 0, 0, 1, 1,
                                           0, 1, 1, 0, 0, 0, 2, 0, 1, 1])

        complexity_obj = statistics.Complexity(spiketrains,
                                               sampling_rate=sampling_rate,
                                               spread=0)

        assert_array_equal(complexity_obj.complexity_histogram,
                           correct_histogram)

        assert_array_equal(
            complexity_obj.time_histogram.magnitude.flatten().astype(int),
            correct_time_histogram)

    def test_complexity_epoch_spread_0(self):

        sampling_rate = 1 / pq.s

        spiketrains = [neo.SpikeTrain([1, 5, 9, 11, 16, 19] * pq.s,
                                      t_stop=20*pq.s),
                       neo.SpikeTrain([1, 4, 8, 12, 16, 18] * pq.s,
                                      t_stop=20*pq.s)]

        complexity_obj = statistics.Complexity(spiketrains,
                                               sampling_rate=sampling_rate,
                                               spread=0)

        self.assertIsInstance(complexity_obj.epoch,
                              neo.Epoch)

    def test_complexity_histogram_spread_1(self):

        sampling_rate = 1 / pq.s

        spiketrains = [neo.SpikeTrain([0, 1, 5, 9, 11, 13, 20] * pq.s,
                                      t_stop=21*pq.s),
                       neo.SpikeTrain([1, 4, 7, 12, 16, 18] * pq.s,
                                      t_stop=21*pq.s)]

        correct_histogram = np.array([9, 5, 1, 2])

        correct_time_histogram = np.array([3, 3, 0, 0, 2, 2, 0, 1, 0, 1, 0,
                                           3, 3, 3, 0, 0, 1, 0, 1, 0, 1])

        complexity_obj = statistics.Complexity(spiketrains,
                                               sampling_rate=sampling_rate,
                                               spread=1)

        assert_array_equal(complexity_obj.complexity_histogram,
                           correct_histogram)

        assert_array_equal(
            complexity_obj.time_histogram.magnitude.flatten().astype(int),
            correct_time_histogram)

    def test_complexity_histogram_spread_2(self):

        sampling_rate = 1 / pq.s

        spiketrains = [neo.SpikeTrain([1, 5, 9, 11, 13, 20] * pq.s,
                                      t_stop=21*pq.s),
                       neo.SpikeTrain([1, 4, 7, 12, 16, 18] * pq.s,
                                      t_stop=21*pq.s)]

        correct_histogram = np.array([5, 0, 1, 1, 0, 0, 0, 1])

        correct_time_histogram = np.array([0, 2, 0, 0, 7, 7, 7, 7, 7, 7, 7,
                                           7, 7, 7, 0, 0, 3, 3, 3, 3, 3])

        complexity_obj = statistics.Complexity(spiketrains,
                                               sampling_rate=sampling_rate,
                                               spread=2)

        assert_array_equal(complexity_obj.complexity_histogram,
                           correct_histogram)

        assert_array_equal(
            complexity_obj.time_histogram.magnitude.flatten().astype(int),
            correct_time_histogram)

    def test_wrong_input_errors(self):
        spiketrains = [neo.SpikeTrain([1, 5, 9, 11, 13, 20] * pq.s,
                                      t_stop=21*pq.s),
                       neo.SpikeTrain([1, 4, 7, 12, 16, 18] * pq.s,
                                      t_stop=21*pq.s)]

        self.assertRaises(ValueError,
                          statistics.Complexity,
                          spiketrains)

        self.assertRaises(ValueError,
                          statistics.Complexity,
                          spiketrains,
                          sampling_rate=1*pq.s,
                          spread=-7)

    def test_sampling_rate_warning(self):
        spiketrains = [neo.SpikeTrain([1, 5, 9, 11, 13, 20] * pq.s,
                                      t_stop=21*pq.s),
                       neo.SpikeTrain([1, 4, 7, 12, 16, 18] * pq.s,
                                      t_stop=21*pq.s)]

        with self.assertWarns(UserWarning):
            statistics.Complexity(spiketrains,
                                  bin_size=1*pq.s,
                                  spread=1)

    def test_binning_for_input_with_rounding_errors(self):

        # a test with inputs divided by 30000 which leads to rounding errors
        # these errors have to be accounted for by proper binning;
        # check if we still get the correct result

        sampling_rate = 333 / pq.s

        spiketrains = [neo.SpikeTrain(np.arange(1000, step=2) * pq.s / 333,
                                      t_stop=30.33333333333 * pq.s),
                       neo.SpikeTrain(np.arange(2000, step=4) * pq.s / 333,
                                      t_stop=30.33333333333 * pq.s)]

        correct_time_histogram = np.zeros(10101)
        correct_time_histogram[:1000:2] = 1
        correct_time_histogram[:2000:4] += 1

        complexity_obj = statistics.Complexity(spiketrains,
                                               sampling_rate=sampling_rate,
                                               spread=1)

        assert_array_equal(
            complexity_obj.time_histogram.magnitude.flatten().astype(int),
            correct_time_histogram)


if __name__ == '__main__':
    unittest.main()
