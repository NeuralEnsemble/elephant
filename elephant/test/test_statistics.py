# -*- coding: utf-8 -*-
"""
Unit tests for the statistics module.

:copyright: Copyright 2014-2024 by the Elephant team, see `doc/authors.rst`.
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
from numpy.testing import (
    assert_array_almost_equal,
    assert_array_equal,
    assert_array_less,
)
import elephant.kernels as kernels
from elephant import statistics
from elephant.spike_train_generation import StationaryPoissonProcess
from elephant.test.test_trials import _create_trials_block
from elephant.trials import TrialsFromBlock


class IsiTestCase(unittest.TestCase):
    def setUp(self):
        self.test_array_2d = np.array(
            [
                [0.3, 0.56, 0.87, 1.23],
                [0.02, 0.71, 1.82, 8.46],
                [0.03, 0.14, 0.15, 0.92],
            ]
        )
        self.targ_array_2d_0 = np.array(
            [[-0.28, 0.15, 0.95, 7.23], [0.01, -0.57, -1.67, -7.54]]
        )
        self.targ_array_2d_1 = np.array(
            [[0.26, 0.31, 0.36], [0.69, 1.11, 6.64], [0.11, 0.01, 0.77]]
        )
        self.targ_array_2d_default = self.targ_array_2d_1

        self.test_array_1d = self.test_array_2d[0, :]
        self.targ_array_1d = self.targ_array_2d_1[0, :]

    def test_isi_with_spiketrain(self):
        st = neo.SpikeTrain(self.test_array_1d, units="ms", t_stop=10.0, t_start=0.29)
        target = pq.Quantity(self.targ_array_1d, "ms")
        res = statistics.isi(st)
        assert_array_almost_equal(res, target, decimal=9)

    def test_isi_with_quantities_1d(self):
        st = pq.Quantity(self.test_array_1d, units="ms")
        target = pq.Quantity(self.targ_array_1d, "ms")
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
            statistics.isi(array)


class IsiCvTestCase(unittest.TestCase):
    def setUp(self):
        self.test_array_regular = np.arange(1, 6)

    def test_cv_isi_regular_spiketrain_is_zero(self):
        st = neo.SpikeTrain(self.test_array_regular, units="ms", t_stop=10.0)
        targ = 0.0
        res = statistics.cv(statistics.isi(st))
        self.assertEqual(res, targ)

    def test_cv_isi_regular_array_is_zero(self):
        st = self.test_array_regular
        targ = 0.0
        res = statistics.cv(statistics.isi(st))
        self.assertEqual(res, targ)


class MeanFiringRateTestCase(unittest.TestCase):
    def setUp(self):
        self.test_array_3d = np.ones([5, 7, 13])
        self.test_array_2d = np.array(
            [
                [0.3, 0.56, 0.87, 1.23],
                [0.02, 0.71, 1.82, 8.46],
                [0.03, 0.14, 0.15, 0.92],
            ]
        )

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
            self.assertRaises(TypeError, statistics.mean_firing_rate, st_invalid)

    def test_mean_firing_rate_with_spiketrain(self):
        st = neo.SpikeTrain(self.test_array_1d, units="ms", t_stop=10.0)
        target = pq.Quantity(self.targ_array_1d / 10.0, "1/ms")
        res = statistics.mean_firing_rate(st)
        assert_array_almost_equal(res, target, decimal=9)

    def test_mean_firing_rate_typical_use_case(self):
        np.random.seed(92)
        st = StationaryPoissonProcess(
            rate=100 * pq.Hz, t_stop=100 * pq.s
        ).generate_spiketrain()
        rate1 = statistics.mean_firing_rate(st)
        rate2 = statistics.mean_firing_rate(st, t_start=st.t_start, t_stop=st.t_stop)
        self.assertEqual(rate1.units, rate2.units)
        self.assertAlmostEqual(rate1.item(), rate2.item())

    def test_mean_firing_rate_with_spiketrain_set_ends(self):
        st = neo.SpikeTrain(self.test_array_1d, units="ms", t_stop=10.0)
        target = pq.Quantity(2 / 0.5, "1/ms")
        res = statistics.mean_firing_rate(st, t_start=0.4 * pq.ms, t_stop=0.9 * pq.ms)
        assert_array_almost_equal(res, target, decimal=9)

    def test_mean_firing_rate_with_quantities_1d(self):
        st = pq.Quantity(self.test_array_1d, units="ms")
        target = pq.Quantity(self.targ_array_1d / self.max_array_1d, "1/ms")
        res = statistics.mean_firing_rate(st)
        assert_array_almost_equal(res, target, decimal=9)

    def test_mean_firing_rate_with_quantities_1d_set_ends(self):
        st = pq.Quantity(self.test_array_1d, units="ms")

        # t_stop is not a Quantity
        self.assertRaises(
            TypeError, statistics.mean_firing_rate, st, t_start=400 * pq.us, t_stop=1.0
        )

        # t_start is not a Quantity
        self.assertRaises(
            TypeError, statistics.mean_firing_rate, st, t_start=0.4, t_stop=1.0 * pq.ms
        )

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
        target = np.sum(self.test_array_3d, None) / 5.0
        res = statistics.mean_firing_rate(st, axis=None, t_stop=5.0)
        assert not isinstance(res, pq.Quantity)
        assert_array_almost_equal(res, target, decimal=9)

    def test_mean_firing_rate_with_plain_array_3d_0(self):
        st = self.test_array_3d
        target = np.sum(self.test_array_3d, 0) / 5.0
        res = statistics.mean_firing_rate(st, axis=0, t_stop=5.0)
        assert not isinstance(res, pq.Quantity)
        assert_array_almost_equal(res, target, decimal=9)

    def test_mean_firing_rate_with_plain_array_3d_1(self):
        st = self.test_array_3d
        target = np.sum(self.test_array_3d, 1) / 5.0
        res = statistics.mean_firing_rate(st, axis=1, t_stop=5.0)
        assert not isinstance(res, pq.Quantity)
        assert_array_almost_equal(res, target, decimal=9)

    def test_mean_firing_rate_with_plain_array_3d_2(self):
        st = self.test_array_3d
        target = np.sum(self.test_array_3d, 2) / 5.0
        res = statistics.mean_firing_rate(st, axis=2, t_stop=5.0)
        assert not isinstance(res, pq.Quantity)
        assert_array_almost_equal(res, target, decimal=9)

    def test_mean_firing_rate_with_plain_array_2d_1_set_ends(self):
        st = self.test_array_2d
        target = np.array([4, 1, 3]) / (1.23 - 0.14)
        res = statistics.mean_firing_rate(st, axis=1, t_start=0.14, t_stop=1.23)
        assert not isinstance(res, pq.Quantity)
        assert_array_almost_equal(res, target, decimal=9)

    def test_mean_firing_rate_with_plain_array_2d_None(self):
        st = self.test_array_2d
        target = self.targ_array_2d_None / self.max_array_2d_None
        res = statistics.mean_firing_rate(st, axis=None)
        assert not isinstance(res, pq.Quantity)
        assert_array_almost_equal(res, target, decimal=9)

    def test_mean_firing_rate_with_plain_array_and_units_start_stop_typeerror(self):
        st = self.test_array_2d
        self.assertRaises(
            TypeError, statistics.mean_firing_rate, st, t_start=pq.Quantity(0, "ms")
        )
        self.assertRaises(
            TypeError, statistics.mean_firing_rate, st, t_stop=pq.Quantity(10, "ms")
        )
        self.assertRaises(
            TypeError,
            statistics.mean_firing_rate,
            st,
            t_start=pq.Quantity(0, "ms"),
            t_stop=pq.Quantity(10, "ms"),
        )
        self.assertRaises(
            TypeError,
            statistics.mean_firing_rate,
            st,
            t_start=pq.Quantity(0, "ms"),
            t_stop=10.0,
        )
        self.assertRaises(
            TypeError,
            statistics.mean_firing_rate,
            st,
            t_start=0.0,
            t_stop=pq.Quantity(10, "ms"),
        )


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
            st = neo.core.SpikeTrain(
                r * pq.ms, t_start=0.0 * pq.ms, t_stop=20.0 * pq.ms
            )
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
            statistics.fanofactor(self.test_spiketrains),
        )

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
        st = neo.core.SpikeTrain([] * pq.ms, t_start=0 * pq.ms, t_stop=1.5 * pq.ms)
        self.assertTrue(np.isnan(statistics.fanofactor(st)))

    def test_fanofactor_spiketrains_same(self):
        # Test with same spiketrains in list
        sts = [self.test_spiketrains[0]] * 3
        self.assertEqual(statistics.fanofactor(sts), 0.0)

    def test_fanofactor_array(self):
        self.assertEqual(
            statistics.fanofactor(self.test_array),
            np.var(self.sp_counts) / np.mean(self.sp_counts),
        )

    def test_fanofactor_array_same(self):
        lst = [self.test_array[0]] * 3
        self.assertEqual(statistics.fanofactor(lst), 0.0)

    def test_fanofactor_quantity(self):
        self.assertEqual(
            statistics.fanofactor(self.test_quantity),
            np.var(self.sp_counts) / np.mean(self.sp_counts),
        )

    def test_fanofactor_quantity_same(self):
        lst = [self.test_quantity[0]] * 3
        self.assertEqual(statistics.fanofactor(lst), 0.0)

    def test_fanofactor_list(self):
        self.assertEqual(
            statistics.fanofactor(self.test_list),
            np.var(self.sp_counts) / np.mean(self.sp_counts),
        )

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
        self.assertRaises(TypeError, statistics.fanofactor, [st1], warn_tolerance=1e-4)


class LVTestCase(unittest.TestCase):
    def setUp(self):
        self.test_seq = [
            1,
            28,
            4,
            47,
            5,
            16,
            2,
            5,
            21,
            12,
            4,
            12,
            59,
            2,
            4,
            18,
            33,
            25,
            2,
            34,
            4,
            1,
            1,
            14,
            8,
            1,
            10,
            1,
            8,
            20,
            5,
            1,
            6,
            5,
            12,
            2,
            8,
            8,
            2,
            8,
            2,
            10,
            2,
            1,
            1,
            2,
            15,
            3,
            20,
            6,
            11,
            6,
            18,
            2,
            5,
            17,
            4,
            3,
            13,
            6,
            1,
            18,
            1,
            16,
            12,
            2,
            52,
            2,
            5,
            7,
            6,
            25,
            6,
            5,
            3,
            15,
            4,
            3,
            16,
            3,
            6,
            5,
            24,
            21,
            3,
            3,
            4,
            8,
            4,
            11,
            5,
            7,
            5,
            6,
            8,
            11,
            33,
            10,
            7,
            4,
        ]

        self.target = 0.971826029994

    def test_lv_with_quantities(self):
        seq = pq.Quantity(self.test_seq, units="ms")
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
        self.test_seq = [
            1,
            28,
            4,
            47,
            5,
            16,
            2,
            5,
            21,
            12,
            4,
            12,
            59,
            2,
            4,
            18,
            33,
            25,
            2,
            34,
            4,
            1,
            1,
            14,
            8,
            1,
            10,
            1,
            8,
            20,
            5,
            1,
            6,
            5,
            12,
            2,
            8,
            8,
            2,
            8,
            2,
            10,
            2,
            1,
            1,
            2,
            15,
            3,
            20,
            6,
            11,
            6,
            18,
            2,
            5,
            17,
            4,
            3,
            13,
            6,
            1,
            18,
            1,
            16,
            12,
            2,
            52,
            2,
            5,
            7,
            6,
            25,
            6,
            5,
            3,
            15,
            4,
            3,
            16,
            3,
            6,
            5,
            24,
            21,
            3,
            3,
            4,
            8,
            4,
            11,
            5,
            7,
            5,
            6,
            8,
            11,
            33,
            10,
            7,
            4,
        ]

        self.target = 2.1845363464753134

    def test_lvr_with_quantities(self):
        seq = pq.Quantity(self.test_seq, units="ms")
        assert_array_almost_equal(statistics.lvr(seq), self.target, decimal=9)
        seq = pq.Quantity(self.test_seq, units="ms").rescale("s", dtype=float)
        assert_array_almost_equal(statistics.lvr(seq), self.target, decimal=9)

    def test_lvr_with_plain_array(self):
        seq = np.array(self.test_seq)
        with self.assertWarns(UserWarning):
            assert_array_almost_equal(statistics.lvr(seq), self.target, decimal=9)

    def test_lvr_with_list(self):
        seq = self.test_seq
        with self.assertWarns(UserWarning):
            assert_array_almost_equal(statistics.lvr(seq), self.target, decimal=9)

    def test_lvr_raise_error(self):
        seq = self.test_seq
        self.assertRaises(ValueError, statistics.lvr, [])
        self.assertRaises(ValueError, statistics.lvr, 1)
        self.assertRaises(ValueError, statistics.lvr, np.array([seq, seq]))
        self.assertRaises(ValueError, statistics.lvr, seq, -1 * pq.ms)

    def test_lvr_refractoriness_kwarg(self):
        seq = np.array(self.test_seq)
        with self.assertWarns(UserWarning):
            assert_array_almost_equal(statistics.lvr(seq, R=5), self.target, decimal=9)

    def test_2short_spike_train(self):
        seq = [1]
        with self.assertWarns(UserWarning):
            # Catches UserWarning: Input size is too small. Please provide
            # an input with more than 1 entry.
            self.assertTrue(math.isnan(statistics.lvr(seq, with_nan=True)))


class CV2TestCase(unittest.TestCase):
    def setUp(self):
        self.test_seq = [
            1,
            28,
            4,
            47,
            5,
            16,
            2,
            5,
            21,
            12,
            4,
            12,
            59,
            2,
            4,
            18,
            33,
            25,
            2,
            34,
            4,
            1,
            1,
            14,
            8,
            1,
            10,
            1,
            8,
            20,
            5,
            1,
            6,
            5,
            12,
            2,
            8,
            8,
            2,
            8,
            2,
            10,
            2,
            1,
            1,
            2,
            15,
            3,
            20,
            6,
            11,
            6,
            18,
            2,
            5,
            17,
            4,
            3,
            13,
            6,
            1,
            18,
            1,
            16,
            12,
            2,
            52,
            2,
            5,
            7,
            6,
            25,
            6,
            5,
            3,
            15,
            4,
            3,
            16,
            3,
            6,
            5,
            24,
            21,
            3,
            3,
            4,
            8,
            4,
            11,
            5,
            7,
            5,
            6,
            8,
            11,
            33,
            10,
            7,
            4,
        ]

        self.target = 1.0022235296529176

    def test_cv2_with_quantities(self):
        seq = pq.Quantity(self.test_seq, units="ms")
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
    @classmethod
    def setUpClass(cls) -> None:
        """
        Run once before tests:
        """

        block = _create_trials_block(n_trials=36)
        cls.block = block
        cls.trial_object = TrialsFromBlock(block, description="trials are segments")

        # create a poisson spike train:
        cls.st_tr = (0, 20.0)  # seconds
        cls.st_dur = cls.st_tr[1] - cls.st_tr[0]  # seconds
        cls.st_margin = 5.0  # seconds
        cls.st_rate = 10.0  # Hertz
        np.random.seed(19)
        duration_effective = cls.st_dur - 2 * cls.st_margin
        st_num_spikes = np.random.poisson(cls.st_rate * duration_effective)
        spike_train = sorted(
            np.random.rand(st_num_spikes) * duration_effective + cls.st_margin
        )
        # convert spike train into neo objects
        cls.spike_train = neo.SpikeTrain(
            spike_train * pq.s, t_start=cls.st_tr[0] * pq.s, t_stop=cls.st_tr[1] * pq.s
        )
        # generation of a multiply used specific kernel
        cls.kernel = kernels.TriangularKernel(sigma=0.03 * pq.s)
        # calculate instantaneous rate
        cls.sampling_period = 0.01 * pq.s
        cls.inst_rate = statistics.instantaneous_rate(
            cls.spike_train, cls.sampling_period, cls.kernel, cutoff=0
        )

    def test_instantaneous_rate_warnings(self):
        with self.assertWarns(UserWarning):
            # Catches warning: The width of the kernel was adjusted to a
            # minimally allowed width.
            statistics.instantaneous_rate(
                self.spike_train, self.sampling_period, self.kernel, cutoff=0
            )

    def test_instantaneous_rate_errors(self):
        self.assertRaises(  # input is not neo.SpikeTrain
            TypeError,
            statistics.instantaneous_rate,
            spiketrains=[1, 2, 3] * pq.s,
            sampling_period=0.01 * pq.ms,
            kernel=self.kernel,
        )
        self.assertRaises(  # sampling period is not time quantity
            TypeError,
            statistics.instantaneous_rate,
            spiketrains=self.spike_train,
            kernel=self.kernel,
            sampling_period=0.01,
        )
        self.assertRaises(  # sampling period is < 0
            ValueError,
            statistics.instantaneous_rate,
            spiketrains=self.spike_train,
            kernel=self.kernel,
            sampling_period=-0.01 * pq.ms,
        )
        self.assertRaises(  # no kernel or kernel='auto'
            TypeError,
            statistics.instantaneous_rate,
            spiketrains=self.spike_train,
            sampling_period=0.01 * pq.ms,
            kernel="NONE",
        )
        self.assertRaises(  # wrong string for kernel='string'
            TypeError,
            statistics.instantaneous_rate,
            spiketrains=self.spike_train,
            sampling_period=0.01 * pq.s,
            kernel="wrong_string",
        )
        self.assertRaises(  # cutoff is not float or int
            TypeError,
            statistics.instantaneous_rate,
            spiketrains=self.spike_train,
            sampling_period=0.01 * pq.ms,
            kernel=self.kernel,
            cutoff=20 * pq.ms,
        )
        self.assertRaises(  # t_start not time quantity
            TypeError,
            statistics.instantaneous_rate,
            spiketrains=self.spike_train,
            sampling_period=0.01 * pq.ms,
            kernel=self.kernel,
            t_start=2,
        )
        self.assertRaises(  # t_stop not time quantity
            TypeError,
            statistics.instantaneous_rate,
            spiketrains=self.spike_train,
            sampling_period=0.01 * pq.ms,
            kernel=self.kernel,
            t_stop=20 * pq.mV,
        )
        self.assertRaises(  # trim is not bool
            TypeError,
            statistics.instantaneous_rate,
            spiketrains=self.spike_train,
            sampling_period=0.01 * pq.ms,
            kernel=self.kernel,
            trim=1,
        )
        self.assertRaises(  # can't estimate a kernel for a list of spiketrains
            ValueError,
            statistics.instantaneous_rate,
            spiketrains=[self.spike_train, self.spike_train],
            sampling_period=10 * pq.ms,
            kernel="auto",
        )

    def test_instantaneous_rate_output(self):
        # return type correct
        self.assertIsInstance(self.inst_rate, neo.core.AnalogSignal)
        # sampling_period input and output same
        self.assertEqual(
            self.inst_rate.sampling_period.simplified, self.sampling_period.simplified
        )
        # return correct units pq.Hz
        self.assertEqual(self.inst_rate.simplified.units, pq.Hz)
        # input and output t_stop same
        self.assertEqual(
            self.spike_train.t_stop.simplified, self.inst_rate.t_stop.simplified
        )
        # input and output t_start same
        self.assertEqual(
            self.inst_rate.t_start.simplified, self.spike_train.t_start.simplified
        )

    def test_instantaneous_rate_rate_estimation_consistency(self):
        """
        Test, whether the integral of the rate estimation curve is (almost)
        equal to the number of spikes of the spike train.
        """
        kernel_types = tuple(
            kern_cls
            for kern_cls in kernels.__dict__.values()
            if isinstance(kern_cls, type)
            and issubclass(kern_cls, kernels.Kernel)
            and kern_cls is not kernels.Kernel
            and kern_cls is not kernels.SymmetricKernel
        )
        # set sigma
        kernels_available = [
            kern_cls(sigma=0.5 * pq.s, invert=False) for kern_cls in kernel_types
        ]
        kernels_available.append("auto")
        kernel_resolution = 0.01 * pq.s
        for kernel in kernels_available:
            border_correction = False
            if isinstance(kernel, kernels.GaussianKernel):
                border_correction = True
            for center_kernel in (False, True):
                rate_estimate = statistics.instantaneous_rate(
                    self.spike_train,
                    sampling_period=kernel_resolution,
                    kernel=kernel,
                    t_start=self.st_tr[0] * pq.s,
                    t_stop=self.st_tr[1] * pq.s,
                    trim=False,
                    center_kernel=center_kernel,
                    border_correction=border_correction,
                )
                num_spikes = len(self.spike_train)
                area_under_curve = spint.cumulative_trapezoid(
                    y=rate_estimate.magnitude[:, 0],
                    x=rate_estimate.times.rescale("s").magnitude,
                )[-1]
                self.assertAlmostEqual(
                    num_spikes, area_under_curve, delta=0.01 * num_spikes
                )

    def test_instantaneous_rate_regression_107(self):
        # Create a spiketrain with t_start=0s, t_stop=2s and a single spike at
        # t=1s. Now choose an asymmetric kernel starting at t=0 to avoid a rise
        # in firing rate before the response onset, so to say to avoid 'looking
        # into the future' from the perspective of the neuron.
        t_spike = 1 * pq.s
        spiketrain = neo.SpikeTrain(
            [t_spike], t_start=0 * pq.s, t_stop=2 * pq.s, units=pq.s
        )
        kernel = kernels.AlphaKernel(200 * pq.ms)
        sampling_period = 0.1 * pq.ms
        rate = statistics.instantaneous_rate(
            spiketrains=spiketrain,
            sampling_period=sampling_period,
            kernel=kernel,
            center_kernel=False,
        )
        # find positive nonezero rate estimates
        rate_nonzero_index = np.nonzero(rate > 1e-6)[0]
        # find times, where the mass is concentrated, i.e. rate is estimated>0
        rate_mass_times = rate.times.rescale(t_spike.units)[rate_nonzero_index]
        # all times, where rate is >0 should occur after response onset
        # (t_spike is at 1s)
        all_after_response_onset = (rate_mass_times >= t_spike).all()
        self.assertTrue(all_after_response_onset)

    def test_instantaneous_rate_regression_288(self):
        # check that instantaneous_rate "works" for kernels with small sigma
        # without triggering an incomprehensible error:
        # ValueError: zero-size array to reduction operation minimum which has
        # no identity
        try:
            np.random.seed(9)
            sampling_period = 200 * pq.ms
            spiketrain = StationaryPoissonProcess(
                10 * pq.Hz, t_start=0 * pq.s, t_stop=10 * pq.s
            ).generate_spiketrain()
            kernel = kernels.AlphaKernel(sigma=5 * pq.ms, invert=True)
            _ = statistics.instantaneous_rate(
                spiketrain, sampling_period=sampling_period, kernel=kernel
            )
        except ValueError:
            self.fail(
                "When providing a kernel on a much smaller time scale "
                "than sampling rate requested the instantaneous rate "
                "estimation will fail on numpy level "
            )

    def test_instantaneous_rate_small_kernel_sigma(self):
        # Test that the instantaneous rate is overestimated when
        # kernel.sigma << sampling_period and center_kernel is True.
        # The setup is set to match the issue 288.
        np.random.seed(9)
        sampling_period = 200 * pq.ms
        sigma = 5 * pq.ms
        rate_expected = 10 * pq.Hz
        spiketrain = StationaryPoissonProcess(
            rate_expected, t_start=0 * pq.s, t_stop=10 * pq.s
        ).generate_spiketrain()

        kernel_types = tuple(
            kern_cls
            for kern_cls in kernels.__dict__.values()
            if isinstance(kern_cls, type)
            and issubclass(kern_cls, kernels.Kernel)
            and kern_cls is not kernels.Kernel
            and kern_cls is not kernels.SymmetricKernel
        )
        for kern_cls, invert in itertools.product(kernel_types, (False, True)):
            kernel = kern_cls(sigma=sigma, invert=invert)
            with self.subTest(kernel=kernel):
                rate = statistics.instantaneous_rate(
                    spiketrain,
                    sampling_period=sampling_period,
                    kernel=kernel,
                    center_kernel=True,
                )
                self.assertGreater(rate.mean(), rate_expected)

    def test_instantaneous_rate_spikes_on_edges(self):
        # this test demonstrates that the trimming (convolve valid mode)
        # removes the edges of the rate estimate, underestimating the true
        # firing rate and thus is not able to reconstruct the number of spikes
        # in a spiketrain (see test_rate_estimation_consistency)
        cutoff = 5
        sampling_period = 0.01 * pq.s
        # with t_spikes = [-5, 5]s the isi is 10s, so 1/isi 0.1 Hz
        t_spikes = np.array([-cutoff, cutoff]) * pq.s
        spiketrain = neo.SpikeTrain(t_spikes, t_start=t_spikes[0], t_stop=t_spikes[-1])
        kernel_types = tuple(
            kern_cls
            for kern_cls in kernels.__dict__.values()
            if isinstance(kern_cls, type)
            and issubclass(kern_cls, kernels.Kernel)
            and kern_cls is not kernels.Kernel
            and kern_cls is not kernels.SymmetricKernel
        )
        kernels_available = [
            kern_cls(sigma=1 * pq.s, invert=False) for kern_cls in kernel_types
        ]
        for kernel in kernels_available:
            for center_kernel in (False, True):
                rate = statistics.instantaneous_rate(
                    spiketrain,
                    sampling_period=sampling_period,
                    kernel=kernel,
                    cutoff=cutoff,
                    trim=True,
                    center_kernel=center_kernel,
                )
                assert_array_almost_equal(rate.magnitude, 0, decimal=2)

    def test_instantaneous_rate_center_kernel(self):
        # this test is obsolete since trimming is now always done by
        # np.fftconvolve, in earlier version trimming was implemented for
        # center_kernel = True
        # This test now verifies, that an already centered kernel is not
        # affected by center_kernel = True.
        cutoff = 5
        sampling_period = 0.01 * pq.s
        t_spikes = np.linspace(-cutoff, cutoff, num=(2 * cutoff + 1)) * pq.s
        spiketrain = neo.SpikeTrain(t_spikes, t_start=t_spikes[0], t_stop=t_spikes[-1])
        kernel = kernels.RectangularKernel(sigma=1 * pq.s)
        assert cutoff > kernel.min_cutoff, "Choose larger cutoff"
        kernel_types = tuple(
            kern_cls
            for kern_cls in kernels.__dict__.values()
            if isinstance(kern_cls, type)
            and issubclass(kern_cls, kernels.SymmetricKernel)
            and kern_cls is not kernels.SymmetricKernel
        )
        kernels_symmetric = [
            kern_cls(sigma=1 * pq.s, invert=False) for kern_cls in kernel_types
        ]
        for kernel in kernels_symmetric:
            for trim in (False, True):
                rate_centered = statistics.instantaneous_rate(
                    spiketrain,
                    sampling_period=sampling_period,
                    kernel=kernel,
                    cutoff=cutoff,
                    trim=trim,
                    center_kernel=True,
                )

                rate_not_centered = statistics.instantaneous_rate(
                    spiketrain,
                    sampling_period=sampling_period,
                    kernel=kernel,
                    cutoff=cutoff,
                    trim=trim,
                    center_kernel=False,
                )
                assert_array_almost_equal(rate_centered, rate_not_centered)

    def test_instantaneous_rate_list_of_spiketrains(self):
        np.random.seed(19)
        duration_effective = self.st_dur - 2 * self.st_margin
        st_num_spikes = np.random.poisson(self.st_rate * duration_effective)
        spike_train2 = sorted(
            np.random.rand(st_num_spikes) * duration_effective + self.st_margin
        )
        spike_train2 = neo.SpikeTrain(
            spike_train2 * pq.s,
            t_start=self.st_tr[0] * pq.s,
            t_stop=self.st_tr[1] * pq.s,
        )

        st_rate_1 = statistics.instantaneous_rate(
            self.spike_train, sampling_period=self.sampling_period, kernel=self.kernel
        )
        st_rate_2 = statistics.instantaneous_rate(
            spike_train2, sampling_period=self.sampling_period, kernel=self.kernel
        )
        rate_concat = np.c_[st_rate_1, st_rate_2]

        combined_rate = statistics.instantaneous_rate(
            [self.spike_train, spike_train2],
            sampling_period=self.sampling_period,
            kernel=self.kernel,
        )
        # 'time_vector.dtype' in instantaneous_rate() is changed from float64
        # to float32 which results in 3e-6 abs difference
        assert_array_almost_equal(
            combined_rate.magnitude, rate_concat.magnitude, decimal=5
        )

    def test_instantaneous_rate_regression_144(self):
        # The following spike train contains spikes that are so close to each
        # other, that the optimal kernel cannot be detected. Therefore, the
        # function should react with a ValueError.
        st = neo.SpikeTrain([2.12, 2.13, 2.15] * pq.s, t_stop=10 * pq.s)
        self.assertRaises(ValueError, statistics.instantaneous_rate, st, 1 * pq.ms)

    def test_instantaneous_rate_regression_245(self):
        # This test makes sure that the correct kernel width is chosen when
        # selecting 'auto' as kernel
        spiketrain = neo.SpikeTrain(
            pq.ms * range(1, 30), t_start=0 * pq.ms, t_stop=30 * pq.ms
        )

        # This is the correct procedure to attain the kernel: first, the result
        # of sskernel retrieves the kernel bandwidth of an optimal Gaussian
        # kernel in terms of its standard deviation sigma, then uses this value
        # directly in the function for creating the Gaussian kernel
        kernel_width_sigma = statistics.optimal_kernel_bandwidth(
            spiketrain.magnitude, times=None, bootstrap=False
        )["optw"]
        kernel = kernels.GaussianKernel(kernel_width_sigma * spiketrain.units)
        result_target = statistics.instantaneous_rate(
            spiketrain, 10 * pq.ms, kernel=kernel
        )

        # Here, we check if the 'auto' argument leads to the same operation. In
        # the regression, it was incorrectly assumed that the sskernel()
        # function returns the actual bandwidth of the kernel, which is defined
        # as approximately bandwidth = sigma * 5.5 = sigma * (2 * 2.75).
        # factor 2.0 connects kernel width with its half width,
        # factor 2.7 connects half width of Gaussian distribution with
        #            99% probability mass with its standard deviation.
        result_automatic = statistics.instantaneous_rate(
            spiketrain, 10 * pq.ms, kernel="auto"
        )

        assert_array_almost_equal(result_target, result_automatic)

    def test_instantaneous_rate_grows_with_sampling_period(self):
        np.random.seed(0)
        rate_expected = 10 * pq.Hz
        spiketrain = StationaryPoissonProcess(
            rate=rate_expected, t_stop=10 * pq.s
        ).generate_spiketrain()
        kernel = kernels.GaussianKernel(sigma=100 * pq.ms)
        rates_mean = []
        for sampling_period in np.linspace(1, 1000, num=10) * pq.ms:
            with self.subTest(sampling_period=sampling_period):
                rate = statistics.instantaneous_rate(
                    spiketrain, sampling_period=sampling_period, kernel=kernel
                )
                rates_mean.append(rate.mean())
        # rate means are greater or equal the expected rate
        assert_array_less(rate_expected, rates_mean)
        # check sorted
        self.assertTrue(np.all(rates_mean[:-1] < rates_mean[1:]))

    def test_instantaneous_rate_regression_360(self):
        # This test check if the resulting rate is centered for a spiketrain
        # with spikes at [-0.0001, 0, 0.0001].
        # Skip RectangularKernel because it doesn't have a strong peak.
        kernel_types = tuple(
            kern_cls
            for kern_cls in kernels.__dict__.values()
            if isinstance(kern_cls, type)
            and issubclass(kern_cls, kernels.SymmetricKernel)
            and kern_cls not in (kernels.SymmetricKernel, kernels.RectangularKernel)
        )
        kernels_symmetric = [
            kern_cls(sigma=50 * pq.ms, invert=False) for kern_cls in kernel_types
        ]

        # first part: a symmetric spiketrain with a symmetric kernel
        spiketrain = neo.SpikeTrain(
            np.array([-0.0001, 0, 0.0001]) * pq.s, t_start=-1, t_stop=1
        )
        for kernel in kernels_symmetric:
            rate = statistics.instantaneous_rate(
                spiketrain, sampling_period=20 * pq.ms, kernel=kernel
            )
            # the peak time must be centered at origin t=0
            self.assertEqual(rate.times[np.argmax(rate)], 0)

        # second part: a single spike at t=0
        periods = [2**exp for exp in range(-3, 6)]
        for period in periods:
            with self.subTest(period=period):
                spiketrain = neo.SpikeTrain(
                    np.array([0]) * pq.s,
                    t_start=-period * 10 * pq.ms,
                    t_stop=period * 10 * pq.ms,
                )
                for kernel in kernels_symmetric:
                    rate = statistics.instantaneous_rate(
                        spiketrain, sampling_period=period * pq.ms, kernel=kernel
                    )
                    self.assertEqual(rate.times[np.argmax(rate)], 0)

    def test_instantaneous_rate_annotations(self):
        spiketrain = neo.SpikeTrain([1, 2], t_stop=2 * pq.s, units=pq.s)
        kernel = kernels.AlphaKernel(sigma=100 * pq.ms)
        rate = statistics.instantaneous_rate(
            spiketrain, sampling_period=10 * pq.ms, kernel=kernel
        )
        kernel_annotation = dict(
            type=type(kernel).__name__, sigma=str(kernel.sigma), invert=kernel.invert
        )
        self.assertIn("kernel", rate.annotations)
        self.assertEqual(rate.annotations["kernel"], kernel_annotation)

    def test_instantaneous_rate_regression_374(self):
        # Check if the last interval is dropped.
        # In this example a spiketrain with t_start=0, t_stop=9.8, and spikes
        # at [9.65, 9.7, 9.75]s is used. When calculating the rate estimate
        # with a sampling_period of 1s, the last interval [9.0, 9.8) should be
        # dropped and not be considered in the calculation.
        spike_times = np.array([9.65, 9.7, 9.75]) * pq.s

        spiketrain = neo.SpikeTrain(spike_times, t_start=0, t_stop=9.8)
        kernel = kernels.GaussianKernel(sigma=250 * pq.ms)
        sampling_period = 1000 * pq.ms
        rate = statistics.instantaneous_rate(
            spiketrain,
            sampling_period=sampling_period,
            kernel=kernel,
            center_kernel=False,
            trim=False,
            cutoff=1,
        )
        assert_array_almost_equal(rate.magnitude, 0)

    def test_instantaneous_rate_rate_times(self):
        # check if the differences between the rate.times is equal to
        # sampling_period
        st = self.spike_train
        periods = [1, 0.99, 0.35, 11, st.duration] * pq.s
        for period in periods:
            rate = statistics.instantaneous_rate(
                st,
                sampling_period=period,
                kernel=self.kernel,
                center_kernel=True,
                trim=False,
            )
            rate_times_diff = np.diff(rate.times)
            period_times = np.full_like(rate_times_diff, period)
            assert_array_almost_equal(rate_times_diff, period_times)

    def test_instantaneous_rate_bin_edges(self):
        # This test checks if the bin edges used to calculate the rate estimate
        # are multiples of the sampling rate. In the following example, the
        # rate maximum is expected to be at 5.785s.
        # See PR#453  https://github.com/NeuralEnsemble/elephant/pull/453
        spike_times = np.array([4.45, 4.895, 5.34, 5.785, 6.23, 6.675, 7.12]) * pq.s
        # add 0.01 s
        shifted_spike_times = spike_times + 0.01 * pq.s

        spiketrain = neo.SpikeTrain(shifted_spike_times, t_start=0, t_stop=10)
        kernel = kernels.GaussianKernel(sigma=500 * pq.ms)
        sampling_period = 445 * pq.ms
        rate = statistics.instantaneous_rate(
            spiketrain,
            sampling_period=sampling_period,
            kernel=kernel,
            center_kernel=True,
            trim=False,
        )
        self.assertAlmostEqual(
            spike_times[3].magnitude.item(), rate.times[rate.argmax()].magnitude.item()
        )

    def test_instantaneous_rate_border_correction(self):
        np.random.seed(0)
        n_spiketrains = 125
        rate = 50.0 * pq.Hz
        t_start = 0.0 * pq.ms
        t_stop = 1000.0 * pq.ms
        sampling_period = 0.1 * pq.ms
        trial_list = StationaryPoissonProcess(
            rate=rate, t_start=t_start, t_stop=t_stop
        ).generate_n_spiketrains(n_spiketrains)
        for correction in (True, False):
            rates = []
            for trial in trial_list:
                # calculate the instantaneous rate, discard extra dimension
                instantaneous_rate = statistics.instantaneous_rate(
                    spiketrains=trial,
                    sampling_period=sampling_period,
                    kernel="auto",
                    border_correction=correction,
                )
                rates.append(instantaneous_rate)
        # The average estimated rate gives the average estimated value of
        # the firing rate in each time bin.
        # Note: the indexing [:, 0] is necessary to get the output an
        # one-dimensional array.
        average_estimated_rate = np.mean(rates, axis=0)[:, 0]
        rtol = 0.05  # Five percent of tolerance
        if correction:
            self.assertLess(np.max(average_estimated_rate), (1.0 + rtol) * rate.item())
            self.assertGreater(
                np.min(average_estimated_rate), (1.0 - rtol) * rate.item()
            )
        else:
            self.assertLess(np.max(average_estimated_rate), (1.0 + rtol) * rate.item())
            # The minimal rate deviates strongly in the uncorrected case.
            self.assertLess(np.min(average_estimated_rate), (1.0 - rtol) * rate.item())

    def test_instantaneous_rate_trials_pool_trials(self):
        kernel = kernels.GaussianKernel(sigma=500 * pq.ms)

        rate = statistics.instantaneous_rate(
            self.trial_object,
            sampling_period=0.1 * pq.ms,
            kernel=kernel,
            pool_spike_trains=False,
            pool_trials=True,
        )
        self.assertIsInstance(rate, neo.core.AnalogSignal)

    def test_instantaneous_rate_list_pool_spike_trains(self):
        kernel = kernels.GaussianKernel(sigma=500 * pq.ms)

        rate = statistics.instantaneous_rate(
            self.trial_object.get_spiketrains_from_trial_as_list(0),
            sampling_period=0.1 * pq.ms,
            kernel=kernel,
            pool_spike_trains=True,
            pool_trials=False,
        )
        self.assertIsInstance(rate, neo.core.AnalogSignal)
        self.assertEqual(rate.magnitude.shape[1], 1)

    def test_instantaneous_rate_list_of_spike_trains(self):
        kernel = kernels.GaussianKernel(sigma=500 * pq.ms)
        rate = statistics.instantaneous_rate(
            self.trial_object.get_spiketrains_from_trial_as_list(0),
            sampling_period=0.1 * pq.ms,
            kernel=kernel,
            pool_spike_trains=False,
            pool_trials=False,
        )
        self.assertIsInstance(rate, neo.core.AnalogSignal)
        self.assertEqual(rate.magnitude.shape[1], 2)


class TimeHistogramTestCase(unittest.TestCase):
    def setUp(self):
        self.spiketrain_a = neo.SpikeTrain(
            [0.5, 0.7, 1.2, 3.1, 4.3, 5.5, 6.7] * pq.s, t_stop=10.0 * pq.s
        )
        self.spiketrain_b = neo.SpikeTrain(
            [0.1, 0.7, 1.2, 2.2, 4.3, 5.5, 8.0] * pq.s, t_stop=10.0 * pq.s
        )
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
        histogram = statistics.time_histogram(
            self.spiketrains, bin_size=pq.s, binary=True
        )
        assert_array_equal(targ, histogram.magnitude[:, 0])

    def test_time_histogram_tstart_tstop(self):
        # Start, stop short range
        targ = np.array([2, 1])
        histogram = statistics.time_histogram(
            self.spiketrains, bin_size=pq.s, t_start=5 * pq.s, t_stop=7 * pq.s
        )
        assert_array_equal(targ, histogram.magnitude[:, 0])

        # Test without t_stop
        targ = np.array([4, 2, 1, 1, 2, 2, 1, 0, 1, 0])
        histogram = statistics.time_histogram(
            self.spiketrains, bin_size=1 * pq.s, t_start=0 * pq.s
        )
        assert_array_equal(targ, histogram.magnitude[:, 0])

        # Test without t_start
        histogram = statistics.time_histogram(
            self.spiketrains, bin_size=1 * pq.s, t_stop=10 * pq.s
        )
        assert_array_equal(targ, histogram.magnitude[:, 0])

    def test_time_histogram_output(self):
        # Normalization mean
        histogram = statistics.time_histogram(
            self.spiketrains, bin_size=pq.s, output="mean"
        )
        targ = np.array([4, 2, 1, 1, 2, 2, 1, 0, 1, 0], dtype=float) / 2
        assert_array_equal(targ.reshape(targ.size, 1), histogram.magnitude)

        # Normalization rate
        histogram = statistics.time_histogram(
            self.spiketrains, bin_size=pq.s, output="rate"
        )
        assert_array_equal(
            histogram.view(pq.Quantity), targ.reshape(targ.size, 1) * 1 / pq.s
        )

        # Normalization unspecified, raises error
        self.assertRaises(
            ValueError,
            statistics.time_histogram,
            self.spiketrains,
            bin_size=pq.s,
            output=" ",
        )

    def test_annotations(self):
        np.random.seed(1)
        spiketrains = StationaryPoissonProcess(
            rate=10 * pq.Hz, t_stop=10 * pq.s
        ).generate_n_spiketrains(n_spiketrains=10)
        for output in ("counts", "mean", "rate"):
            histogram = statistics.time_histogram(
                spiketrains, bin_size=3 * pq.ms, output=output
            )
            self.assertIn("normalization", histogram.annotations)
            self.assertEqual(histogram.annotations["normalization"], output)


class ComplexityTestCase(unittest.TestCase):
    def test_complexity_pdf_deprecated(self):
        spiketrain_a = neo.SpikeTrain(
            [0.5, 0.7, 1.2, 2.3, 4.3, 5.5, 6.7] * pq.s, t_stop=10.0 * pq.s
        )
        spiketrain_b = neo.SpikeTrain(
            [0.5, 0.7, 1.2, 2.3, 4.3, 5.5, 8.0] * pq.s, t_stop=10.0 * pq.s
        )
        spiketrain_c = neo.SpikeTrain(
            [0.5, 0.7, 1.2, 2.3, 4.3, 5.5, 8.0] * pq.s, t_stop=10.0 * pq.s
        )
        spiketrains = [spiketrain_a, spiketrain_b, spiketrain_c]
        # runs the previous function which will be deprecated
        targ = np.array([0.92, 0.01, 0.01, 0.06])

        complexity = statistics.complexity_pdf(spiketrains, bin_size=0.1 * pq.s)
        assert_array_equal(targ, complexity.magnitude[:, 0])
        self.assertEqual(1, complexity.magnitude[:, 0].sum())
        self.assertEqual(len(spiketrains) + 1, len(complexity))
        self.assertIsInstance(complexity, neo.AnalogSignal)
        self.assertEqual(complexity.units, 1 * pq.dimensionless)

    def test_complexity_pdf(self):
        spiketrain_a = neo.SpikeTrain(
            [0.5, 0.7, 1.2, 2.3, 4.3, 5.5, 6.7] * pq.s, t_stop=10.0 * pq.s
        )
        spiketrain_b = neo.SpikeTrain(
            [0.5, 0.7, 1.2, 2.3, 4.3, 5.5, 8.0] * pq.s, t_stop=10.0 * pq.s
        )
        spiketrain_c = neo.SpikeTrain(
            [0.5, 0.7, 1.2, 2.3, 4.3, 5.5, 8.0] * pq.s, t_stop=10.0 * pq.s
        )
        spiketrains = [spiketrain_a, spiketrain_b, spiketrain_c]
        # runs the previous function which will be deprecated
        targ = np.array([0.92, 0.01, 0.01, 0.06])
        complexity_obj = statistics.Complexity(spiketrains, bin_size=0.1 * pq.s)
        pdf = complexity_obj.pdf()
        assert_array_equal(targ, complexity_obj.pdf().magnitude[:, 0])
        self.assertEqual(1, pdf.magnitude[:, 0].sum())
        self.assertEqual(len(spiketrains) + 1, len(pdf))
        self.assertIsInstance(pdf, neo.AnalogSignal)
        self.assertEqual(pdf.units, 1 * pq.dimensionless)

    def test_complexity_histogram_spread_0(self):
        sampling_rate = 1 / pq.s

        spiketrains = [
            neo.SpikeTrain([1, 5, 9, 11, 16, 19] * pq.s, t_stop=20 * pq.s),
            neo.SpikeTrain([1, 4, 8, 12, 16, 18] * pq.s, t_stop=20 * pq.s),
        ]

        correct_histogram = np.array([10, 8, 2])

        correct_time_histogram = np.array(
            [0, 2, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 2, 0, 1, 1]
        )

        complexity_obj = statistics.Complexity(
            spiketrains, sampling_rate=sampling_rate, spread=0
        )

        assert_array_equal(complexity_obj.complexity_histogram, correct_histogram)

        assert_array_equal(
            complexity_obj.time_histogram.magnitude.flatten().astype(int),
            correct_time_histogram,
        )

    def test_complexity_histogram_spread_0_nonbinary(self):
        sampling_rate = 1 / pq.s

        spiketrains = [
            neo.SpikeTrain([1, 5, 5, 9, 11, 16, 19] * pq.s, t_stop=20 * pq.s),
            neo.SpikeTrain([1, 4, 8, 12, 16, 16, 18] * pq.s, t_stop=20 * pq.s),
        ]

        correct_histogram = np.array([10, 7, 2, 1])

        correct_time_histogram = np.array(
            [0, 2, 0, 0, 1, 2, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 3, 0, 1, 1]
        )

        complexity_obj = statistics.Complexity(
            spiketrains, sampling_rate=sampling_rate, binary=False, spread=0
        )

        assert_array_equal(complexity_obj.complexity_histogram, correct_histogram)

        assert_array_equal(
            complexity_obj.time_histogram.magnitude.flatten().astype(int),
            correct_time_histogram,
        )

    def test_complexity_epoch_spread_0(self):
        sampling_rate = 1 / pq.s

        spiketrains = [
            neo.SpikeTrain([1, 5, 9, 11, 16, 19] * pq.s, t_stop=20 * pq.s),
            neo.SpikeTrain([1, 4, 8, 12, 16, 18] * pq.s, t_stop=20 * pq.s),
        ]

        complexity_obj = statistics.Complexity(
            spiketrains, sampling_rate=sampling_rate, spread=0
        )

        self.assertIsInstance(complexity_obj.epoch, neo.Epoch)

    def test_complexity_histogram_spread_1(self):
        sampling_rate = 1 / pq.s

        spiketrains = [
            neo.SpikeTrain([0, 1, 5, 9, 11, 13, 20] * pq.s, t_stop=21 * pq.s),
            neo.SpikeTrain([1, 4, 7, 12, 16, 18] * pq.s, t_stop=21 * pq.s),
        ]

        correct_histogram = np.array([9, 5, 1, 2])

        correct_time_histogram = np.array(
            [3, 3, 0, 0, 2, 2, 0, 1, 0, 1, 0, 3, 3, 3, 0, 0, 1, 0, 1, 0, 1]
        )

        complexity_obj = statistics.Complexity(
            spiketrains, sampling_rate=sampling_rate, spread=1
        )

        assert_array_equal(complexity_obj.complexity_histogram, correct_histogram)

        assert_array_equal(
            complexity_obj.time_histogram.magnitude.flatten().astype(int),
            correct_time_histogram,
        )

    def test_complexity_histogram_spread_1_nonbinary(self):
        sampling_rate = 1 / pq.s

        spiketrains = [
            neo.SpikeTrain([0, 1, 5, 5, 9, 11, 13, 20] * pq.s, t_stop=21 * pq.s),
            neo.SpikeTrain([1, 4, 7, 12, 16, 16, 18] * pq.s, t_stop=21 * pq.s),
        ]

        correct_histogram = np.array([9, 4, 1, 3])

        correct_time_histogram = np.array(
            [3, 3, 0, 0, 3, 3, 0, 1, 0, 1, 0, 3, 3, 3, 0, 0, 2, 0, 1, 0, 1]
        )

        complexity_obj = statistics.Complexity(
            spiketrains, sampling_rate=sampling_rate, binary=False, spread=1
        )

        assert_array_equal(complexity_obj.complexity_histogram, correct_histogram)

        assert_array_equal(
            complexity_obj.time_histogram.magnitude.flatten().astype(int),
            correct_time_histogram,
        )

    def test_complexity_histogram_spread_2(self):
        sampling_rate = 1 / pq.s

        spiketrains = [
            neo.SpikeTrain([1, 5, 9, 11, 13, 20] * pq.s, t_stop=21 * pq.s),
            neo.SpikeTrain([1, 4, 7, 12, 16, 18] * pq.s, t_stop=21 * pq.s),
        ]

        correct_histogram = np.array([5, 0, 1, 1, 0, 0, 0, 1])

        correct_time_histogram = np.array(
            [0, 2, 0, 0, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 0, 0, 3, 3, 3, 3, 3]
        )

        complexity_obj = statistics.Complexity(
            spiketrains, sampling_rate=sampling_rate, spread=2
        )

        assert_array_equal(complexity_obj.complexity_histogram, correct_histogram)

        assert_array_equal(
            complexity_obj.time_histogram.magnitude.flatten().astype(int),
            correct_time_histogram,
        )

    def test_complexity_histogram_spread_2_nonbinary(self):
        sampling_rate = 1 / pq.s

        spiketrains = [
            neo.SpikeTrain([1, 5, 5, 9, 11, 13, 20] * pq.s, t_stop=21 * pq.s),
            neo.SpikeTrain([1, 4, 7, 12, 16, 16, 18] * pq.s, t_stop=21 * pq.s),
        ]

        correct_histogram = np.array([5, 0, 1, 0, 1, 0, 0, 0, 1])

        correct_time_histogram = np.array(
            [0, 2, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 4, 4, 4, 4, 4]
        )

        complexity_obj = statistics.Complexity(
            spiketrains, sampling_rate=sampling_rate, binary=False, spread=2
        )

        assert_array_equal(complexity_obj.complexity_histogram, correct_histogram)

        assert_array_equal(
            complexity_obj.time_histogram.magnitude.flatten().astype(int),
            correct_time_histogram,
        )

    def test_wrong_input_errors(self):
        spiketrains = [
            neo.SpikeTrain([1, 5, 9, 11, 13, 20] * pq.s, t_stop=21 * pq.s),
            neo.SpikeTrain([1, 4, 7, 12, 16, 18] * pq.s, t_stop=21 * pq.s),
        ]

        self.assertRaises(ValueError, statistics.Complexity, spiketrains)

        self.assertRaises(
            ValueError,
            statistics.Complexity,
            spiketrains,
            sampling_rate=1 * pq.s,
            spread=-7,
        )

    def test_sampling_rate_warning(self):
        spiketrains = [
            neo.SpikeTrain([1, 5, 9, 11, 13, 20] * pq.s, t_stop=21 * pq.s),
            neo.SpikeTrain([1, 4, 7, 12, 16, 18] * pq.s, t_stop=21 * pq.s),
        ]

        with self.assertWarns(UserWarning):
            statistics.Complexity(spiketrains, bin_size=1 * pq.s, spread=1)

    def test_binning_for_input_with_rounding_errors(self):
        # a test with inputs divided by 30000 which leads to rounding errors
        # these errors have to be accounted for by proper binning;
        # check if we still get the correct result

        sampling_rate = 333 / pq.s

        spiketrains = [
            neo.SpikeTrain(
                np.arange(1000, step=2) * pq.s / 333, t_stop=30.33333333333 * pq.s
            ),
            neo.SpikeTrain(
                np.arange(2000, step=4) * pq.s / 333, t_stop=30.33333333333 * pq.s
            ),
        ]

        correct_time_histogram = np.zeros(10101)
        correct_time_histogram[:1000:2] = 1
        correct_time_histogram[:2000:4] += 1

        complexity_obj = statistics.Complexity(
            spiketrains, sampling_rate=sampling_rate, spread=1
        )

        assert_array_equal(
            complexity_obj.time_histogram.magnitude.flatten().astype(int),
            correct_time_histogram,
        )


if __name__ == "__main__":
    unittest.main()
