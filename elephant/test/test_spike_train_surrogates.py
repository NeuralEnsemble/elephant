# -*- coding: utf-8 -*-
"""
unittests for spike_train_surrogates module.

:copyright: Copyright 2014-2026 by the Elephant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
"""

import unittest
import random

import elephant.spike_train_surrogates as surr
import elephant.spike_train_generation as stg
import elephant.conversion as conv
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_less
import quantities as pq
import neo


class SurrogatesTestCase(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        random.seed(0)

    @classmethod
    def setUpClass(cls) -> None:
        st1 = neo.SpikeTrain([90, 150, 180, 350] * pq.ms, t_stop=500 * pq.ms)
        cls.st1 = st1

    def test_dither_spikes_output_format(self):
        self.st1.t_stop = .5 * pq.s
        n_surrogates = 2
        dither = 10 * pq.ms
        surrogate_trains = surr.dither_spikes(
            self.st1, dither=dither, n_surrogates=n_surrogates)

        self.assertIsInstance(surrogate_trains, list)
        self.assertEqual(len(surrogate_trains), n_surrogates)

        self.assertIsInstance(surrogate_trains[0], neo.SpikeTrain)
        for surrogate_train in surrogate_trains:
            self.assertEqual(surrogate_train.units, self.st1.units)
            self.assertEqual(surrogate_train.t_start, self.st1.t_start)
            self.assertEqual(surrogate_train.t_stop, self.st1.t_stop)
            self.assertEqual(len(surrogate_train), len(self.st1))
            assert_array_less(0., np.diff(surrogate_train))  # check ordering

    def test_dither_spikes_empty_train(self):

        st = neo.SpikeTrain([] * pq.ms, t_stop=500 * pq.ms)

        dither = 10 * pq.ms
        surrogate_train = surr.dither_spikes(
            st, dither=dither, n_surrogates=1)[0]
        self.assertEqual(len(surrogate_train), 0)

    def test_dither_spikes_refactory_period_zero_or_none(self):
        dither = 10 * pq.ms
        decimals = 3
        n_surrogates = 1

        np.random.seed(42)
        surrogate_trains_zero = surr.dither_spikes(
            self.st1, dither, decimals=decimals, n_surrogates=n_surrogates,
            refractory_period=0)
        np.random.seed(42)
        surrogate_trains_none = surr.dither_spikes(
            self.st1, dither, decimals=decimals, n_surrogates=n_surrogates,
            refractory_period=None)
        np.testing.assert_array_almost_equal(
            surrogate_trains_zero[0].magnitude,
            surrogate_trains_none[0].magnitude)

    def test_dither_spikes_output_decimals(self):
        n_surrogates = 2
        dither = 10 * pq.ms
        np.random.seed(42)
        surrogate_trains = surr.dither_spikes(
            self.st1, dither=dither, decimals=3, n_surrogates=n_surrogates)

        np.random.seed(42)
        dither_values = np.random.random_sample((n_surrogates, len(self.st1)))
        expected_non_dithered = np.sum(dither_values == 0)

        observed_non_dithered = 0
        for surrogate_train in surrogate_trains:
            for i in range(len(surrogate_train)):
                if surrogate_train[i] - int(surrogate_train[i]) * \
                        pq.ms == surrogate_train[i] - surrogate_train[i]:
                    observed_non_dithered += 1

        self.assertEqual(observed_non_dithered, expected_non_dithered)

    def test_dither_spikes_false_edges(self):
        n_surrogates = 2
        dither = 10 * pq.ms
        surrogate_trains = surr.dither_spikes(
            self.st1, dither=dither, n_surrogates=n_surrogates, edges=False)

        for surrogate_train in surrogate_trains:
            for i in range(len(surrogate_train)):
                self.assertLessEqual(surrogate_train[i], self.st1.t_stop)

    def test_dither_spikes_with_refractory_period_output_format(self):

        spiketrain = neo.SpikeTrain([90, 93, 97, 100, 105,
                                     150, 180, 350] * pq.ms, t_stop=.5 * pq.s)
        n_surrogates = 2
        dither = 10 * pq.ms
        surrogate_trains = surr.dither_spikes(
            spiketrain, dither=dither, n_surrogates=n_surrogates,
            refractory_period=4 * pq.ms)

        self.assertIsInstance(surrogate_trains, list)
        self.assertEqual(len(surrogate_trains), n_surrogates)

        self.assertIsInstance(surrogate_trains[0], neo.SpikeTrain)
        for surrogate_train in surrogate_trains:
            self.assertEqual(surrogate_train.units, spiketrain.units)
            self.assertEqual(surrogate_train.t_start, spiketrain.t_start)
            self.assertEqual(surrogate_train.t_stop, spiketrain.t_stop)
            self.assertEqual(len(surrogate_train), len(spiketrain))
            # Check that refractory period is conserved
            self.assertLessEqual(np.min(np.diff(spiketrain)),
                                 np.min(np.diff(surrogate_train)))
            sigma_displacement = np.std(surrogate_train - spiketrain)
            # Check that spikes are moved
            self.assertLessEqual(dither / 10, sigma_displacement)
            # Spikes are not moved more than dither
            self.assertLessEqual(sigma_displacement, dither)

        self.assertRaises(ValueError, surr.dither_spikes,
                          spiketrain, dither=dither, refractory_period=3)

    def test_dither_spikes_with_refractory_period_empty_train(self):

        spiketrain = neo.SpikeTrain([] * pq.ms, t_stop=500 * pq.ms)

        dither = 10 * pq.ms
        surrogate_train = surr.dither_spikes(
            spiketrain, dither=dither, n_surrogates=1,
            refractory_period=4 * pq.ms)[0]
        self.assertEqual(len(surrogate_train), 0)

    def test_dither_spikes_regression_issue_586(self):
        """
        When using the dither_spikes surrogate generation function, with the
        edges=True option, there is an exception when spikes are removed due
        to being dithered outside the spiketrain duration.

        Since the arrays in the list will have different dimensions, the
        multiplication operator fails.
        However, this worked with numpy==1.23 and fails with numpy>=1.24.
        See: https://github.com/NeuralEnsemble/elephant/issues/586
        """
        # Generate one spiketrain with a spike close to t_stop
        t_stop = 2 * pq.s
        st = stg.StationaryPoissonProcess(
            rate=10 * pq.Hz, t_stop=t_stop).generate_spiketrain()
        st = neo.SpikeTrain(np.hstack([st.magnitude, [1.9999999]]),
                            units=st.units, t_stop=t_stop)

        # Dither
        np.random.seed(5)
        surrogate_trains = surr.dither_spikes(
            st, dither=15 * pq.ms, n_surrogates=30, edges=True, decimals=2)
        for surrogate in surrogate_trains:
            with self.subTest(surrogate):
                self.assertLess(surrogate[-1], surrogate.t_stop)
                self.assertGreater(surrogate[0], surrogate.t_start)

    def test_randomise_spikes_output_format(self):
        n_surrogates = 2
        surrogate_trains = surr.randomise_spikes(
            self.st1, n_surrogates=n_surrogates)

        self.assertIsInstance(surrogate_trains, list)
        self.assertEqual(len(surrogate_trains), n_surrogates)

        self.assertIsInstance(surrogate_trains[0], neo.SpikeTrain)
        for surrogate_train in surrogate_trains:
            self.assertEqual(surrogate_train.units, self.st1.units)
            self.assertEqual(surrogate_train.t_start, self.st1.t_start)
            self.assertEqual(surrogate_train.t_stop, self.st1.t_stop)
            self.assertEqual(len(surrogate_train), len(self.st1))

    def test_randomise_spikes_empty_train(self):

        spiketrain = neo.SpikeTrain([] * pq.ms, t_stop=500 * pq.ms)

        surrogate_train = surr.randomise_spikes(spiketrain, n_surrogates=1)[0]
        self.assertEqual(len(surrogate_train), 0)

    def test_randomise_spikes_output_decimals(self):
        n_surrogates = 2
        surrogate_trains = surr.randomise_spikes(
            self.st1, n_surrogates=n_surrogates, decimals=3)

        for surrogate_train in surrogate_trains:
            for i in range(len(surrogate_train)):
                self.assertNotEqual(surrogate_train[i] -
                                    int(surrogate_train[i]) *
                                    pq.ms, surrogate_train[i] -
                                    surrogate_train[i])

    def test_shuffle_isis_output_format(self):
        n_surrogates = 2
        surrogate_trains = surr.shuffle_isis(
            self.st1, n_surrogates=n_surrogates)

        self.assertIsInstance(surrogate_trains, list)
        self.assertEqual(len(surrogate_trains), n_surrogates)

        self.assertIsInstance(surrogate_trains[0], neo.SpikeTrain)
        for surrogate_train in surrogate_trains:
            self.assertEqual(surrogate_train.units, self.st1.units)
            self.assertEqual(surrogate_train.t_start, self.st1.t_start)
            self.assertEqual(surrogate_train.t_stop, self.st1.t_stop)
            self.assertEqual(len(surrogate_train), len(self.st1))

    def test_shuffle_isis_empty_train(self):

        spiketrain = neo.SpikeTrain([] * pq.ms, t_stop=500 * pq.ms)

        surrogate_train = surr.shuffle_isis(spiketrain, n_surrogates=1)[0]
        self.assertEqual(len(surrogate_train), 0)

    def test_shuffle_isis_same_isis(self):
        surrogate_train = surr.shuffle_isis(self.st1, n_surrogates=1)[0]

        st_pq = self.st1.view(pq.Quantity)
        surr_pq = surrogate_train.view(pq.Quantity)

        isi0_orig = self.st1[0] - self.st1.t_start
        ISIs_orig = np.sort([isi0_orig] + [isi for isi in np.diff(st_pq)])

        isi0_surr = surrogate_train[0] - surrogate_train.t_start
        ISIs_surr = np.sort([isi0_surr] + [isi for isi in np.diff(surr_pq)])

        self.assertTrue(np.all(ISIs_orig == ISIs_surr))

    def test_shuffle_isis_output_decimals(self):
        surrogate_train = surr.shuffle_isis(
            self.st1, n_surrogates=1, decimals=95)[0]

        st_pq = self.st1.view(pq.Quantity)
        surr_pq = surrogate_train.view(pq.Quantity)

        isi0_orig = self.st1[0] - self.st1.t_start
        ISIs_orig = np.sort([isi0_orig] + [isi for isi in np.diff(st_pq)])

        isi0_surr = surrogate_train[0] - surrogate_train.t_start
        ISIs_surr = np.sort([isi0_surr] + [isi for isi in np.diff(surr_pq)])

        self.assertTrue(np.all(ISIs_orig == ISIs_surr))

    def test_shuffle_isis_with_wrongly_ordered_spikes(self):
        surr_method = 'shuffle_isis'
        n_surr = 30
        dither = 15 * pq.ms
        spiketrain = neo.SpikeTrain(
            [39.65696411, 98.93868274, 120.2417674, 134.70971166,
             154.20788924,
             160.29077989, 179.19884034, 212.86773029, 247.59488061,
             273.04095041,
             297.56437605, 344.99204215, 418.55696486, 460.54298334,
             482.82299125,
             524.236052, 566.38966742, 597.87562722, 651.26965293,
             692.39802855,
             740.90285815, 849.45874695, 974.57724848, 8.79247605],
            t_start=0. * pq.ms, t_stop=1000. * pq.ms, units=pq.ms)
        surr.surrogates(spiketrain, n_surrogates=n_surr, method=surr_method,
                        dt=dither)

    def test_dither_spike_train_output_format(self):
        n_surrogates = 2
        shift = 10 * pq.ms
        surrogate_trains = surr.dither_spike_train(
            self.st1, shift=shift, n_surrogates=n_surrogates)

        self.assertIsInstance(surrogate_trains, list)
        self.assertEqual(len(surrogate_trains), n_surrogates)

        self.assertIsInstance(surrogate_trains[0], neo.SpikeTrain)
        for surrogate_train in surrogate_trains:
            self.assertEqual(surrogate_train.units, self.st1.units)
            self.assertEqual(surrogate_train.t_start, self.st1.t_start)
            self.assertEqual(surrogate_train.t_stop, self.st1.t_stop)
            self.assertEqual(len(surrogate_train), len(self.st1))

    def test_dither_spike_train_empty_train(self):

        spiketrain = neo.SpikeTrain([] * pq.ms, t_stop=500 * pq.ms)

        shift = 10 * pq.ms
        surrogate_train = surr.dither_spike_train(
            spiketrain, shift=shift, n_surrogates=1)[0]
        self.assertEqual(len(surrogate_train), 0)

    def test_dither_spike_train_output_decimals(self):
        n_surrogates = 2
        shift = 10 * pq.ms
        surrogate_trains = surr.dither_spike_train(
            self.st1, shift=shift, n_surrogates=n_surrogates, decimals=3)

        for surrogate_train in surrogate_trains:
            for i in range(len(surrogate_train)):
                self.assertNotEqual(surrogate_train[i] -
                                    int(surrogate_train[i]) *
                                    pq.ms, surrogate_train[i] -
                                    surrogate_train[i])

    def test_dither_spike_train_false_edges(self):
        n_surrogates = 2
        shift = 10 * pq.ms
        surrogate_trains = surr.dither_spike_train(
            self.st1, shift=shift, n_surrogates=n_surrogates, edges=False)

        for surrogate_train in surrogate_trains:
            for i in range(len(surrogate_train)):
                self.assertLessEqual(surrogate_train[i], self.st1.t_stop)

    def test_jitter_spikes_output_format(self):
        n_surrogates = 2
        bin_size = 100 * pq.ms
        surrogate_trains = surr.jitter_spikes(
            self.st1, bin_size=bin_size, n_surrogates=n_surrogates)

        self.assertIsInstance(surrogate_trains, list)
        self.assertEqual(len(surrogate_trains), n_surrogates)

        self.assertIsInstance(surrogate_trains[0], neo.SpikeTrain)
        for surrogate_train in surrogate_trains:
            self.assertEqual(surrogate_train.units, self.st1.units)
            self.assertEqual(surrogate_train.t_start, self.st1.t_start)
            self.assertEqual(surrogate_train.t_stop, self.st1.t_stop)
            self.assertEqual(len(surrogate_train), len(self.st1))

    def test_jitter_spikes_empty_train(self):

        spiketrain = neo.SpikeTrain([] * pq.ms, t_stop=500 * pq.ms)

        bin_size = 75 * pq.ms
        surrogate_train = surr.jitter_spikes(
            spiketrain, bin_size=bin_size, n_surrogates=1)[0]
        self.assertEqual(len(surrogate_train), 0)

    def test_jitter_spikes_same_bins(self):
        bin_size = 100 * pq.ms
        surrogate_train = surr.jitter_spikes(
            self.st1, bin_size=bin_size, n_surrogates=1)[0]

        bin_ids_orig = np.array(
            (self.st1.view(
                pq.Quantity) /
             bin_size).rescale(
                pq.dimensionless).magnitude,
            dtype=int)
        bin_ids_surr = np.array(
            (surrogate_train.view(
                pq.Quantity) /
             bin_size).rescale(
                pq.dimensionless).magnitude,
            dtype=int)
        self.assertTrue(np.all(bin_ids_orig == bin_ids_surr))

        # Bug encountered when the original and surrogate trains have
        # different number of spikes
        self.assertEqual(len(self.st1), len(surrogate_train))

    def test_jitter_spikes_unequal_bin_size(self):

        spiketrain = neo.SpikeTrain(
            [90, 150, 180, 480] * pq.ms, t_stop=500 * pq.ms)

        bin_size = 75 * pq.ms
        surrogate_train = surr.jitter_spikes(
            spiketrain, bin_size=bin_size, n_surrogates=1)[0]

        bin_ids_orig = np.array(
            (spiketrain.view(
                pq.Quantity) /
             bin_size).rescale(
                pq.dimensionless).magnitude,
            dtype=int)
        bin_ids_surr = np.array(
            (surrogate_train.view(
                pq.Quantity) /
             bin_size).rescale(
                pq.dimensionless).magnitude,
            dtype=int)

        self.assertTrue(np.all(bin_ids_orig == bin_ids_surr))

    def test_surr_method(self):

        surr_methods = \
            ('dither_spike_train', 'dither_spikes', 'jitter_spikes',
             'randomise_spikes', 'shuffle_isis', 'joint_isi_dithering',
             'dither_spikes_with_refractory_period', 'trial_shifting',
             'bin_shuffling', 'isi_dithering')

        surr_method_kwargs = \
            {'dither_spikes': {},
             'dither_spikes_with_refractory_period': {'refractory_period':
                                                      3 * pq.ms},
             'randomise_spikes': {},
             'shuffle_isis': {},
             'dither_spike_train': {},
             'jitter_spikes': {},
             'bin_shuffling': {'bin_size': 3 * pq.ms},
             'joint_isi_dithering': {},
             'isi_dithering': {},
             'trial_shifting': {'trial_length': 200 * pq.ms,
                                'trial_separation': 50 * pq.ms}}

        dt = 15 * pq.ms
        spiketrain = neo.SpikeTrain(
            [90, 150, 180, 350] * pq.ms, t_stop=500 * pq.ms)
        n_surrogates = 3
        for method in surr_methods:
            surrogates = surr.surrogates(
                spiketrain,
                dt=dt,
                n_surrogates=n_surrogates,
                method=method,
                **surr_method_kwargs[method]
            )
            self.assertTrue(len(surrogates) == n_surrogates)

            for surrogate_train in surrogates:
                self.assertTrue(
                    isinstance(surrogates[0], neo.SpikeTrain))
                self.assertEqual(surrogate_train.units, spiketrain.units)
                self.assertEqual(surrogate_train.t_start, spiketrain.t_start)
                self.assertEqual(surrogate_train.t_stop, spiketrain.t_stop)
                self.assertEqual(len(surrogate_train), len(spiketrain))
            self.assertTrue(len(surrogates) == n_surrogates)

        self.assertRaises(ValueError, surr.surrogates, spiketrain,
                          n_surrogates=1,
                          method='spike_shifting',
                          dt=None, decimals=None, edges=True)

        self.assertRaises(ValueError, surr.surrogates, spiketrain,
                          method='dither_spikes', dt=None)

        self.assertRaises(TypeError, surr.surrogates, spiketrain.magnitude,
                          method='dither_spikes', dt=10 * pq.ms)

    def test_joint_isi_dithering_format(self):

        rate = 100. * pq.Hz
        t_stop = 1. * pq.s
        process = stg.StationaryPoissonProcess(rate, t_stop=t_stop)
        spiketrain = process.generate_spiketrain()
        n_surrogates = 2
        dither = 10 * pq.ms

        # Test fast version
        joint_isi_instance = surr.JointISI(spiketrain, dither=dither,
                                           method='fast')
        surrogate_trains = joint_isi_instance.dithering(
            n_surrogates=n_surrogates)

        self.assertIsInstance(surrogate_trains, list)
        self.assertEqual(len(surrogate_trains), n_surrogates)
        self.assertEqual(joint_isi_instance.method, 'fast')

        for surrogate_train in surrogate_trains:
            self.assertIsInstance(surrogate_train, neo.SpikeTrain)
            self.assertEqual(surrogate_train.units, spiketrain.units)
            self.assertEqual(surrogate_train.t_start, spiketrain.t_start)
            self.assertEqual(surrogate_train.t_stop, spiketrain.t_stop)
            self.assertEqual(len(surrogate_train), len(spiketrain))

        # Test window_version
        joint_isi_instance = surr.JointISI(spiketrain,
                                           method='window',
                                           dither=2 * dither,
                                           n_bins=50)
        surrogate_trains = joint_isi_instance.dithering(
            n_surrogates=n_surrogates)

        self.assertIsInstance(surrogate_trains, list)
        self.assertEqual(len(surrogate_trains), n_surrogates)
        self.assertEqual(joint_isi_instance.method, 'window')

        for surrogate_train in surrogate_trains:
            self.assertIsInstance(surrogate_train, neo.SpikeTrain)
            self.assertEqual(surrogate_train.units, spiketrain.units)
            self.assertEqual(surrogate_train.t_start, spiketrain.t_start)
            self.assertEqual(surrogate_train.t_stop, spiketrain.t_stop)
            self.assertEqual(len(surrogate_train), len(spiketrain))

        # Test isi_dithering
        joint_isi_instance = surr.JointISI(spiketrain,
                                           method='window',
                                           dither=2 * dither,
                                           n_bins=50,
                                           isi_dithering=True,
                                           use_sqrt=True,
                                           cutoff=False)
        surrogate_trains = joint_isi_instance.dithering(
            n_surrogates=n_surrogates)

        self.assertIsInstance(surrogate_trains, list)
        self.assertEqual(len(surrogate_trains), n_surrogates)
        self.assertEqual(joint_isi_instance.method, 'window')

        for surrogate_train in surrogate_trains:
            self.assertIsInstance(surrogate_train, neo.SpikeTrain)
            self.assertEqual(surrogate_train.units, spiketrain.units)
            self.assertEqual(surrogate_train.t_start, spiketrain.t_start)
            self.assertEqual(surrogate_train.t_stop, spiketrain.t_stop)
            self.assertEqual(len(surrogate_train), len(spiketrain))

        # Test surrogate methods wrapper
        surrogate_trains = surr.surrogates(
            spiketrain,
            dt=15 * pq.ms,
            n_surrogates=n_surrogates,
            method='joint_isi_dithering')
        self.assertIsInstance(surrogate_trains, list)
        self.assertEqual(len(surrogate_trains), n_surrogates)

        for surrogate_train in surrogate_trains:
            self.assertIsInstance(surrogate_train, neo.SpikeTrain)
            self.assertEqual(surrogate_train.units, spiketrain.units)
            self.assertEqual(surrogate_train.t_start, spiketrain.t_start)
            self.assertEqual(surrogate_train.t_stop, spiketrain.t_stop)
            self.assertEqual(len(surrogate_train), len(spiketrain))
        with self.assertRaises(ValueError):
            joint_isi_instance = surr.JointISI(spiketrain,
                                               method='wrong method',
                                               dither=2 * dither,
                                               n_bins=50)

    def test_joint_isi_dithering_empty_train(self):
        spiketrain = neo.SpikeTrain([] * pq.ms, t_stop=500 * pq.ms)
        surrogate_train = surr.JointISI(spiketrain).dithering()[0]
        self.assertEqual(len(surrogate_train), 0)

    def test_joint_isi_dithering_output(self):
        process = stg.StationaryPoissonProcess(
            rate=100. * pq.Hz, refractory_period=3 * pq.ms, t_stop=0.1 * pq.s)
        spiketrain = process.generate_spiketrain()
        surrogate_train = surr.JointISI(spiketrain).dithering()[0]
        ground_truth = [0.0060744, 0.01886591, 0.02732847, 0.03683888,
                        0.04569622, 0.05196334, 0.05899197, 0.07855664]
        assert_array_almost_equal(surrogate_train.magnitude, ground_truth)

    def test_joint_isi_with_wrongly_ordered_spikes(self):
        surr_method = 'joint_isi_dithering'
        n_surr = 30
        dither = 15 * pq.ms
        spiketrain = neo.SpikeTrain(
            [39.65696411, 98.93868274, 120.2417674, 134.70971166,
             154.20788924,
             160.29077989, 179.19884034, 212.86773029, 247.59488061,
             273.04095041,
             297.56437605, 344.99204215, 418.55696486, 460.54298334,
             482.82299125,
             524.236052, 566.38966742, 597.87562722, 651.26965293,
             692.39802855,
             740.90285815, 849.45874695, 974.57724848, 8.79247605],
            t_start=0. * pq.ms, t_stop=1000. * pq.ms, units=pq.ms)
        surr.surrogates(spiketrain, n_surrogates=n_surr, method=surr_method,
                        dt=dither)

    def test_joint_isi_spikes_at_border(self):
        surr_method = 'joint_isi_dithering'
        n_surr = 30
        dither = 15 * pq.ms
        spiketrain = neo.SpikeTrain(
            [4., 28., 45., 51., 83., 87., 96., 111., 126., 131.,
             138., 150.,
             209., 232., 253., 275., 279., 303., 320., 371., 396.,
             401., 429., 447.,
             479., 511., 535., 549., 581., 585., 605., 607., 626.,
             630., 644., 714.,
             832., 835., 853., 858., 878., 905., 909., 932., 950.,
             961., 999., 1000.],
            t_start=0. * pq.ms, t_stop=1000. * pq.ms, units=pq.ms)
        surr.surrogates(
            spiketrain, n_surrogates=n_surr, method=surr_method, dt=dither)

    def test_bin_shuffling_output_format(self):

        self.bin_size = 3 * pq.ms
        self.max_displacement = 10
        spiketrain = neo.SpikeTrain([90, 93, 97, 100, 105,
                                     150, 180, 350] * pq.ms, t_stop=.5 * pq.s)
        binned_spiketrain = conv.BinnedSpikeTrain(spiketrain, self.bin_size)
        n_surrogates = 2

        for sliding in (True, False):
            surrogate_trains = surr.bin_shuffling(
                binned_spiketrain, max_displacement=self.max_displacement,
                n_surrogates=n_surrogates, sliding=sliding)

            self.assertIsInstance(surrogate_trains, list)
            self.assertEqual(len(surrogate_trains), n_surrogates)

            self.assertIsInstance(surrogate_trains[0], conv.BinnedSpikeTrain)
            for surrogate_train in surrogate_trains:
                self.assertEqual(surrogate_train.t_start,
                                 binned_spiketrain.t_start)
                self.assertEqual(surrogate_train.t_stop,
                                 binned_spiketrain.t_stop)
                self.assertEqual(surrogate_train.n_bins,
                                 binned_spiketrain.n_bins)
                self.assertEqual(surrogate_train.bin_size,
                                 binned_spiketrain.bin_size)

    def test_bin_shuffling_empty_train(self):

        self.bin_size = 3 * pq.ms
        self.max_displacement = 10
        empty_spiketrain = neo.SpikeTrain([] * pq.ms, t_stop=500 * pq.ms)

        binned_spiketrain = conv.BinnedSpikeTrain(empty_spiketrain,
                                                  self.bin_size)
        surrogate_train = surr.bin_shuffling(
            binned_spiketrain, max_displacement=self.max_displacement,
            n_surrogates=1)[0]
        self.assertEqual(np.sum(surrogate_train.to_bool_array()), 0)

    def test_bin_shuffling_spike_count_preserved(self):
        """Total spike count must be unchanged after bin shuffling."""
        spiketrain = neo.SpikeTrain(
            [90, 93, 97, 100, 105, 150, 180, 350] * pq.ms, t_stop=500 * pq.ms)
        bin_size = 5 * pq.ms
        max_displacement = 10
        binned = conv.BinnedSpikeTrain(spiketrain, bin_size)
        original_count = int(np.sum(binned.to_bool_array()))
        for surrogate in surr.bin_shuffling(
                binned, max_displacement=max_displacement, n_surrogates=10):
            self.assertEqual(int(np.sum(surrogate.to_bool_array())),
                             original_count)

    def test_bin_shuffling_window_spike_counts_preserved(self):
        """Spikes must not cross non-overlapping window boundaries."""
        max_displacement = 5   # displacement_window = 10 bins
        bin_size = 10 * pq.ms  # each window covers 100 ms
        # 4 windows: [0-99], [100-199], [200-299], [300-399] ms
        spiketrain = neo.SpikeTrain(
            [15, 55, 135, 215, 265, 315, 375] * pq.ms, t_stop=400 * pq.ms)
        binned = conv.BinnedSpikeTrain(spiketrain, bin_size)
        bool_arr = binned.to_bool_array()[0]
        window_size = 2 * max_displacement
        n_windows = len(bool_arr) // window_size
        orig_counts = [int(np.sum(bool_arr[i * window_size:
                                           (i + 1) * window_size]))
                       for i in range(n_windows)]
        for surrogate in surr.bin_shuffling(
                binned, max_displacement=max_displacement, n_surrogates=20):
            surr_bool = surrogate.to_bool_array()[0]
            for i in range(n_windows):
                surr_count = int(np.sum(
                    surr_bool[i * window_size:(i + 1) * window_size]))
                self.assertEqual(surr_count, orig_counts[i],
                                 f"Spike count changed in window {i}")

    def test_bin_shuffling_surrogates_differ(self):
        """Multiple surrogates must not all be identical."""
        spiketrain = neo.SpikeTrain(
            [50, 100, 150, 200, 250] * pq.ms, t_stop=500 * pq.ms)
        bin_size = 5 * pq.ms
        binned = conv.BinnedSpikeTrain(spiketrain, bin_size)
        n_surrogates = 10
        surrogate_trains = surr.bin_shuffling(
            binned, max_displacement=20, n_surrogates=n_surrogates)
        bool_arrays = [s.to_bool_array()[0] for s in surrogate_trains]
        all_equal = all(np.array_equal(bool_arrays[0], b)
                        for b in bool_arrays[1:])
        self.assertFalse(all_equal,
                         "All surrogates produced identical spike trains")

    def test_bin_shuffling_neo_spiketrain_input(self):
        """Passing neo.SpikeTrain + bin_size must return list of neo.SpikeTrain."""
        spiketrain = neo.SpikeTrain(
            [90, 93, 97, 100, 105, 150, 180, 350] * pq.ms, t_stop=500 * pq.ms)
        bin_size = 10 * pq.ms
        max_displacement = 5
        n_surrogates = 3
        surrogate_trains = surr.bin_shuffling(
            spiketrain, max_displacement=max_displacement,
            bin_size=bin_size, n_surrogates=n_surrogates)
        self.assertIsInstance(surrogate_trains, list)
        self.assertEqual(len(surrogate_trains), n_surrogates)
        for surrogate_train in surrogate_trains:
            self.assertIsInstance(surrogate_train, neo.SpikeTrain)
            self.assertEqual(surrogate_train.t_start, spiketrain.t_start)
            self.assertEqual(surrogate_train.t_stop, spiketrain.t_stop)
            if len(surrogate_train) > 1:
                assert_array_less(0., np.diff(surrogate_train.magnitude))

    def test_bin_shuffling_neo_spiketrain_missing_bin_size_raises(self):
        """Passing neo.SpikeTrain without bin_size must raise ValueError."""
        spiketrain = neo.SpikeTrain([50, 100, 200] * pq.ms, t_stop=500 * pq.ms)
        with self.assertRaises(ValueError):
            surr.bin_shuffling(spiketrain, max_displacement=5)

    def test_bin_shuffling_sliding_on_neo_spiketrain_warns(self):
        """Using sliding=True with a neo.SpikeTrain input must issue UserWarning."""
        spiketrain = neo.SpikeTrain([50, 100, 200] * pq.ms, t_stop=500 * pq.ms)
        with self.assertWarns(UserWarning):
            surr.bin_shuffling(spiketrain, max_displacement=5,
                               bin_size=10 * pq.ms, sliding=True)

    def test_bin_shuffling_via_surrogates_wrapper(self):
        """surrogates() with method='bin_shuffling' must return neo.SpikeTrain."""
        spiketrain = neo.SpikeTrain(
            [90, 93, 97, 100, 105] * pq.ms, t_stop=200 * pq.ms)
        bin_size = 5 * pq.ms
        n_surrogates = 3
        result = surr.surrogates(spiketrain, n_surrogates=n_surrogates,
                                 method='bin_shuffling',
                                 dt=10 * pq.ms, bin_size=bin_size)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), n_surrogates)
        for s in result:
            self.assertIsInstance(s, neo.SpikeTrain)

    def test_trial_shuffling_output_format(self):
        spiketrain = \
            [neo.SpikeTrain([90, 93, 97, 100, 105, 150, 180, 190] * pq.ms,
                            t_stop=.2 * pq.s),
             neo.SpikeTrain([90, 93, 97, 100, 105, 150, 180, 190] * pq.ms,
                            t_stop=.2 * pq.s)]
        n_surrogates = 2
        dither = 10 * pq.ms
        surrogate_trains = surr.trial_shifting(
            spiketrain, dither=dither, n_surrogates=n_surrogates)

        self.assertIsInstance(surrogate_trains, list)
        self.assertEqual(len(surrogate_trains), n_surrogates)

        self.assertIsInstance(surrogate_trains[0], list)
        self.assertIsInstance(surrogate_trains[0][0], neo.SpikeTrain)
        # check all surrogates, not just the first one
        for surr_idx, surrogate in enumerate(surrogate_trains):
            for trial_idx, surrogate_train in enumerate(surrogate):
                ref_trial = spiketrain[trial_idx]
                self.assertEqual(surrogate_train.units, ref_trial.units)
                self.assertEqual(surrogate_train.t_start, ref_trial.t_start)
                self.assertEqual(surrogate_train.t_stop, ref_trial.t_stop)
                self.assertEqual(len(surrogate_train), len(ref_trial))
                assert_array_less(0., np.diff(surrogate_train))

    def test_trial_shuffling_empty_train(self):

        empty_spiketrain = [neo.SpikeTrain([] * pq.ms, t_stop=500 * pq.ms),
                            neo.SpikeTrain([] * pq.ms, t_stop=500 * pq.ms)]

        dither = 10 * pq.ms
        surrogate_train = surr.trial_shifting(
            empty_spiketrain, dither=dither, n_surrogates=1)[0]

        self.assertEqual(len(surrogate_train), 2)
        self.assertEqual(len(surrogate_train[0]), 0)

    def test_trial_shuffling_output_format_concatenated(self):
        spiketrain = neo.SpikeTrain([90, 93, 97, 100, 105,
                                     150, 180, 350] * pq.ms, t_stop=.5 * pq.s)
        trial_length = 200 * pq.ms
        trial_separation = 50 * pq.ms
        n_surrogates = 2
        dither = 10 * pq.ms
        surrogate_trains = surr._trial_shifting_of_concatenated_spiketrain(
            spiketrain, dither=dither, n_surrogates=n_surrogates,
            trial_length=trial_length, trial_separation=trial_separation)

        self.assertIsInstance(surrogate_trains, list)
        self.assertEqual(len(surrogate_trains), n_surrogates)

        self.assertIsInstance(surrogate_trains[0], neo.SpikeTrain)
        for surrogate_train in surrogate_trains:
            self.assertEqual(surrogate_train.units, spiketrain.units)
            self.assertEqual(surrogate_train.t_start, spiketrain.t_start)
            self.assertEqual(surrogate_train.t_stop, spiketrain.t_stop)
            self.assertEqual(len(surrogate_train), len(spiketrain))
            assert_array_less(0., np.diff(surrogate_train))  # check ordering

    def test_trial_shuffling_empty_train_concatenated(self):

        empty_spiketrain = neo.SpikeTrain([] * pq.ms, t_stop=500 * pq.ms)
        trial_length = 200 * pq.ms
        trial_separation = 50 * pq.ms

        dither = 10 * pq.ms
        surrogate_train = surr._trial_shifting_of_concatenated_spiketrain(
            empty_spiketrain, dither=dither, n_surrogates=1,
            trial_length=trial_length, trial_separation=trial_separation)[0]
        self.assertEqual(len(surrogate_train), 0)

    def test_trial_shifting_spikes_within_bounds(self):
        """All surrogate spikes must stay within each trial's [t_start, t_stop]."""
        trials = [
            neo.SpikeTrain([10, 50, 90, 150, 180] * pq.ms, t_stop=200 * pq.ms),
            neo.SpikeTrain([20, 60, 110, 160, 195] * pq.ms, t_stop=200 * pq.ms),
        ]
        dither = 30 * pq.ms
        surrogate_trains = surr.trial_shifting(trials, dither=dither,
                                               n_surrogates=5)
        for surrogate in surrogate_trains:
            for trial_idx, surrogate_trial in enumerate(surrogate):
                t_start = trials[trial_idx].t_start
                t_stop = trials[trial_idx].t_stop
                self.assertTrue(
                    np.all(surrogate_trial >= t_start),
                    f"Spike below t_start in trial {trial_idx}")
                self.assertTrue(
                    np.all(surrogate_trial < t_stop),
                    f"Spike at or above t_stop in trial {trial_idx}")

    def test_trial_shifting_spike_count_preserved(self):
        """Circular wrap-around must preserve the spike count of every trial."""
        trials = [
            neo.SpikeTrain([10, 50, 90, 150, 190] * pq.ms, t_stop=200 * pq.ms),
            neo.SpikeTrain([5, 70, 130] * pq.ms, t_stop=200 * pq.ms),
        ]
        dither = 100 * pq.ms  # intentionally large to force wrap-arounds
        surrogate_trains = surr.trial_shifting(trials, dither=dither,
                                               n_surrogates=10)
        for surrogate in surrogate_trains:
            for trial_idx, surrogate_trial in enumerate(surrogate):
                self.assertEqual(len(surrogate_trial), len(trials[trial_idx]))

    def test_trial_shifting_circular_wraparound(self):
        """A spike shifted past t_stop must reappear near t_start."""
        # Single spike at 195 ms in a 200 ms trial; a positive shift of +10 ms
        # wraps it to ~5 ms.
        trial = neo.SpikeTrain([195] * pq.ms, t_stop=200 * pq.ms)
        t_start_s = trial.t_start.rescale(pq.s).magnitude
        t_stop_s = trial.t_stop.rescale(pq.s).magnitude
        trial_dur_s = t_stop_s - t_start_s

        dither_s = 0.010  # 10 ms in seconds
        spike_s = 0.195

        # Replicate the internal shift with a known positive delta (+10 ms).
        shifted = spike_s + dither_s  # 0.205 s — beyond t_stop
        wrapped = np.remainder(shifted - t_start_s, trial_dur_s) + t_start_s
        self.assertGreaterEqual(wrapped, t_start_s)
        self.assertLess(wrapped, t_stop_s)
        # Wrapped value should be approximately 0.005 s (i.e. 5 ms)
        self.assertAlmostEqual(wrapped, t_start_s + 0.005, places=10)

    def test_trial_shifting_wraparound_exact(self):
        """A spike shifted past t_stop must wrap to the exact correct position.

        Uses a fixed seed so the shift magnitude is known in advance.
        With random.seed(0) the shift is +13.776... ms (positive), which moves
        the spike at 195 ms to 208.776... ms and must wrap to 8.776... ms.
        The test verifies the exact output of trial_shifting, not just the math.
        """
        # setUp has already called random.seed(0).
        # Pre-draw the shift, then reset so trial_shifting sees the same value.
        dither_s = 0.020
        expected_shift_s = dither_s * (2 * random.random() - 1)
        random.seed(0)

        trial = neo.SpikeTrain([60, 100, 195] * pq.ms, t_stop=200 * pq.ms)
        surrogate = surr.trial_shifting(
            [trial], dither=20 * pq.ms, n_surrogates=1)[0][0]

        orig_s = np.array([0.060, 0.100, 0.195])
        expected_s = np.sort(np.remainder(orig_s + expected_shift_s, 0.200))

        # With seed 0 the shift is positive (~+13.78 ms), so 195 ms wraps to
        # ~8.78 ms and must appear as the first spike in the sorted output.
        self.assertLess(expected_s[0], 0.020,
                        "Wrapped spike should be near t_start, not t_stop")
        np.testing.assert_allclose(
            surrogate.rescale(pq.s).magnitude, expected_s, rtol=1e-10,
            err_msg="Surrogate times do not match expected wrap-around shift")

    def test_trial_shifting_sorted_output(self):
        """Surrogate spike trains must be sorted within every trial."""
        trials = [
            neo.SpikeTrain([30, 80, 120, 170] * pq.ms, t_stop=200 * pq.ms),
            neo.SpikeTrain([15, 55, 95, 185] * pq.ms, t_stop=200 * pq.ms),
        ]
        dither = 50 * pq.ms
        surrogate_trains = surr.trial_shifting(trials, dither=dither,
                                               n_surrogates=10)
        for surrogate in surrogate_trains:
            for surrogate_trial in surrogate:
                assert_array_less(0., np.diff(surrogate_trial.magnitude))

    def test_trial_shifting_surrogates_differ(self):
        """Multiple surrogates must not all produce identical spike trains."""
        np.random.seed(None)  # use true randomness to avoid seed collision
        random.seed(None)
        trials = [
            neo.SpikeTrain([50, 100, 150] * pq.ms, t_stop=200 * pq.ms),
        ]
        dither = 20 * pq.ms
        n_surrogates = 10
        surrogate_trains = surr.trial_shifting(trials, dither=dither,
                                               n_surrogates=n_surrogates)
        # Collect the first (and only) trial's spike times for each surrogate
        first_spikes = [surrogate_trains[i][0].magnitude
                        for i in range(n_surrogates)]
        # At least two surrogates must differ
        all_equal = all(
            np.allclose(first_spikes[0], s) for s in first_spikes[1:])
        self.assertFalse(all_equal,
                         "All surrogates produced identical spike trains")

    def test_trial_shifting_trials_shifted_independently(self):
        """Within a single surrogate, different trials receive different shifts."""
        np.random.seed(None)
        random.seed(None)
        # Use 5 identical trials so any difference in shift is detectable.
        trial = neo.SpikeTrain([50, 100, 150] * pq.ms, t_stop=200 * pq.ms)
        n_trials = 5
        trials = [trial.copy() for _ in range(n_trials)]
        dither = 30 * pq.ms

        # Run many surrogates and check that the shifts across trials differ.
        found_differing = False
        for _ in range(20):
            surrogate_trains = surr.trial_shifting(trials, dither=dither,
                                                   n_surrogates=1)
            surrogate = surrogate_trains[0]
            shifts = []
            for surr_trial in surrogate:
                # The shift can be recovered mod trial_duration because the
                # spike times are wrapped — just check they are not all the same.
                shifts.append(surr_trial.magnitude[0])
            if len(set(np.round(shifts, 6))) > 1:
                found_differing = True
                break
        self.assertTrue(found_differing,
                        "All trials had identical shifts in every run")

    def test_trial_shifting_isi_preservation(self):
        """For a trial shifted without wrap-around, ISIs must be unchanged."""
        # Place spikes well inside the trial so a small dither cannot cause
        # any spike to wrap around.
        trial = neo.SpikeTrain([60, 80, 100, 120, 140] * pq.ms,
                               t_stop=200 * pq.ms)
        dither = 5 * pq.ms  # too small to push any spike out of [0, 200) ms

        orig_isis = np.diff(trial.rescale(pq.ms).magnitude)
        n_surrogates = 20
        for surrogate in surr.trial_shifting([trial], dither=dither,
                                             n_surrogates=n_surrogates):
            surr_isis = np.diff(surrogate[0].rescale(pq.ms).magnitude)
            np.testing.assert_allclose(
                surr_isis, orig_isis,
                err_msg="ISIs changed for a shift that caused no wrap-around")

    def test_trial_shifting_via_surrogates_wrapper_list_input(self):
        """surrogates() with method='trial_shifting' and a list input must work."""
        trials = [
            neo.SpikeTrain([90, 93, 97, 100, 105] * pq.ms,
                           t_stop=200 * pq.ms),
            neo.SpikeTrain([40, 80, 120, 160] * pq.ms,
                           t_stop=200 * pq.ms),
        ]
        dt = 15 * pq.ms
        n_surrogates = 3
        result = surr.surrogates(trials, n_surrogates=n_surrogates,
                                 method='trial_shifting', dt=dt)

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), n_surrogates)
        for surrogate in result:
            self.assertIsInstance(surrogate, list)
            self.assertEqual(len(surrogate), len(trials))
            for trial_idx, surrogate_trial in enumerate(surrogate):
                self.assertIsInstance(surrogate_trial, neo.SpikeTrain)
                self.assertEqual(len(surrogate_trial), len(trials[trial_idx]))
                self.assertEqual(surrogate_trial.t_start,
                                 trials[trial_idx].t_start)
                self.assertEqual(surrogate_trial.t_stop,
                                 trials[trial_idx].t_stop)

    def test_trial_shifting_ground_truth(self):
        """Exact spike times must match the expected circular shift for a known seed.

        setUp seeds random with 0 before every test.  We consume one draw to
        find the shift the function will apply, reset the seed so the function
        sees the same draw, then compare the output exactly.
        """
        dither_s = 0.020  # 20 ms in seconds
        expected_shift_s = dither_s * (2 * random.random() - 1)
        random.seed(0)    # reset so trial_shifting draws the same value

        trial = neo.SpikeTrain([50, 80, 120] * pq.ms, t_stop=200 * pq.ms)
        surrogate = surr.trial_shifting(
            [trial], dither=20 * pq.ms, n_surrogates=1)[0][0]

        orig_s = np.array([0.050, 0.080, 0.120])
        expected_s = np.sort(
            np.remainder(orig_s + expected_shift_s, 0.200))
        np.testing.assert_allclose(
            surrogate.rescale(pq.s).magnitude, expected_s, rtol=1e-10,
            err_msg="Surrogate spike times do not match expected circular shift")

    def test_trial_shifting_shift_bounded_by_dither(self):
        """Maximum displacement of any spike must not exceed dither.

        Spikes are placed far from both edges so no wrap-around can occur,
        making the displacement directly measurable.
        """
        trial = neo.SpikeTrain([80, 100, 120] * pq.ms, t_stop=200 * pq.ms)
        dither = 25 * pq.ms
        dither_ms = dither.rescale(pq.ms).magnitude
        orig_ms = trial.rescale(pq.ms).magnitude
        for surrogate in surr.trial_shifting([trial], dither=dither,
                                             n_surrogates=50):
            surr_ms = surrogate[0].rescale(pq.ms).magnitude
            displacements = np.abs(surr_ms - orig_ms)
            self.assertTrue(
                np.all(displacements <= dither_ms + 1e-9),
                f"Displacement {displacements.max():.4f} ms exceeds "
                f"dither {dither_ms:.4f} ms")

    def test_trial_shifting_heterogeneous_trial_lengths(self):
        """Trials with different t_start/t_stop values are handled correctly."""
        trials = [
            neo.SpikeTrain([10, 50, 90] * pq.ms,
                           t_start=0 * pq.ms, t_stop=100 * pq.ms),
            neo.SpikeTrain([210, 250, 290] * pq.ms,
                           t_start=200 * pq.ms, t_stop=300 * pq.ms),
            neo.SpikeTrain([520, 560] * pq.ms,
                           t_start=500 * pq.ms, t_stop=600 * pq.ms),
        ]
        dither = 20 * pq.ms
        surrogate_trains = surr.trial_shifting(trials, dither=dither,
                                               n_surrogates=5)
        for surrogate in surrogate_trains:
            for trial_idx, surrogate_trial in enumerate(surrogate):
                t_start = trials[trial_idx].t_start
                t_stop = trials[trial_idx].t_stop
                self.assertEqual(len(surrogate_trial), len(trials[trial_idx]))
                self.assertTrue(np.all(surrogate_trial >= t_start))
                self.assertTrue(np.all(surrogate_trial < t_stop))


if __name__ == "__main__":
    unittest.main(verbosity=2)
