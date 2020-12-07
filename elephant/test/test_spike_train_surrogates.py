# -*- coding: utf-8 -*-
"""
unittests for spike_train_surrogates module.

:copyright: Copyright 2014-2020 by the Elephant team, see `doc/authors.rst`.
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

    def test_dither_spikes_output_format(self):

        spiketrain = neo.SpikeTrain([90, 93, 97, 100, 105,
                                     150, 180, 350] * pq.ms, t_stop=.5 * pq.s)
        spiketrain.t_stop = .5 * pq.s
        n_surrogates = 2
        dither = 10 * pq.ms
        surrogate_trains = surr.dither_spikes(
            spiketrain, dither=dither, n_surrogates=n_surrogates)

        self.assertIsInstance(surrogate_trains, list)
        self.assertEqual(len(surrogate_trains), n_surrogates)

        self.assertIsInstance(surrogate_trains[0], neo.SpikeTrain)
        for surrogate_train in surrogate_trains:
            self.assertEqual(surrogate_train.units, spiketrain.units)
            self.assertEqual(surrogate_train.t_start, spiketrain.t_start)
            self.assertEqual(surrogate_train.t_stop, spiketrain.t_stop)
            self.assertEqual(len(surrogate_train), len(spiketrain))
            assert_array_less(0., np.diff(surrogate_train))  # check ordering

    def test_dither_spikes_empty_train(self):

        st = neo.SpikeTrain([] * pq.ms, t_stop=500 * pq.ms)

        dither = 10 * pq.ms
        surrogate_train = surr.dither_spikes(
            st, dither=dither, n_surrogates=1)[0]
        self.assertEqual(len(surrogate_train), 0)

    def test_dither_spikes_output_decimals(self):

        st = neo.SpikeTrain([90, 150, 180, 350] * pq.ms, t_stop=500 * pq.ms)

        n_surrogates = 2
        dither = 10 * pq.ms
        np.random.seed(42)
        surrogate_trains = surr.dither_spikes(
            st, dither=dither, decimals=3, n_surrogates=n_surrogates)

        np.random.seed(42)
        dither_values = np.random.random_sample((n_surrogates, len(st)))
        expected_non_dithered = np.sum(dither_values == 0)

        observed_non_dithered = 0
        for surrogate_train in surrogate_trains:
            for i in range(len(surrogate_train)):
                if surrogate_train[i] - int(surrogate_train[i]) * \
                        pq.ms == surrogate_train[i] - surrogate_train[i]:
                    observed_non_dithered += 1

        self.assertEqual(observed_non_dithered, expected_non_dithered)

    def test_dither_spikes_false_edges(self):

        st = neo.SpikeTrain([90, 150, 180, 350] * pq.ms, t_stop=500 * pq.ms)

        n_surrogates = 2
        dither = 10 * pq.ms
        surrogate_trains = surr.dither_spikes(
            st, dither=dither, n_surrogates=n_surrogates, edges=False)

        for surrogate_train in surrogate_trains:
            for i in range(len(surrogate_train)):
                self.assertLessEqual(surrogate_train[i], st.t_stop)

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

    def test_randomise_spikes_output_format(self):

        spiketrain = neo.SpikeTrain(
            [90, 150, 180, 350] * pq.ms, t_stop=500 * pq.ms)

        n_surrogates = 2
        surrogate_trains = surr.randomise_spikes(
            spiketrain, n_surrogates=n_surrogates)

        self.assertIsInstance(surrogate_trains, list)
        self.assertEqual(len(surrogate_trains), n_surrogates)

        self.assertIsInstance(surrogate_trains[0], neo.SpikeTrain)
        for surrogate_train in surrogate_trains:
            self.assertEqual(surrogate_train.units, spiketrain.units)
            self.assertEqual(surrogate_train.t_start, spiketrain.t_start)
            self.assertEqual(surrogate_train.t_stop, spiketrain.t_stop)
            self.assertEqual(len(surrogate_train), len(spiketrain))

    def test_randomise_spikes_empty_train(self):

        spiketrain = neo.SpikeTrain([] * pq.ms, t_stop=500 * pq.ms)

        surrogate_train = surr.randomise_spikes(spiketrain, n_surrogates=1)[0]
        self.assertEqual(len(surrogate_train), 0)

    def test_randomise_spikes_output_decimals(self):
        spiketrain = neo.SpikeTrain(
            [90, 150, 180, 350] * pq.ms, t_stop=500 * pq.ms)

        n_surrogates = 2
        surrogate_trains = surr.randomise_spikes(
            spiketrain, n_surrogates=n_surrogates, decimals=3)

        for surrogate_train in surrogate_trains:
            for i in range(len(surrogate_train)):
                self.assertNotEqual(surrogate_train[i] -
                                    int(surrogate_train[i]) *
                                    pq.ms, surrogate_train[i] -
                                    surrogate_train[i])

    def test_shuffle_isis_output_format(self):

        spiketrain = neo.SpikeTrain(
            [90, 150, 180, 350] * pq.ms, t_stop=500 * pq.ms)

        n_surrogates = 2
        surrogate_trains = surr.shuffle_isis(
            spiketrain, n_surrogates=n_surrogates)

        self.assertIsInstance(surrogate_trains, list)
        self.assertEqual(len(surrogate_trains), n_surrogates)

        self.assertIsInstance(surrogate_trains[0], neo.SpikeTrain)
        for surrogate_train in surrogate_trains:
            self.assertEqual(surrogate_train.units, spiketrain.units)
            self.assertEqual(surrogate_train.t_start, spiketrain.t_start)
            self.assertEqual(surrogate_train.t_stop, spiketrain.t_stop)
            self.assertEqual(len(surrogate_train), len(spiketrain))

    def test_shuffle_isis_empty_train(self):

        spiketrain = neo.SpikeTrain([] * pq.ms, t_stop=500 * pq.ms)

        surrogate_train = surr.shuffle_isis(spiketrain, n_surrogates=1)[0]
        self.assertEqual(len(surrogate_train), 0)

    def test_shuffle_isis_same_isis(self):

        spiketrain = neo.SpikeTrain(
            [90, 150, 180, 350] * pq.ms, t_stop=500 * pq.ms)

        surrogate_train = surr.shuffle_isis(spiketrain, n_surrogates=1)[0]

        st_pq = spiketrain.view(pq.Quantity)
        surr_pq = surrogate_train.view(pq.Quantity)

        isi0_orig = spiketrain[0] - spiketrain.t_start
        ISIs_orig = np.sort([isi0_orig] + [isi for isi in np.diff(st_pq)])

        isi0_surr = surrogate_train[0] - surrogate_train.t_start
        ISIs_surr = np.sort([isi0_surr] + [isi for isi in np.diff(surr_pq)])

        self.assertTrue(np.all(ISIs_orig == ISIs_surr))

    def test_shuffle_isis_output_decimals(self):

        spiketrain = neo.SpikeTrain(
            [90, 150, 180, 350] * pq.ms, t_stop=500 * pq.ms)

        surrogate_train = surr.shuffle_isis(
            spiketrain, n_surrogates=1, decimals=95)[0]

        st_pq = spiketrain.view(pq.Quantity)
        surr_pq = surrogate_train.view(pq.Quantity)

        isi0_orig = spiketrain[0] - spiketrain.t_start
        ISIs_orig = np.sort([isi0_orig] + [isi for isi in np.diff(st_pq)])

        isi0_surr = surrogate_train[0] - surrogate_train.t_start
        ISIs_surr = np.sort([isi0_surr] + [isi for isi in np.diff(surr_pq)])

        self.assertTrue(np.all(ISIs_orig == ISIs_surr))

    def test_shuffle_isis_with_wrongly_ordered_spikes(self):
        surr_method = 'shuffle_isis'
        n_surr = 30
        dither = 15 * pq.ms
        spiketrain = neo.SpikeTrain(
            [39.65696411,  98.93868274, 120.2417674,  134.70971166,
             154.20788924,
             160.29077989, 179.19884034, 212.86773029, 247.59488061,
             273.04095041,
             297.56437605, 344.99204215, 418.55696486, 460.54298334,
             482.82299125,
             524.236052,   566.38966742, 597.87562722, 651.26965293,
             692.39802855,
             740.90285815, 849.45874695, 974.57724848,   8.79247605],
            t_start=0.*pq.ms, t_stop=1000.*pq.ms, units=pq.ms)
        surr.surrogates(spiketrain, n_surrogates=n_surr, method=surr_method,
                        dt=dither)

    def test_dither_spike_train_output_format(self):

        spiketrain = neo.SpikeTrain(
            [90, 150, 180, 350] * pq.ms, t_stop=500 * pq.ms)

        n_surrogates = 2
        shift = 10 * pq.ms
        surrogate_trains = surr.dither_spike_train(
            spiketrain, shift=shift, n_surrogates=n_surrogates)

        self.assertIsInstance(surrogate_trains, list)
        self.assertEqual(len(surrogate_trains), n_surrogates)

        self.assertIsInstance(surrogate_trains[0], neo.SpikeTrain)
        for surrogate_train in surrogate_trains:
            self.assertEqual(surrogate_train.units, spiketrain.units)
            self.assertEqual(surrogate_train.t_start, spiketrain.t_start)
            self.assertEqual(surrogate_train.t_stop, spiketrain.t_stop)
            self.assertEqual(len(surrogate_train), len(spiketrain))

    def test_dither_spike_train_empty_train(self):

        spiketrain = neo.SpikeTrain([] * pq.ms, t_stop=500 * pq.ms)

        shift = 10 * pq.ms
        surrogate_train = surr.dither_spike_train(
            spiketrain, shift=shift, n_surrogates=1)[0]
        self.assertEqual(len(surrogate_train), 0)

    def test_dither_spike_train_output_decimals(self):
        st = neo.SpikeTrain([90, 150, 180, 350] * pq.ms, t_stop=500 * pq.ms)

        n_surrogates = 2
        shift = 10 * pq.ms
        surrogate_trains = surr.dither_spike_train(
            st, shift=shift, n_surrogates=n_surrogates, decimals=3)

        for surrogate_train in surrogate_trains:
            for i in range(len(surrogate_train)):
                self.assertNotEqual(surrogate_train[i] -
                                    int(surrogate_train[i]) *
                                    pq.ms, surrogate_train[i] -
                                    surrogate_train[i])

    def test_dither_spike_train_false_edges(self):

        spiketrain = neo.SpikeTrain(
            [90, 150, 180, 350] * pq.ms, t_stop=500 * pq.ms)

        n_surrogates = 2
        shift = 10 * pq.ms
        surrogate_trains = surr.dither_spike_train(
            spiketrain, shift=shift, n_surrogates=n_surrogates, edges=False)

        for surrogate_train in surrogate_trains:
            for i in range(len(surrogate_train)):
                self.assertLessEqual(surrogate_train[i], spiketrain.t_stop)

    def test_jitter_spikes_output_format(self):

        spiketrain = neo.SpikeTrain(
            [90, 150, 180, 350] * pq.ms, t_stop=500 * pq.ms)

        n_surrogates = 2
        bin_size = 100 * pq.ms
        surrogate_trains = surr.jitter_spikes(
            spiketrain, bin_size=bin_size, n_surrogates=n_surrogates)

        self.assertIsInstance(surrogate_trains, list)
        self.assertEqual(len(surrogate_trains), n_surrogates)

        self.assertIsInstance(surrogate_trains[0], neo.SpikeTrain)
        for surrogate_train in surrogate_trains:
            self.assertEqual(surrogate_train.units, spiketrain.units)
            self.assertEqual(surrogate_train.t_start, spiketrain.t_start)
            self.assertEqual(surrogate_train.t_stop, spiketrain.t_stop)
            self.assertEqual(len(surrogate_train), len(spiketrain))

    def test_jitter_spikes_empty_train(self):

        spiketrain = neo.SpikeTrain([] * pq.ms, t_stop=500 * pq.ms)

        bin_size = 75 * pq.ms
        surrogate_train = surr.jitter_spikes(
            spiketrain, bin_size=bin_size, n_surrogates=1)[0]
        self.assertEqual(len(surrogate_train), 0)

    def test_jitter_spikes_same_bins(self):

        spiketrain = neo.SpikeTrain(
            [90, 150, 180, 350] * pq.ms, t_stop=500 * pq.ms)

        bin_size = 100 * pq.ms
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

        # Bug encountered when the original and surrogate trains have
        # different number of spikes
        self.assertEqual(len(spiketrain), len(surrogate_train))

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
                                                      3*pq.ms},
             'randomise_spikes': {},
             'shuffle_isis': {},
             'dither_spike_train': {},
             'jitter_spikes': {},
             'bin_shuffling': {'bin_size': 3*pq.ms},
             'joint_isi_dithering': {},
             'isi_dithering': {},
             'trial_shifting': {'trial_length': 200*pq.ms,
                                'trial_separation': 50*pq.ms}}

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
                          method='dither_spikes', dt=10*pq.ms)

    def test_joint_isi_dithering_format(self):

        rate = 100. * pq.Hz
        t_stop = 1. * pq.s
        spiketrain = stg.homogeneous_poisson_process(rate, t_stop=t_stop)
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
            dt=15*pq.ms,
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
        spiketrain = stg.homogeneous_poisson_process(
            rate=100. * pq.Hz,
            refractory_period=3 * pq.ms,
            t_stop=0.1 * pq.s)
        surrogate_train = surr.JointISI(spiketrain).dithering()[0]
        ground_truth = [0.005571, 0.018363, 0.026825, 0.036336, 0.045193,
                        0.05146, 0.058489, 0.078053]
        assert_array_almost_equal(surrogate_train.magnitude, ground_truth)

    def test_joint_isi_with_wrongly_ordered_spikes(self):
        surr_method = 'joint_isi_dithering'
        n_surr = 30
        dither = 15 * pq.ms
        spiketrain = neo.SpikeTrain(
            [39.65696411,  98.93868274, 120.2417674,  134.70971166,
             154.20788924,
             160.29077989, 179.19884034, 212.86773029, 247.59488061,
             273.04095041,
             297.56437605, 344.99204215, 418.55696486, 460.54298334,
             482.82299125,
             524.236052,   566.38966742, 597.87562722, 651.26965293,
             692.39802855,
             740.90285815, 849.45874695, 974.57724848,   8.79247605],
            t_start=0.*pq.ms, t_stop=1000.*pq.ms, units=pq.ms)
        surr.surrogates(spiketrain, n_surrogates=n_surr, method=surr_method,
                        dt=dither)

    def test_joint_isi_spikes_at_border(self):
        surr_method = 'joint_isi_dithering'
        n_surr = 30
        dither = 15 * pq.ms
        spiketrain = neo.SpikeTrain(
            [4.,   28.,   45.,  51.,   83.,   87.,   96., 111., 126.,  131.,
             138.,  150.,
             209.,  232.,  253.,  275.,  279.,  303.,  320.,  371.,  396.,
             401.,  429.,  447.,
             479.,  511.,  535.,  549.,  581.,  585.,  605.,  607.,  626.,
             630.,  644.,  714.,
             832.,  835.,  853.,  858.,  878.,  905.,  909.,  932.,  950.,
             961.,  999.,  1000.],
            t_start=0.*pq.ms, t_stop=1000.*pq.ms, units=pq.ms)
        surr.surrogates(
            spiketrain, n_surrogates=n_surr, method=surr_method, dt=dither)

    def test_bin_shuffling_output_format(self):

        self.bin_size = 3*pq.ms
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

    def test_trial_shuffling_output_format(self):
        spiketrain = \
            [neo.SpikeTrain([90, 93, 97, 100, 105, 150, 180, 190] * pq.ms,
                            t_stop=.2 * pq.s),
             neo.SpikeTrain([90, 93, 97, 100, 105, 150, 180, 190] * pq.ms,
                            t_stop=.2 * pq.s)]
        # trial_length = 200 * pq.ms
        # trial_separation = 50 * pq.ms
        n_surrogates = 2
        dither = 10 * pq.ms
        surrogate_trains = surr.trial_shifting(
            spiketrain, dither=dither, n_surrogates=n_surrogates)

        self.assertIsInstance(surrogate_trains, list)
        self.assertEqual(len(surrogate_trains), n_surrogates)

        self.assertIsInstance(surrogate_trains[0], list)
        self.assertIsInstance(surrogate_trains[0][0], neo.SpikeTrain)
        for surrogate_train in surrogate_trains[0]:
            self.assertEqual(surrogate_train.units, spiketrain[0].units)
            self.assertEqual(surrogate_train.t_start, spiketrain[0].t_start)
            self.assertEqual(surrogate_train.t_stop, spiketrain[0].t_stop)
            self.assertEqual(len(surrogate_train), len(spiketrain[0]))
            assert_array_less(0., np.diff(surrogate_train))  # check ordering

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


def suite():
    suite = unittest.makeSuite(SurrogatesTestCase, 'test')
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())
