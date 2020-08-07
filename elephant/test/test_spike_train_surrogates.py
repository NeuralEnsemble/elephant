# -*- coding: utf-8 -*-
"""
unittests for spike_train_surrogates module.

:copyright: Copyright 2014-2016 by the Elephant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
"""

import unittest
import elephant.spike_train_surrogates as surr
import elephant.spike_train_generation as stg
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_less
import quantities as pq
import neo
import random


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
        print(surrogate_trains)

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

        spiketrain = neo.SpikeTrain(
            [90, 150, 180, 350] * pq.ms, t_stop=500 * pq.ms)
        n_surrogates = 2
        surrogate_trains = surr.surrogates(
            spiketrain,
            dt=3 * pq.ms,
            n_surrogates=n_surrogates,
            method='shuffle_isis',
            edges=False)

        self.assertRaises(ValueError, surr.surrogates, spiketrain, n=1,
                          surr_method='spike_shifting',
                          dt=None, decimals=None, edges=True)
        self.assertTrue(len(surrogate_trains) == n_surrogates)

        n_surrogates2 = 4
        surrogate_trains2 = surr.surrogates(
            spiketrain,
            dt=5 * pq.ms,
            n_surrogates=n_surrogates2,
            method='dither_spike_train',
            edges=True)

        for surrogate_train in surrogate_trains:
            self.assertTrue(isinstance(surrogate_trains[0], neo.SpikeTrain))
            self.assertEqual(surrogate_train.units, spiketrain.units)
            self.assertEqual(surrogate_train.t_start, spiketrain.t_start)
            self.assertEqual(surrogate_train.t_stop, spiketrain.t_stop)
            self.assertEqual(len(surrogate_train), len(spiketrain))
        self.assertTrue(len(surrogate_trains) == n_surrogates)

        for surrogate_train in surrogate_trains2:
            self.assertTrue(isinstance(surrogate_trains2[0], neo.SpikeTrain))
            self.assertEqual(surrogate_train.units, spiketrain.units)
            self.assertEqual(surrogate_train.t_start, spiketrain.t_start)
            self.assertEqual(surrogate_train.t_stop, spiketrain.t_stop)
            self.assertEqual(len(surrogate_train), len(spiketrain))
        self.assertTrue(len(surrogate_trains2) == n_surrogates2)

    def test_joint_isi_dithering_format(self):

        rate = 100. * pq.Hz
        t_stop = 1. * pq.s
        spiketrain = stg.homogeneous_poisson_process(rate, t_stop=t_stop)
        n_surrogates = 2
        dither = 10 * pq.ms

        # Test fast version
        joint_isi_instance = surr.JointISI(spiketrain, dither=dither)
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

        # Test surrogate methods wrapper
        surrogate_trains = surr.surrogates(
            spiketrain,
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


def suite():
    suite = unittest.makeSuite(SurrogatesTestCase, 'test')
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())
