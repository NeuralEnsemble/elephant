# -*- coding: utf-8 -*-
"""
unittests for spike_train_surrogates module.

:copyright: Copyright 2014-2024 by the Elephant team, see `doc/authors.rst`.
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
        self.st1.t_stop = 0.5 * pq.s
        n_surrogates = 2
        dither = 10 * pq.ms
        surrogate_trains = surr.dither_spikes(
            self.st1, dither=dither, n_surrogates=n_surrogates
        )

        self.assertIsInstance(surrogate_trains, list)
        self.assertEqual(len(surrogate_trains), n_surrogates)

        self.assertIsInstance(surrogate_trains[0], neo.SpikeTrain)
        for surrogate_train in surrogate_trains:
            self.assertEqual(surrogate_train.units, self.st1.units)
            self.assertEqual(surrogate_train.t_start, self.st1.t_start)
            self.assertEqual(surrogate_train.t_stop, self.st1.t_stop)
            self.assertEqual(len(surrogate_train), len(self.st1))
            assert_array_less(0.0, np.diff(surrogate_train))  # check ordering

    def test_dither_spikes_empty_train(self):
        st = neo.SpikeTrain([] * pq.ms, t_stop=500 * pq.ms)

        dither = 10 * pq.ms
        surrogate_train = surr.dither_spikes(st, dither=dither, n_surrogates=1)[0]
        self.assertEqual(len(surrogate_train), 0)

    def test_dither_spikes_refactory_period_zero_or_none(self):
        dither = 10 * pq.ms
        decimals = 3
        n_surrogates = 1

        np.random.seed(42)
        surrogate_trains_zero = surr.dither_spikes(
            self.st1,
            dither,
            decimals=decimals,
            n_surrogates=n_surrogates,
            refractory_period=0,
        )
        np.random.seed(42)
        surrogate_trains_none = surr.dither_spikes(
            self.st1,
            dither,
            decimals=decimals,
            n_surrogates=n_surrogates,
            refractory_period=None,
        )
        np.testing.assert_array_almost_equal(
            surrogate_trains_zero[0].magnitude, surrogate_trains_none[0].magnitude
        )

    def test_dither_spikes_output_decimals(self):
        n_surrogates = 2
        dither = 10 * pq.ms
        np.random.seed(42)
        surrogate_trains = surr.dither_spikes(
            self.st1, dither=dither, decimals=3, n_surrogates=n_surrogates
        )

        np.random.seed(42)
        dither_values = np.random.random_sample((n_surrogates, len(self.st1)))
        expected_non_dithered = np.sum(dither_values == 0)

        observed_non_dithered = 0
        for surrogate_train in surrogate_trains:
            for i in range(len(surrogate_train)):
                if (
                    surrogate_train[i] - int(surrogate_train[i]) * pq.ms
                    == surrogate_train[i] - surrogate_train[i]
                ):
                    observed_non_dithered += 1

        self.assertEqual(observed_non_dithered, expected_non_dithered)

    def test_dither_spikes_false_edges(self):
        n_surrogates = 2
        dither = 10 * pq.ms
        surrogate_trains = surr.dither_spikes(
            self.st1, dither=dither, n_surrogates=n_surrogates, edges=False
        )

        for surrogate_train in surrogate_trains:
            for i in range(len(surrogate_train)):
                self.assertLessEqual(surrogate_train[i], self.st1.t_stop)

    def test_dither_spikes_with_refractory_period_output_format(self):
        spiketrain = neo.SpikeTrain(
            [90, 93, 97, 100, 105, 150, 180, 350] * pq.ms, t_stop=0.5 * pq.s
        )
        n_surrogates = 2
        dither = 10 * pq.ms
        surrogate_trains = surr.dither_spikes(
            spiketrain,
            dither=dither,
            n_surrogates=n_surrogates,
            refractory_period=4 * pq.ms,
        )

        self.assertIsInstance(surrogate_trains, list)
        self.assertEqual(len(surrogate_trains), n_surrogates)

        self.assertIsInstance(surrogate_trains[0], neo.SpikeTrain)
        for surrogate_train in surrogate_trains:
            self.assertEqual(surrogate_train.units, spiketrain.units)
            self.assertEqual(surrogate_train.t_start, spiketrain.t_start)
            self.assertEqual(surrogate_train.t_stop, spiketrain.t_stop)
            self.assertEqual(len(surrogate_train), len(spiketrain))
            # Check that refractory period is conserved
            self.assertLessEqual(
                np.min(np.diff(spiketrain)), np.min(np.diff(surrogate_train))
            )
            sigma_displacement = np.std(surrogate_train - spiketrain)
            # Check that spikes are moved
            self.assertLessEqual(dither / 10, sigma_displacement)
            # Spikes are not moved more than dither
            self.assertLessEqual(sigma_displacement, dither)

        self.assertRaises(
            ValueError,
            surr.dither_spikes,
            spiketrain,
            dither=dither,
            refractory_period=3,
        )

    def test_dither_spikes_with_refractory_period_empty_train(self):
        spiketrain = neo.SpikeTrain([] * pq.ms, t_stop=500 * pq.ms)

        dither = 10 * pq.ms
        surrogate_train = surr.dither_spikes(
            spiketrain, dither=dither, n_surrogates=1, refractory_period=4 * pq.ms
        )[0]
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
            rate=10 * pq.Hz, t_stop=t_stop
        ).generate_spiketrain()
        st = neo.SpikeTrain(
            np.hstack([st.magnitude, [1.9999999]]), units=st.units, t_stop=t_stop
        )

        # Dither
        np.random.seed(5)
        surrogate_trains = surr.dither_spikes(
            st, dither=15 * pq.ms, n_surrogates=30, edges=True, decimals=2
        )
        for surrogate in surrogate_trains:
            with self.subTest(surrogate):
                self.assertLess(surrogate[-1], surrogate.t_stop)
                self.assertGreater(surrogate[0], surrogate.t_start)

    def test_randomise_spikes_output_format(self):
        n_surrogates = 2
        surrogate_trains = surr.randomise_spikes(self.st1, n_surrogates=n_surrogates)

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
            self.st1, n_surrogates=n_surrogates, decimals=3
        )

        for surrogate_train in surrogate_trains:
            for i in range(len(surrogate_train)):
                self.assertNotEqual(
                    surrogate_train[i] - int(surrogate_train[i]) * pq.ms,
                    surrogate_train[i] - surrogate_train[i],
                )

    def test_shuffle_isis_output_format(self):
        n_surrogates = 2
        surrogate_trains = surr.shuffle_isis(self.st1, n_surrogates=n_surrogates)

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
        surrogate_train = surr.shuffle_isis(self.st1, n_surrogates=1, decimals=95)[0]

        st_pq = self.st1.view(pq.Quantity)
        surr_pq = surrogate_train.view(pq.Quantity)

        isi0_orig = self.st1[0] - self.st1.t_start
        ISIs_orig = np.sort([isi0_orig] + [isi for isi in np.diff(st_pq)])

        isi0_surr = surrogate_train[0] - surrogate_train.t_start
        ISIs_surr = np.sort([isi0_surr] + [isi for isi in np.diff(surr_pq)])

        self.assertTrue(np.all(ISIs_orig == ISIs_surr))

    def test_shuffle_isis_with_wrongly_ordered_spikes(self):
        surr_method = "shuffle_isis"
        n_surr = 30
        dither = 15 * pq.ms
        spiketrain = neo.SpikeTrain(
            [
                39.65696411,
                98.93868274,
                120.2417674,
                134.70971166,
                154.20788924,
                160.29077989,
                179.19884034,
                212.86773029,
                247.59488061,
                273.04095041,
                297.56437605,
                344.99204215,
                418.55696486,
                460.54298334,
                482.82299125,
                524.236052,
                566.38966742,
                597.87562722,
                651.26965293,
                692.39802855,
                740.90285815,
                849.45874695,
                974.57724848,
                8.79247605,
            ],
            t_start=0.0 * pq.ms,
            t_stop=1000.0 * pq.ms,
            units=pq.ms,
        )
        surr.surrogates(spiketrain, n_surrogates=n_surr, method=surr_method, dt=dither)

    def test_dither_spike_train_output_format(self):
        n_surrogates = 2
        shift = 10 * pq.ms
        surrogate_trains = surr.dither_spike_train(
            self.st1, shift=shift, n_surrogates=n_surrogates
        )

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
            spiketrain, shift=shift, n_surrogates=1
        )[0]
        self.assertEqual(len(surrogate_train), 0)

    def test_dither_spike_train_output_decimals(self):
        n_surrogates = 2
        shift = 10 * pq.ms
        surrogate_trains = surr.dither_spike_train(
            self.st1, shift=shift, n_surrogates=n_surrogates, decimals=3
        )

        for surrogate_train in surrogate_trains:
            for i in range(len(surrogate_train)):
                self.assertNotEqual(
                    surrogate_train[i] - int(surrogate_train[i]) * pq.ms,
                    surrogate_train[i] - surrogate_train[i],
                )

    def test_dither_spike_train_false_edges(self):
        n_surrogates = 2
        shift = 10 * pq.ms
        surrogate_trains = surr.dither_spike_train(
            self.st1, shift=shift, n_surrogates=n_surrogates, edges=False
        )

        for surrogate_train in surrogate_trains:
            for i in range(len(surrogate_train)):
                self.assertLessEqual(surrogate_train[i], self.st1.t_stop)

    def test_jitter_spikes_output_format(self):
        n_surrogates = 2
        bin_size = 100 * pq.ms
        surrogate_trains = surr.jitter_spikes(
            self.st1, bin_size=bin_size, n_surrogates=n_surrogates
        )

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
            spiketrain, bin_size=bin_size, n_surrogates=1
        )[0]
        self.assertEqual(len(surrogate_train), 0)

    def test_jitter_spikes_same_bins(self):
        bin_size = 100 * pq.ms
        surrogate_train = surr.jitter_spikes(
            self.st1, bin_size=bin_size, n_surrogates=1
        )[0]

        bin_ids_orig = np.array(
            (self.st1.view(pq.Quantity) / bin_size).rescale(pq.dimensionless).magnitude,
            dtype=int,
        )
        bin_ids_surr = np.array(
            (surrogate_train.view(pq.Quantity) / bin_size)
            .rescale(pq.dimensionless)
            .magnitude,
            dtype=int,
        )
        self.assertTrue(np.all(bin_ids_orig == bin_ids_surr))

        # Bug encountered when the original and surrogate trains have
        # different number of spikes
        self.assertEqual(len(self.st1), len(surrogate_train))

    def test_jitter_spikes_unequal_bin_size(self):
        spiketrain = neo.SpikeTrain([90, 150, 180, 480] * pq.ms, t_stop=500 * pq.ms)

        bin_size = 75 * pq.ms
        surrogate_train = surr.jitter_spikes(
            spiketrain, bin_size=bin_size, n_surrogates=1
        )[0]

        bin_ids_orig = np.array(
            (spiketrain.view(pq.Quantity) / bin_size)
            .rescale(pq.dimensionless)
            .magnitude,
            dtype=int,
        )
        bin_ids_surr = np.array(
            (surrogate_train.view(pq.Quantity) / bin_size)
            .rescale(pq.dimensionless)
            .magnitude,
            dtype=int,
        )

        self.assertTrue(np.all(bin_ids_orig == bin_ids_surr))

    def test_surr_method(self):
        surr_methods = (
            "dither_spike_train",
            "dither_spikes",
            "jitter_spikes",
            "randomise_spikes",
            "shuffle_isis",
            "joint_isi_dithering",
            "dither_spikes_with_refractory_period",
            "trial_shifting",
            "bin_shuffling",
            "isi_dithering",
        )

        surr_method_kwargs = {
            "dither_spikes": {},
            "dither_spikes_with_refractory_period": {"refractory_period": 3 * pq.ms},
            "randomise_spikes": {},
            "shuffle_isis": {},
            "dither_spike_train": {},
            "jitter_spikes": {},
            "bin_shuffling": {"bin_size": 3 * pq.ms},
            "joint_isi_dithering": {},
            "isi_dithering": {},
            "trial_shifting": {
                "trial_length": 200 * pq.ms,
                "trial_separation": 50 * pq.ms,
            },
        }

        dt = 15 * pq.ms
        spiketrain = neo.SpikeTrain([90, 150, 180, 350] * pq.ms, t_stop=500 * pq.ms)
        n_surrogates = 3
        for method in surr_methods:
            surrogates = surr.surrogates(
                spiketrain,
                dt=dt,
                n_surrogates=n_surrogates,
                method=method,
                **surr_method_kwargs[method],
            )
            self.assertTrue(len(surrogates) == n_surrogates)

            for surrogate_train in surrogates:
                self.assertTrue(isinstance(surrogates[0], neo.SpikeTrain))
                self.assertEqual(surrogate_train.units, spiketrain.units)
                self.assertEqual(surrogate_train.t_start, spiketrain.t_start)
                self.assertEqual(surrogate_train.t_stop, spiketrain.t_stop)
                self.assertEqual(len(surrogate_train), len(spiketrain))
            self.assertTrue(len(surrogates) == n_surrogates)

        self.assertRaises(
            ValueError,
            surr.surrogates,
            spiketrain,
            n_surrogates=1,
            method="spike_shifting",
            dt=None,
            decimals=None,
            edges=True,
        )

        self.assertRaises(
            ValueError, surr.surrogates, spiketrain, method="dither_spikes", dt=None
        )

        self.assertRaises(
            TypeError,
            surr.surrogates,
            spiketrain.magnitude,
            method="dither_spikes",
            dt=10 * pq.ms,
        )

    def test_joint_isi_dithering_format(self):
        rate = 100.0 * pq.Hz
        t_stop = 1.0 * pq.s
        process = stg.StationaryPoissonProcess(rate, t_stop=t_stop)
        spiketrain = process.generate_spiketrain()
        n_surrogates = 2
        dither = 10 * pq.ms

        # Test fast version
        joint_isi_instance = surr.JointISI(spiketrain, dither=dither, method="fast")
        surrogate_trains = joint_isi_instance.dithering(n_surrogates=n_surrogates)

        self.assertIsInstance(surrogate_trains, list)
        self.assertEqual(len(surrogate_trains), n_surrogates)
        self.assertEqual(joint_isi_instance.method, "fast")

        for surrogate_train in surrogate_trains:
            self.assertIsInstance(surrogate_train, neo.SpikeTrain)
            self.assertEqual(surrogate_train.units, spiketrain.units)
            self.assertEqual(surrogate_train.t_start, spiketrain.t_start)
            self.assertEqual(surrogate_train.t_stop, spiketrain.t_stop)
            self.assertEqual(len(surrogate_train), len(spiketrain))

        # Test window_version
        joint_isi_instance = surr.JointISI(
            spiketrain, method="window", dither=2 * dither, n_bins=50
        )
        surrogate_trains = joint_isi_instance.dithering(n_surrogates=n_surrogates)

        self.assertIsInstance(surrogate_trains, list)
        self.assertEqual(len(surrogate_trains), n_surrogates)
        self.assertEqual(joint_isi_instance.method, "window")

        for surrogate_train in surrogate_trains:
            self.assertIsInstance(surrogate_train, neo.SpikeTrain)
            self.assertEqual(surrogate_train.units, spiketrain.units)
            self.assertEqual(surrogate_train.t_start, spiketrain.t_start)
            self.assertEqual(surrogate_train.t_stop, spiketrain.t_stop)
            self.assertEqual(len(surrogate_train), len(spiketrain))

        # Test isi_dithering
        joint_isi_instance = surr.JointISI(
            spiketrain,
            method="window",
            dither=2 * dither,
            n_bins=50,
            isi_dithering=True,
            use_sqrt=True,
            cutoff=False,
        )
        surrogate_trains = joint_isi_instance.dithering(n_surrogates=n_surrogates)

        self.assertIsInstance(surrogate_trains, list)
        self.assertEqual(len(surrogate_trains), n_surrogates)
        self.assertEqual(joint_isi_instance.method, "window")

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
            method="joint_isi_dithering",
        )
        self.assertIsInstance(surrogate_trains, list)
        self.assertEqual(len(surrogate_trains), n_surrogates)

        for surrogate_train in surrogate_trains:
            self.assertIsInstance(surrogate_train, neo.SpikeTrain)
            self.assertEqual(surrogate_train.units, spiketrain.units)
            self.assertEqual(surrogate_train.t_start, spiketrain.t_start)
            self.assertEqual(surrogate_train.t_stop, spiketrain.t_stop)
            self.assertEqual(len(surrogate_train), len(spiketrain))
        with self.assertRaises(ValueError):
            joint_isi_instance = surr.JointISI(
                spiketrain, method="wrong method", dither=2 * dither, n_bins=50
            )

    def test_joint_isi_dithering_empty_train(self):
        spiketrain = neo.SpikeTrain([] * pq.ms, t_stop=500 * pq.ms)
        surrogate_train = surr.JointISI(spiketrain).dithering()[0]
        self.assertEqual(len(surrogate_train), 0)

    def test_joint_isi_dithering_output(self):
        process = stg.StationaryPoissonProcess(
            rate=100.0 * pq.Hz, refractory_period=3 * pq.ms, t_stop=0.1 * pq.s
        )
        spiketrain = process.generate_spiketrain()
        surrogate_train = surr.JointISI(spiketrain).dithering()[0]
        ground_truth = [
            0.0060744,
            0.01886591,
            0.02732847,
            0.03683888,
            0.04569622,
            0.05196334,
            0.05899197,
            0.07855664,
        ]
        assert_array_almost_equal(surrogate_train.magnitude, ground_truth)

    def test_joint_isi_with_wrongly_ordered_spikes(self):
        surr_method = "joint_isi_dithering"
        n_surr = 30
        dither = 15 * pq.ms
        spiketrain = neo.SpikeTrain(
            [
                39.65696411,
                98.93868274,
                120.2417674,
                134.70971166,
                154.20788924,
                160.29077989,
                179.19884034,
                212.86773029,
                247.59488061,
                273.04095041,
                297.56437605,
                344.99204215,
                418.55696486,
                460.54298334,
                482.82299125,
                524.236052,
                566.38966742,
                597.87562722,
                651.26965293,
                692.39802855,
                740.90285815,
                849.45874695,
                974.57724848,
                8.79247605,
            ],
            t_start=0.0 * pq.ms,
            t_stop=1000.0 * pq.ms,
            units=pq.ms,
        )
        surr.surrogates(spiketrain, n_surrogates=n_surr, method=surr_method, dt=dither)

    def test_joint_isi_spikes_at_border(self):
        surr_method = "joint_isi_dithering"
        n_surr = 30
        dither = 15 * pq.ms
        spiketrain = neo.SpikeTrain(
            [
                4.0,
                28.0,
                45.0,
                51.0,
                83.0,
                87.0,
                96.0,
                111.0,
                126.0,
                131.0,
                138.0,
                150.0,
                209.0,
                232.0,
                253.0,
                275.0,
                279.0,
                303.0,
                320.0,
                371.0,
                396.0,
                401.0,
                429.0,
                447.0,
                479.0,
                511.0,
                535.0,
                549.0,
                581.0,
                585.0,
                605.0,
                607.0,
                626.0,
                630.0,
                644.0,
                714.0,
                832.0,
                835.0,
                853.0,
                858.0,
                878.0,
                905.0,
                909.0,
                932.0,
                950.0,
                961.0,
                999.0,
                1000.0,
            ],
            t_start=0.0 * pq.ms,
            t_stop=1000.0 * pq.ms,
            units=pq.ms,
        )
        surr.surrogates(spiketrain, n_surrogates=n_surr, method=surr_method, dt=dither)

    def test_bin_shuffling_output_format(self):
        self.bin_size = 3 * pq.ms
        self.max_displacement = 10
        spiketrain = neo.SpikeTrain(
            [90, 93, 97, 100, 105, 150, 180, 350] * pq.ms, t_stop=0.5 * pq.s
        )
        binned_spiketrain = conv.BinnedSpikeTrain(spiketrain, self.bin_size)
        n_surrogates = 2

        for sliding in (True, False):
            surrogate_trains = surr.bin_shuffling(
                binned_spiketrain,
                max_displacement=self.max_displacement,
                n_surrogates=n_surrogates,
                sliding=sliding,
            )

            self.assertIsInstance(surrogate_trains, list)
            self.assertEqual(len(surrogate_trains), n_surrogates)

            self.assertIsInstance(surrogate_trains[0], conv.BinnedSpikeTrain)
            for surrogate_train in surrogate_trains:
                self.assertEqual(surrogate_train.t_start, binned_spiketrain.t_start)
                self.assertEqual(surrogate_train.t_stop, binned_spiketrain.t_stop)
                self.assertEqual(surrogate_train.n_bins, binned_spiketrain.n_bins)
                self.assertEqual(surrogate_train.bin_size, binned_spiketrain.bin_size)

    def test_bin_shuffling_empty_train(self):
        self.bin_size = 3 * pq.ms
        self.max_displacement = 10
        empty_spiketrain = neo.SpikeTrain([] * pq.ms, t_stop=500 * pq.ms)

        binned_spiketrain = conv.BinnedSpikeTrain(empty_spiketrain, self.bin_size)
        surrogate_train = surr.bin_shuffling(
            binned_spiketrain, max_displacement=self.max_displacement, n_surrogates=1
        )[0]
        self.assertEqual(np.sum(surrogate_train.to_bool_array()), 0)

    def test_trial_shuffling_output_format(self):
        spiketrain = [
            neo.SpikeTrain(
                [90, 93, 97, 100, 105, 150, 180, 190] * pq.ms, t_stop=0.2 * pq.s
            ),
            neo.SpikeTrain(
                [90, 93, 97, 100, 105, 150, 180, 190] * pq.ms, t_stop=0.2 * pq.s
            ),
        ]
        # trial_length = 200 * pq.ms
        # trial_separation = 50 * pq.ms
        n_surrogates = 2
        dither = 10 * pq.ms
        surrogate_trains = surr.trial_shifting(
            spiketrain, dither=dither, n_surrogates=n_surrogates
        )

        self.assertIsInstance(surrogate_trains, list)
        self.assertEqual(len(surrogate_trains), n_surrogates)

        self.assertIsInstance(surrogate_trains[0], list)
        self.assertIsInstance(surrogate_trains[0][0], neo.SpikeTrain)
        for surrogate_train in surrogate_trains[0]:
            self.assertEqual(surrogate_train.units, spiketrain[0].units)
            self.assertEqual(surrogate_train.t_start, spiketrain[0].t_start)
            self.assertEqual(surrogate_train.t_stop, spiketrain[0].t_stop)
            self.assertEqual(len(surrogate_train), len(spiketrain[0]))
            assert_array_less(0.0, np.diff(surrogate_train))  # check ordering

    def test_trial_shuffling_empty_train(self):
        empty_spiketrain = [
            neo.SpikeTrain([] * pq.ms, t_stop=500 * pq.ms),
            neo.SpikeTrain([] * pq.ms, t_stop=500 * pq.ms),
        ]

        dither = 10 * pq.ms
        surrogate_train = surr.trial_shifting(
            empty_spiketrain, dither=dither, n_surrogates=1
        )[0]

        self.assertEqual(len(surrogate_train), 2)
        self.assertEqual(len(surrogate_train[0]), 0)

    def test_trial_shuffling_output_format_concatenated(self):
        spiketrain = neo.SpikeTrain(
            [90, 93, 97, 100, 105, 150, 180, 350] * pq.ms, t_stop=0.5 * pq.s
        )
        trial_length = 200 * pq.ms
        trial_separation = 50 * pq.ms
        n_surrogates = 2
        dither = 10 * pq.ms
        surrogate_trains = surr._trial_shifting_of_concatenated_spiketrain(
            spiketrain,
            dither=dither,
            n_surrogates=n_surrogates,
            trial_length=trial_length,
            trial_separation=trial_separation,
        )

        self.assertIsInstance(surrogate_trains, list)
        self.assertEqual(len(surrogate_trains), n_surrogates)

        self.assertIsInstance(surrogate_trains[0], neo.SpikeTrain)
        for surrogate_train in surrogate_trains:
            self.assertEqual(surrogate_train.units, spiketrain.units)
            self.assertEqual(surrogate_train.t_start, spiketrain.t_start)
            self.assertEqual(surrogate_train.t_stop, spiketrain.t_stop)
            self.assertEqual(len(surrogate_train), len(spiketrain))
            assert_array_less(0.0, np.diff(surrogate_train))  # check ordering

    def test_trial_shuffling_empty_train_concatenated(self):
        empty_spiketrain = neo.SpikeTrain([] * pq.ms, t_stop=500 * pq.ms)
        trial_length = 200 * pq.ms
        trial_separation = 50 * pq.ms

        dither = 10 * pq.ms
        surrogate_train = surr._trial_shifting_of_concatenated_spiketrain(
            empty_spiketrain,
            dither=dither,
            n_surrogates=1,
            trial_length=trial_length,
            trial_separation=trial_separation,
        )[0]
        self.assertEqual(len(surrogate_train), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
