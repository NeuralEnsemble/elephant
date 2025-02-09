# -*- coding: utf-8 -*-
"""
Unit tests for the phase analysis module.

:copyright: Copyright 2014-2024 by the Elephant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
"""

from __future__ import division, print_function

import unittest

import numpy as np
import quantities as pq
import scipy.io
from neo import SpikeTrain, AnalogSignal
from numpy.ma.testutils import assert_allclose

import elephant.phase_analysis
from elephant.datasets import download_datasets


class SpikeTriggeredPhaseTestCase(unittest.TestCase):
    def setUp(self):
        tlen0 = 100 * pq.s
        f0 = 20.0 * pq.Hz
        fs0 = 1 * pq.ms
        t0 = (
            np.arange(0, tlen0.rescale(pq.s).magnitude, fs0.rescale(pq.s).magnitude)
            * pq.s
        )
        self.anasig0 = AnalogSignal(
            np.sin(2 * np.pi * (f0 * t0).simplified.magnitude),
            units=pq.mV,
            t_start=0 * pq.ms,
            sampling_period=fs0,
        )
        self.st0 = SpikeTrain(
            np.arange(50, tlen0.rescale(pq.ms).magnitude - 50, 50) * pq.ms,
            t_start=0 * pq.ms,
            t_stop=tlen0,
        )
        self.st1 = SpikeTrain(
            [100.0, 100.1, 100.2, 100.3, 100.9, 101.0] * pq.ms,
            t_start=0 * pq.ms,
            t_stop=tlen0,
        )

    def test_perfect_locking_one_spiketrain_one_signal(self):
        phases, amps, times = elephant.phase_analysis.spike_triggered_phase(
            elephant.signal_processing.hilbert(self.anasig0), self.st0, interpolate=True
        )

        assert_allclose(phases[0], -np.pi / 2.0)
        assert_allclose(amps[0], 1, atol=0.1)
        assert_allclose(times[0].magnitude, self.st0.magnitude)
        self.assertEqual(len(phases[0]), len(self.st0))
        self.assertEqual(len(amps[0]), len(self.st0))
        self.assertEqual(len(times[0]), len(self.st0))

    def test_perfect_locking_many_spiketrains_many_signals(self):
        phases, amps, times = elephant.phase_analysis.spike_triggered_phase(
            [
                elephant.signal_processing.hilbert(self.anasig0),
                elephant.signal_processing.hilbert(self.anasig0),
            ],
            [self.st0, self.st0],
            interpolate=True,
        )

        assert_allclose(phases[0], -np.pi / 2.0)
        assert_allclose(amps[0], 1, atol=0.1)
        assert_allclose(times[0].magnitude, self.st0.magnitude)
        self.assertEqual(len(phases[0]), len(self.st0))
        self.assertEqual(len(amps[0]), len(self.st0))
        self.assertEqual(len(times[0]), len(self.st0))

    def test_perfect_locking_one_spiketrains_many_signals(self):
        phases, amps, times = elephant.phase_analysis.spike_triggered_phase(
            [
                elephant.signal_processing.hilbert(self.anasig0),
                elephant.signal_processing.hilbert(self.anasig0),
            ],
            [self.st0],
            interpolate=True,
        )

        assert_allclose(phases[0], -np.pi / 2.0)
        assert_allclose(amps[0], 1, atol=0.1)
        assert_allclose(times[0].magnitude, self.st0.magnitude)
        self.assertEqual(len(phases[0]), len(self.st0))
        self.assertEqual(len(amps[0]), len(self.st0))
        self.assertEqual(len(times[0]), len(self.st0))

    def test_perfect_locking_many_spiketrains_one_signal(self):
        phases, amps, times = elephant.phase_analysis.spike_triggered_phase(
            elephant.signal_processing.hilbert(self.anasig0),
            [self.st0, self.st0],
            interpolate=True,
        )

        assert_allclose(phases[0], -np.pi / 2.0)
        assert_allclose(amps[0], 1, atol=0.1)
        assert_allclose(times[0].magnitude, self.st0.magnitude)
        self.assertEqual(len(phases[0]), len(self.st0))
        self.assertEqual(len(amps[0]), len(self.st0))
        self.assertEqual(len(times[0]), len(self.st0))

    def test_interpolate(self):
        phases_int, _, _ = elephant.phase_analysis.spike_triggered_phase(
            elephant.signal_processing.hilbert(self.anasig0), self.st1, interpolate=True
        )

        self.assertLess(phases_int[0][0], phases_int[0][1])
        self.assertLess(phases_int[0][1], phases_int[0][2])
        self.assertLess(phases_int[0][2], phases_int[0][3])
        self.assertLess(phases_int[0][3], phases_int[0][4])
        self.assertLess(phases_int[0][4], phases_int[0][5])

        phases_noint, _, _ = elephant.phase_analysis.spike_triggered_phase(
            elephant.signal_processing.hilbert(self.anasig0),
            self.st1,
            interpolate=False,
        )

        self.assertEqual(phases_noint[0][0], phases_noint[0][1])
        self.assertEqual(phases_noint[0][1], phases_noint[0][2])
        self.assertEqual(phases_noint[0][2], phases_noint[0][3])
        self.assertEqual(phases_noint[0][3], phases_noint[0][4])
        self.assertNotEqual(phases_noint[0][4], phases_noint[0][5])

        # Verify that when using interpolation and the spike sits on the sample
        # of the Hilbert transform, this is the same result as when not using
        # interpolation with a spike slightly to the right
        self.assertEqual(phases_noint[0][2], phases_int[0][0])
        self.assertEqual(phases_noint[0][4], phases_int[0][0])

    def test_inconsistent_numbers_spiketrains_hilbert(self):
        self.assertRaises(
            ValueError,
            elephant.phase_analysis.spike_triggered_phase,
            [
                elephant.signal_processing.hilbert(self.anasig0),
                elephant.signal_processing.hilbert(self.anasig0),
            ],
            [self.st0, self.st0, self.st0],
            False,
        )

        self.assertRaises(
            ValueError,
            elephant.phase_analysis.spike_triggered_phase,
            [
                elephant.signal_processing.hilbert(self.anasig0),
                elephant.signal_processing.hilbert(self.anasig0),
            ],
            [self.st0, self.st0, self.st0],
            False,
        )

    def test_spike_earlier_than_hilbert(self):
        # This is a spike clearly outside the bounds
        st = SpikeTrain([-50, 50], units="s", t_start=-100 * pq.s, t_stop=100 * pq.s)
        phases_noint, _, _ = elephant.phase_analysis.spike_triggered_phase(
            elephant.signal_processing.hilbert(self.anasig0), st, interpolate=False
        )
        self.assertEqual(len(phases_noint[0]), 1)

        # This is a spike right on the border (start of the signal is at 0s,
        # spike sits at t=0s). By definition of intervals in
        # Elephant (left borders inclusive, right borders exclusive), this
        # spike is to be considered.
        st = SpikeTrain([0, 50], units="s", t_start=-100 * pq.s, t_stop=100 * pq.s)
        phases_noint, _, _ = elephant.phase_analysis.spike_triggered_phase(
            elephant.signal_processing.hilbert(self.anasig0), st, interpolate=False
        )
        self.assertEqual(len(phases_noint[0]), 2)

    def test_spike_later_than_hilbert(self):
        # This is a spike clearly outside the bounds
        st = SpikeTrain([1, 250], units="s", t_start=-1 * pq.s, t_stop=300 * pq.s)
        phases_noint, _, _ = elephant.phase_analysis.spike_triggered_phase(
            elephant.signal_processing.hilbert(self.anasig0), st, interpolate=False
        )
        self.assertEqual(len(phases_noint[0]), 1)

        # This is a spike right on the border (length of the signal is 100s,
        # spike sits at t=100s). However, by definition of intervals in
        # Elephant (left borders inclusive, right borders exclusive), this
        # spike is not to be considered.
        st = SpikeTrain([1, 100], units="s", t_start=-1 * pq.s, t_stop=200 * pq.s)
        phases_noint, _, _ = elephant.phase_analysis.spike_triggered_phase(
            elephant.signal_processing.hilbert(self.anasig0), st, interpolate=False
        )
        self.assertEqual(len(phases_noint[0]), 1)

    # This test handles the correct dealing with input signals that have
    # different time units, including a CompoundUnit
    def test_regression_269(self):
        # This is a spike train on a 30KHz sampling, one spike at 1s, one just
        # before the end of the signal
        cu = pq.CompoundUnit("1/30000.*s")
        st = SpikeTrain(
            [30000.0, (self.anasig0.t_stop - 1 * pq.s).rescale(cu).magnitude],
            units=pq.CompoundUnit("1/30000.*s"),
            t_start=-1 * pq.s,
            t_stop=300 * pq.s,
        )
        phases_noint, _, _ = elephant.phase_analysis.spike_triggered_phase(
            elephant.signal_processing.hilbert(self.anasig0), st, interpolate=False
        )
        self.assertEqual(len(phases_noint[0]), 2)


class MeanVectorTestCase(unittest.TestCase):
    def setUp(self):
        self.tolerance = 1e-15
        self.n_samples = 200
        # create a sample with all values equal to a random phase-lock phi
        self.lock_value_phi = np.random.uniform(-np.pi, np.pi, 1)
        self.dataset1 = np.ones(self.n_samples) * self.lock_value_phi
        # create a evenly spaced / uniform distribution
        self.dataset2 = np.arange(0, 2 * np.pi, (2 * np.pi) / self.n_samples)
        # create a random distribution
        self.dataset3 = np.random.uniform(-np.pi, np.pi, self.n_samples)

    def testMeanVector_direction_is_phi_and_length_is_1(self):
        """
        Test if the mean vector length is 1 and if the mean direction is phi
        for a sample with all phases equal to phi on the unit circle.

        """
        theta_bar_1, r_1 = elephant.phase_analysis.mean_phase_vector(self.dataset1)
        # mean direction must be phi
        self.assertAlmostEqual(theta_bar_1, self.lock_value_phi, delta=self.tolerance)
        # mean vector length must be almost equal 1
        self.assertAlmostEqual(r_1, 1, delta=self.tolerance)

    def testMeanVector_length_is_0(self):
        """
        Test if the mean vector length  is 0 for a evenly spaced distribution
        on the unit circle.
        """
        theta_bar_2, r_2 = elephant.phase_analysis.mean_phase_vector(self.dataset2)
        # mean vector length must be almost equal 0
        self.assertAlmostEqual(r_2, 0, delta=self.tolerance)

    def testMeanVector_ranges_of_direction_and_length(self):
        """
        Test if the range of the mean vector direction follows numpy standard
        and is within (-pi, pi].
        Test if the range of the mean vector length is within [0, 1].
        """
        theta_bar_3, r_3 = elephant.phase_analysis.mean_phase_vector(self.dataset3)
        # mean vector direction
        self.assertTrue(-np.pi < theta_bar_3 <= np.pi)
        # mean vector length
        self.assertTrue(0 <= r_3 <= 1)


class PhaseDifferenceTestCase(unittest.TestCase):
    def setUp(self):
        self.tolerance = 1e-15
        self.n_samples = 200

    def testPhaseDifference_in_range_minus_pi_to_pi(self):
        """
        Test if the range of the phase difference is within [-pi, pi] for
        random pairs of alpha and beta.
        """
        alpha = np.random.uniform(-np.pi, np.pi, self.n_samples)
        beta = np.random.uniform(-np.pi, np.pi, self.n_samples)

        phase_diff = elephant.phase_analysis.phase_difference(alpha, beta)
        self.assertTrue((-np.pi <= phase_diff).all() and (phase_diff <= np.pi).all())

    def testPhaseDifference_is_delta(self):
        """
        Test if the phase difference is random delta for random pairs of
        alpha and beta, where beta is a copy of alpha shifted by delta.
        """
        delta = np.random.uniform(-np.pi, np.pi, self.n_samples)
        alpha = np.random.uniform(-np.pi, np.pi, self.n_samples)
        _beta = alpha - delta
        beta = np.arctan2(np.sin(_beta), np.cos(_beta))

        phase_diff = elephant.phase_analysis.phase_difference(alpha, beta)
        np.testing.assert_allclose(phase_diff, delta, atol=1e-10)


class PhaseLockingValueTestCase(unittest.TestCase):
    def setUp(self):
        self.tolerance = 1e-15
        self.phase_shift = np.pi / 4
        self.num_time_points = 1000
        self.num_trials = 100

        # create two random uniform distributions (all trials are identical)
        self.signal_x = np.full(
            [self.num_trials, self.num_time_points],
            np.random.uniform(-np.pi, np.pi, self.num_time_points),
        )
        self.signal_y = np.full(
            [self.num_trials, self.num_time_points],
            np.random.uniform(-np.pi, np.pi, self.num_time_points),
        )

        # create two random uniform distributions, where all trails are random
        self.random_x = np.random.uniform(-np.pi, np.pi, (1000, self.num_time_points))
        self.random_y = np.random.uniform(-np.pi, np.pi, (1000, self.num_time_points))

        # simple samples of different shapes to assert ErrorRaising
        self.simple_x = np.array([[0, -np.pi, np.pi], [0, -np.pi, np.pi]])
        self.simple_y = np.array([0, -np.pi, np.pi])
        self.simple_z = np.array([0, np.pi, np.pi / 2, -np.pi])

    def testPhaseLockingValue_identical_signals_both_identical_trials(self):
        """
        Test if the PLV's are 1, when 2 identical signals with identical
        trials are passed. PLV's needed to be 1, due to the constant phase
        difference of 0 across trials at each time-point.
        """
        list1_plv_t = elephant.phase_analysis.phase_locking_value(
            self.signal_x, self.signal_x
        )
        target_plv_r_is_one = np.ones_like(list1_plv_t)
        np.testing.assert_allclose(list1_plv_t, target_plv_r_is_one, self.tolerance)

    def testPhaseLockingValue_different_signals_both_identical_trials(self):
        """
        Test if the PLV's are 1, when 2 different signals are passed, where
        within each signal the trials are identical. PLV's needed to be 1,
        due to a constant phase difference across trials, which may vary for
        different time-points.
        """
        list2_plv_t = elephant.phase_analysis.phase_locking_value(
            self.signal_x, self.signal_y
        )
        target_plv_r_is_one = np.ones_like(list2_plv_t)
        np.testing.assert_allclose(list2_plv_t, target_plv_r_is_one, atol=3e-15)

    def testPhaseLockingValue_different_signals_both_different_trials(self):
        """
        Test if the PLV's are close to 0, when 2 different signals are passed,
        where both have different trials, which are all randomly distributed.
        The PLV's needed to be close to 0, do to a random
        phase difference across trials for each time-point.
        """
        list3_plv_t = elephant.phase_analysis.phase_locking_value(
            self.random_x, self.random_y
        )
        target_plv_is_zero = np.zeros_like(list3_plv_t)
        # use default value from np.allclose() for atol=1e-8 to prevent failure
        np.testing.assert_allclose(
            list3_plv_t, target_plv_is_zero, rtol=1e-2, atol=1.1e-1
        )

    def testPhaseLockingValue_raise_Error_if_trial_number_is_different(self):
        """
        Test if a ValueError is raised, when the signals have different
        number of trails.
        """
        # different numbers of trails
        np.testing.assert_raises(
            ValueError,
            elephant.phase_analysis.phase_locking_value,
            self.simple_x,
            self.simple_y,
        )

    def testPhaseLockingValue_raise_Error_if_trial_lengths_are_different(self):
        """
        Test if a ValueError is raised, when within a trail-pair of the signals
        the trial-lengths are different.
        """
        # different lengths in a trail pair
        np.testing.assert_raises(
            ValueError,
            elephant.phase_analysis.phase_locking_value,
            self.simple_y,
            self.simple_z,
        )


class WeightedPhaseLagIndexTestCase(unittest.TestCase):
    files_to_download_ground_truth = None
    files_to_download_artificial = None
    files_to_download_real = None

    @classmethod
    def setUpClass(cls):
        np.random.seed(73)

        # The files from G-Node GIN 'elephant-data' repository will be
        # downloaded once into a local temporary directory
        # and then loaded/ read for each test function individually.

        cls.tmp_path = {}
        # REAL DATA
        real_data_path = (
            "unittest/phase_analysis/weighted_phase_lag_index/data/wpli_real_data"
        )
        cls.files_to_download_real = (
            (
                "i140703-001_ch01_slice_TS_ON_to_GO_ON_correct_trials.mat",
                "0e76454c58208cab710e672d04de5168",
            ),
            (
                "i140703-001_ch02_slice_TS_ON_to_GO_ON_correct_trials.mat",
                "b06059e5222e91eb640caad0aba15b7f",
            ),
            (
                "i140703-001_cross_spectrum_of_channel_1_and_2_of_slice_"
                "TS_ON_to_GO_ON_corect_trials.mat",
                "2687ef63a4a456971a5dcc621b02e9a9",
            ),
        )
        for filename, checksum in cls.files_to_download_real:
            # files will be downloaded to ELEPHANT_TMP_DIR
            cls.tmp_path[filename] = {
                "filename": filename,
                "path": download_datasets(
                    f"{real_data_path}/{filename}", checksum=checksum
                ),
            }
        # ARTIFICIAL DATA
        artificial_data_path = (
            "unittest/phase_analysis/"
            "weighted_phase_lag_index/data/wpli_specific_artificial_dataset"
        )
        cls.files_to_download_artificial = (
            ("artificial_LFPs_1.mat", "4b99b15f89c0b9a0eb6fc14e9009436f"),
            ("artificial_LFPs_2.mat", "7144976b5f871fa62f4a831f530deee4"),
        )
        for filename, checksum in cls.files_to_download_artificial:
            # files will be downloaded to ELEPHANT_TMP_DIR
            cls.tmp_path[filename] = {
                "filename": filename,
                "path": download_datasets(
                    f"{artificial_data_path}/{filename}", checksum=checksum
                ),
            }
        # GROUND TRUTH DATA
        ground_truth_data_path = (
            "unittest/phase_analysis/weighted_phase_lag_index/data/wpli_ground_truth"
        )
        cls.files_to_download_ground_truth = (
            (
                "ground_truth_WPLI_from_ft_connectivity_wpli_with_real_LFPs_R2G.csv",
                "4d9a7b7afab7d107023956077ab11fef",
            ),
            (
                "ground_truth_WPLI_from_ft_connectivity_wpli_with_artificial_LFPs.csv",
                "92988f475333d7badbe06b3f23abe494",
            ),
        )
        for filename, checksum in cls.files_to_download_ground_truth:
            # files will be downloaded into ELEPHANT_TMP_DIR
            cls.tmp_path[filename] = {
                "filename": filename,
                "path": download_datasets(
                    f"{ground_truth_data_path}/{filename}", checksum=checksum
                ),
            }

    def setUp(self):
        self.tolerance = 1e-15

        # load real/artificial LFP-dataset for ground-truth consistency checks
        # real LFP-dataset
        dataset1_real = scipy.io.loadmat(
            f"{self.tmp_path[self.files_to_download_real[0][0]]['path']}",
            squeeze_me=True,
        )
        dataset2_real = scipy.io.loadmat(
            f"{self.tmp_path[self.files_to_download_real[1][0]]['path']}",
            squeeze_me=True,
        )

        # get relevant values
        self.lfps1_real = dataset1_real["lfp_matrix"] * pq.uV
        self.sf1_real = dataset1_real["sf"] * pq.Hz
        self.lfps2_real = dataset2_real["lfp_matrix"] * pq.uV
        self.sf2_real = dataset2_real["sf"] * pq.Hz
        # create AnalogSignals from the real dataset
        self.lfps1_real_AnalogSignal = AnalogSignal(
            signal=self.lfps1_real, sampling_rate=self.sf1_real
        )
        self.lfps2_real_AnalogSignal = AnalogSignal(
            signal=self.lfps2_real, sampling_rate=self.sf2_real
        )

        # artificial LFP-dataset
        dataset1_path = (
            f"{self.tmp_path[self.files_to_download_artificial[0][0]]['path']}"
        )
        dataset1_artificial = scipy.io.loadmat(dataset1_path, squeeze_me=True)
        dataset2_path = (
            f"{self.tmp_path[self.files_to_download_artificial[1][0]]['path']}"
        )
        dataset2_artificial = scipy.io.loadmat(dataset2_path, squeeze_me=True)
        # get relevant values
        self.lfps1_artificial = dataset1_artificial["lfp_matrix"] * pq.uV
        self.sf1_artificial = dataset1_artificial["sf"] * pq.Hz
        self.lfps2_artificial = dataset2_artificial["lfp_matrix"] * pq.uV
        self.sf2_artificial = dataset2_artificial["sf"] * pq.Hz
        # create AnalogSignals from the artificial dataset
        self.lfps1_artificial_AnalogSignal = AnalogSignal(
            signal=self.lfps1_artificial, sampling_rate=self.sf1_artificial
        )
        self.lfps2_artificial_AnalogSignal = AnalogSignal(
            signal=self.lfps2_artificial, sampling_rate=self.sf2_artificial
        )

        # load ground-truth reference calculated by:
        # Matlab package 'FieldTrip': ft_connectivity_wpli()
        self.wpli_ground_truth_ft_connectivity_wpli_real = np.loadtxt(
            f"{self.tmp_path[self.files_to_download_ground_truth[0][0]]['path']}",  # noqa
            delimiter=",",
            dtype=np.float64,
        )
        self.wpli_ground_truth_ft_connectivity_artificial = np.loadtxt(
            f"{self.tmp_path[self.files_to_download_ground_truth[1][0]]['path']}",  # noqa
            delimiter=",",
            dtype=np.float64,
        )

    def test_WPLI_ground_truth_consistency_real_LFP_dataset(self):
        """
        Test if the WPLI is consistent with the reference implementation
        ft_connectivity_wpli() of the MATLAB-package FieldTrip using
        LFP-dataset cuttings from the multielectrode-grasp  G-Node GIN
        repository, which can be found here:
        https://doi.gin.g-node.org/10.12751/g-node.f83565/
        The cutting was performed with this python-script:
        multielectrode_grasp_i140703-001_cutting_script_TS_ON_to_GO_ON.py
        which is available on https://gin.g-node.org/INM-6/elephant-data
        in folder validation/phase_analysis/weighted_phase_lag_index/scripts,
        where also the MATLAB-script for ground-truth generation is located.
        """
        # Quantity-input
        with self.subTest(msg="Quantity input"):
            freq, wpli = elephant.phase_analysis.weighted_phase_lag_index(
                self.lfps1_real, self.lfps2_real, self.sf1_real
            )
            np.testing.assert_allclose(
                wpli, self.wpli_ground_truth_ft_connectivity_wpli_real, equal_nan=True
            )
        # np.array-input
        with self.subTest(msg="np.array input"):
            freq, wpli = elephant.phase_analysis.weighted_phase_lag_index(
                self.lfps1_real.magnitude, self.lfps2_real.magnitude, self.sf1_real
            )
            np.testing.assert_allclose(
                wpli, self.wpli_ground_truth_ft_connectivity_wpli_real, equal_nan=True
            )
        # neo.AnalogSignal-input
        with self.subTest(msg="neo.AnalogSignal input"):
            freq, wpli = elephant.phase_analysis.weighted_phase_lag_index(
                self.lfps1_real_AnalogSignal, self.lfps2_real_AnalogSignal
            )
            np.testing.assert_allclose(
                wpli, self.wpli_ground_truth_ft_connectivity_wpli_real, equal_nan=True
            )

    def test_WPLI_ground_truth_consistency_artificial_LFP_dataset(self):
        """
        Test if the WPLI is consistent with the ground truth generated with
        multi-sine artificial LFP-datasets.
        The generation was performed with this python-script:
        generate_artificial_datasets_for_ground_truth_of_wpli.py
        which is available on https://gin.g-node.org/INM-6/elephant-data
        in folder validation/phase_analysis/weighted_phase_lag_index/scripts,
        where also the MATLAB-script for ground-truth generation is located.
        """
        # Quantity-input
        with self.subTest(msg="Quantity input"):
            freq, wpli = elephant.phase_analysis.weighted_phase_lag_index(
                self.lfps1_artificial,
                self.lfps2_artificial,
                self.sf1_artificial,
                absolute_value=False,
            )
            np.testing.assert_allclose(
                wpli,
                self.wpli_ground_truth_ft_connectivity_artificial,
                atol=1e-14,
                rtol=1e-12,
                equal_nan=True,
            )
        # np.array-input
        with self.subTest(msg="np.array input"):
            freq, wpli = elephant.phase_analysis.weighted_phase_lag_index(
                self.lfps1_artificial.magnitude,
                self.lfps2_artificial.magnitude,
                self.sf1_artificial,
                absolute_value=False,
            )
            np.testing.assert_allclose(
                wpli,
                self.wpli_ground_truth_ft_connectivity_artificial,
                atol=1e-14,
                rtol=1e-12,
                equal_nan=True,
            )
        # neo.AnalogSignal-input
        with self.subTest(msg="neo.AnalogSignal input"):
            freq, wpli = elephant.phase_analysis.weighted_phase_lag_index(
                self.lfps1_artificial_AnalogSignal,
                self.lfps2_artificial_AnalogSignal,
                absolute_value=False,
            )
            np.testing.assert_allclose(
                wpli,
                self.wpli_ground_truth_ft_connectivity_artificial,
                atol=1e-14,
                rtol=1e-12,
                equal_nan=True,
            )

    def test_WPLI_is_zero(self):
        """
        Test if WPLI is close to zero at frequency f=70Hz for the multi-sine
        artificial LFP dataset. White noise prevents arbitrary approximation.
        """
        # Quantity-input
        with self.subTest(msg="Quantity input"):
            freq, wpli = elephant.phase_analysis.weighted_phase_lag_index(
                self.lfps1_artificial,
                self.lfps2_artificial,
                self.sf1_artificial,
                absolute_value=False,
            )
            np.testing.assert_allclose(
                wpli[freq == 70], 0, atol=0.004, rtol=self.tolerance
            )
        # np.array-input
        with self.subTest(msg="np.array input"):
            freq, wpli = elephant.phase_analysis.weighted_phase_lag_index(
                self.lfps1_artificial.magnitude,
                self.lfps2_artificial.magnitude,
                self.sf1_artificial,
                absolute_value=False,
            )
            np.testing.assert_allclose(
                wpli[freq == 70], 0, atol=0.004, rtol=self.tolerance
            )
        # neo.AnalogSignal-input
        with self.subTest(msg="neo.AnalogSignal input"):
            freq, wpli = elephant.phase_analysis.weighted_phase_lag_index(
                self.lfps1_artificial_AnalogSignal,
                self.lfps2_artificial_AnalogSignal,
                absolute_value=False,
            )
            np.testing.assert_allclose(
                wpli[freq == 70], 0, atol=0.004, rtol=self.tolerance
            )

    def test_WPLI_is_one(self):
        """
        Test if WPLI is one at frequency f=16Hz and 36Hz for the multi-sine
        artificial LFP dataset.
        """
        # Quantity-input
        with self.subTest(msg="Quantity input"):
            freq, wpli = elephant.phase_analysis.weighted_phase_lag_index(
                self.lfps1_artificial,
                self.lfps2_artificial,
                self.sf1_artificial,
                absolute_value=False,
            )
            mask = (freq == 16) | (freq == 36)
            np.testing.assert_allclose(
                wpli[mask], 1, atol=self.tolerance, rtol=self.tolerance
            )
        # np.array-input
        with self.subTest(msg="np.array input"):
            freq, wpli = elephant.phase_analysis.weighted_phase_lag_index(
                self.lfps1_artificial.magnitude,
                self.lfps2_artificial.magnitude,
                self.sf1_artificial,
                absolute_value=False,
            )
            mask = (freq == 16) | (freq == 36)
            np.testing.assert_allclose(
                wpli[mask], 1, atol=self.tolerance, rtol=self.tolerance
            )
        # neo.AnalogSignal-input
        with self.subTest(msg="neo.AnalogSignal input"):
            freq, wpli = elephant.phase_analysis.weighted_phase_lag_index(
                self.lfps1_artificial_AnalogSignal,
                self.lfps2_artificial_AnalogSignal,
                absolute_value=False,
            )
            mask = (freq == 16) | (freq == 36)
            np.testing.assert_allclose(
                wpli[mask], 1, atol=self.tolerance, rtol=self.tolerance
            )

    def test_WPLI_is_minus_one(self):
        """
        Test if WPLI is minus one at frequency f=52Hz and 100Hz
        for the multi-sine artificial LFP dataset.
        """
        # Quantity-input
        with self.subTest(msg="Quantity input"):
            freq, wpli = elephant.phase_analysis.weighted_phase_lag_index(
                self.lfps1_artificial,
                self.lfps2_artificial,
                self.sf1_artificial,
                absolute_value=False,
            )
            mask = (freq == 52) | (freq == 100)
            np.testing.assert_allclose(
                wpli[mask], -1, atol=self.tolerance, rtol=self.tolerance
            )
        # np.array-input
        with self.subTest(msg="np.array input"):
            freq, wpli = elephant.phase_analysis.weighted_phase_lag_index(
                self.lfps1_artificial.magnitude,
                self.lfps2_artificial.magnitude,
                self.sf1_artificial,
                absolute_value=False,
            )
            np.testing.assert_allclose(
                wpli[mask], -1, atol=self.tolerance, rtol=self.tolerance
            )
        # neo.AnalogSignal-input
        with self.subTest(msg="neo.AnalogSignal input"):
            freq, wpli = elephant.phase_analysis.weighted_phase_lag_index(
                self.lfps1_artificial_AnalogSignal,
                self.lfps2_artificial_AnalogSignal,
                absolute_value=False,
            )
            np.testing.assert_allclose(
                wpli[mask], -1, atol=self.tolerance, rtol=self.tolerance
            )

    def test_WPLI_raises_error_if_signals_have_different_shapes(self):
        """
        Test if WPLI raises a ValueError, when the signals have different
        number of trails or different trial lengths.
        """
        # simple samples of different shapes to assert ErrorRaising
        trials2_length3 = np.array([[0, -1, 1], [0, -1, 1]]) * pq.uV
        trials1_length3 = np.array([[0, -1, 1]]) * pq.uV
        trials1_length4 = np.array([[0, 1, 1 / 2, -1]]) * pq.uV
        sampling_frequency = 250 * pq.Hz
        trials2_length3_analogsignal = AnalogSignal(
            signal=trials2_length3, sampling_rate=sampling_frequency
        )
        trials1_length3_analogsignal = AnalogSignal(
            signal=trials1_length3, sampling_rate=sampling_frequency
        )
        trials1_length4_analogsignal = AnalogSignal(
            signal=trials1_length4, sampling_rate=sampling_frequency
        )

        # different numbers of trails
        with self.subTest(msg="diff. trial numbers & Quantity input"):
            np.testing.assert_raises(
                ValueError,
                elephant.phase_analysis.weighted_phase_lag_index,
                trials2_length3,
                trials1_length3,
                sampling_frequency,
            )
        with self.subTest(msg="diff. trial numbers & np.array input"):
            np.testing.assert_raises(
                ValueError,
                elephant.phase_analysis.weighted_phase_lag_index,
                trials2_length3.magnitude,
                trials1_length3.magnitude,
                sampling_frequency,
            )
        with self.subTest(msg="diff. trial numbers & neo.AnalogSignal input"):
            np.testing.assert_raises(
                ValueError,
                elephant.phase_analysis.weighted_phase_lag_index,
                trials2_length3_analogsignal,
                trials1_length3_analogsignal,
            )
        # different lengths in a trail pair
        with self.subTest(msg="diff. trial lengths & Quantity input"):
            np.testing.assert_raises(
                ValueError,
                elephant.phase_analysis.weighted_phase_lag_index,
                trials1_length3,
                trials1_length4,
                sampling_frequency,
            )
        with self.subTest(msg="diff. trial lengths & np.array input"):
            np.testing.assert_raises(
                ValueError,
                elephant.phase_analysis.weighted_phase_lag_index,
                trials1_length3.magnitude,
                trials1_length4.magnitude,
                sampling_frequency,
            )
        with self.subTest(msg="diff. trial lengths & neo.AnalogSignal input"):
            np.testing.assert_raises(
                ValueError,
                elephant.phase_analysis.weighted_phase_lag_index,
                trials1_length3_analogsignal,
                trials1_length4_analogsignal,
            )

    @staticmethod
    def test_WPLI_raises_error_if_AnalogSignals_have_diff_sampling_rate():
        """
        Test if WPLI raises a ValueError, when the AnalogSignals have different
        sampling rates.
        """
        signal_x_250_hz = AnalogSignal(
            signal=np.random.random([40, 2100]),
            units=pq.mV,
            sampling_rate=0.25 * pq.kHz,
        )
        signal_y_1000_hz = AnalogSignal(
            signal=np.random.random([40, 2100]), units=pq.mV, sampling_rate=1000 * pq.Hz
        )
        np.testing.assert_raises(
            ValueError,
            elephant.phase_analysis.weighted_phase_lag_index,
            signal_x_250_hz,
            signal_y_1000_hz,
        )

    def test_WPLI_raises_error_if_sampling_rate_not_given(self):
        """
        Test if WPLI raises a ValueError, when the sampling rate is not given
        for np.array() or Quantity input.
        """
        signal_x = np.random.random([40, 2100]) * pq.mV
        signal_y = np.random.random([40, 2100]) * pq.mV
        with self.subTest(msg="Quantity-input"):
            np.testing.assert_raises(
                ValueError,
                elephant.phase_analysis.weighted_phase_lag_index,
                signal_x,
                signal_y,
            )
        with self.subTest(msg="np.array-input"):
            np.testing.assert_raises(
                ValueError,
                elephant.phase_analysis.weighted_phase_lag_index,
                signal_x.magnitude,
                signal_y.magnitude,
            )


if __name__ == "__main__":
    unittest.main()
