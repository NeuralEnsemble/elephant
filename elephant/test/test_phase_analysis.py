# -*- coding: utf-8 -*-
"""
Unit tests for the phase analysis module.

:copyright: Copyright 2014-2020 by the Elephant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
"""
from __future__ import division, print_function

import unittest

from neo import SpikeTrain, AnalogSignal
import numpy as np
import quantities as pq

import elephant.phase_analysis

from numpy.ma.testutils import assert_allclose


class SpikeTriggeredPhaseTestCase(unittest.TestCase):

    def setUp(self):
        tlen0 = 100 * pq.s
        f0 = 20. * pq.Hz
        fs0 = 1 * pq.ms
        t0 = np.arange(
            0, tlen0.rescale(pq.s).magnitude,
            fs0.rescale(pq.s).magnitude) * pq.s
        self.anasig0 = AnalogSignal(
            np.sin(2 * np.pi * (f0 * t0).simplified.magnitude),
            units=pq.mV, t_start=0 * pq.ms, sampling_period=fs0)
        self.st0 = SpikeTrain(
            np.arange(50, tlen0.rescale(pq.ms).magnitude - 50, 50) * pq.ms,
            t_start=0 * pq.ms, t_stop=tlen0)
        self.st1 = SpikeTrain(
            [100., 100.1, 100.2, 100.3, 100.9, 101.] * pq.ms,
            t_start=0 * pq.ms, t_stop=tlen0)

    def test_perfect_locking_one_spiketrain_one_signal(self):
        phases, amps, times = elephant.phase_analysis.spike_triggered_phase(
            elephant.signal_processing.hilbert(self.anasig0),
            self.st0,
            interpolate=True)

        assert_allclose(phases[0], - np.pi / 2.)
        assert_allclose(amps[0], 1, atol=0.1)
        assert_allclose(times[0].magnitude, self.st0.magnitude)
        self.assertEqual(len(phases[0]), len(self.st0))
        self.assertEqual(len(amps[0]), len(self.st0))
        self.assertEqual(len(times[0]), len(self.st0))

    def test_perfect_locking_many_spiketrains_many_signals(self):
        phases, amps, times = elephant.phase_analysis.spike_triggered_phase(
            [
                elephant.signal_processing.hilbert(self.anasig0),
                elephant.signal_processing.hilbert(self.anasig0)],
            [self.st0, self.st0],
            interpolate=True)

        assert_allclose(phases[0], -np.pi / 2.)
        assert_allclose(amps[0], 1, atol=0.1)
        assert_allclose(times[0].magnitude, self.st0.magnitude)
        self.assertEqual(len(phases[0]), len(self.st0))
        self.assertEqual(len(amps[0]), len(self.st0))
        self.assertEqual(len(times[0]), len(self.st0))

    def test_perfect_locking_one_spiketrains_many_signals(self):
        phases, amps, times = elephant.phase_analysis.spike_triggered_phase(
            [
                elephant.signal_processing.hilbert(self.anasig0),
                elephant.signal_processing.hilbert(self.anasig0)],
            [self.st0],
            interpolate=True)

        assert_allclose(phases[0], -np.pi / 2.)
        assert_allclose(amps[0], 1, atol=0.1)
        assert_allclose(times[0].magnitude, self.st0.magnitude)
        self.assertEqual(len(phases[0]), len(self.st0))
        self.assertEqual(len(amps[0]), len(self.st0))
        self.assertEqual(len(times[0]), len(self.st0))

    def test_perfect_locking_many_spiketrains_one_signal(self):
        phases, amps, times = elephant.phase_analysis.spike_triggered_phase(
            elephant.signal_processing.hilbert(self.anasig0),
            [self.st0, self.st0],
            interpolate=True)

        assert_allclose(phases[0], -np.pi / 2.)
        assert_allclose(amps[0], 1, atol=0.1)
        assert_allclose(times[0].magnitude, self.st0.magnitude)
        self.assertEqual(len(phases[0]), len(self.st0))
        self.assertEqual(len(amps[0]), len(self.st0))
        self.assertEqual(len(times[0]), len(self.st0))

    def test_interpolate(self):
        phases_int, _, _ = elephant.phase_analysis.spike_triggered_phase(
            elephant.signal_processing.hilbert(self.anasig0),
            self.st1,
            interpolate=True)

        self.assertLess(phases_int[0][0], phases_int[0][1])
        self.assertLess(phases_int[0][1], phases_int[0][2])
        self.assertLess(phases_int[0][2], phases_int[0][3])
        self.assertLess(phases_int[0][3], phases_int[0][4])
        self.assertLess(phases_int[0][4], phases_int[0][5])

        phases_noint, _, _ = elephant.phase_analysis.spike_triggered_phase(
            elephant.signal_processing.hilbert(self.anasig0),
            self.st1,
            interpolate=False)

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
            ValueError, elephant.phase_analysis.spike_triggered_phase,
            [
                elephant.signal_processing.hilbert(self.anasig0),
                elephant.signal_processing.hilbert(self.anasig0)],
            [self.st0, self.st0, self.st0], False)

        self.assertRaises(
            ValueError, elephant.phase_analysis.spike_triggered_phase,
            [
                elephant.signal_processing.hilbert(self.anasig0),
                elephant.signal_processing.hilbert(self.anasig0)],
            [self.st0, self.st0, self.st0], False)

    def test_spike_earlier_than_hilbert(self):
        # This is a spike clearly outside the bounds
        st = SpikeTrain(
            [-50, 50],
            units='s', t_start=-100 * pq.s, t_stop=100 * pq.s)
        phases_noint, _, _ = elephant.phase_analysis.spike_triggered_phase(
            elephant.signal_processing.hilbert(self.anasig0),
            st,
            interpolate=False)
        self.assertEqual(len(phases_noint[0]), 1)

        # This is a spike right on the border (start of the signal is at 0s,
        # spike sits at t=0s). By definition of intervals in
        # Elephant (left borders inclusive, right borders exclusive), this
        # spike is to be considered.
        st = SpikeTrain(
            [0, 50],
            units='s', t_start=-100 * pq.s, t_stop=100 * pq.s)
        phases_noint, _, _ = elephant.phase_analysis.spike_triggered_phase(
            elephant.signal_processing.hilbert(self.anasig0),
            st,
            interpolate=False)
        self.assertEqual(len(phases_noint[0]), 2)

    def test_spike_later_than_hilbert(self):
        # This is a spike clearly outside the bounds
        st = SpikeTrain(
            [1, 250],
            units='s', t_start=-1 * pq.s, t_stop=300 * pq.s)
        phases_noint, _, _ = elephant.phase_analysis.spike_triggered_phase(
            elephant.signal_processing.hilbert(self.anasig0),
            st,
            interpolate=False)
        self.assertEqual(len(phases_noint[0]), 1)

        # This is a spike right on the border (length of the signal is 100s,
        # spike sits at t=100s). However, by definition of intervals in
        # Elephant (left borders inclusive, right borders exclusive), this
        # spike is not to be considered.
        st = SpikeTrain(
            [1, 100],
            units='s', t_start=-1 * pq.s, t_stop=200 * pq.s)
        phases_noint, _, _ = elephant.phase_analysis.spike_triggered_phase(
            elephant.signal_processing.hilbert(self.anasig0),
            st,
            interpolate=False)
        self.assertEqual(len(phases_noint[0]), 1)

    # This test handles the correct dealing with input signals that have
    # different time units, including a CompoundUnit
    def test_regression_269(self):
        # This is a spike train on a 30KHz sampling, one spike at 1s, one just
        # before the end of the signal
        cu = pq.CompoundUnit("1/30000.*s")
        st = SpikeTrain(
            [30000., (self.anasig0.t_stop - 1 * pq.s).rescale(cu).magnitude],
            units=pq.CompoundUnit("1/30000.*s"),
            t_start=-1 * pq.s, t_stop=300 * pq.s)
        phases_noint, _, _ = elephant.phase_analysis.spike_triggered_phase(
            elephant.signal_processing.hilbert(self.anasig0),
            st,
            interpolate=False)
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
        theta_bar_1, r_1 = elephant.phase_analysis.mean_phase_vector(
            self.dataset1)
        # mean direction must be phi
        self.assertAlmostEqual(theta_bar_1, self.lock_value_phi,
                               delta=self.tolerance)
        # mean vector length must be almost equal 1
        self.assertAlmostEqual(r_1, 1, delta=self.tolerance)

    def testMeanVector_length_is_0(self):
        """
        Test if the mean vector length  is 0 for a evenly spaced distribution
        on the unit circle.
        """
        theta_bar_2, r_2 = elephant.phase_analysis.mean_phase_vector(
            self.dataset2)
        # mean vector length must be almost equal 0
        self.assertAlmostEqual(r_2, 0, delta=self.tolerance)

    def testMeanVector_ranges_of_direction_and_length(self):
        """
        Test if the range of the mean vector direction follows numpy standard
        and is within (-pi, pi].
        Test if the range of the mean vector length is within [0, 1].
        """
        theta_bar_3, r_3 = elephant.phase_analysis.mean_phase_vector(
            self.dataset3)
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
        self.assertTrue((-np.pi <= phase_diff).all()
                        and (phase_diff <= np.pi).all())

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
        self.signal_x = \
            np.full([self.num_trials, self.num_time_points],
                    np.random.uniform(-np.pi, np.pi, self.num_time_points))
        self.signal_y = \
            np.full([self.num_trials, self.num_time_points],
                    np.random.uniform(-np.pi, np.pi, self.num_time_points))

        # create two random uniform distributions, where all trails are random
        self.random_x = np.random.uniform(
            -np.pi, np.pi, (1000, self.num_time_points))
        self.random_y = np.random.uniform(
            -np.pi, np.pi, (1000, self.num_time_points))

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
        list1_plv_t = \
            elephant.phase_analysis.phase_locking_value(self.signal_x,
                                                        self.signal_x)
        target_plv_r_is_one = np.ones_like(list1_plv_t)
        np.testing.assert_allclose(list1_plv_t, target_plv_r_is_one,
                                   self.tolerance)

    def testPhaseLockingValue_different_signals_both_identical_trials(self):
        """
        Test if the PLV's are 1, when 2 different signals are passed, where
        within each signal the trials are identical. PLV's needed to be 1,
        due to a constant phase difference across trials, which may vary for
        different time-points.
        """
        list2_plv_t = elephant.phase_analysis.phase_locking_value(
            self.signal_x, self.signal_y)
        target_plv_r_is_one = np.ones_like(list2_plv_t)
        np.testing.assert_allclose(list2_plv_t, target_plv_r_is_one,
                                   atol=3e-15)

    def testPhaseLockingValue_different_signals_both_different_trials(self):
        """
        Test if the PLV's are close to 0, when 2 different signals are passed,
        where both have different trials, which are all randomly distributed.
        The PLV's needed to be close to 0, do to a random
        phase difference across trials for each time-point.
        """
        list3_plv_t = elephant.phase_analysis.phase_locking_value(
            self.random_x, self.random_y)
        target_plv_is_zero = np.zeros_like(list3_plv_t)
        # use default value from np.allclose() for atol=1e-8 to prevent failure
        np.testing.assert_allclose(list3_plv_t, target_plv_is_zero,
                                   rtol=1e-2, atol=1.1e-1)

    def testPhaseLockingValue_raise_Error_if_trial_number_is_different(self):
        """
        Test if a ValueError is raised, when the signals have different
        number of trails.
        """
        # different numbers of trails
        np.testing.assert_raises(
            ValueError, elephant.phase_analysis.phase_locking_value,
            self.simple_x, self.simple_y)

    def testPhaseLockingValue_raise_Error_if_trial_lengths_are_different(self):
        """
        Test if a ValueError is raised, when within a trail-pair of the signals
        the trial-lengths are different.
        """
        # different lengths in a trail pair
        np.testing.assert_raises(
            ValueError, elephant.phase_analysis.phase_locking_value,
            self.simple_y, self.simple_z)


if __name__ == '__main__':
    unittest.main()
