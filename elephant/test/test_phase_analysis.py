# -*- coding: utf-8 -*-
"""
Unit tests for the phase analysis module.

:copyright: Copyright 2016 by the Elephant team, see `doc/authors.rst`.
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
        # create a nonuniform-distribution at a random phase-lock phi
        phi = np.random.random(1)[0] * 2 * np.pi
        self.lock_value_phi = np.arctan2(np.sin(phi), np.cos(phi))
        self.dataset1 = np.ones(self.n_samples) * self.lock_value_phi
        # create a evenly spaced / uniform distribution
        self.dataset2 = np.arange(0, 2*np.pi, (2*np.pi) / self.n_samples)
        # create a random distribution
        self.dataset3 = np.random.random(self.n_samples) * 2 * np.pi

    def testMeanVector_direction_is_phi_and_length_is_1(self):
        """
        Test if the mean vector length of a homogenous sample with all phases
        equal phi on the unit circle is 1 and if the mean direction is phi.

        """
        theta_bar_1, r_1 = elephant.phase_analysis.mean_vector(self.dataset1,
                                                               axis=0)
        # mean direction must be phi
        self.assertAlmostEqual(theta_bar_1, self.lock_value_phi,
                               delta=self.tolerance)
        # mean vector length must be almost equal 1
        self.assertAlmostEqual(r_1, 1, delta=self.tolerance)

    def testMeanVector_length_is_0(self):
        """
        Test if the mean vector length of a evenly spaced distribution on the
        unit circle is 0.
        """
        theta_bar_2, r_2 = elephant.phase_analysis.mean_vector(self.dataset2,
                                                               axis=0)
        # mean vector length must be almost equal 0
        self.assertAlmostEqual(r_2, 0, delta=self.tolerance)

    def testMeanVector_ranges_of_direction_and_length(self):
        """
        Test if the range of the mean vector direction follows numpy standard
        and is within (-pi, pi].
        Test if the range of the mean vector length is within [0, 1].
        """
        theta_bar_3, r_3 = elephant.phase_analysis.mean_vector(self.dataset3,
                                                               axis=0)
        # mean vector direction
        self.assertTrue(-np.pi < theta_bar_3 <= np.pi)
        # mean vector length
        self.assertTrue(0 <= r_3 <= 1)


class PhaseDifferenceTestCase(unittest.TestCase):
    def setUp(self):
        self.tolerance = 1e-15
        self.n_samples = 200

    def testPhaseDifference_abs_alpha_minus_beta_smaller_pi(self):
        alpha = np.array([0.8, 0.6, 0.2, -0.2]) * np.pi
        beta = np.array([0.6, 0.8, -0.2, 0.2]) * np.pi
        target_phase_diff = np.array([0.2, -0.2, 0.4, -0.4]) * np.pi

        phase_diff = elephant.phase_analysis.phase_difference(alpha, beta)
        np.testing.assert_allclose(phase_diff, target_phase_diff,
                                   self.tolerance)

    def testPhaseDifference_abs_alpha_minus_beta_greater_pi(self):
        alpha = np.array([0.8, -0.8, 0.3, -0.8]) * np.pi
        beta = np.array([-0.8, 0.8, -0.8, 0.3]) * np.pi
        target_phase_diff = np.array([-0.4, 0.4, -0.9, 0.9]) * np.pi

        phase_diff = elephant.phase_analysis.phase_difference(alpha, beta)
        np.testing.assert_allclose(phase_diff, target_phase_diff,
                                    self.tolerance)

    def testPhaseDifference_in_range_minus_pi_to_pi(self):
        sign_1 = 1 if np.random.random(1) < 0.5 else -1
        sign_2 = 1 if np.random.random(1) < 0.5 else -1
        alpha = sign_1 * np.random.random(1) * np.pi
        beta = sign_2 * np.random.random(1) * np.pi

        phase_diff = elephant.phase_analysis.phase_difference(alpha, beta)
        self.assertTrue(-np.pi <= phase_diff <= np.pi)

    def testPhaseDifference_for_arrays(self):
        delta = np.random.random(1) * np.pi
        alpha = np.random.random(self.n_samples) * 2 * np.pi
        alpha = np.arctan2(np.sin(alpha), np.cos(alpha))
        beta = alpha - delta
        beta = np.arctan2(np.sin(beta), np.cos(beta))

        phase_diff = elephant.phase_analysis.phase_difference(alpha, beta)
        target_phase_diff = np.ones_like(phase_diff) * delta
        np.testing.assert_allclose(phase_diff, target_phase_diff,
                                   3 * self.tolerance)
        # NOTE: tolerance must be increased to 3e-15 to prevent failure
        

class PhaseLockingValueTestCase(unittest.TestCase):
    def setUp(self):
        self.tolerance = 1e-15
        self.phase_shift = np.pi/4
        self.num_time_points = 1000
        self.num_trials = 100

        # create randomly two uniform distributions in the half-open interval
        # of [-pi, pi)
        one_x_trial = np.random.uniform(-np.pi, np.pi, self.num_time_points)
        self.signal_x = np.empty([self.num_trials, self.num_time_points])
        self.signal_x[:] = one_x_trial

        one_y_trial = np.random.uniform(-np.pi, np.pi, self.num_time_points)
        self.signal_y = np.empty([self.num_trials, self.num_time_points])
        self.signal_y[:] = one_y_trial

        # create phase-shifted trials of signal_x,
        # to create later phase differences, which vary a lot across trials
        shift_steps = \
            np.reshape(np.arange(0, 2*np.pi, (2*np.pi)/self.num_trials),
                       (self.num_trials, 1))
        self.shifted_signal_x = np.copy(self.signal_x) + shift_steps
        # keep phases in range of [-pi, pi)
        self.shifted_signal_x = np.arctan2(np.sin(self.shifted_signal_x),
                                           np.cos(self.shifted_signal_x))

    def testPhaseLockingValue_identical_signals_both_homogeneous_trials(self):
        """
        Test if the PLV's are 1, when 2 identical signals with homogenous 
        trials are passed. PLV's needed to be 1, due to the constant phase 
        difference of 0 across trials at each time-point.
        """
        list1_plv_t = \
            elephant.phase_analysis.phase_locking_value(self.signal_x,
                                                        self.signal_x)
        target_plv_r_is_one = np.ones_like(list1_plv_t)
        np.testing.assert_allclose(list1_plv_t, target_plv_r_is_one,
                                   self.tolerance)

    def testPhaseLockingValue_different_signals_both_homogenous_trials(self):
        """
        Test if the PLV's are 1, when 2 different signals with homogenous
        trials are passed. PLV's needed to be 1, due to a constant phase
        difference across trials, which may vary for different time-points.
        """
        list2_plv_t = elephant.phase_analysis.phase_locking_value(
            self.signal_x, self.signal_y)
        target_plv_r_is_one = np.ones_like(list2_plv_t)
        np.testing.assert_allclose(list2_plv_t, target_plv_r_is_one,
                                   3 * self.tolerance)
        # NOTE: tolerance must be increased to 3e-15 to prevent failure

    def testPhaseLockingValue_different_signals_one_heterogeneous_trials(self):
        """
        Test if the PLV's are 0, when 2 different signals (original & shifted
        version) are passed, where one has homogenous trials and the other
        heterogeneous trials. In the shifted version each trial got shifted by
        a variable step (steps are evenly spaced/distributed).
        The PLV's needed to be 0, do to a variable (evenly spaced/distributed)
        phase difference across trials for each time-point.
        """
        list3_plv_t = elephant.phase_analysis.phase_locking_value(
            self.signal_x, self.shifted_signal_x)
        target_plv_r_is_zero = np.zeros_like(list3_plv_t)
        # use default value from np.allclose() for atol=1e-8 to prevent failure
        np.testing.assert_allclose(list3_plv_t, target_plv_r_is_zero,
                                   rtol=self.tolerance, atol=1e-08)


if __name__ == '__main__':
    unittest.main()
