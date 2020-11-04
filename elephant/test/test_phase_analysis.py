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
        self.n_sampels = 200
        # create a phase-locked 'non'-distribution
        self.lock_value = np.random.random(1)[0] * 2 * np.pi
        self.dataset1 = np.full((1, self.n_sampels), self.lock_value)[0]
        # create a evenly spaced / uniform distribution
        self.dataset2 = [i * 2 * np.pi / self.n_sampels
                         for i in range(self.n_sampels)]

    def testMeanVector(self):
        theta_bar_1, r_1 = \
            elephant.phase_analysis.mean_vector(self.dataset1)
        theta_bar_2, r_2 = \
            elephant.phase_analysis.mean_vector(self.dataset2)
        # mean direction
        self.assertAlmostEqual(theta_bar_1, self.lock_value,
                               delta=self.tolerance)

        # mean vector length
        self.assertAlmostEqual(r_1, 1, delta=self.tolerance)
        self.assertAlmostEqual(r_2, 0, delta=self.tolerance)


class AngularDifferenceTestCase(unittest.TestCase):
    def setUp(self):
        self.tolerance = 1e-15

    def testADiff_ABS_AlphaMinusBeta_SmallerPi(self):
        adiff_1 = elephant.phase_analysis.phase_difference(0.8 * np.pi,
                                                             0.6 * np.pi)
        self.assertAlmostEqual(adiff_1, 0.2*np.pi, delta=self.tolerance)
        adiff_2 = elephant.phase_analysis.phase_difference(0.6 * np.pi,
                                                             0.8 * np.pi)
        self.assertAlmostEqual(adiff_2, -0.2*np.pi, delta=self.tolerance)
        adiff_3 = elephant.phase_analysis.phase_difference(0.2 * np.pi,
                                                             -0.2 * np.pi)
        self.assertAlmostEqual(adiff_3, 0.4 * np.pi, delta=self.tolerance)
        adiff_4 = elephant.phase_analysis.phase_difference(-0.2 * np.pi,
                                                             0.2 * np.pi)
        self.assertAlmostEqual(adiff_4, -0.4 * np.pi, delta=self.tolerance)

    def testADiff_ABS_AlphaMinusBeta_GreaterPi(self):
        adiff_1 = elephant.phase_analysis.phase_difference(0.8 * np.pi,
                                                             -0.8 * np.pi)
        self.assertAlmostEqual(adiff_1, -0.4 * np.pi, delta=self.tolerance)
        adiff_2 = elephant.phase_analysis.phase_difference(-0.8 * np.pi,
                                                             0.8 * np.pi)
        self.assertAlmostEqual(adiff_2, 0.4 * np.pi, delta=self.tolerance)
        adiff_3 = elephant.phase_analysis.phase_difference(0.3 * np.pi,
                                                             -0.8 * np.pi)
        self.assertAlmostEqual(adiff_3, -0.9 * np.pi, delta=self.tolerance)
        adiff_4 = elephant.phase_analysis.phase_difference(-0.8 * np.pi,
                                                             0.3 * np.pi)
        self.assertAlmostEqual(adiff_4, 0.9 * np.pi, delta=self.tolerance)

    def testADiff_in_range_MinusPi_and_Pi(self):
        sign_1 = 1 if np.random.random(1) < 0.5 else -1
        sign_2 = 1 if np.random.random(1) < 0.5 else -1
        alpha = sign_1 * np.random.random(1) * np.pi
        beta = sign_2 * np.random.random(1) * np.pi
        adiff = elephant.phase_analysis.phase_difference(alpha, beta)
        self.assertTrue(-np.pi <= adiff <= np.pi)


class PhaseLockingValueTestCase(unittest.TestCase):
    def setUp(self):
        self.tolerance = 1e-15
        self.phase_shift = np.pi/4
        self.num_time_points = 1000
        # time in ms
        t_start = 0
        t_stop = 1000
        self.time = np.linspace(t_start, t_stop, self.num_time_points)
        # frequency in Hz
        self.frequency = 1
        self.num_trials = 100
        # create discrete representations of the phases of sin() and cos()
        one_sine_trial = np.array((2 * np.pi * self.frequency * self.time)
                                  % (2 * np.pi))
        # change phases range from [0, 2pi] to [-pi, pi]
        one_sine_trial = np.where(one_sine_trial <= np.pi, one_sine_trial,
                                  one_sine_trial - 2*np.pi)
        self.signal_x_sine = np.empty([self.num_trials, self.num_time_points])
        self.signal_x_sine[:] = one_sine_trial

        one_cos_trial = np.array((2 * np.pi * self.frequency * self.time
                                  + np.pi/2) % (2 * np.pi))
        # change phases range from [0, 2pi] to [-pi, pi]
        one_cos_trial = np.where(one_cos_trial <= np.pi, one_cos_trial,
                                 one_cos_trial - 2*np.pi)
        self.signal_y_cos = np.empty([self.num_trials, self.num_time_points])
        self.signal_y_cos[:] = one_cos_trial

        # create phase-shifted trials of signal_x_sine,
        # to create later phase differences, which vary a lot across trials
        self.shifted_signal_x = np.copy(self.signal_x_sine)
        for i, trial in enumerate(self.shifted_signal_x):
            trial += i/self.num_trials * 2 * np.pi
        # change phases range from [0, 2pi] to [-pi, pi]
        self.shifted_signal_x = \
            np.where(self.shifted_signal_x <= np.pi, self.shifted_signal_x,
                     self.shifted_signal_x - 2*np.pi)

        # print(f"sine: {self.signal_x_sine}")
        # print(f"cos: {self.signal_y_cos}")
        # print(f"shifted sine: {self.shifted_signal_x}")

    def testPhaseLockingValue_sineMinusSine(self):
        # example 1: sine minus sine
        list1_plv_t = \
            elephant.phase_analysis.phase_locking_value(self.signal_x_sine,
                                                        self.signal_x_sine)
        for i in range(len(list1_plv_t)):
            plv_r_i = list1_plv_t[i]
            self.assertAlmostEqual(plv_r_i, 1, delta=self.tolerance)

    def testPhaseLockingValue_sineMinusCos(self):
        # example 2: sine minus cos
        list2_plv_t = elephant.phase_analysis.phase_locking_value(
            self.signal_x_sine, self.signal_y_cos)
        for i in range(len(list2_plv_t)):
            plv_r_i = list2_plv_t[i]
            self.assertAlmostEqual(plv_r_i, 1, delta=self.tolerance)

    def testPhaseLockingValue_shuffledSineMinusShuffledSine(self):
        # example 3: shuffled sine minus shuffled sine
        list3_plv_t = elephant.phase_analysis.phase_locking_value(
            self.signal_x_sine, self.shifted_signal_x)
        for i in range(len(list3_plv_t)):
            plv_r_i = list3_plv_t[i]
            self.assertAlmostEqual(plv_r_i, 0, delta=self.tolerance)


if __name__ == '__main__':
    unittest.main()
