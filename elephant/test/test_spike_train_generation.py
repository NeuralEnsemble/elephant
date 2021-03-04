# -*- coding: utf-8 -*-
"""
Unit tests for the spike_train_generation module.

:copyright: Copyright 2014-2020 by the Elephant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
"""

from __future__ import division

import os
import unittest
import warnings

import neo
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_allclose
import quantities as pq
from scipy.stats import expon
from scipy.stats import kstest, poisson

import elephant.spike_train_generation as stgen
from elephant.statistics import isi, instantaneous_rate
from elephant import kernels


def pdiff(a, b):
    """Difference between a and b as a fraction of a

    i.e. abs((a - b)/a)
    """
    return abs((a - b) / a)


class AnalogSignalThresholdDetectionTestCase(unittest.TestCase):

    def setUp(self):
        # Load membrane potential simulated using Brian2
        # according to make_spike_extraction_test_data.py.
        curr_dir = os.path.dirname(os.path.realpath(__file__))
        raw_data_file_loc = os.path.join(
            curr_dir, 'spike_extraction_test_data.txt')
        raw_data = []
        with open(raw_data_file_loc, 'r') as f:
            for x in f.readlines():
                raw_data.append(float(x))
        self.vm = neo.AnalogSignal(
            raw_data, units=pq.V, sampling_period=0.1 * pq.ms)
        self.true_time_stamps = [0.0123, 0.0354, 0.0712, 0.1191, 0.1694,
                                 0.2200, 0.2711] * pq.s

    def test_threshold_detection(self):
        # Test whether spikes are extracted at the correct times from
        # an analog signal.

        spike_train = stgen.threshold_detection(self.vm)
        try:
            len(spike_train)
        # Handles an error in Neo related to some zero length
        # spike trains being treated as unsized objects.
        except TypeError:
            warnings.warn(
                ("The spike train may be an unsized object. This may be"
                 " related to an issue in Neo with some zero-length SpikeTrain"
                 " objects. Bypassing this by creating an empty SpikeTrain"
                 " object."))
            spike_train = neo.SpikeTrain([], t_start=spike_train.t_start,
                                         t_stop=spike_train.t_stop,
                                         units=spike_train.units)

        # Does threshold_detection gives the correct number of spikes?
        self.assertEqual(len(spike_train), len(self.true_time_stamps))
        # Does threshold_detection gives the correct times for the spikes?
        try:
            assert_array_almost_equal(spike_train, self.true_time_stamps)
        except AttributeError:  # If numpy version too old to have allclose
            self.assertTrue(np.array_equal(spike_train, self.true_time_stamps))

    def test_peak_detection_threshold(self):
        # Test for empty SpikeTrain when threshold is too high
        result = stgen.threshold_detection(self.vm, threshold=30 * pq.mV)
        self.assertEqual(len(result), 0)


class AnalogSignalPeakDetectionTestCase(unittest.TestCase):

    def setUp(self):
        curr_dir = os.path.dirname(os.path.realpath(__file__))
        raw_data_file_loc = os.path.join(
            curr_dir, 'spike_extraction_test_data.txt')
        raw_data = []
        with open(raw_data_file_loc, 'r') as f:
            for x in f.readlines():
                raw_data.append(float(x))
        self.vm = neo.AnalogSignal(
            raw_data, units=pq.V, sampling_period=0.1 * pq.ms)
        self.true_time_stamps = [0.0124, 0.0354, 0.0713, 0.1192, 0.1695,
                                 0.2201, 0.2711] * pq.s

    def test_peak_detection_time_stamps(self):
        # Test with default arguments
        result = stgen.peak_detection(self.vm)
        self.assertEqual(len(self.true_time_stamps), len(result))
        self.assertIsInstance(result, neo.core.SpikeTrain)

        try:
            assert_array_almost_equal(result, self.true_time_stamps)
        except AttributeError:
            self.assertTrue(np.array_equal(result, self.true_time_stamps))

    def test_peak_detection_threshold(self):
        # Test for empty SpikeTrain when threshold is too high
        result = stgen.peak_detection(self.vm, threshold=30 * pq.mV)
        self.assertEqual(len(result), 0)


class AnalogSignalSpikeExtractionTestCase(unittest.TestCase):

    def setUp(self):
        curr_dir = os.path.dirname(os.path.realpath(__file__))
        raw_data_file_loc = os.path.join(
            curr_dir, 'spike_extraction_test_data.txt')
        raw_data = []
        with open(raw_data_file_loc, 'r') as f:
            for x in f.readlines():
                raw_data.append(float(x))
        self.vm = neo.AnalogSignal(
            raw_data, units=pq.V, sampling_period=0.1 * pq.ms)
        self.first_spike = np.array([-0.04084546, -0.03892033, -0.03664779,
                                     -0.03392689, -0.03061474, -0.02650277,
                                     -0.0212756, -0.01443531, -0.00515365,
                                     0.00803962, 0.02797951, -0.07,
                                     -0.06974495, -0.06950466, -0.06927778,
                                     -0.06906314, -0.06885969, -0.06866651,
                                     -0.06848277, -0.06830773, -0.06814071,
                                     -0.06798113, -0.06782843, -0.06768213,
                                     -0.06754178, -0.06740699, -0.06727737,
                                     -0.06715259, -0.06703235, -0.06691635])

    def test_spike_extraction_waveform(self):
        spike_train = stgen.spike_extraction(self.vm.reshape(-1),
                                             interval=(-1 * pq.ms, 2 * pq.ms))
        try:
            assert_array_almost_equal(
                spike_train.waveforms[0][0].magnitude.reshape(-1),
                self.first_spike)
        except AttributeError:
            self.assertTrue(
                np.array_equal(spike_train.waveforms[0][0].magnitude,
                               self.first_spike))


class HomogeneousPoissonProcessTestCase(unittest.TestCase):

    def test_statistics(self):
        # This is a statistical test that has a non-zero chance of failure
        # during normal operation. Thus, we set the random seed to a value that
        # creates a realization passing the test.

        for rate in [123.0 * pq.Hz, 0.123 * pq.kHz]:
            for t_stop in [2345 * pq.ms, 2.345 * pq.s]:
                # zero refractory period should act as no refractory period
                for refractory_period in (None, 0 * pq.ms):
                    np.random.seed(seed=12345)
                    spiketrain = stgen.homogeneous_poisson_process(
                        rate, t_stop=t_stop,
                        refractory_period=refractory_period)
                    intervals = isi(spiketrain)

                    expected_mean_isi = 1. / rate.simplified
                    self.assertAlmostEqual(
                        expected_mean_isi.magnitude,
                        intervals.mean().simplified.magnitude,
                        places=3)

                    expected_first_spike = 0 * pq.ms
                    self.assertLess(
                        spiketrain[0] - expected_first_spike,
                        7 * expected_mean_isi)

                    expected_last_spike = t_stop
                    self.assertLess(expected_last_spike -
                                    spiketrain[-1], 7 * expected_mean_isi)

                    # Kolmogorov-Smirnov test
                    D, p = kstest(
                        intervals.rescale(t_stop.units),
                        "expon",
                        # args are (loc, scale)
                        args=(0,
                              expected_mean_isi.rescale(t_stop.units)),
                        alternative='two-sided')
                    self.assertGreater(p, 0.001)
                    self.assertLess(D, 0.12)

    def test_zero_refractory_period(self):
        rate = 10 * pq.Hz
        t_stop = 20 * pq.s
        np.random.seed(27)
        sp1 = stgen.homogeneous_poisson_process(rate, t_stop=t_stop,
                                                as_array=True)
        np.random.seed(27)
        sp2 = stgen.homogeneous_poisson_process(rate, t_stop=t_stop,
                                                refractory_period=0 * pq.ms,
                                                as_array=True)
        assert_array_almost_equal(sp1, sp2)

    def test_t_start_and_t_stop(self):
        rate = 10 * pq.Hz
        t_start = 17 * pq.ms
        t_stop = 2 * pq.s
        for refractory_period in (None, 3 * pq.ms):
            spiketrain = stgen.homogeneous_poisson_process(
                rate=rate, t_start=t_start, t_stop=t_stop,
                refractory_period=refractory_period)
            self.assertEqual(spiketrain.t_start, t_start)
            self.assertEqual(spiketrain.t_stop, t_stop)

    def test_zero_rate(self):
        for refractory_period in (None, 3 * pq.ms):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # RuntimeWarning: divide by zero encountered in true_divide
                # mean_interval = 1 / rate.magnitude, when rate == 0 Hz.
                sp = stgen.homogeneous_poisson_process(
                    rate=0 * pq.Hz, t_stop=10 * pq.s,
                    refractory_period=refractory_period)
                self.assertEqual(sp.size, 0)

    def test_nondecrease_spike_times(self):
        for refractory_period in (None, 3 * pq.ms):
            np.random.seed(27)
            spiketrain = stgen.homogeneous_poisson_process(
                rate=10 * pq.Hz, t_stop=1000 * pq.s,
                refractory_period=refractory_period)
            diffs = np.diff(spiketrain.times)
            self.assertTrue((diffs >= 0).all())

    def test_compare_with_as_array(self):
        rate = 10 * pq.Hz
        t_stop = 10 * pq.s
        for refractory_period in (None, 3 * pq.ms):
            np.random.seed(27)
            spiketrain = stgen.homogeneous_poisson_process(
                rate=rate, t_stop=t_stop, refractory_period=refractory_period)
            self.assertIsInstance(spiketrain, neo.SpikeTrain)
            np.random.seed(27)
            spiketrain_array = stgen.homogeneous_poisson_process(
                rate=rate, t_stop=t_stop, refractory_period=refractory_period,
                as_array=True)
            # don't check with isinstance: Quantity is a subclass of np.ndarray
            self.assertTrue(isinstance(spiketrain_array, np.ndarray))
            assert_array_almost_equal(spiketrain.times.magnitude,
                                      spiketrain_array)

    def test_effective_rate_refractory_period(self):
        np.random.seed(27)
        rate_expected = 10 * pq.Hz
        refractory_period = 90 * pq.ms  # 10 ms of effective ISI
        spiketrain = stgen.homogeneous_poisson_process(
            rate_expected, t_stop=1000 * pq.s,
            refractory_period=refractory_period)
        rate_obtained = len(spiketrain) / spiketrain.t_stop
        rate_obtained = rate_obtained.simplified
        self.assertAlmostEqual(rate_expected.simplified,
                               rate_obtained.simplified, places=1)
        intervals = isi(spiketrain)
        isi_mean_expected = 1. / rate_expected
        self.assertAlmostEqual(isi_mean_expected.simplified,
                               intervals.mean().simplified, places=3)

    def test_invalid(self):
        rate = 10 * pq.Hz
        for refractory_period in (None, 3 * pq.ms):
            # t_stop < t_start
            hpp = stgen.homogeneous_poisson_process
            self.assertRaises(
                ValueError, hpp, rate=rate, t_start=5 * pq.ms,
                t_stop=1 * pq.ms, refractory_period=refractory_period)

            # no units provided for rate, t_stop
            self.assertRaises(ValueError, hpp, rate=10,
                              refractory_period=refractory_period)
            self.assertRaises(ValueError, hpp, rate=rate, t_stop=5,
                              refractory_period=refractory_period)

        # no units provided for refractory_period
        self.assertRaises(ValueError, hpp, rate=rate, refractory_period=2)


class InhomogeneousGammaTestCase(unittest.TestCase):
    def setUp(self):
        rate_list = [[20]] * 1000 + [[200]] * 1000
        self.rate_profile = neo.AnalogSignal(
            rate_list * pq.Hz, sampling_period=0.001 * pq.s)
        rate_0 = [[0]] * 1000
        self.rate_profile_0 = neo.AnalogSignal(
            rate_0 * pq.Hz, sampling_period=0.001 * pq.s)
        rate_negative = [[-1]] * 1000
        self.rate_profile_negative = neo.AnalogSignal(
            rate_negative * pq.Hz, sampling_period=0.001 * pq.s)

    def test_statistics(self):
        # This is a statistical test that has a non-zero chance of failure
        # during normal operation. Thus, we set the random seed to a value that
        # creates a realization passing the test.
        np.random.seed(seed=12345)

        shape_factor = 2.5

        for rate in [self.rate_profile, self.rate_profile.rescale(pq.kHz)]:
            spiketrain = stgen.inhomogeneous_gamma_process(
                rate, shape_factor=shape_factor)
            intervals = isi(spiketrain)

            # Computing expected statistics and percentiles
            expected_spike_count = (np.sum(
                rate) * rate.sampling_period).simplified
            percentile_count = poisson.ppf(.999, expected_spike_count)
            expected_min_isi = (1 / np.min(rate))
            expected_max_isi = (1 / np.max(rate))
            percentile_min_isi = expon.ppf(.999, expected_min_isi)
            percentile_max_isi = expon.ppf(.999, expected_max_isi)

            # Testing (each should fail 1 every 1000 times)
            self.assertLess(spiketrain.size, percentile_count)
            self.assertLess(np.min(intervals), percentile_min_isi)
            self.assertLess(np.max(intervals), percentile_max_isi)

            # Testing t_start t_stop
            self.assertEqual(rate.t_stop, spiketrain.t_stop)
            self.assertEqual(rate.t_start, spiketrain.t_start)

        # Testing type
        spiketrain_as_array = stgen.inhomogeneous_gamma_process(
            rate, shape_factor=shape_factor, as_array=True)
        self.assertTrue(isinstance(spiketrain_as_array, np.ndarray))
        self.assertTrue(isinstance(spiketrain, neo.SpikeTrain))

        # check error if rate has wrong format
        self.assertRaises(
            ValueError, stgen.inhomogeneous_gamma_process,
            rate=[0.1, 2.],
            shape_factor=shape_factor)

        # check error if negative values in rate
        self.assertRaises(
            ValueError, stgen.inhomogeneous_gamma_process,
            rate=neo.AnalogSignal([-0.1, 10.] * pq.Hz,
                                  sampling_period=0.001 * pq.s),
            shape_factor=shape_factor)

        # check error if rate is empty
        self.assertRaises(
            ValueError, stgen.inhomogeneous_gamma_process,
            rate=neo.AnalogSignal([] * pq.Hz, sampling_period=0.001 * pq.s),
            shape_factor=shape_factor)

    def test_recovered_firing_rate_profile(self):
        np.random.seed(54)
        t_start = 0 * pq.s
        t_stop = 4 * np.round(np.pi, decimals=3) * pq.s  # 2 full periods
        sampling_period = 0.001 * pq.s

        # an arbitrary rate profile
        profile = 0.5 * (1 + np.sin(np.arange(t_start.item(), t_stop.item(),
                                              sampling_period.item())))

        time_generation = 0
        n_trials = 200
        rtol = 0.05  # 5% of deviation allowed
        kernel = kernels.RectangularKernel(sigma=0.25 * pq.s)
        for rate in (10 * pq.Hz, 100 * pq.Hz):
            rate_profile = neo.AnalogSignal(rate * profile,
                                            sampling_period=sampling_period)
            # the recovered firing rate profile should not depend on the
            # shape factor; here we test float and integer values of the shape
            # factor: the method supports float values that is not trivial
            # for inhomogeneous gamma process generation
            for shape_factor in (1, 2.5, 10.):

                spiketrains = \
                    [stgen.inhomogeneous_gamma_process(
                        rate_profile, shape_factor=shape_factor)
                     for _ in range(n_trials)]
                rate_recovered = instantaneous_rate(
                    spiketrains,
                    sampling_period=sampling_period,
                    kernel=kernel,
                    t_start=t_start,
                    t_stop=t_stop, trim=True).sum(axis=1) / n_trials

                rate_recovered = rate_recovered.flatten().magnitude
                trim = (rate_profile.shape[0] - rate_recovered.shape[0]) // 2
                rate_profile_valid = rate_profile.magnitude.squeeze()
                rate_profile_valid = rate_profile_valid[trim: -trim - 1]
                assert_allclose(rate_recovered, rate_profile_valid,
                                rtol=0, atol=rtol * rate.item())


class InhomogeneousPoissonProcessTestCase(unittest.TestCase):
    def setUp(self):
        rate_list = [[20]] * 1000 + [[200]] * 1000
        self.rate_profile = neo.AnalogSignal(
            rate_list * pq.Hz, sampling_period=0.001 * pq.s)
        rate_0 = [[0]] * 1000
        self.rate_profile_0 = neo.AnalogSignal(
            rate_0 * pq.Hz, sampling_period=0.001 * pq.s)
        rate_negative = [[-1]] * 1000
        self.rate_profile_negative = neo.AnalogSignal(
            rate_negative * pq.Hz, sampling_period=0.001 * pq.s)

    def test_statistics(self):
        # This is a statistical test that has a non-zero chance of failure
        # during normal operation. Thus, we set the random seed to a value that
        # creates a realization passing the test.
        np.random.seed(seed=12345)

        for rate in (self.rate_profile, self.rate_profile.rescale(pq.kHz)):
            for refractory_period in (3 * pq.ms, None):
                spiketrain = stgen.inhomogeneous_poisson_process(
                    rate, refractory_period=refractory_period)
                intervals = isi(spiketrain)

                # Computing expected statistics and percentiles
                expected_spike_count = (np.sum(
                    rate) * rate.sampling_period).simplified
                percentile_count = poisson.ppf(.999, expected_spike_count)
                expected_min_isi = (1 / np.min(rate))
                expected_max_isi = (1 / np.max(rate))
                percentile_min_isi = expon.ppf(.999, expected_min_isi)
                percentile_max_isi = expon.ppf(.999, expected_max_isi)

                # Check that minimal ISI is greater than the refractory_period
                if refractory_period is not None:
                    self.assertGreater(np.min(intervals), refractory_period)

                # Testing (each should fail 1 every 1000 times)
                self.assertLess(spiketrain.size, percentile_count)
                self.assertLess(np.min(intervals), percentile_min_isi)
                self.assertLess(np.max(intervals), percentile_max_isi)

                # Testing t_start t_stop
                self.assertEqual(rate.t_stop, spiketrain.t_stop)
                self.assertEqual(rate.t_start, spiketrain.t_start)

        # Testing type
        spiketrain_as_array = stgen.inhomogeneous_poisson_process(
            rate, as_array=True)
        self.assertTrue(isinstance(spiketrain_as_array, np.ndarray))
        self.assertTrue(isinstance(spiketrain, neo.SpikeTrain))

        # Testing type for refractory period
        refractory_period = 3 * pq.ms
        spiketrain = stgen.inhomogeneous_poisson_process(
            rate, refractory_period=refractory_period)
        spiketrain_as_array = stgen.inhomogeneous_poisson_process(
            rate, as_array=True, refractory_period=refractory_period)
        self.assertTrue(isinstance(spiketrain_as_array, np.ndarray))
        self.assertTrue(isinstance(spiketrain, neo.SpikeTrain))

        # Check that to high refractory period raises error
        self.assertRaises(
            ValueError, stgen.inhomogeneous_poisson_process,
            self.rate_profile,
            refractory_period=1000 * pq.ms)

    def test_effective_rate_refractory_period(self):
        np.random.seed(27)
        rate_expected = 10 * pq.Hz
        refractory_period = 90 * pq.ms  # 10 ms of effective ISI
        rates = neo.AnalogSignal(np.repeat(rate_expected, 1000), units=pq.Hz,
                                 t_start=0 * pq.ms, sampling_rate=1 * pq.Hz)
        spiketrain = stgen.inhomogeneous_poisson_process(
            rates, refractory_period=refractory_period)
        rate_obtained = len(spiketrain) / spiketrain.t_stop
        self.assertAlmostEqual(rate_expected, rate_obtained.simplified,
                               places=1)
        intervals_inhomo = isi(spiketrain)
        isi_mean_expected = 1. / rate_expected
        self.assertAlmostEqual(isi_mean_expected.simplified,
                               intervals_inhomo.mean().simplified,
                               places=3)

    def test_zero_rate(self):
        for refractory_period in (3 * pq.ms, None):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # RuntimeWarning: divide by zero encountered in true_divide
                # mean_interval = 1 / rate.magnitude, when rate == 0 Hz.
                spiketrain = stgen.inhomogeneous_poisson_process(
                    self.rate_profile_0, refractory_period=refractory_period)
            self.assertEqual(spiketrain.size, 0)

    def test_negative_rates(self):
        for refractory_period in (3 * pq.ms, None):
            self.assertRaises(
                ValueError, stgen.inhomogeneous_poisson_process,
                self.rate_profile_negative,
                refractory_period=refractory_period)


class HomogeneousGammaProcessTestCase(unittest.TestCase):

    def test_statistics(self):
        # This is a statistical test that has a non-zero chance of failure
        # during normal operation. Thus, we set the random seed to a value that
        # creates a realization passing the test.
        np.random.seed(seed=12345)

        a = 3.0
        for b in (67.0 * pq.Hz, 0.067 * pq.kHz):
            for t_stop in (2345 * pq.ms, 2.345 * pq.s):
                spiketrain = stgen.homogeneous_gamma_process(
                    a, b, t_stop=t_stop)
                intervals = isi(spiketrain)

                expected_spike_count = int((b / a * t_stop).simplified)
                # should fail about 1 time in 1000
                self.assertLess(
                    pdiff(expected_spike_count, spiketrain.size), 0.25)

                expected_mean_isi = (a / b).rescale(pq.ms)
                self.assertLess(
                    pdiff(expected_mean_isi, intervals.mean()), 0.3)

                expected_first_spike = 0 * pq.ms
                self.assertLess(
                    spiketrain[0] - expected_first_spike,
                    4 * expected_mean_isi)

                expected_last_spike = t_stop
                self.assertLess(expected_last_spike -
                                spiketrain[-1], 4 * expected_mean_isi)

                # Kolmogorov-Smirnov test
                D, p = kstest(intervals.rescale(t_stop.units),
                              "gamma",
                              # args are (a, loc, scale)
                              args=(a, 0, (1 / b).rescale(t_stop.units)),
                              alternative='two-sided')
                self.assertGreater(p, 0.001)
                self.assertLess(D, 0.25)

    def test_compare_with_as_array(self):
        a = 3.
        b = 10 * pq.Hz
        np.random.seed(27)
        spiketrain = stgen.homogeneous_gamma_process(a=a, b=b)
        self.assertIsInstance(spiketrain, neo.SpikeTrain)
        np.random.seed(27)
        spiketrain_array = stgen.homogeneous_gamma_process(a=a, b=b,
                                                           as_array=True)
        # don't check with isinstance: pq.Quantity is a subclass of np.ndarray
        self.assertTrue(isinstance(spiketrain_array, np.ndarray))
        assert_array_almost_equal(spiketrain.times.magnitude, spiketrain_array)


class _n_poisson_TestCase(unittest.TestCase):

    def setUp(self):
        self.n = 4
        self.rate = 10 * pq.Hz
        self.rates = np.arange(1, self.n + 1) * pq.Hz
        self.t_stop = 10000 * pq.ms

    def test_poisson(self):

        # Check the output types for input rate + n number of neurons
        pp = stgen._n_poisson(
            rate=self.rate,
            t_stop=self.t_stop,
            n_spiketrains=self.n)
        self.assertIsInstance(pp, list)
        self.assertIsInstance(pp[0], neo.core.spiketrain.SpikeTrain)
        self.assertEqual(pp[0].simplified.units, 1000 * pq.ms)
        self.assertEqual(len(pp), self.n)

        # Check the output types for input list of rates
        pp = stgen._n_poisson(rate=self.rates, t_stop=self.t_stop)
        self.assertIsInstance(pp, list)
        self.assertIsInstance(pp[0], neo.core.spiketrain.SpikeTrain)
        self.assertEqual(pp[0].simplified.units, 1000 * pq.ms)
        self.assertEqual(len(pp), self.n)

    def test_poisson_error(self):

        # Dimensionless rate
        self.assertRaises(
            ValueError, stgen._n_poisson, rate=5, t_stop=self.t_stop)
        # Negative rate
        self.assertRaises(
            ValueError, stgen._n_poisson, rate=-5 * pq.Hz, t_stop=self.t_stop)
        # Negative value when rate is a list
        self.assertRaises(
            ValueError, stgen._n_poisson, rate=[-5, 3] * pq.Hz,
            t_stop=self.t_stop)
        # Negative n
        self.assertRaises(
            ValueError, stgen._n_poisson, rate=self.rate, t_stop=self.t_stop,
            n_spiketrains=-1)
        # t_start>t_stop
        self.assertRaises(
            ValueError, stgen._n_poisson, rate=self.rate, t_start=4 * pq.ms,
            t_stop=3 * pq.ms, n_spiketrains=3)


class singleinteractionprocess_TestCase(unittest.TestCase):
    def setUp(self):
        self.n = 4
        self.rate = 10 * pq.Hz
        self.rates = np.arange(1, self.n + 1) * pq.Hz
        self.t_stop = 10000 * pq.ms
        self.rate_c = 1 * pq.Hz

    def test_sip(self):

        # Generate an example SIP mode
        sip, coinc = stgen.single_interaction_process(
            n_spiketrains=self.n, t_stop=self.t_stop, rate=self.rate,
            coincidence_rate=self.rate_c, return_coincidences=True)

        # Check the output types
        self.assertEqual(type(sip), list)
        self.assertEqual(type(sip[0]), neo.core.spiketrain.SpikeTrain)
        self.assertEqual(type(coinc[0]), neo.core.spiketrain.SpikeTrain)
        self.assertEqual(sip[0].simplified.units, 1000 * pq.ms)
        self.assertEqual(coinc[0].simplified.units, 1000 * pq.ms)

        # Check the output length
        self.assertEqual(len(sip), self.n)
        self.assertEqual(
            len(coinc[0]), (self.rate_c * self.t_stop).simplified.magnitude)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Generate an example SIP mode giving a list of rates as imput
            sip, coinc = stgen.single_interaction_process(
                t_stop=self.t_stop, rate=self.rates,
                coincidence_rate=self.rate_c, return_coincidences=True)

        # Check the output types
        self.assertEqual(type(sip), list)
        self.assertEqual(type(sip[0]), neo.core.spiketrain.SpikeTrain)
        self.assertEqual(type(coinc[0]), neo.core.spiketrain.SpikeTrain)
        self.assertEqual(sip[0].simplified.units, 1000 * pq.ms)
        self.assertEqual(coinc[0].simplified.units, 1000 * pq.ms)

        # Check the output length
        self.assertEqual(len(sip), self.n)
        self.assertEqual(
            len(coinc[0]),
            (self.rate_c * self.t_stop).rescale(pq.dimensionless))

        # Generate an example SIP mode stochastic number of coincidences
        sip = stgen.single_interaction_process(
            n_spiketrains=self.n,
            t_stop=self.t_stop,
            rate=self.rate,
            coincidence_rate=self.rate_c,
            coincidences='stochastic',
            return_coincidences=False)

        # Check the output types
        self.assertEqual(type(sip), list)
        self.assertEqual(type(sip[0]), neo.core.spiketrain.SpikeTrain)
        self.assertEqual(sip[0].simplified.units, 1000 * pq.ms)

    def test_sip_error(self):
        # Negative rate
        self.assertRaises(
            ValueError, stgen.single_interaction_process, n_spiketrains=self.n,
            rate=-5 * pq.Hz,
            coincidence_rate=self.rate_c, t_stop=self.t_stop)
        # Negative coincidence rate
        self.assertRaises(
            ValueError, stgen.single_interaction_process, n_spiketrains=self.n,
            rate=self.rate, coincidence_rate=-3 * pq.Hz, t_stop=self.t_stop)
        # Negative value when rate is a list
        self.assertRaises(
            ValueError, stgen.single_interaction_process, n_spiketrains=self.n,
            rate=[-5, 3, 4, 2] * pq.Hz, coincidence_rate=self.rate_c,
            t_stop=self.t_stop)
        # Negative n
        self.assertRaises(
            ValueError, stgen.single_interaction_process, n_spiketrains=-1,
            rate=self.rate, coincidence_rate=self.rate_c, t_stop=self.t_stop)
        # Rate_c < rate
        self.assertRaises(
            ValueError,
            stgen.single_interaction_process,
            n_spiketrains=self.n,
            rate=self.rate,
            coincidence_rate=self.rate + 1 * pq.Hz,
            t_stop=self.t_stop)


class cppTestCase(unittest.TestCase):
    def test_cpp_hom(self):
        # testing output with generic inputs
        amplitude_distribution = np.array([0, .9, .1])
        t_stop = 10 * 1000 * pq.ms
        t_start = 5 * 1000 * pq.ms
        rate = 3 * pq.Hz
        cpp_hom = stgen.cpp(rate, amplitude_distribution,
                            t_stop, t_start=t_start)
        # testing the ouput formats
        self.assertEqual(
            [type(train) for train in cpp_hom],
            [neo.SpikeTrain] * len(cpp_hom))
        self.assertEqual(cpp_hom[0].simplified.units, 1000 * pq.ms)
        self.assertEqual(type(cpp_hom), list)
        # testing quantities format of the output
        self.assertEqual(
            [train.simplified.units for train in cpp_hom],
            [1000 * pq.ms] * len(cpp_hom))
        # testing output t_start t_stop
        for st in cpp_hom:
            self.assertEqual(st.t_stop, t_stop)
            self.assertEqual(st.t_start, t_start)
        self.assertEqual(len(cpp_hom), len(amplitude_distribution) - 1)

        # testing the units
        t_stop = 10000 * pq.ms
        t_start = 5 * 1000 * pq.ms
        rate = 3 * pq.Hz
        cpp_unit = stgen.cpp(rate, amplitude_distribution,
                             t_stop, t_start=t_start)

        self.assertEqual(cpp_unit[0].units, t_stop.units)
        self.assertEqual(cpp_unit[0].t_stop.units, t_stop.units)
        self.assertEqual(cpp_unit[0].t_start.units, t_stop.units)

        # testing output without copy of spikes
        amplitude_distribution = np.array([1])
        t_stop = 10 * 1000 * pq.ms
        t_start = 5 * 1000 * pq.ms
        rate = 3 * pq.Hz
        cpp_hom_empty = stgen.cpp(
            rate, amplitude_distribution, t_stop, t_start=t_start)

        self.assertEqual(
            [len(train) for train in cpp_hom_empty], [0] * len(cpp_hom_empty))

        # testing output with rate equal to 0
        amplitude_distribution = np.array([0, .9, .1])
        t_stop = 10 * 1000 * pq.ms
        t_start = 5 * 1000 * pq.ms
        rate = 0 * pq.Hz
        cpp_hom_empty_r = stgen.cpp(
            rate, amplitude_distribution, t_stop, t_start=t_start)
        self.assertEqual(
            [len(train) for train in cpp_hom_empty_r], [0] * len(
                cpp_hom_empty_r))

        # testing output with same spike trains in output
        amplitude_distribution = np.array([0., 0., 1.])
        t_stop = 10 * 1000 * pq.ms
        t_start = 5 * 1000 * pq.ms
        rate = 3 * pq.Hz
        cpp_hom_eq = stgen.cpp(
            rate, amplitude_distribution, t_stop, t_start=t_start)

        self.assertTrue(
            np.allclose(cpp_hom_eq[0].magnitude, cpp_hom_eq[1].magnitude))

    def test_cpp_hom_errors(self):
        # testing raises of ValueError (wrong inputs)
        # testing empty amplitude
        self.assertRaises(
            ValueError,
            stgen.cpp,
            amplitude_distribution=[],
            t_stop=10 * 1000 * pq.ms,
            rate=3 * pq.Hz)

        # testing sum of amplitude>1
        self.assertRaises(
            ValueError,
            stgen.cpp,
            amplitude_distribution=[1, 1, 1],
            t_stop=10 * 1000 * pq.ms,
            rate=3 * pq.Hz)
        # testing negative value in the amplitude
        self.assertRaises(
            ValueError, stgen.cpp, amplitude_distribution=[-1, 1, 1],
            t_stop=10 * 1000 * pq.ms,
            rate=3 * pq.Hz)
        # test negative rate
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Catches RuntimeWarning: invalid value encountered in sqrt
            # number = np.ceil(n + 3 * np.sqrt(n)), when `n` == -3 Hz.
            self.assertRaises(
                ValueError, stgen.cpp, amplitude_distribution=[0, 1, 0],
                t_stop=10 * 1000 * pq.ms,
                rate=-3 * pq.Hz)
        # test wrong unit for rate
        self.assertRaises(
            ValueError,
            stgen.cpp,
            amplitude_distribution=[0, 1, 0],
            t_stop=10 * 1000 * pq.ms,
            rate=3 * 1000 * pq.ms)

        # testing raises of AttributeError (missing input units)
        # Testing missing unit to t_stop
        self.assertRaises(
            ValueError, stgen.cpp, amplitude_distribution=[0, 1, 0], t_stop=10,
            rate=3 * pq.Hz)
        # Testing missing unit to t_start
        self.assertRaises(
            ValueError,
            stgen.cpp,
            amplitude_distribution=[0, 1, 0],
            t_stop=10 * 1000 * pq.ms,
            rate=3 * pq.Hz,
            t_start=3)
        # testing rate missing unit
        self.assertRaises(
            AttributeError, stgen.cpp, amplitude_distribution=[0, 1, 0],
            t_stop=10 * 1000 * pq.ms,
            rate=3)

    def test_cpp_het(self):
        # testing output with generic inputs
        amplitude_distribution = np.array([0, .9, .1])
        t_stop = 10 * 1000 * pq.ms
        t_start = 5 * 1000 * pq.ms
        rate = [3, 4] * pq.Hz
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Catch RuntimeWarning: divide by zero encountered in true_divide
            # mean_interval = 1 / rate.magnitude, when rate == 0 Hz.
            cpp_het = stgen.cpp(rate, amplitude_distribution,
                                t_stop, t_start=t_start)
            # testing the ouput formats
            self.assertEqual(
                [type(train) for train in cpp_het],
                [neo.SpikeTrain] * len(cpp_het))
            self.assertEqual(cpp_het[0].simplified.units, 1000 * pq.ms)
            self.assertEqual(type(cpp_het), list)
            # testing units
            self.assertEqual(
                [train.simplified.units for train in cpp_het],
                [1000 * pq.ms] * len(cpp_het))
            # testing output t_start and t_stop
            for st in cpp_het:
                self.assertEqual(st.t_stop, t_stop)
                self.assertEqual(st.t_start, t_start)
            # testing the number of output spiketrains
            self.assertEqual(len(cpp_het), len(amplitude_distribution) - 1)
            self.assertEqual(len(cpp_het), len(rate))

            # testing the units
            t_stop = 10000 * pq.ms
            t_start = 5 * 1000 * pq.ms
            rate = [3, 4] * pq.Hz
            cpp_unit = stgen.cpp(
                rate, amplitude_distribution, t_stop, t_start=t_start)

            self.assertEqual(cpp_unit[0].units, t_stop.units)
            self.assertEqual(cpp_unit[0].t_stop.units, t_stop.units)
            self.assertEqual(cpp_unit[0].t_start.units, t_stop.units)
            # testing without copying any spikes
            amplitude_distribution = np.array([1, 0, 0])
            t_stop = 10 * 1000 * pq.ms
            t_start = 5 * 1000 * pq.ms
            rate = [3, 4] * pq.Hz
            cpp_het_empty = stgen.cpp(
                rate, amplitude_distribution, t_stop, t_start=t_start)

            self.assertEqual(len(cpp_het_empty[0]), 0)

            # testing output with rate equal to 0
            amplitude_distribution = np.array([0, .9, .1])
            t_stop = 10 * 1000 * pq.ms
            t_start = 5 * 1000 * pq.ms
            rate = [0, 0] * pq.Hz
            cpp_het_empty_r = stgen.cpp(
                rate, amplitude_distribution, t_stop, t_start=t_start)
            self.assertEqual(
                [len(train) for train in cpp_het_empty_r], [0] * len(
                    cpp_het_empty_r))

            # testing completely sync spiketrains
            amplitude_distribution = np.array([0, 0, 1])
            t_stop = 10 * 1000 * pq.ms
            t_start = 5 * 1000 * pq.ms
            rate = [3, 3] * pq.Hz
            cpp_het_eq = stgen.cpp(
                rate, amplitude_distribution, t_stop, t_start=t_start)

            self.assertTrue(np.allclose(
                cpp_het_eq[0].magnitude, cpp_het_eq[1].magnitude))

    def test_cpp_het_err(self):
        # testing raises of ValueError (wrong inputs)
        # testing empty amplitude
        self.assertRaises(
            ValueError,
            stgen.cpp,
            amplitude_distribution=[],
            t_stop=10 * 1000 * pq.ms,
            rate=[3, 4] * pq.Hz)
        # testing sum amplitude>1
        self.assertRaises(
            ValueError,
            stgen.cpp,
            amplitude_distribution=[1, 1, 1],
            t_stop=10 * 1000 * pq.ms,
            rate=[3, 4] * pq.Hz)
        # testing amplitude negative value
        self.assertRaises(
            ValueError, stgen.cpp, amplitude_distribution=[-1, 1, 1],
            t_stop=10 * 1000 * pq.ms,
            rate=[3, 4] * pq.Hz)
        # testing negative rate
        self.assertRaises(ValueError, stgen.cpp, amplitude_distribution=[
            0, 1, 0], t_stop=10 * 1000 * pq.ms, rate=[-3, 4] * pq.Hz)
        # testing empty rate
        self.assertRaises(
            ValueError,
            stgen.cpp,
            amplitude_distribution=[0, 1, 0],
            t_stop=10 * 1000 * pq.ms,
            rate=[] * pq.Hz)
        # testing empty amplitude
        self.assertRaises(
            ValueError,
            stgen.cpp,
            amplitude_distribution=[],
            t_stop=10 * 1000 * pq.ms,
            rate=[3, 4] * pq.Hz)
        # testing different len(A)-1 and len(rate)
        self.assertRaises(
            ValueError,
            stgen.cpp,
            amplitude_distribution=[0, 1],
            t_stop=10 * 1000 * pq.ms,
            rate=[3, 4] * pq.Hz)
        # testing rate with different unit from Hz
        self.assertRaises(
            ValueError,
            stgen.cpp,
            amplitude_distribution=[0, 1],
            t_stop=10 * 1000 * pq.ms,
            rate=[3, 4] * 1000 * pq.ms)
        # Testing analytical constrain between amplitude and rate
        self.assertRaises(
            ValueError,
            stgen.cpp,
            amplitude_distribution=[0, 0, 1],
            t_stop=10 * 1000 * pq.ms,
            rate=[3, 4] * pq.Hz,
            t_start=3)

        # testing raises of AttributeError (missing input units)
        # Testing missing unit to t_stop
        self.assertRaises(
            ValueError, stgen.cpp, amplitude_distribution=[0, 1, 0], t_stop=10,
            rate=[3, 4] * pq.Hz)
        # Testing missing unit to t_start
        self.assertRaises(
            ValueError,
            stgen.cpp,
            amplitude_distribution=[0, 1, 0],
            t_stop=10 * 1000 * pq.ms,
            rate=[3, 4] * pq.Hz,
            t_start=3)
        # Testing missing unit to rate
        self.assertRaises(
            AttributeError, stgen.cpp, amplitude_distribution=[0, 1, 0],
            t_stop=10 * 1000 * pq.ms,
            rate=[3, 4])

    def test_cpp_jttered(self):
        # testing output with generic inputs
        amplitude_distribution = np.array([0, .9, .1])
        t_stop = 10 * 1000 * pq.ms
        t_start = 5 * 1000 * pq.ms
        rate = 3 * pq.Hz
        cpp_shift = stgen.cpp(
            rate,
            amplitude_distribution,
            t_stop,
            t_start=t_start,
            shift=3 * pq.ms)
        # testing the ouput formats
        self.assertEqual(
            [type(train) for train in cpp_shift], [neo.SpikeTrain] * len(
                cpp_shift))
        self.assertEqual(cpp_shift[0].simplified.units, 1000 * pq.ms)
        self.assertEqual(type(cpp_shift), list)
        # testing quantities format of the output
        self.assertEqual(
            [train.simplified.units for train in cpp_shift],
            [1000 * pq.ms] * len(cpp_shift))
        # testing output t_start t_stop
        for spiketrain in cpp_shift:
            self.assertEqual(spiketrain.t_stop, t_stop)
            self.assertEqual(spiketrain.t_start, t_start)
        self.assertEqual(len(cpp_shift), len(amplitude_distribution) - 1)


class HomogeneousPoissonProcessWithRefrPeriodTestCase(unittest.TestCase):

    def test_invalid(self):
        rate = 10 * pq.Hz
        # t_stop < t_start
        hpp = stgen.homogeneous_poisson_process
        self.assertRaises(ValueError, hpp, rate=rate, t_start=5 * pq.ms,
                          t_stop=1 * pq.ms, refractory_period=3 * pq.ms)

        # no units provided
        self.assertRaises(ValueError, hpp, rate=10,
                          refractory_period=3 * pq.ms)
        self.assertRaises(ValueError, hpp, rate=rate, t_stop=5,
                          refractory_period=3 * pq.ms)
        self.assertRaises(ValueError, hpp, rate=rate, refractory_period=2)


if __name__ == '__main__':
    unittest.main()
