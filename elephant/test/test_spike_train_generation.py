# -*- coding: utf-8 -*-
"""
Unit tests for the spike_train_generation module.

:copyright: Copyright 2014-2024 by the Elephant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
"""

from __future__ import division

import os
import unittest
import warnings

import neo
from neo.core.spiketrainlist import SpikeTrainList
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_allclose
import quantities as pq
from scipy.stats import expon, kstest, poisson, variation

from elephant.spike_train_generation import (
    StationaryPoissonProcess,
    threshold_detection,
    peak_detection,
    spike_extraction,
    AbstractPointProcess,
    StationaryGammaProcess,
    StationaryLogNormalProcess,
    NonStationaryPoissonProcess,
    NonStationaryGammaProcess,
    StationaryInverseGaussianProcess,
    _n_poisson,
    single_interaction_process,
    cpp,
    homogeneous_gamma_process,
    homogeneous_poisson_process,
    inhomogeneous_poisson_process,
    inhomogeneous_gamma_process,
)
from elephant.statistics import isi, instantaneous_rate
from elephant import kernels


def pdiff(a, b):
    """Difference between a and b as a fraction of a

    i.e. abs((a - b)/a)
    """
    return abs((a - b) / a)


class ThresholdDetectionTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load membrane potential simulated using Brian2
        # according to make_spike_extraction_test_data.py.
        curr_dir = os.path.dirname(os.path.realpath(__file__))
        raw_data_file_loc = os.path.join(curr_dir, "spike_extraction_test_data.txt")
        raw_data = []
        with open(raw_data_file_loc, "r") as f:
            for x in f.readlines():
                raw_data.append(float(x))
        cls.vm = neo.AnalogSignal(raw_data, units=pq.V, sampling_period=0.1 * pq.ms)
        cls.vm_3d = neo.AnalogSignal(
            np.array([raw_data, raw_data, raw_data]).T,
            units=pq.V,
            sampling_period=0.1 * pq.ms,
        )
        cls.true_time_stamps = [
            0.0123,
            0.0354,
            0.0712,
            0.1191,
            0.1694,
            0.2200,
            0.2711,
        ] * pq.s

    def test_threshold_detection(self):
        # Test whether spikes are extracted at the correct times from
        # an analog signal.

        spike_train = threshold_detection(self.vm)
        try:
            len(spike_train)
        # Handles an error in Neo related to some zero length
        # spike trains being treated as unsized objects.
        except TypeError:
            warnings.warn(
                (
                    "The spike train may be an unsized object. This may be"
                    " related to an issue in Neo with some zero-length SpikeTrain"
                    " objects. Bypassing this by creating an empty SpikeTrain"
                    " object."
                )
            )
            spike_train = neo.SpikeTrain(
                [],
                t_start=spike_train.t_start,
                t_stop=spike_train.t_stop,
                units=spike_train.units,
            )

        # Does threshold_detection gives the correct number of spikes?
        self.assertEqual(len(spike_train), len(self.true_time_stamps))
        # Does threshold_detection gives the correct times for the spikes?
        try:
            assert_array_almost_equal(spike_train, self.true_time_stamps)
        except AttributeError:  # If numpy version too old to have allclose
            self.assertTrue(np.array_equal(spike_train, self.true_time_stamps))

    def test_threshold_detection_threshold(self):
        # Test for empty SpikeTrain when threshold is too high
        result = threshold_detection(self.vm, threshold=30 * pq.mV)
        self.assertEqual(len(result), 0)

    def test_threshold_raise_type_error(self):
        with self.assertRaises(TypeError):
            threshold_detection(self.vm, threshold=30)

    def test_sign_raise_value_error(self):
        with self.assertRaises(ValueError):
            threshold_detection(self.vm, sign="wrong input")

    def test_return_is_neo_spike_train(self):
        self.assertIsInstance(threshold_detection(self.vm), neo.core.SpikeTrain)

    def test_signal_raise_type_error(self):
        with self.assertRaises(TypeError):
            threshold_detection(self.vm.magnitude)

    def test_always_return_as_list(self):
        self.assertIsInstance(
            threshold_detection(self.vm, always_as_list=True), SpikeTrainList
        )

    def test_analog_signal_multiple_channels(self):
        list_of_spike_trains = threshold_detection(self.vm_3d)
        self.assertEqual(len(list_of_spike_trains), 3)
        for spike_train in list_of_spike_trains:
            with self.subTest(value=spike_train):
                self.assertIsInstance(spike_train, neo.SpikeTrain)
        self.assertIsInstance(list_of_spike_trains, SpikeTrainList)

    def test_empty_analog_signal(self):
        empty_analog_signal = neo.AnalogSignal([], units="V", sampling_period=1 * pq.ms)
        self.assertEqual(empty_analog_signal.shape, (0, 1))
        self.assertIsInstance(
            threshold_detection(empty_analog_signal), neo.core.SpikeTrain
        )


class PeakDetectionTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        curr_dir = os.path.dirname(os.path.realpath(__file__))
        raw_data_file_loc = os.path.join(curr_dir, "spike_extraction_test_data.txt")
        raw_data = []
        with open(raw_data_file_loc, "r") as f:
            for x in f.readlines():
                raw_data.append(float(x))
        cls.vm = neo.AnalogSignal(raw_data, units=pq.V, sampling_period=0.1 * pq.ms)
        cls.vm_3d = neo.AnalogSignal(
            np.array([raw_data, raw_data, raw_data]).T,
            units=pq.V,
            sampling_period=0.1 * pq.ms,
        )
        cls.true_time_stamps = [
            0.0124,
            0.0354,
            0.0713,
            0.1192,
            0.1695,
            0.2201,
            0.2711,
        ] * pq.s

    def test_peak_detection_validate_result(self):
        # Test with default arguments
        result = peak_detection(self.vm)
        self.assertEqual(len(self.true_time_stamps), len(result))

        try:
            assert_array_almost_equal(result, self.true_time_stamps)
        except AttributeError:
            self.assertTrue(np.array_equal(result, self.true_time_stamps))

    def test_peak_detection_threshold(self):
        # Test for empty SpikeTrain when threshold is too high
        result = peak_detection(self.vm, threshold=30 * pq.mV)
        self.assertEqual(len(result), 0)

    def test_threshold_raise_type_error(self):
        with self.assertRaises(TypeError):
            peak_detection(self.vm, threshold=30)

    def test_sign_raise_value_error(self):
        with self.assertRaises(ValueError):
            peak_detection(self.vm, sign="wrong input")

    def test_return_is_neo_spike_train(self):
        self.assertIsInstance(peak_detection(self.vm), neo.core.SpikeTrain)

    def test_signal_raise_type_error(self):
        with self.assertRaises(TypeError):
            peak_detection(self.vm.magnitude)

    def test_always_return_as_list(self):
        self.assertIsInstance(
            peak_detection(self.vm, always_as_list=True), SpikeTrainList
        )

    def test_analog_signal_multiple_channels(self):
        list_of_spike_trains = peak_detection(self.vm_3d)
        self.assertEqual(len(list_of_spike_trains), 3)
        for spike_train in list_of_spike_trains:
            with self.subTest(value=spike_train):
                self.assertIsInstance(spike_train, neo.SpikeTrain)

    def test_analog_signal_multiple_channels_as_array(self):
        list_of_spike_trains = peak_detection(self.vm_3d, as_array=True)
        self.assertEqual(len(list_of_spike_trains), 3)
        for spike_train in list_of_spike_trains:
            with self.subTest(value=spike_train):
                self.assertIsInstance(spike_train, np.ndarray)

    def test_analog_signal_single_channel_as_array(self):
        array = peak_detection(self.vm, as_array=True)
        self.assertIsInstance(array, np.ndarray)
        self.assertEqual(array.ndim, 1)


class SpikeExtractionTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        curr_dir = os.path.dirname(os.path.realpath(__file__))
        raw_data_file_loc = os.path.join(curr_dir, "spike_extraction_test_data.txt")
        raw_data = []
        with open(raw_data_file_loc, "r") as f:
            for x in f.readlines():
                raw_data.append(float(x))
        cls.vm = neo.AnalogSignal(raw_data, units=pq.V, sampling_period=0.1 * pq.ms)
        cls.vm_3d = neo.AnalogSignal(
            np.array([raw_data, raw_data, raw_data]).T,
            units=pq.V,
            sampling_period=0.1 * pq.ms,
        )
        cls.first_spike = np.array(
            [
                -0.04084546,
                -0.03892033,
                -0.03664779,
                -0.03392689,
                -0.03061474,
                -0.02650277,
                -0.0212756,
                -0.01443531,
                -0.00515365,
                0.00803962,
                0.02797951,
                -0.07,
                -0.06974495,
                -0.06950466,
                -0.06927778,
                -0.06906314,
                -0.06885969,
                -0.06866651,
                -0.06848277,
                -0.06830773,
                -0.06814071,
                -0.06798113,
                -0.06782843,
                -0.06768213,
                -0.06754178,
                -0.06740699,
                -0.06727737,
                -0.06715259,
                -0.06703235,
                -0.06691635,
            ]
        )

    def test_spike_extraction_waveform(self):
        spike_train = spike_extraction(self.vm, interval=(-1 * pq.ms, 2 * pq.ms))

        assert_array_almost_equal(
            spike_train.waveforms[0][0].magnitude.reshape(-1), self.first_spike
        )

    def test_threshold_raise_type_error(self):
        with self.assertRaises(TypeError):
            spike_extraction(self.vm, threshold=30)

    def test_sign_raise_value_error(self):
        with self.assertRaises(ValueError):
            spike_extraction(self.vm, sign="wrong input")

    def test_return_is_neo_spike_train(self):
        self.assertIsInstance(spike_extraction(self.vm), neo.core.SpikeTrain)

    def test_signal_raise_type_error(self):
        with self.assertRaises(TypeError):
            spike_extraction(self.vm.magnitude)

    def test_always_return_as_list(self):
        self.assertIsInstance(
            spike_extraction(self.vm, always_as_list=True), SpikeTrainList
        )

    def test_analog_signal_multiple_channels(self):
        list_of_spike_trains = spike_extraction(self.vm_3d)
        self.assertEqual(len(list_of_spike_trains), 3)
        for spike_train in list_of_spike_trains:
            with self.subTest(value=spike_train):
                self.assertIsInstance(spike_train, neo.SpikeTrain)
        self.assertIsInstance(list_of_spike_trains, SpikeTrainList)


class AbstractPointProcessTestCase(unittest.TestCase):
    def test_not_implemented_error(self):
        process = AbstractPointProcess()
        self.assertRaises(NotImplementedError, process._generate_spiketrain_as_array)


class StationaryPoissonProcessTestCase(unittest.TestCase):
    def test_statistics(self):
        # This is a statistical test that has a non-zero chance of failure
        # during normal operation. Thus, we set the random seed to a value that
        # creates a realization passing the test.

        for rate in [123.0 * pq.Hz, 0.123 * pq.kHz]:
            for t_stop in [2345 * pq.ms, 2.345 * pq.s]:
                for refractory_period in (None, 3.0 * pq.ms):
                    np.random.seed(seed=123456)
                    spiketrain_old = homogeneous_poisson_process(
                        rate, t_stop=t_stop, refractory_period=refractory_period
                    )
                    np.random.seed(seed=123456)

                    spiketrain = StationaryPoissonProcess(
                        rate,
                        t_stop=t_stop,
                        refractory_period=refractory_period,
                        equilibrium=False,
                    ).generate_spiketrain()
                    assert_array_almost_equal(
                        spiketrain_old.magnitude, spiketrain.magnitude
                    )
                    intervals = isi(spiketrain)

                    expected_mean_isi = 1.0 / rate.simplified
                    self.assertAlmostEqual(
                        expected_mean_isi.magnitude,
                        intervals.mean().simplified.magnitude,
                        places=3,
                    )

                    expected_first_spike = 0 * pq.ms
                    self.assertLess(
                        spiketrain[0] - expected_first_spike, 7 * expected_mean_isi
                    )

                    expected_last_spike = t_stop
                    self.assertLess(
                        expected_last_spike - spiketrain[-1], 7 * expected_mean_isi
                    )

                    if refractory_period is None:
                        # Kolmogorov-Smirnov test
                        D, p = kstest(
                            intervals.rescale(t_stop.units).magnitude,
                            "expon",
                            # args are (loc, scale)
                            args=(
                                0.0,
                                expected_mean_isi.rescale(t_stop.units).magnitude,
                            ),
                            alternative="two-sided",
                        )
                    else:
                        refractory_period = refractory_period.rescale(
                            t_stop.units
                        ).item()
                        measured_rate = (
                            1.0 / expected_mean_isi.rescale(t_stop.units).item()
                        )
                        effective_rate = measured_rate / (
                            1.0 - measured_rate * refractory_period
                        )

                        # Kolmogorov-Smirnov test
                        D, p = kstest(
                            intervals.rescale(t_stop.units).magnitude,
                            "expon",
                            # args are (loc, scale)
                            args=(refractory_period, 1.0 / effective_rate),
                            alternative="two-sided",
                        )
                    self.assertGreater(p, 0.001)
                    self.assertLess(D, 0.12)

    def test_zero_refractory_period(self):
        rate = 10 * pq.Hz
        t_stop = 20 * pq.s

        np.random.seed(27)
        sp1 = StationaryPoissonProcess(rate, t_stop=t_stop).generate_spiketrain(
            as_array=True
        )

        np.random.seed(27)
        sp2 = StationaryPoissonProcess(
            rate, t_stop=t_stop, refractory_period=0.0 * pq.ms
        ).generate_spiketrain(as_array=True)

        assert_array_almost_equal(sp1, sp2)

    def test_t_start_and_t_stop(self):
        rate = 10 * pq.Hz
        t_start = 17 * pq.ms
        t_stop = 2 * pq.s

        sp1 = StationaryPoissonProcess(
            rate, t_start=t_start, t_stop=t_stop
        ).generate_spiketrain()

        sp2 = StationaryPoissonProcess(
            rate, t_start=t_start, t_stop=t_stop, refractory_period=3 * pq.ms
        ).generate_spiketrain()

        for spiketrain in (sp1, sp2):
            self.assertEqual(spiketrain.t_start, t_start)
            self.assertEqual(spiketrain.t_stop, t_stop)

    def test_zero_rate(self):
        for refractory_period in (None, 3 * pq.ms):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # RuntimeWarning: divide by zero encountered in true_divide
                # mean_interval = 1 / rate.magnitude, when rate == 0 Hz.
                spiketrain = StationaryPoissonProcess(
                    rate=0 * pq.Hz,
                    t_stop=10 * pq.s,
                    refractory_period=refractory_period,
                ).generate_spiketrain()
                self.assertEqual(spiketrain.size, 0)

    def test_nondecrease_spike_times(self):
        for refractory_period in (None, 3 * pq.ms):
            np.random.seed(27)

            spiketrain = StationaryPoissonProcess(
                rate=10 * pq.Hz, t_stop=1000 * pq.s, refractory_period=refractory_period
            ).generate_spiketrain()
            diffs = np.diff(spiketrain.times)
            self.assertTrue((diffs >= 0).all())

    def test_compare_with_as_array(self):
        rate = 10 * pq.Hz
        t_stop = 10 * pq.s
        for refractory_period in (None, 3 * pq.ms):
            process = StationaryPoissonProcess(
                rate=rate, t_stop=t_stop, refractory_period=refractory_period
            )
            np.random.seed(27)
            spiketrain = process.generate_spiketrain()
            self.assertIsInstance(spiketrain, neo.SpikeTrain)
            np.random.seed(27)
            spiketrain_array = process.generate_spiketrain().as_array()
            # don't check with isinstance: Quantity is a subclass of np.ndarray
            self.assertTrue(isinstance(spiketrain_array, np.ndarray))
            assert_array_almost_equal(spiketrain.times.magnitude, spiketrain_array)

    def test_effective_rate_refractory_period(self):
        np.random.seed(27)
        rate_expected = 10 * pq.Hz
        refractory_period = 90 * pq.ms  # 10 ms of effective ISI
        spiketrain = StationaryPoissonProcess(
            rate_expected, t_stop=1000 * pq.s, refractory_period=refractory_period
        ).generate_spiketrain()
        rate_obtained = len(spiketrain) / spiketrain.t_stop
        rate_obtained = rate_obtained.simplified
        self.assertAlmostEqual(
            rate_expected.simplified, rate_obtained.simplified, places=1
        )
        intervals = isi(spiketrain)
        isi_mean_expected = 1.0 / rate_expected
        self.assertAlmostEqual(
            isi_mean_expected.simplified, intervals.mean().simplified, places=3
        )

    def test_invalid(self):
        rate = 10 * pq.Hz
        for refractory_period in (None, 3 * pq.ms):
            # t_stop < t_start

            hpp = StationaryPoissonProcess
            self.assertRaises(
                ValueError,
                hpp,
                rate=rate,
                t_start=5 * pq.ms,
                t_stop=1 * pq.ms,
                refractory_period=refractory_period,
            )
            # no units provided for rate, t_stop
            self.assertRaises(
                ValueError, hpp, rate=10, refractory_period=refractory_period
            )
            self.assertRaises(
                ValueError,
                hpp,
                rate=rate,
                t_stop=5,
                refractory_period=refractory_period,
            )
            # no units provided for refractory_period
            self.assertRaises(ValueError, hpp, rate=rate, refractory_period=2)
        self.assertRaises(
            ValueError, StationaryPoissonProcess, rate, refractory_period=1.0 * pq.s
        )


class StationaryGammaProcessTestCase(unittest.TestCase):
    def test_statistics(self):
        # This is a statistical test that has a non-zero chance of failure
        # during normal operation. Thus, we set the random seed to a value that
        # creates a realization passing the test.
        a = 3.0
        for b in (67.0 * pq.Hz, 0.067 * pq.kHz):
            for t_stop in (2345 * pq.ms, 2.345 * pq.s):
                np.random.seed(seed=12345)
                spiketrain_old = homogeneous_gamma_process(a, b, t_stop=t_stop)
                np.random.seed(seed=12345)
                spiketrain = StationaryGammaProcess(
                    rate=b / a, shape_factor=a, t_stop=t_stop, equilibrium=False
                ).generate_spiketrain()
                assert_allclose(spiketrain_old.magnitude, spiketrain.magnitude)

                intervals = isi(spiketrain)

                expected_spike_count = int((b / a * t_stop).simplified)
                # should fail about 1 time in 1000
                self.assertLess(pdiff(expected_spike_count, spiketrain.size), 0.25)

                expected_mean_isi = (a / b).rescale(pq.ms)
                self.assertLess(pdiff(expected_mean_isi, intervals.mean()), 0.3)

                expected_first_spike = 0 * pq.ms
                self.assertLess(
                    spiketrain[0] - expected_first_spike, 4 * expected_mean_isi
                )

                expected_last_spike = t_stop
                self.assertLess(
                    expected_last_spike - spiketrain[-1], 4 * expected_mean_isi
                )

                # Kolmogorov-Smirnov test
                D, p = kstest(
                    intervals.rescale(t_stop.units),
                    "gamma",
                    # args are (a, loc, scale)
                    args=(a, 0, (1 / b).rescale(t_stop.units)),
                    alternative="two-sided",
                )
                self.assertGreater(p, 0.001)
                self.assertLess(D, 0.25)

    def test_compare_with_as_array(self):
        a = 3.0
        b = 10 * pq.Hz
        np.random.seed(27)
        spiketrain = StationaryGammaProcess(
            rate=b / a, shape_factor=a, equilibrium=False
        ).generate_spiketrain()
        self.assertIsInstance(spiketrain, neo.SpikeTrain)
        np.random.seed(27)
        spiketrain_array = StationaryGammaProcess(
            rate=b / a, shape_factor=a, equilibrium=False
        ).generate_spiketrain(as_array=True)
        # don't check with isinstance: pq.Quantity is a subclass of np.ndarray
        self.assertTrue(isinstance(spiketrain_array, np.ndarray))
        assert_array_almost_equal(spiketrain.times.magnitude, spiketrain_array)


class StationaryLogNormalProcessTestCase(unittest.TestCase):
    def test_statistics(self):
        # This is a statistical test that has a non-zero chance of failure
        # during normal operation. Thus, we set the random seed to a value that
        # creates a realization passing the test.
        sigma = 1.2
        for rate in (67.0 * pq.Hz, 0.067 * pq.kHz):
            for t_stop in (2345 * pq.ms, 2.345 * pq.s):
                np.random.seed(seed=123456)
                spiketrain = StationaryLogNormalProcess(
                    rate=rate, sigma=sigma, t_stop=t_stop, equilibrium=False
                ).generate_spiketrain()

                intervals = isi(spiketrain)

                expected_spike_count = int((rate * t_stop).simplified)
                # should fail about 1 time in 1000
                self.assertLess(pdiff(expected_spike_count, spiketrain.size), 0.25)

                expected_mean_isi = (1 / rate).rescale(pq.ms)
                self.assertLess(pdiff(expected_mean_isi, intervals.mean()), 0.3)

                expected_first_spike = 0 * pq.ms
                self.assertLess(
                    spiketrain[0] - expected_first_spike, 4 * expected_mean_isi
                )

                expected_last_spike = t_stop
                self.assertLess(
                    expected_last_spike - spiketrain[-1], 4 * expected_mean_isi
                )

                # Kolmogorov-Smirnov test
                D, p = kstest(
                    intervals.rescale(t_stop.units),
                    "lognorm",
                    # args are (s, loc, scale)
                    args=(
                        sigma,
                        0,
                        (1 / rate * np.exp(-(sigma**2) / 2)).rescale(t_stop.units),
                    ),
                    alternative="two-sided",
                )
                self.assertGreater(p, 0.001)
                self.assertLess(D, 0.25)

    def test_compare_with_as_array(self):
        sigma = 1.2
        rate = 10 * pq.Hz
        np.random.seed(27)
        spiketrain = StationaryLogNormalProcess(
            rate=rate, sigma=sigma, equilibrium=False
        ).generate_spiketrain()
        self.assertIsInstance(spiketrain, neo.SpikeTrain)
        np.random.seed(27)
        spiketrain_array = StationaryLogNormalProcess(
            rate=rate, sigma=sigma, equilibrium=False
        ).generate_spiketrain(as_array=True)
        # don't check with isinstance: pq.Quantity is a subclass of np.ndarray
        self.assertTrue(isinstance(spiketrain_array, np.ndarray))
        assert_array_almost_equal(spiketrain.times.magnitude, spiketrain_array)


class StationaryInverseGaussianProcessTestCase(unittest.TestCase):
    def test_statistics(self):
        # This is a statistical test that has a non-zero chance of failure
        # during normal operation. Thus, we set the random seed to a value that
        # creates a realization passing the test.
        cv = 0.9
        for rate in (67.0 * pq.Hz, 0.067 * pq.kHz):
            for t_stop in (2345 * pq.ms, 2.345 * pq.s):
                np.random.seed(seed=123456)
                spiketrain = StationaryInverseGaussianProcess(
                    rate=rate, cv=cv, t_stop=t_stop, equilibrium=False
                ).generate_spiketrain()

                intervals = isi(spiketrain)

                expected_spike_count = int((rate * t_stop).simplified)
                # should fail about 1 time in 1000

                self.assertLess(pdiff(expected_spike_count, spiketrain.size), 0.25)

                expected_mean_isi = (1 / rate).rescale(pq.ms)
                self.assertLess(pdiff(expected_mean_isi, intervals.mean()), 0.3)

                expected_first_spike = 0 * pq.ms
                self.assertLess(
                    spiketrain[0] - expected_first_spike, 4 * expected_mean_isi
                )

                expected_last_spike = t_stop
                self.assertLess(
                    expected_last_spike - spiketrain[-1], 4 * expected_mean_isi
                )

                # Kolmogorov-Smirnov test
                D, p = kstest(
                    intervals.rescale(t_stop.units),
                    "invgauss",
                    # args are (mu, loc, scale)
                    args=(cv**2, 0, (1 / (rate * cv**2)).rescale(t_stop.units)),
                    alternative="two-sided",
                )
                self.assertGreater(p, 0.001)
                self.assertLess(D, 0.25)

    def test_compare_with_as_array(self):
        cv = 1.2
        rate = 10 * pq.Hz
        np.random.seed(27)
        spiketrain = StationaryInverseGaussianProcess(
            rate=rate, cv=cv, equilibrium=False
        ).generate_spiketrain()
        self.assertIsInstance(spiketrain, neo.SpikeTrain)
        np.random.seed(27)
        spiketrain_array = StationaryInverseGaussianProcess(
            rate=rate, cv=cv, equilibrium=False
        ).generate_spiketrain(as_array=True)
        # don't check with isinstance: pq.Quantity is a subclass of np.ndarray
        self.assertTrue(isinstance(spiketrain_array, np.ndarray))
        assert_array_almost_equal(spiketrain.times.magnitude, spiketrain_array)


class FirstSpikeCvTestCase(unittest.TestCase):
    def setUp(self):
        np.random.seed(987654321)
        self.rate = 100.0 * pq.Hz
        self.t_stop = 10.0 * pq.s
        self.n_spiketrains = 10

        # can only have CV equal to 1.
        self.poisson_process = StationaryPoissonProcess(
            rate=self.rate, t_stop=self.t_stop
        )

        # choose all further processes to have CV of 1/2
        # CV = 1 - rate * refractory_period
        self.poisson_refractory_period_ordinary = StationaryPoissonProcess(
            rate=self.rate,
            refractory_period=0.5 / self.rate,
            t_stop=self.t_stop,
            equilibrium=False,
        )

        self.poisson_refractory_period_equilibrium = StationaryPoissonProcess(
            rate=self.rate,
            refractory_period=0.5 / self.rate,
            t_stop=self.t_stop,
            equilibrium=True,
        )

        # CV = 1 / sqrt(shape_factor)
        self.gamma_process_ordinary = StationaryGammaProcess(
            rate=self.rate, shape_factor=4, t_stop=self.t_stop, equilibrium=False
        )

        self.gamma_process_equilibrium = StationaryGammaProcess(
            rate=self.rate, shape_factor=4, t_stop=self.t_stop, equilibrium=True
        )

        # CV = sqrt(exp(sigma**2) - 1)
        self.log_normal_process_ordinary = StationaryLogNormalProcess(
            rate=self.rate,
            sigma=np.sqrt(np.log(5.0 / 4.0)),
            t_stop=self.t_stop,
            equilibrium=False,
        )

        self.log_normal_process_equilibrium = StationaryLogNormalProcess(
            rate=self.rate,
            sigma=np.sqrt(np.log(5.0 / 4.0)),
            t_stop=self.t_stop,
            equilibrium=True,
        )

        self.inverse_gaussian_process_ordinary = StationaryInverseGaussianProcess(
            rate=self.rate, cv=1 / 2, t_stop=self.t_stop, equilibrium=False
        )

        self.inverse_gaussian_process_equilibrium = StationaryInverseGaussianProcess(
            rate=self.rate, cv=1 / 2, t_stop=self.t_stop, equilibrium=True
        )

    def test_cv(self):
        processes = (
            self.poisson_process,
            self.poisson_refractory_period_ordinary,
            self.gamma_process_ordinary,
            self.log_normal_process_ordinary,
            self.inverse_gaussian_process_ordinary,
        )
        for process in processes:
            if process is self.poisson_process:
                self.assertAlmostEqual(1.0, process.expected_cv)

                # test the general expected-cv function
                self.assertAlmostEqual(1.0, super(type(process), process).expected_cv)
            else:
                self.assertAlmostEqual(0.5, process.expected_cv)
                # test the general expected-cv function
                self.assertAlmostEqual(0.5, super(type(process), process).expected_cv)
            spiketrains = process.generate_n_spiketrains(
                n_spiketrains=self.n_spiketrains, as_array=True
            )

            cvs = [variation(np.diff(spiketrain)) for spiketrain in spiketrains]
            mean_cv = np.mean(cvs)

            assert_allclose(process.expected_cv, mean_cv, atol=0.01)

    def test_first_spike(self):
        ordinary_processes = (
            self.poisson_refractory_period_ordinary,
            self.gamma_process_ordinary,
            self.log_normal_process_ordinary,
            self.inverse_gaussian_process_ordinary,
        )
        equilibrium_processes = (
            self.poisson_refractory_period_equilibrium,
            self.gamma_process_equilibrium,
            self.log_normal_process_equilibrium,
            self.inverse_gaussian_process_equilibrium,
        )

        for ordinary_process, equilibrium_process in zip(
            ordinary_processes, equilibrium_processes
        ):
            ordinary_spiketrains = ordinary_process.generate_n_spiketrains(
                self.n_spiketrains
            )
            equilbrium_spiketrains = equilibrium_process.generate_n_spiketrains(
                self.n_spiketrains
            )
            first_spikes_ordinary = [
                spiketrain[0].item() for spiketrain in ordinary_spiketrains
            ]
            first_spikes_equilibrium = [
                spiketrain[0].item() for spiketrain in equilbrium_spiketrains
            ]
            mean_first_spike_ordinary = np.mean(first_spikes_ordinary)
            mean_first_spike_equilibrium = np.mean(first_spikes_equilibrium)

            # for regular spike trains (CV=0.5 here) the first spike
            # in equilibrium is on average than in the ordinary case
            self.assertLess(mean_first_spike_equilibrium, mean_first_spike_ordinary)


class NonStationaryPoissonProcessTestCase(unittest.TestCase):
    def setUp(self):
        rate_list = [[20]] * 1000 + [[200]] * 1000
        self.rate_profile = neo.AnalogSignal(
            rate_list * pq.Hz, sampling_period=0.001 * pq.s
        )
        rate_0 = [[0]] * 1000
        self.rate_profile_0 = neo.AnalogSignal(
            rate_0 * pq.Hz, sampling_period=0.001 * pq.s
        )
        rate_negative = [[-1]] * 1000
        self.rate_profile_negative = neo.AnalogSignal(
            rate_negative * pq.Hz, sampling_period=0.001 * pq.s
        )

    def test_statistics(self):
        # This is a statistical test that has a non-zero chance of failure
        # during normal operation. Thus, we set the random seed to a value that
        # creates a realization passing the test.
        for rate in (self.rate_profile, self.rate_profile.rescale(pq.kHz)):
            for refractory_period in (3 * pq.ms, None):
                np.random.seed(seed=12345)
                spiketrain_old = inhomogeneous_poisson_process(
                    rate, refractory_period=refractory_period
                )
                np.random.seed(seed=12345)

                process = NonStationaryPoissonProcess
                spiketrain = process(
                    rate, refractory_period=refractory_period
                ).generate_spiketrain()

                assert_allclose(spiketrain_old.magnitude, spiketrain.magnitude)

                intervals = isi(spiketrain)

                # Computing expected statistics and percentiles
                expected_spike_count = (np.sum(rate) * rate.sampling_period).simplified
                percentile_count = poisson.ppf(0.999, expected_spike_count)
                expected_min_isi = 1 / np.min(rate)
                expected_max_isi = 1 / np.max(rate)
                percentile_min_isi = expon.ppf(0.999, expected_min_isi)
                percentile_max_isi = expon.ppf(0.999, expected_max_isi)

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
        spiketrain_as_array = NonStationaryPoissonProcess(rate).generate_spiketrain(
            as_array=True
        )
        self.assertTrue(isinstance(spiketrain_as_array, np.ndarray))
        self.assertTrue(isinstance(spiketrain, neo.SpikeTrain))

        # Testing type for refractory period
        refractory_period = 3 * pq.ms
        spiketrain = NonStationaryPoissonProcess(
            rate, refractory_period=refractory_period
        ).generate_spiketrain()
        spiketrain_as_array = NonStationaryPoissonProcess(
            rate, refractory_period=refractory_period
        ).generate_spiketrain(as_array=True)
        self.assertTrue(isinstance(spiketrain_as_array, np.ndarray))
        self.assertTrue(isinstance(spiketrain, neo.SpikeTrain))

        # Check that to high refractory period raises error
        self.assertRaises(
            ValueError,
            NonStationaryPoissonProcess,
            self.rate_profile,
            refractory_period=1000 * pq.ms,
        )

    def test_effective_rate_refractory_period(self):
        np.random.seed(27)
        rate_expected = 10 * pq.Hz
        refractory_period = 90 * pq.ms  # 10 ms of effective ISI
        rates = neo.AnalogSignal(
            np.repeat(rate_expected, 1000),
            units=pq.Hz,
            t_start=0 * pq.ms,
            sampling_rate=1 * pq.Hz,
        )
        spiketrain = NonStationaryPoissonProcess(
            rates, refractory_period=refractory_period
        ).generate_spiketrain()
        rate_obtained = len(spiketrain) / spiketrain.t_stop
        self.assertAlmostEqual(
            rate_expected.simplified.item(), rate_obtained.simplified.item(), places=1
        )
        intervals_inhomo = isi(spiketrain)
        isi_mean_expected = 1.0 / rate_expected
        self.assertAlmostEqual(
            isi_mean_expected.simplified, intervals_inhomo.mean().simplified, places=3
        )

    def test_zero_rate(self):
        for refractory_period in (3 * pq.ms, None):
            process = NonStationaryPoissonProcess
            spiketrain = process(
                self.rate_profile_0, refractory_period=refractory_period
            ).generate_spiketrain()
            self.assertEqual(spiketrain.size, 0)
        self.assertRaises(
            ValueError,
            NonStationaryPoissonProcess,
            self.rate_profile,
            refractory_period=5,
        )

    def test_negative_rates(self):
        for refractory_period in (3 * pq.ms, None):
            process = NonStationaryPoissonProcess
            self.assertRaises(
                ValueError,
                process,
                self.rate_profile_negative,
                refractory_period=refractory_period,
            )


class NonStationaryGammaTestCase(unittest.TestCase):
    def setUp(self):
        rate_list = [[20]] * 1000 + [[200]] * 1000
        self.rate_profile = neo.AnalogSignal(
            rate_list * pq.Hz, sampling_period=0.001 * pq.s
        )
        rate_0 = [[0]] * 1000
        self.rate_profile_0 = neo.AnalogSignal(
            rate_0 * pq.Hz, sampling_period=0.001 * pq.s
        )
        rate_negative = [[-1]] * 1000
        self.rate_profile_negative = neo.AnalogSignal(
            rate_negative * pq.Hz, sampling_period=0.001 * pq.s
        )

    def test_statistics(self):
        # This is a statistical test that has a non-zero chance of failure
        # during normal operation. Thus, we set the random seed to a value that
        # creates a realization passing the test.
        shape_factor = 2.5

        for rate in [self.rate_profile, self.rate_profile.rescale(pq.kHz)]:
            np.random.seed(seed=12345)
            spiketrain_old = inhomogeneous_gamma_process(
                rate, shape_factor=shape_factor
            )
            np.random.seed(seed=12345)
            spiketrain = NonStationaryGammaProcess(
                rate, shape_factor=shape_factor
            ).generate_spiketrain()
            assert_allclose(spiketrain_old.magnitude, spiketrain.magnitude)

            intervals = isi(spiketrain)

            # Computing expected statistics and percentiles
            expected_spike_count = (np.sum(rate) * rate.sampling_period).simplified
            percentile_count = poisson.ppf(0.999, expected_spike_count)
            expected_min_isi = 1 / np.min(rate)
            expected_max_isi = 1 / np.max(rate)
            percentile_min_isi = expon.ppf(0.999, expected_min_isi)
            percentile_max_isi = expon.ppf(0.999, expected_max_isi)

            # Testing (each should fail 1 every 1000 times)
            self.assertLess(spiketrain.size, percentile_count)
            self.assertLess(np.min(intervals), percentile_min_isi)
            self.assertLess(np.max(intervals), percentile_max_isi)

            # Testing t_start t_stop
            self.assertEqual(rate.t_stop, spiketrain.t_stop)
            self.assertEqual(rate.t_start, spiketrain.t_start)

        # Testing type
        spiketrain_as_array = NonStationaryGammaProcess(
            rate, shape_factor=shape_factor
        ).generate_spiketrain(as_array=True)
        self.assertTrue(isinstance(spiketrain_as_array, np.ndarray))
        self.assertTrue(isinstance(spiketrain, neo.SpikeTrain))

        # check error if rate has wrong format
        self.assertRaises(
            ValueError,
            NonStationaryGammaProcess,
            rate_signal=[0.1, 2.0],
            shape_factor=shape_factor,
        )

        # check error if negative values in rate
        self.assertRaises(
            ValueError,
            NonStationaryGammaProcess,
            rate_signal=neo.AnalogSignal(
                [-0.1, 10.0] * pq.Hz, sampling_period=0.001 * pq.s
            ),
            shape_factor=shape_factor,
        )

        # check error if rate is empty
        self.assertRaises(
            ValueError,
            NonStationaryGammaProcess,
            rate_signal=neo.AnalogSignal([] * pq.Hz, sampling_period=0.001 * pq.s),
            shape_factor=shape_factor,
        )

    def test_recovered_firing_rate_profile(self):
        np.random.seed(54)
        t_start = 0 * pq.s
        t_stop = 2 * np.round(np.pi, decimals=3) * pq.s  # 2 full periods
        sampling_period = 0.001 * pq.s

        # an arbitrary rate profile
        profile = 0.5 * (
            1 + np.sin(np.arange(t_start.item(), t_stop.item(), sampling_period.item()))
        )

        n_trials = 100
        rtol = 0.1  # 10% of deviation allowed
        kernel = kernels.RectangularKernel(sigma=0.25 * pq.s)
        for rate in (10 * pq.Hz, 50 * pq.Hz):
            rate_profile = neo.AnalogSignal(
                rate * profile, sampling_period=sampling_period
            )
            # the recovered firing rate profile should not depend on the
            # shape factor; here we test float and integer values of the shape
            # factor: the method supports float values that is not trivial
            # for inhomogeneous gamma process generation
            for shape_factor in (2.5, 10.0):
                spiketrains = NonStationaryGammaProcess(
                    rate_profile, shape_factor=shape_factor
                ).generate_n_spiketrains(n_trials)
                rate_recovered = (
                    instantaneous_rate(
                        spiketrains,
                        sampling_period=sampling_period,
                        kernel=kernel,
                        t_start=t_start,
                        t_stop=t_stop,
                        trim=True,
                    ).sum(axis=1)
                    / n_trials
                )

                rate_recovered = rate_recovered.flatten().magnitude
                trim = (rate_profile.shape[0] - rate_recovered.shape[0]) // 2
                rate_profile_valid = rate_profile.magnitude.squeeze()
                rate_profile_valid = rate_profile_valid[trim:-trim]
                assert_allclose(
                    rate_recovered, rate_profile_valid, rtol=0, atol=rtol * rate.item()
                )


class NPoissonTestCase(unittest.TestCase):
    def setUp(self):
        self.n = 4
        self.rate = 10 * pq.Hz
        self.rates = np.arange(1, self.n + 1) * pq.Hz
        self.t_stop = 10000 * pq.ms

    def test_poisson(self):
        # Check the output types for input rate + n number of neurons
        pp = _n_poisson(rate=self.rate, t_stop=self.t_stop, n_spiketrains=self.n)
        self.assertIsInstance(pp, list)
        self.assertIsInstance(pp[0], neo.core.spiketrain.SpikeTrain)
        self.assertEqual(pp[0].simplified.units, 1000 * pq.ms)
        self.assertEqual(len(pp), self.n)

        # Check the output types for input list of rates
        pp = _n_poisson(rate=self.rates, t_stop=self.t_stop)
        self.assertIsInstance(pp, list)
        self.assertIsInstance(pp[0], neo.core.spiketrain.SpikeTrain)
        self.assertEqual(pp[0].simplified.units, 1000 * pq.ms)
        self.assertEqual(len(pp), self.n)

    def test_poisson_error(self):
        # Dimensionless rate
        self.assertRaises(ValueError, _n_poisson, rate=5, t_stop=self.t_stop)
        # Negative rate
        self.assertRaises(ValueError, _n_poisson, rate=-5 * pq.Hz, t_stop=self.t_stop)
        # Negative value when rate is a list
        self.assertRaises(
            ValueError, _n_poisson, rate=[-5, 3] * pq.Hz, t_stop=self.t_stop
        )
        # Negative n
        self.assertRaises(
            ValueError, _n_poisson, rate=self.rate, t_stop=self.t_stop, n_spiketrains=-1
        )
        # t_start>t_stop
        self.assertRaises(
            ValueError,
            _n_poisson,
            rate=self.rate,
            t_start=4 * pq.ms,
            t_stop=3 * pq.ms,
            n_spiketrains=3,
        )


class SingleInteractionProcessTestCase(unittest.TestCase):
    def setUp(self):
        self.n = 4
        self.rate = 10 * pq.Hz
        self.rates = np.arange(1, self.n + 1) * pq.Hz
        self.t_stop = 10000 * pq.ms
        self.rate_c = 1 * pq.Hz

    def format_check(self, sip, coinc):
        self.assertEqual(type(sip), list)
        self.assertEqual(type(sip[0]), neo.core.spiketrain.SpikeTrain)
        self.assertEqual(type(coinc[0]), neo.core.spiketrain.SpikeTrain)
        self.assertEqual(sip[0].simplified.units, 1000 * pq.ms)
        self.assertEqual(coinc[0].simplified.units, 1000 * pq.ms)

        # Check the output length
        self.assertEqual(len(sip), self.n)

    def test_sip(self):
        # Generate an example SIP mode
        sip, coinc = single_interaction_process(
            n_spiketrains=self.n,
            t_stop=self.t_stop,
            rate=self.rate,
            coincidence_rate=self.rate_c,
            return_coincidences=True,
        )

        # Check the output types
        self.format_check(sip, coinc)
        self.assertEqual(
            len(coinc[0]), (self.rate_c * self.t_stop).simplified.magnitude
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Generate an example SIP mode giving a list of rates as imput
            sip, coinc = single_interaction_process(
                t_stop=self.t_stop,
                rate=self.rates,
                coincidence_rate=self.rate_c,
                return_coincidences=True,
            )

        # Check the output types
        self.format_check(sip, coinc)
        self.assertEqual(
            len(coinc[0]), (self.rate_c * self.t_stop).rescale(pq.dimensionless)
        )

        # Generate an example SIP mode stochastic number of coincidences
        sip = single_interaction_process(
            n_spiketrains=self.n,
            t_stop=self.t_stop,
            rate=self.rate,
            coincidence_rate=self.rate_c,
            coincidences="stochastic",
            return_coincidences=False,
        )

        # Check the output types
        self.assertEqual(type(sip), list)
        self.assertEqual(type(sip[0]), neo.core.spiketrain.SpikeTrain)
        self.assertEqual(sip[0].simplified.units, 1000 * pq.ms)

    def test_sip_error(self):
        # Negative rate
        self.assertRaises(
            ValueError,
            single_interaction_process,
            n_spiketrains=self.n,
            rate=-5 * pq.Hz,
            coincidence_rate=self.rate_c,
            t_stop=self.t_stop,
        )
        # Negative coincidence rate
        self.assertRaises(
            ValueError,
            single_interaction_process,
            n_spiketrains=self.n,
            rate=self.rate,
            coincidence_rate=-3 * pq.Hz,
            t_stop=self.t_stop,
        )
        # Negative value when rate is a list
        self.assertRaises(
            ValueError,
            single_interaction_process,
            n_spiketrains=self.n,
            rate=[-5, 3, 4, 2] * pq.Hz,
            coincidence_rate=self.rate_c,
            t_stop=self.t_stop,
        )
        # Negative n
        self.assertRaises(
            ValueError,
            single_interaction_process,
            n_spiketrains=-1,
            rate=self.rate,
            coincidence_rate=self.rate_c,
            t_stop=self.t_stop,
        )
        # Rate_c < rate
        self.assertRaises(
            ValueError,
            single_interaction_process,
            n_spiketrains=self.n,
            rate=self.rate,
            coincidence_rate=self.rate + 1 * pq.Hz,
            t_stop=self.t_stop,
        )


class CppTestCase(unittest.TestCase):
    def format_check(self, cpp, amplitude_distribution, t_start, t_stop):
        self.assertEqual([type(train) for train in cpp], [neo.SpikeTrain] * len(cpp))
        self.assertEqual(cpp[0].simplified.units, 1000 * pq.ms)
        self.assertEqual(type(cpp), list)
        # testing quantities format of the output
        self.assertEqual(
            [train.simplified.units for train in cpp], [1000 * pq.ms] * len(cpp)
        )
        # testing output t_start t_stop
        for spiketrain in cpp:
            self.assertEqual(spiketrain.t_stop, t_stop)
            self.assertEqual(spiketrain.t_start, t_start)
        self.assertEqual(len(cpp), len(amplitude_distribution) - 1)

    def test_cpp_hom(self):
        # testing output with generic inputs
        amplitude_distribution = np.array([0, 0.9, 0.1])
        t_stop = 10 * 1000 * pq.ms
        t_start = 5 * 1000 * pq.ms
        rate = 3 * pq.Hz
        cpp_hom = cpp(rate, amplitude_distribution, t_stop, t_start=t_start)
        # testing the output formats
        self.format_check(cpp_hom, amplitude_distribution, t_start, t_stop)

        # testing the units
        t_stop = 10000 * pq.ms
        t_start = 5 * 1000 * pq.ms
        rate = 3 * pq.Hz
        cpp_unit = cpp(rate, amplitude_distribution, t_stop, t_start=t_start)

        self.assertEqual(cpp_unit[0].units, t_stop.units)
        self.assertEqual(cpp_unit[0].t_stop.units, t_stop.units)
        self.assertEqual(cpp_unit[0].t_start.units, t_stop.units)

        # testing output without copy of spikes
        amplitude_distribution = np.array([1])
        t_stop = 10 * 1000 * pq.ms
        t_start = 5 * 1000 * pq.ms
        rate = 3 * pq.Hz
        cpp_hom_empty = cpp(rate, amplitude_distribution, t_stop, t_start=t_start)

        self.assertEqual(
            [len(train) for train in cpp_hom_empty], [0] * len(cpp_hom_empty)
        )

        # testing output with rate equal to 0
        amplitude_distribution = np.array([0, 0.9, 0.1])
        t_stop = 10 * 1000 * pq.ms
        t_start = 5 * 1000 * pq.ms
        rate = 0 * pq.Hz
        cpp_hom_empty_r = cpp(rate, amplitude_distribution, t_stop, t_start=t_start)
        self.assertEqual(
            [len(train) for train in cpp_hom_empty_r], [0] * len(cpp_hom_empty_r)
        )

        # testing output with same spike trains in output
        amplitude_distribution = np.array([0.0, 0.0, 1.0])
        t_stop = 10 * 1000 * pq.ms
        t_start = 5 * 1000 * pq.ms
        rate = 3 * pq.Hz
        cpp_hom_eq = cpp(rate, amplitude_distribution, t_stop, t_start=t_start)

        self.assertTrue(np.allclose(cpp_hom_eq[0].magnitude, cpp_hom_eq[1].magnitude))

    def test_cpp_hom_errors(self):
        # testing raises of ValueError (wrong inputs)
        # testing empty amplitude
        self.assertRaises(
            ValueError,
            cpp,
            amplitude_distribution=[],
            t_stop=10 * 1000 * pq.ms,
            rate=3 * pq.Hz,
        )

        # testing sum of amplitude>1
        self.assertRaises(
            ValueError,
            cpp,
            amplitude_distribution=[1, 1, 1],
            t_stop=10 * 1000 * pq.ms,
            rate=3 * pq.Hz,
        )
        # testing negative value in the amplitude
        self.assertRaises(
            ValueError,
            cpp,
            amplitude_distribution=[-1, 1, 1],
            t_stop=10 * 1000 * pq.ms,
            rate=3 * pq.Hz,
        )
        # test negative rate
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Catches RuntimeWarning: invalid value encountered in sqrt
            # number = np.ceil(n + 3 * np.sqrt(n)), when `n` == -3 Hz.
            self.assertRaises(
                ValueError,
                cpp,
                amplitude_distribution=[0, 1, 0],
                t_stop=10 * 1000 * pq.ms,
                rate=-3 * pq.Hz,
            )
        # test wrong unit for rate
        self.assertRaises(
            ValueError,
            cpp,
            amplitude_distribution=[0, 1, 0],
            t_stop=10 * 1000 * pq.ms,
            rate=3 * 1000 * pq.ms,
        )

        # testing raises of AttributeError (missing input units)
        # Testing missing unit to t_stop
        self.assertRaises(
            ValueError, cpp, amplitude_distribution=[0, 1, 0], t_stop=10, rate=3 * pq.Hz
        )
        # Testing missing unit to t_start
        self.assertRaises(
            ValueError,
            cpp,
            amplitude_distribution=[0, 1, 0],
            t_stop=10 * 1000 * pq.ms,
            rate=3 * pq.Hz,
            t_start=3,
        )
        # testing rate missing unit
        self.assertRaises(
            AttributeError,
            cpp,
            amplitude_distribution=[0, 1, 0],
            t_stop=10 * 1000 * pq.ms,
            rate=3,
        )

    def test_cpp_het(self):
        # testing output with generic inputs
        amplitude_distribution = np.array([0, 0.9, 0.1])
        t_stop = 10 * 1000 * pq.ms
        t_start = 5 * 1000 * pq.ms
        rate = [3, 4] * pq.Hz
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Catch RuntimeWarning: divide by zero encountered in true_divide
            # mean_interval = 1 / rate.magnitude, when rate == 0 Hz.
            cpp_het = cpp(rate, amplitude_distribution, t_stop, t_start=t_start)
            # testing the output formats
            self.format_check(cpp_het, amplitude_distribution, t_start, t_stop)
            self.assertEqual(len(cpp_het), len(rate))

            # testing the units
            t_stop = 10000 * pq.ms
            t_start = 5 * 1000 * pq.ms
            rate = [3, 4] * pq.Hz
            cpp_unit = cpp(rate, amplitude_distribution, t_stop, t_start=t_start)

            self.assertEqual(cpp_unit[0].units, t_stop.units)
            self.assertEqual(cpp_unit[0].t_stop.units, t_stop.units)
            self.assertEqual(cpp_unit[0].t_start.units, t_stop.units)
            # testing without copying any spikes
            amplitude_distribution = np.array([1, 0, 0])
            t_stop = 10 * 1000 * pq.ms
            t_start = 5 * 1000 * pq.ms
            rate = [3, 4] * pq.Hz
            cpp_het_empty = cpp(rate, amplitude_distribution, t_stop, t_start=t_start)

            self.assertEqual(len(cpp_het_empty[0]), 0)

            # testing output with rate equal to 0
            amplitude_distribution = np.array([0, 0.9, 0.1])
            t_stop = 10 * 1000 * pq.ms
            t_start = 5 * 1000 * pq.ms
            rate = [0, 0] * pq.Hz
            cpp_het_empty_r = cpp(rate, amplitude_distribution, t_stop, t_start=t_start)
            self.assertEqual(
                [len(train) for train in cpp_het_empty_r], [0] * len(cpp_het_empty_r)
            )

            # testing completely synchronous spike trains
            amplitude_distribution = np.array([0, 0, 1])
            t_stop = 10 * 1000 * pq.ms
            t_start = 5 * 1000 * pq.ms
            rate = [3, 3] * pq.Hz
            cpp_het_eq = cpp(rate, amplitude_distribution, t_stop, t_start=t_start)

            self.assertTrue(
                np.allclose(cpp_het_eq[0].magnitude, cpp_het_eq[1].magnitude)
            )

    def test_cpp_het_err(self):
        # testing raises of ValueError (wrong inputs)
        # testing empty amplitude
        self.assertRaises(
            ValueError,
            cpp,
            amplitude_distribution=[],
            t_stop=10 * 1000 * pq.ms,
            rate=[3, 4] * pq.Hz,
        )
        # testing sum amplitude>1
        self.assertRaises(
            ValueError,
            cpp,
            amplitude_distribution=[1, 1, 1],
            t_stop=10 * 1000 * pq.ms,
            rate=[3, 4] * pq.Hz,
        )
        # testing amplitude negative value
        self.assertRaises(
            ValueError,
            cpp,
            amplitude_distribution=[-1, 1, 1],
            t_stop=10 * 1000 * pq.ms,
            rate=[3, 4] * pq.Hz,
        )
        # testing negative rate
        self.assertRaises(
            ValueError,
            cpp,
            amplitude_distribution=[0, 1, 0],
            t_stop=10 * 1000 * pq.ms,
            rate=[-3, 4] * pq.Hz,
        )
        # testing empty rate
        self.assertRaises(
            ValueError,
            cpp,
            amplitude_distribution=[0, 1, 0],
            t_stop=10 * 1000 * pq.ms,
            rate=[] * pq.Hz,
        )
        # testing empty amplitude
        self.assertRaises(
            ValueError,
            cpp,
            amplitude_distribution=[],
            t_stop=10 * 1000 * pq.ms,
            rate=[3, 4] * pq.Hz,
        )
        # testing different len(A)-1 and len(rate)
        self.assertRaises(
            ValueError,
            cpp,
            amplitude_distribution=[0, 1],
            t_stop=10 * 1000 * pq.ms,
            rate=[3, 4] * pq.Hz,
        )
        # testing rate with different unit from Hz
        self.assertRaises(
            ValueError,
            cpp,
            amplitude_distribution=[0, 1],
            t_stop=10 * 1000 * pq.ms,
            rate=[3, 4] * 1000 * pq.ms,
        )
        # Testing analytical constrain between amplitude and rate
        self.assertRaises(
            ValueError,
            cpp,
            amplitude_distribution=[0, 0, 1],
            t_stop=10 * 1000 * pq.ms,
            rate=[3, 4] * pq.Hz,
            t_start=3,
        )

        # testing raises of AttributeError (missing input units)
        # Testing missing unit to t_stop
        self.assertRaises(
            ValueError,
            cpp,
            amplitude_distribution=[0, 1, 0],
            t_stop=10,
            rate=[3, 4] * pq.Hz,
        )
        # Testing missing unit to t_start
        self.assertRaises(
            ValueError,
            cpp,
            amplitude_distribution=[0, 1, 0],
            t_stop=10 * 1000 * pq.ms,
            rate=[3, 4] * pq.Hz,
            t_start=3,
        )
        # Testing missing unit to rate
        self.assertRaises(
            AttributeError,
            cpp,
            amplitude_distribution=[0, 1, 0],
            t_stop=10 * 1000 * pq.ms,
            rate=[3, 4],
        )

    def test_cpp_jttered(self):
        # testing output with generic inputs
        amplitude_distribution = np.array([0, 0.9, 0.1])
        t_stop = 10 * 1000 * pq.ms
        t_start = 5 * 1000 * pq.ms
        rate = 3 * pq.Hz
        cpp_shift = cpp(
            rate, amplitude_distribution, t_stop, t_start=t_start, shift=3 * pq.ms
        )
        # testing the output formats
        self.format_check(cpp_shift, amplitude_distribution, t_start, t_stop)


if __name__ == "__main__":
    unittest.main()
