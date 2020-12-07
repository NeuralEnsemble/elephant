# -*- coding: utf-8 -*-
"""
Unit tests for the signal_processing module.

:copyright: Copyright 2014-2020 by the Elephant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
"""
from __future__ import division, print_function

import unittest

import neo
import numpy as np
import quantities as pq
import scipy.signal as spsig
import scipy.stats
from numpy.ma.testutils import assert_array_equal, assert_allclose
from numpy.testing.utils import assert_array_almost_equal

import elephant.signal_processing


class PairwiseCrossCorrelationTest(unittest.TestCase):
    # Set parameters
    sampling_period = 0.02 * pq.s
    sampling_rate = 1. / sampling_period
    n_samples = 2018
    times = np.arange(n_samples) * sampling_period
    freq = 1. * pq.Hz

    def test_cross_correlation_freqs(self):
        """
        Sine vs cosine for different frequencies
        Note, that accuracy depends on N and min(f).
        E.g., f=0.1 and N=2018 only has an accuracy on the order decimal=1
        """
        freq_arr = np.linspace(0.5, 15, 8) * pq.Hz
        signal = np.zeros((self.n_samples, 3))
        for freq in freq_arr:
            signal[:, 0] = np.sin(2. * np.pi * freq * self.times)
            signal[:, 1] = np.cos(2. * np.pi * freq * self.times)
            signal[:, 2] = np.cos(2. * np.pi * freq * self.times + 0.2)
            # Convert signal to neo.AnalogSignal
            signal_neo = neo.AnalogSignal(signal, units='mV',
                                          t_start=0. * pq.ms,
                                          sampling_rate=self.sampling_rate,
                                          dtype=float)
            rho = elephant.signal_processing.cross_correlation_function(
                signal_neo, [[0, 1], [0, 2]])
            # Cross-correlation of sine and cosine should be sine
            assert_array_almost_equal(
                rho.magnitude[:, 0], np.sin(2. * np.pi * freq * rho.times),
                decimal=2)
            self.assertEqual(rho.shape, (signal.shape[0], 2))  # 2 pairs

    def test_cross_correlation_nlags(self):
        """
        Sine vs cosine for specific nlags
        """
        nlags = 30
        signal = np.zeros((self.n_samples, 2))
        signal[:, 0] = 0.2 * np.sin(2. * np.pi * self.freq * self.times)
        signal[:, 1] = 5.3 * np.cos(2. * np.pi * self.freq * self.times)
        # Convert signal to neo.AnalogSignal
        signal = neo.AnalogSignal(signal, units='mV', t_start=0. * pq.ms,
                                  sampling_rate=self.sampling_rate,
                                  dtype=float)
        rho = elephant.signal_processing.cross_correlation_function(
            signal, [0, 1], n_lags=nlags)
        # Test if vector of lags tau has correct length
        assert len(rho.times) == 2 * int(nlags) + 1

    def test_cross_correlation_phi(self):
        """
        Sine with phase shift phi vs cosine
        """
        phi = np.pi / 6.
        signal = np.zeros((self.n_samples, 2))
        signal[:, 0] = 0.2 * np.sin(2. * np.pi * self.freq * self.times + phi)
        signal[:, 1] = 5.3 * np.cos(2. * np.pi * self.freq * self.times)
        # Convert signal to neo.AnalogSignal
        signal = neo.AnalogSignal(signal, units='mV', t_start=0. * pq.ms,
                                  sampling_rate=self.sampling_rate,
                                  dtype=float)
        rho = elephant.signal_processing.cross_correlation_function(
            signal, [0, 1])
        # Cross-correlation of sine and cosine should be sine + phi
        assert_array_almost_equal(rho.magnitude[:, 0], np.sin(
            2. * np.pi * self.freq * rho.times + phi), decimal=2)

    def test_cross_correlation_envelope(self):
        """
        Envelope of sine vs cosine
        """
        # Sine with phase shift phi vs cosine for different frequencies
        nlags = 800  # nlags need to be smaller than N/2 b/c border effects
        signal = np.zeros((self.n_samples, 2))
        signal[:, 0] = 0.2 * np.sin(2. * np.pi * self.freq * self.times)
        signal[:, 1] = 5.3 * np.cos(2. * np.pi * self.freq * self.times)
        # Convert signal to neo.AnalogSignal
        signal = neo.AnalogSignal(signal, units='mV', t_start=0. * pq.ms,
                                  sampling_rate=self.sampling_rate,
                                  dtype=float)
        envelope = elephant.signal_processing.cross_correlation_function(
            signal, [0, 1], n_lags=nlags, hilbert_envelope=True)
        # Envelope should be one for sinusoidal function
        assert_array_almost_equal(envelope, np.ones_like(envelope), decimal=2)

    def test_cross_correlation_biased(self):
        signal = np.c_[np.sin(2. * np.pi * self.freq * self.times),
                       np.cos(2. * np.pi * self.freq * self.times)] * pq.mV
        signal = neo.AnalogSignal(signal, t_start=0. * pq.ms,
                                  sampling_rate=self.sampling_rate)
        raw = elephant.signal_processing.cross_correlation_function(
            signal, [0, 1], scaleopt='none'
        )
        biased = elephant.signal_processing.cross_correlation_function(
            signal, [0, 1], scaleopt='biased'
        )
        assert_array_almost_equal(biased, raw / biased.shape[0])

    def test_cross_correlation_coeff(self):
        signal = np.c_[np.sin(2. * np.pi * self.freq * self.times),
                       np.cos(2. * np.pi * self.freq * self.times)] * pq.mV
        signal = neo.AnalogSignal(signal, t_start=0. * pq.ms,
                                  sampling_rate=self.sampling_rate)
        normalized = elephant.signal_processing.cross_correlation_function(
            signal, [0, 1], scaleopt='coeff'
        )
        sig1, sig2 = signal.magnitude.T
        target_numpy = np.correlate(sig1, sig2, mode="same")
        target_numpy /= np.sqrt((sig1 ** 2).sum() * (sig2 ** 2).sum())
        target_numpy = np.expand_dims(target_numpy, axis=1)
        assert_array_almost_equal(normalized.magnitude,
                                  target_numpy,
                                  decimal=3)

    def test_cross_correlation_coeff_autocorr(self):
        # Numpy/Matlab equivalent
        signal = np.sin(2. * np.pi * self.freq * self.times)
        signal = signal[:, np.newaxis] * pq.mV
        signal = neo.AnalogSignal(signal, t_start=0. * pq.ms,
                                  sampling_rate=self.sampling_rate)
        normalized = elephant.signal_processing.cross_correlation_function(
            signal, [0, 0], scaleopt='coeff'
        )
        # auto-correlation at zero lag should equal 1
        self.assertAlmostEqual(normalized[normalized.shape[0] // 2], 1)


class ZscoreTestCase(unittest.TestCase):

    def setUp(self):
        self.test_seq1 = [1, 28, 4, 47, 5, 16, 2, 5, 21, 12,
                          4, 12, 59, 2, 4, 18, 33, 25, 2, 34,
                          4, 1, 1, 14, 8, 1, 10, 1, 8, 20,
                          5, 1, 6, 5, 12, 2, 8, 8, 2, 8,
                          2, 10, 2, 1, 1, 2, 15, 3, 20, 6,
                          11, 6, 18, 2, 5, 17, 4, 3, 13, 6,
                          1, 18, 1, 16, 12, 2, 52, 2, 5, 7,
                          6, 25, 6, 5, 3, 15, 4, 3, 16, 3,
                          6, 5, 24, 21, 3, 3, 4, 8, 4, 11,
                          5, 7, 5, 6, 8, 11, 33, 10, 7, 4]
        self.test_seq2 = [6, 3, 0, 0, 18, 4, 14, 98, 3, 56,
                          7, 4, 6, 9, 11, 16, 13, 3, 2, 15,
                          24, 1, 0, 7, 4, 4, 9, 24, 12, 11,
                          9, 7, 9, 8, 5, 2, 7, 12, 15, 17,
                          3, 7, 2, 1, 0, 17, 2, 6, 3, 32,
                          22, 19, 11, 8, 5, 4, 3, 2, 7, 21,
                          24, 2, 5, 10, 11, 14, 6, 8, 4, 12,
                          6, 5, 2, 22, 25, 19, 16, 22, 13, 2,
                          19, 20, 17, 19, 2, 4, 1, 3, 5, 23,
                          20, 15, 4, 7, 10, 14, 15, 15, 20, 1]

    def test_zscore_single_dup(self):
        """
        Test z-score on a single AnalogSignal, asking to return a
        duplicate.
        """
        signal = neo.AnalogSignal(
            self.test_seq1, units='mV',
            t_start=0. * pq.ms, sampling_rate=1000. * pq.Hz, dtype=float)

        m = np.mean(self.test_seq1)
        s = np.std(self.test_seq1)
        target = (self.test_seq1 - m) / s
        assert_array_equal(target, scipy.stats.zscore(self.test_seq1))

        result = elephant.signal_processing.zscore(signal, inplace=False)
        assert_array_almost_equal(
            result.magnitude, target.reshape(-1, 1), decimal=9)

        self.assertEqual(result.units, pq.Quantity(1. * pq.dimensionless))

        # Assert original signal is untouched
        self.assertEqual(signal[0].magnitude, self.test_seq1[0])

    def test_zscore_single_inplace(self):
        """
        Test z-score on a single AnalogSignal, asking for an inplace
        operation.
        """
        signal = neo.AnalogSignal(
            self.test_seq1, units='mV',
            t_start=0. * pq.ms, sampling_rate=1000. * pq.Hz, dtype=float)

        m = np.mean(self.test_seq1)
        s = np.std(self.test_seq1)
        target = (self.test_seq1 - m) / s

        result = elephant.signal_processing.zscore(signal, inplace=True)

        assert_array_almost_equal(
            result.magnitude, target.reshape(-1, 1), decimal=9)

        self.assertEqual(result.units, pq.Quantity(1. * pq.dimensionless))

        # Assert original signal is overwritten
        self.assertEqual(signal[0].magnitude, target[0])

    def test_zscore_single_multidim_dup(self):
        """
        Test z-score on a single AnalogSignal with multiple dimensions, asking
        to return a duplicate.
        """
        signal = neo.AnalogSignal(
            np.transpose(
                np.vstack([self.test_seq1, self.test_seq2])), units='mV',
            t_start=0. * pq.ms, sampling_rate=1000. * pq.Hz, dtype=float)

        m = np.mean(signal.magnitude, axis=0, keepdims=True)
        s = np.std(signal.magnitude, axis=0, keepdims=True)
        target = (signal.magnitude - m) / s

        assert_array_almost_equal(
            elephant.signal_processing.zscore(
                signal, inplace=False).magnitude, target, decimal=9)

        # Assert original signal is untouched
        self.assertEqual(signal[0, 0].magnitude, self.test_seq1[0])

    def test_zscore_array_annotations(self):
        signal = neo.AnalogSignal(
            self.test_seq1, units='mV',
            t_start=0. * pq.ms, sampling_rate=1000. * pq.Hz,
            array_annotations=dict(valid=True, my_list=[0]))
        zscored = elephant.signal_processing.zscore(signal, inplace=False)
        self.assertDictEqual(signal.array_annotations,
                             zscored.array_annotations)

    def test_zscore_single_multidim_inplace(self):
        """
        Test z-score on a single AnalogSignal with multiple dimensions, asking
        for an inplace operation.
        """
        signal = neo.AnalogSignal(
            np.vstack([self.test_seq1, self.test_seq2]), units='mV',
            t_start=0. * pq.ms, sampling_rate=1000. * pq.Hz, dtype=float)

        m = np.mean(signal.magnitude, axis=0, keepdims=True)
        s = np.std(signal.magnitude, axis=0, keepdims=True)
        ground_truth = np.divide(signal.magnitude - m, s,
                                 out=np.zeros_like(signal.magnitude),
                                 where=s != 0)
        result = elephant.signal_processing.zscore(signal, inplace=True)

        assert_array_almost_equal(result.magnitude, ground_truth, decimal=8)

        # Assert original signal is overwritten
        self.assertAlmostEqual(signal[0, 0].magnitude, ground_truth[0, 0])

    def test_zscore_single_dup_int(self):
        """
        Test if the z-score is correctly calculated even if the input is an
        AnalogSignal of type int, asking for a duplicate (duplicate should
        be of type float).
        """
        signal = neo.AnalogSignal(
            self.test_seq1, units='mV',
            t_start=0. * pq.ms, sampling_rate=1000. * pq.Hz, dtype=int)

        m = np.mean(self.test_seq1)
        s = np.std(self.test_seq1)
        target = (self.test_seq1 - m) / s

        assert_array_almost_equal(
            elephant.signal_processing.zscore(signal, inplace=False).magnitude,
            target.reshape(-1, 1), decimal=9)

        # Assert original signal is untouched
        self.assertEqual(signal.magnitude[0], self.test_seq1[0])

    def test_zscore_single_inplace_int(self):
        """
        Test if the z-score is correctly calculated even if the input is an
        AnalogSignal of type int, asking for an inplace operation.
        """
        m = np.mean(self.test_seq1)
        s = np.std(self.test_seq1)
        target = (self.test_seq1 - m) / s

        signal = neo.AnalogSignal(
            self.test_seq1, units='mV',
            t_start=0. * pq.ms, sampling_rate=1000. * pq.Hz, dtype=int)
        zscored = elephant.signal_processing.zscore(signal, inplace=True)

        assert_array_almost_equal(zscored.magnitude.squeeze(), target)

    def test_zscore_list_dup(self):
        """
        Test zscore on a list of AnalogSignal objects, asking to return a
        duplicate.
        """
        signal1 = neo.AnalogSignal(
            np.transpose(np.vstack([self.test_seq1, self.test_seq1])),
            units='mV',
            t_start=0. * pq.ms, sampling_rate=1000. * pq.Hz, dtype=float)
        signal2 = neo.AnalogSignal(
            np.transpose(np.vstack([self.test_seq1, self.test_seq2])),
            units='mV',
            t_start=0. * pq.ms, sampling_rate=1000. * pq.Hz, dtype=float)
        signal_list = [signal1, signal2]

        m = np.mean(np.hstack([self.test_seq1, self.test_seq1]))
        s = np.std(np.hstack([self.test_seq1, self.test_seq1]))
        target11 = (self.test_seq1 - m) / s
        target21 = (self.test_seq1 - m) / s
        m = np.mean(np.hstack([self.test_seq1, self.test_seq2]))
        s = np.std(np.hstack([self.test_seq1, self.test_seq2]))
        target12 = (self.test_seq1 - m) / s
        target22 = (self.test_seq2 - m) / s

        # Call elephant function
        result = elephant.signal_processing.zscore(signal_list, inplace=False)

        assert_array_almost_equal(
            result[0].magnitude,
            np.transpose(np.vstack([target11, target12])), decimal=9)
        assert_array_almost_equal(
            result[1].magnitude,
            np.transpose(np.vstack([target21, target22])), decimal=9)

        # Assert original signal is untouched
        self.assertEqual(signal1.magnitude[0, 0], self.test_seq1[0])
        self.assertEqual(signal2.magnitude[0, 1], self.test_seq2[0])

    def test_zscore_list_inplace(self):
        """
        Test zscore on a list of AnalogSignal objects, asking for an
        inplace operation.
        """
        signal1 = neo.AnalogSignal(
            np.transpose(np.vstack([self.test_seq1, self.test_seq1])),
            units='mV',
            t_start=0. * pq.ms, sampling_rate=1000. * pq.Hz, dtype=float)
        signal2 = neo.AnalogSignal(
            np.transpose(np.vstack([self.test_seq1, self.test_seq2])),
            units='mV',
            t_start=0. * pq.ms, sampling_rate=1000. * pq.Hz, dtype=float)
        signal_list = [signal1, signal2]

        m = np.mean(np.hstack([self.test_seq1, self.test_seq1]))
        s = np.std(np.hstack([self.test_seq1, self.test_seq1]))
        target11 = (self.test_seq1 - m) / s
        target21 = (self.test_seq1 - m) / s
        m = np.mean(np.hstack([self.test_seq1, self.test_seq2]))
        s = np.std(np.hstack([self.test_seq1, self.test_seq2]))
        target12 = (self.test_seq1 - m) / s
        target22 = (self.test_seq2 - m) / s

        # Call elephant function
        result = elephant.signal_processing.zscore(signal_list, inplace=True)

        assert_array_almost_equal(
            result[0].magnitude,
            np.transpose(np.vstack([target11, target12])), decimal=9)
        assert_array_almost_equal(
            result[1].magnitude,
            np.transpose(np.vstack([target21, target22])), decimal=9)

        # Assert original signal is overwritten
        self.assertEqual(signal1[0, 0].magnitude, target11[0])
        self.assertEqual(signal2[0, 0].magnitude, target21[0])

    def test_wrong_input(self):
        # wrong type
        self.assertRaises(TypeError, elephant.signal_processing.zscore,
                          signal=[1, 2] * pq.uV)
        # units mismatch
        asig1 = neo.AnalogSignal([0, 1], units=pq.uV, sampling_rate=1 * pq.ms)
        asig2 = neo.AnalogSignal([0, 1], units=pq.V, sampling_rate=1 * pq.ms)
        self.assertRaises(ValueError, elephant.signal_processing.zscore,
                          signal=[asig1, asig2])


class ButterTestCase(unittest.TestCase):

    def test_butter_filter_type(self):
        """
        Test if correct type of filtering is performed according to how cut-off
        frequencies are given
        """
        # generate white noise AnalogSignal
        noise = neo.AnalogSignal(
            np.random.normal(size=5000),
            sampling_rate=1000 * pq.Hz, units='mV')

        # test high-pass filtering: power at the lowest frequency
        # should be almost zero
        # Note: the default detrend function of scipy.signal.welch() seems to
        # cause artificial finite power at the lowest frequencies. Here I avoid
        # this by using an identity function for detrending
        filtered_noise = elephant.signal_processing.butter(
            noise, 250.0 * pq.Hz, None)
        _, psd = spsig.welch(filtered_noise.T, nperseg=1024, fs=1000.0,
                             detrend=lambda x: x)
        self.assertAlmostEqual(psd[0, 0], 0)

        # test low-pass filtering: power at the highest frequency
        # should be almost zero
        filtered_noise = elephant.signal_processing.butter(
            noise, None, 250.0 * pq.Hz)
        _, psd = spsig.welch(filtered_noise.T, nperseg=1024, fs=1000.0)
        self.assertAlmostEqual(psd[0, -1], 0)

        # test band-pass filtering: power at the lowest and highest frequencies
        # should be almost zero
        filtered_noise = elephant.signal_processing.butter(
            noise, 200.0 * pq.Hz, 300.0 * pq.Hz)
        _, psd = spsig.welch(filtered_noise.T, nperseg=1024, fs=1000.0,
                             detrend=lambda x: x)
        self.assertAlmostEqual(psd[0, 0], 0)
        self.assertAlmostEqual(psd[0, -1], 0)

        # test band-stop filtering: power at the intermediate frequency
        # should be almost zero
        filtered_noise = elephant.signal_processing.butter(
            noise, 400.0 * pq.Hz, 100.0 * pq.Hz)
        _, psd = spsig.welch(filtered_noise.T, nperseg=1024, fs=1000.0)
        self.assertAlmostEqual(psd[0, 256], 0)

    def test_butter_filter_function(self):
        """
        `elephant.signal_processing.butter` return values test for all
        available filters (result has to be almost equal):
            * lfilter
            * filtfilt
            * sosfiltfilt
        """
        # generate white noise AnalogSignal
        noise = neo.AnalogSignal(
            np.random.normal(size=5000),
            sampling_rate=1000 * pq.Hz, units='mV',
            array_annotations=dict(valid=True, my_list=[0]))

        kwds = {'signal': noise, 'highpass_freq': 250.0 * pq.Hz,
                'lowpass_freq': None, 'filter_function': 'filtfilt'}
        filtered_noise = elephant.signal_processing.butter(**kwds)
        _, psd_filtfilt = spsig.welch(
            filtered_noise.T, nperseg=1024, fs=1000.0, detrend=lambda x: x)

        kwds['filter_function'] = 'lfilter'
        filtered_noise = elephant.signal_processing.butter(**kwds)
        _, psd_lfilter = spsig.welch(
            filtered_noise.T, nperseg=1024, fs=1000.0, detrend=lambda x: x)

        kwds['filter_function'] = 'sosfiltfilt'
        filtered_noise = elephant.signal_processing.butter(**kwds)
        _, psd_sosfiltfilt = spsig.welch(
            filtered_noise.T, nperseg=1024, fs=1000.0, detrend=lambda x: x)

        self.assertAlmostEqual(psd_filtfilt[0, 0], psd_lfilter[0, 0])
        self.assertAlmostEqual(psd_filtfilt[0, 0], psd_sosfiltfilt[0, 0])

        # Test if array_annotations are preserved
        self.assertDictEqual(noise.array_annotations,
                             filtered_noise.array_annotations)

    def test_butter_invalid_filter_function(self):
        # generate a dummy AnalogSignal
        anasig_dummy = neo.AnalogSignal(
            np.zeros(5000), sampling_rate=1000 * pq.Hz, units='mV')
        # test exception upon invalid filtfunc string
        kwds = {'signal': anasig_dummy, 'highpass_freq': 250.0 * pq.Hz,
                'filter_function': 'invalid_filter'}
        self.assertRaises(
            ValueError, elephant.signal_processing.butter, **kwds)

    def test_butter_missing_cutoff_freqs(self):
        # generate a dummy AnalogSignal
        anasig_dummy = neo.AnalogSignal(
            np.zeros(5000), sampling_rate=1000 * pq.Hz, units='mV')
        # test a case where no cut-off frequencies are given
        kwds = {'signal': anasig_dummy, 'highpass_freq': None,
                'lowpass_freq': None}
        self.assertRaises(
            ValueError, elephant.signal_processing.butter, **kwds)

    def test_butter_input_types(self):
        # generate white noise data of different types
        noise_np = np.random.normal(size=5000)
        noise_pq = noise_np * pq.mV
        noise = neo.AnalogSignal(noise_pq, sampling_rate=1000.0 * pq.Hz)

        # check input as NumPy ndarray
        filtered_noise_np = elephant.signal_processing.butter(
            noise_np, 400.0, 100.0, sampling_frequency=1000.0)
        self.assertTrue(isinstance(filtered_noise_np, np.ndarray))
        self.assertFalse(isinstance(filtered_noise_np, pq.quantity.Quantity))
        self.assertFalse(isinstance(filtered_noise_np, neo.AnalogSignal))
        self.assertEqual(filtered_noise_np.shape, noise_np.shape)

        # check input as Quantity array
        filtered_noise_pq = elephant.signal_processing.butter(
            noise_pq, 400.0 * pq.Hz, 100.0 * pq.Hz, sampling_frequency=1000.0)
        self.assertTrue(isinstance(filtered_noise_pq, pq.quantity.Quantity))
        self.assertFalse(isinstance(filtered_noise_pq, neo.AnalogSignal))
        self.assertEqual(filtered_noise_pq.shape, noise_pq.shape)

        # check input as neo AnalogSignal
        filtered_noise = elephant.signal_processing.butter(noise,
                                                           400.0 * pq.Hz,
                                                           100.0 * pq.Hz)
        self.assertTrue(isinstance(filtered_noise, neo.AnalogSignal))
        self.assertEqual(filtered_noise.shape, noise.shape)

        # check if the results from different input types are identical
        self.assertTrue(np.all(
            filtered_noise_pq.magnitude == filtered_noise_np))
        self.assertTrue(np.all(
            filtered_noise.magnitude[:, 0] == filtered_noise_np))

    def test_butter_axis(self):
        noise = np.random.normal(size=(4, 5000))
        filtered_noise = elephant.signal_processing.butter(
            noise, 250.0, sampling_frequency=1000.0)
        filtered_noise_transposed = elephant.signal_processing.butter(
            noise.T, 250.0, sampling_frequency=1000.0, axis=0)
        self.assertTrue(np.all(filtered_noise == filtered_noise_transposed.T))

    def test_butter_multidim_input(self):
        noise_pq = np.random.normal(size=(4, 5000)) * pq.mV
        noise_neo = neo.AnalogSignal(
            noise_pq.T, sampling_rate=1000.0 * pq.Hz)
        noise_neo1d = neo.AnalogSignal(
            noise_pq[0], sampling_rate=1000.0 * pq.Hz)
        filtered_noise_pq = elephant.signal_processing.butter(
            noise_pq, 250.0, sampling_frequency=1000.0)
        filtered_noise_neo = elephant.signal_processing.butter(
            noise_neo, 250.0)
        filtered_noise_neo1d = elephant.signal_processing.butter(
            noise_neo1d, 250.0)
        self.assertTrue(np.all(
            filtered_noise_pq.magnitude == filtered_noise_neo.T.magnitude))
        self.assertTrue(np.all(
            filtered_noise_neo1d.magnitude[:, 0] ==
            filtered_noise_neo.magnitude[:, 0]))


class HilbertTestCase(unittest.TestCase):

    def setUp(self):
        # Generate test data of a harmonic function over a long time
        time = np.arange(0, 1000, 0.1) * pq.ms
        freq = 10 * pq.Hz

        self.amplitude = np.array([
            np.linspace(1, 10, len(time)),
            np.linspace(1, 10, len(time)),
            np.ones((len(time))),
            np.ones((len(time))) * 10.]).T
        self.phase = np.array([
            (time * freq).simplified.magnitude * 2. * np.pi,
            (time * freq).simplified.magnitude * 2. * np.pi + np.pi / 2,
            (time * freq).simplified.magnitude * 2. * np.pi + np.pi,
            (time * freq).simplified.magnitude * 2. * 2. * np.pi]).T

        self.phase = np.mod(self.phase + np.pi, 2. * np.pi) - np.pi

        # rising amplitude cosine, random ampl. sine, flat inverse cosine,
        # flat cosine at double frequency
        sigs = np.vstack([
            self.amplitude[:, 0] * np.cos(self.phase[:, 0]),
            self.amplitude[:, 1] * np.cos(self.phase[:, 1]),
            self.amplitude[:, 2] * np.cos(self.phase[:, 2]),
            self.amplitude[:, 3] * np.cos(self.phase[:, 3])])

        array_annotations = dict(my_list=np.arange(sigs.shape[0]))
        self.long_signals = neo.AnalogSignal(
            sigs.T, units='mV',
            t_start=0. * pq.ms,
            sampling_rate=(len(time) / (time[-1] - time[0])).rescale(pq.Hz),
            dtype=float,
            array_annotations=array_annotations)

        # Generate test data covering a single oscillation cycle in 1s only
        phases = np.arange(0, 2 * np.pi, np.pi / 256)
        sigs = np.vstack([
            np.sin(phases),
            np.cos(phases),
            np.sin(2 * phases),
            np.cos(2 * phases)])

        self.one_period = neo.AnalogSignal(
            sigs.T, units=pq.mV,
            sampling_rate=len(phases) * pq.Hz)

    def test_hilbert_pad_type_error(self):
        """
        Tests if incorrect pad_type raises ValueError.
        """
        padding = 'wrong_type'

        self.assertRaises(
            ValueError, elephant.signal_processing.hilbert,
            self.long_signals, N=padding)

    def test_hilbert_output_shape(self):
        """
        Tests if the length of the output is identical to the original signal,
        and the dimension is dimensionless.
        """
        true_shape = np.shape(self.long_signals)
        output = elephant.signal_processing.hilbert(
            self.long_signals, padding='nextpow')
        self.assertEqual(np.shape(output), true_shape)
        self.assertEqual(output.units, pq.dimensionless)
        output = elephant.signal_processing.hilbert(
            self.long_signals, padding=16384)
        self.assertEqual(np.shape(output), true_shape)
        self.assertEqual(output.units, pq.dimensionless)

    def test_hilbert_array_annotations(self):
        output = elephant.signal_processing.hilbert(self.long_signals,
                                                    padding='nextpow')
        # Test if array_annotations are preserved
        self.assertSetEqual(set(output.array_annotations.keys()), {"my_list"})
        assert_array_equal(output.array_annotations['my_list'],
                           self.long_signals.array_annotations['my_list'])

    def test_hilbert_theoretical_long_signals(self):
        """
        Tests the output of the hilbert function with regard to amplitude and
        phase of long test signals
        """
        # Performing test using all pad types
        for padding in ['nextpow', 'none', 16384]:

            h = elephant.signal_processing.hilbert(
                self.long_signals, padding=padding)

            phase = np.angle(h.magnitude)
            amplitude = np.abs(h.magnitude)
            real_value = np.real(h.magnitude)

            # The real part should be equal to the original long_signals
            assert_array_almost_equal(
                real_value,
                self.long_signals.magnitude,
                decimal=14)

            # Test only in the middle half of the array (border effects)
            ind1 = int(len(h.times) / 4)
            ind2 = int(3 * len(h.times) / 4)

            # Calculate difference in phase between signal and original phase
            # and use smaller of any two phase differences
            phasediff = np.abs(phase[ind1:ind2, :] - self.phase[ind1:ind2, :])
            phasediff[phasediff >= np.pi] = \
                2 * np.pi - phasediff[phasediff >= np.pi]

            # Calculate difference in amplitude between signal and original
            # amplitude
            amplitudediff = \
                amplitude[ind1:ind2, :] - self.amplitude[ind1:ind2, :]
#
            assert_allclose(phasediff, 0, atol=0.1)
            assert_allclose(amplitudediff, 0, atol=0.5)

    def test_hilbert_theoretical_one_period(self):
        """
        Tests the output of the hilbert function with regard to amplitude and
        phase of a short signal covering one cycle (more accurate estimate).

        This unit test is adapted from the scipy library of the hilbert()
        function.
        """

        # Precision of testing
        decimal = 14

        # Performing test using both pad types
        for padding in ['nextpow', 'none', 512]:

            h = elephant.signal_processing.hilbert(
                self.one_period, padding=padding)

            amplitude = np.abs(h.magnitude)
            phase = np.angle(h.magnitude)
            real_value = np.real(h.magnitude)

            # The real part should be equal to the original long_signals:
            assert_array_almost_equal(
                real_value,
                self.one_period.magnitude,
                decimal=decimal)

            # The absolute value should be 1 everywhere, for this input:
            assert_array_almost_equal(
                amplitude,
                np.ones(self.one_period.magnitude.shape),
                decimal=decimal)

            # For the 'slow' sine - the phase should go from -pi/2 to pi/2 in
            # the first 256 bins:
            assert_array_almost_equal(
                phase[:256, 0],
                np.arange(-np.pi / 2, np.pi / 2, np.pi / 256),
                decimal=decimal)
            # For the 'slow' cosine - the phase should go from 0 to pi in the
            # same interval:
            assert_array_almost_equal(
                phase[:256, 1],
                np.arange(0, np.pi, np.pi / 256),
                decimal=decimal)
            # The 'fast' sine should make this phase transition in half the
            # time:
            assert_array_almost_equal(
                phase[:128, 2],
                np.arange(-np.pi / 2, np.pi / 2, np.pi / 128),
                decimal=decimal)
            # The 'fast' cosine should make this phase transition in half the
            # time:
            assert_array_almost_equal(
                phase[:128, 3],
                np.arange(0, np.pi, np.pi / 128),
                decimal=decimal)


class WaveletTestCase(unittest.TestCase):
    def setUp(self):
        # generate a 10-sec test data of pure 50 Hz cosine wave
        self.fs = 1000.0
        self.times = np.arange(0, 10.0, 1 / self.fs)
        self.test_freq1 = 50.0
        self.test_freq2 = 60.0
        self.test_data1 = np.cos(2 * np.pi * self.test_freq1 * self.times)
        self.test_data2 = np.sin(2 * np.pi * self.test_freq2 * self.times)
        self.test_data_arr = np.vstack([self.test_data1, self.test_data2])
        self.test_data = neo.AnalogSignal(
            self.test_data_arr.T * pq.mV, t_start=self.times[0] * pq.s,
            t_stop=self.times[-1] * pq.s, sampling_period=(1 / self.fs) * pq.s)
        self.true_phase1 = np.angle(
            self.test_data1 +
            1j *
            np.sin(
                2 *
                np.pi *
                self.test_freq1 *
                self.times))
        self.true_phase2 = np.angle(
            self.test_data2 -
            1j *
            np.cos(
                2 *
                np.pi *
                self.test_freq2 *
                self.times))
        self.wt_freqs = [10, 20, 30]

    def test_wavelet_errors(self):
        """
        Tests if errors are raised as expected.
        """
        # too high center frequency
        kwds = {'signal': self.test_data, 'freq': self.fs / 2}
        self.assertRaises(
            ValueError, elephant.signal_processing.wavelet_transform, **kwds)
        kwds = {
            'signal': self.test_data_arr,
            'freq': self.fs / 2,
            'fs': self.fs}
        self.assertRaises(
            ValueError, elephant.signal_processing.wavelet_transform, **kwds)

        # too high center frequency in a list
        kwds = {'signal': self.test_data, 'freq': [self.fs / 10, self.fs / 2]}
        self.assertRaises(
            ValueError, elephant.signal_processing.wavelet_transform, **kwds)
        kwds = {'signal': self.test_data_arr,
                'freq': [self.fs / 10, self.fs / 2], 'fs': self.fs}
        self.assertRaises(
            ValueError, elephant.signal_processing.wavelet_transform, **kwds)

        # nco is not positive
        kwds = {'signal': self.test_data, 'freq': self.fs / 10, 'nco': 0}
        self.assertRaises(
            ValueError, elephant.signal_processing.wavelet_transform, **kwds)

    def test_wavelet_io(self):
        """
        Tests the data type and data shape of the output is consistent with
        that of the input, and also test the consistency between the outputs
        of different types
        """
        # check the shape of the result array
        # --- case of single center frequency
        wt = elephant.signal_processing.wavelet_transform(self.test_data,
                                                          self.fs / 10)
        self.assertTrue(wt.ndim == self.test_data.ndim)
        self.assertTrue(wt.shape[0] == self.test_data.shape[0])  # time axis
        self.assertTrue(wt.shape[1] == self.test_data.shape[1])  # channel axis

        wt_arr = elephant.signal_processing.wavelet_transform(
            self.test_data_arr, self.fs / 10, sampling_frequency=self.fs)
        self.assertTrue(wt_arr.ndim == self.test_data.ndim)
        # channel axis
        self.assertTrue(wt_arr.shape[0] == self.test_data_arr.shape[0])
        # time axis
        self.assertTrue(wt_arr.shape[1] == self.test_data_arr.shape[1])

        wt_arr1d = elephant.signal_processing.wavelet_transform(
            self.test_data1, self.fs / 10, sampling_frequency=self.fs)
        self.assertTrue(wt_arr1d.ndim == self.test_data1.ndim)
        # time axis
        self.assertTrue(wt_arr1d.shape[0] == self.test_data1.shape[0])

        # --- case of multiple center frequencies
        wt = elephant.signal_processing.wavelet_transform(
            self.test_data, self.wt_freqs)
        self.assertTrue(wt.ndim == self.test_data.ndim + 1)
        self.assertTrue(wt.shape[0] == self.test_data.shape[0])  # time axis
        self.assertTrue(wt.shape[1] == self.test_data.shape[1])  # channel axis
        self.assertTrue(wt.shape[2] == len(self.wt_freqs))  # frequency axis

        wt_arr = elephant.signal_processing.wavelet_transform(
            self.test_data_arr, self.wt_freqs, sampling_frequency=self.fs)
        self.assertTrue(wt_arr.ndim == self.test_data_arr.ndim + 1)
        # channel axis
        self.assertTrue(wt_arr.shape[0] == self.test_data_arr.shape[0])
        # frequency axis
        self.assertTrue(wt_arr.shape[1] == len(self.wt_freqs))
        # time axis
        self.assertTrue(wt_arr.shape[2] == self.test_data_arr.shape[1])

        wt_arr1d = elephant.signal_processing.wavelet_transform(
            self.test_data1, self.wt_freqs, sampling_frequency=self.fs)
        self.assertTrue(wt_arr1d.ndim == self.test_data1.ndim + 1)
        # frequency axis
        self.assertTrue(wt_arr1d.shape[0] == len(self.wt_freqs))
        # time axis
        self.assertTrue(wt_arr1d.shape[1] == self.test_data1.shape[0])

        # check that the result does not depend on data type
        self.assertTrue(np.all(wt[:, 0, :] == wt_arr[0, :, :].T))  # channel 0
        self.assertTrue(np.all(wt[:, 1, :] == wt_arr[1, :, :].T))  # channel 1

        # check the data contents in the case where freq is given as a list
        # Note: there seems to be a bug in np.fft since NumPy 1.14.1, which
        # causes that the values of wt_1freq[:, 0] and wt_3freqs[:, 0, 0] are
        # not exactly equal, even though they use the same center frequency for
        # wavelet transform (in NumPy 1.13.1, they become identical). Here we
        # only check that they are almost equal.
        wt_1freq = elephant.signal_processing.wavelet_transform(
            self.test_data, self.wt_freqs[0])
        wt_3freqs = elephant.signal_processing.wavelet_transform(
            self.test_data, self.wt_freqs)
        assert_array_almost_equal(wt_1freq[:, 0], wt_3freqs[:, 0, 0],
                                  decimal=12)

    def test_wavelet_amplitude(self):
        """
        Tests amplitude properties of the obtained wavelet transform
        """
        # check that the amplitude of WT of a sinusoid is (almost) constant
        wt = elephant.signal_processing.wavelet_transform(self.test_data,
                                                          self.test_freq1)
        # take a middle segment in order to avoid edge effects
        amp = np.abs(wt[int(len(wt) / 3):int(len(wt) // 3 * 2), 0])
        mean_amp = amp.mean()
        assert_array_almost_equal((amp - mean_amp) / mean_amp,
                                  np.zeros_like(amp), decimal=6)

        # check that the amplitude of WT is (almost) zero when center frequency
        # is considerably different from signal frequency
        wt_low = elephant.signal_processing.wavelet_transform(
            self.test_data, self.test_freq1 / 10)
        amp_low = np.abs(wt_low[int(len(wt) / 3):int(len(wt) // 3 * 2), 0])
        assert_array_almost_equal(amp_low, np.zeros_like(amp), decimal=6)

        # check that zero padding hardly affect the result
        wt_padded = elephant.signal_processing.wavelet_transform(
            self.test_data, self.test_freq1, zero_padding=False)
        amp_padded = np.abs(
            wt_padded[int(len(wt) / 3):int(len(wt) // 3 * 2), 0])
        assert_array_almost_equal(amp_padded, amp, decimal=9)

    def test_wavelet_phase(self):
        """
        Tests phase properties of the obtained wavelet transform
        """
        # check that the phase of WT is (almost) same as that of the original
        # sinusoid
        wt = elephant.signal_processing.wavelet_transform(self.test_data,
                                                          self.test_freq1)
        phase = np.angle(wt[int(len(wt) / 3):int(len(wt) // 3 * 2), 0])
        true_phase = self.true_phase1[int(len(wt) / 3):int(len(wt) // 3 * 2)]
        assert_array_almost_equal(np.exp(1j * phase), np.exp(1j * true_phase),
                                  decimal=6)

        # check that zero padding hardly affect the result
        wt_padded = elephant.signal_processing.wavelet_transform(
            self.test_data, self.test_freq1, zero_padding=False)
        phase_padded = np.angle(
            wt_padded[int(len(wt) / 3):int(len(wt) // 3 * 2), 0])
        assert_array_almost_equal(
            np.exp(
                1j * phase_padded),
            np.exp(
                1j * phase),
            decimal=9)


class DerivativeTestCase(unittest.TestCase):

    def setUp(self):
        self.fs = 1000.0
        self.tmin = 0.0
        self.tmax = 10.0
        self.times = np.arange(self.tmin, self.tmax, 1 / self.fs)
        self.test_data1 = np.cos(2 * np.pi * self.times)
        self.test_data2 = np.vstack(
            [np.cos(2 * np.pi * self.times), np.sin(2 * np.pi * self.times)]).T
        self.test_signal1 = neo.AnalogSignal(
            self.test_data1 * pq.mV, t_start=self.times[0] * pq.s,
            t_stop=self.times[-1] * pq.s, sampling_period=(1 / self.fs) * pq.s)
        self.test_signal2 = neo.AnalogSignal(
            self.test_data2 * pq.mV, t_start=self.times[0] * pq.s,
            t_stop=self.times[-1] * pq.s, sampling_period=(1 / self.fs) * pq.s)

    def test_derivative_invalid_signal(self):
        """Test derivative on non-AnalogSignal"""
        kwds = {'signal': np.arange(5)}
        self.assertRaises(
            TypeError, elephant.signal_processing.derivative, **kwds)

    def test_derivative_units(self):
        """Test derivative returns AnalogSignal with correct units"""
        derivative = elephant.signal_processing.derivative(
            self.test_signal1)
        self.assertTrue(isinstance(derivative, neo.AnalogSignal))
        self.assertEqual(
            derivative.units,
            self.test_signal1.units / self.test_signal1.times.units)

    def test_derivative_times(self):
        """Test derivative returns AnalogSignal with correct times"""
        derivative = elephant.signal_processing.derivative(
            self.test_signal1)
        self.assertTrue(isinstance(derivative, neo.AnalogSignal))

        # test that sampling period is correct
        self.assertEqual(
            derivative.sampling_period,
            1 / self.fs * self.test_signal1.times.units)

        # test that all times are correct
        target_times = self.times[:-1] * self.test_signal1.times.units \
            + derivative.sampling_period / 2
        assert_array_almost_equal(derivative.times, target_times)

        # test that t_start and t_stop are correct
        self.assertEqual(derivative.t_start, target_times[0])
        assert_array_almost_equal(
            derivative.t_stop,
            target_times[-1] + derivative.sampling_period)

    def test_derivative_values(self):
        """Test derivative returns AnalogSignal with correct values"""
        derivative1 = elephant.signal_processing.derivative(
            self.test_signal1)
        derivative2 = elephant.signal_processing.derivative(
            self.test_signal2)
        self.assertTrue(isinstance(derivative1, neo.AnalogSignal))
        self.assertTrue(isinstance(derivative2, neo.AnalogSignal))

        # single channel
        assert_array_almost_equal(
            derivative1.magnitude,
            np.vstack([np.diff(self.test_data1)]).T / (1 / self.fs))

        # multi channel
        assert_array_almost_equal(derivative2.magnitude, np.vstack([
            np.diff(self.test_data2[:, 0]),
            np.diff(self.test_data2[:, 1])]).T / (1 / self.fs))


class RAUCTestCase(unittest.TestCase):

    def setUp(self):
        self.fs = 1000.0
        self.tmin = 0.0
        self.tmax = 10.0
        self.times = np.arange(self.tmin, self.tmax, 1 / self.fs)
        self.test_data1 = np.cos(2 * np.pi * self.times)
        self.test_data2 = np.vstack(
            [np.cos(2 * np.pi * self.times), np.sin(2 * np.pi * self.times)]).T
        self.test_signal1 = neo.AnalogSignal(
            self.test_data1 * pq.mV, t_start=self.times[0] * pq.s,
            t_stop=self.times[-1] * pq.s, sampling_period=(1 / self.fs) * pq.s)
        self.test_signal2 = neo.AnalogSignal(
            self.test_data2 * pq.mV, t_start=self.times[0] * pq.s,
            t_stop=self.times[-1] * pq.s, sampling_period=(1 / self.fs) * pq.s)

    def test_rauc_invalid_signal(self):
        """Test rauc on non-AnalogSignal"""
        kwds = {'signal': np.arange(5)}
        self.assertRaises(
            ValueError, elephant.signal_processing.rauc, **kwds)

    def test_rauc_invalid_bin_duration(self):
        """Test rauc on bad bin duration"""
        kwds = {'signal': self.test_signal1, 'bin_duration': 'bad'}
        self.assertRaises(
            ValueError, elephant.signal_processing.rauc, **kwds)

    def test_rauc_invalid_baseline(self):
        """Test rauc on bad baseline"""
        kwds = {'signal': self.test_signal1, 'baseline': 'bad'}
        self.assertRaises(
            ValueError, elephant.signal_processing.rauc, **kwds)

    def test_rauc_units(self):
        """Test rauc returns Quantity or AnalogSignal with correct units"""

        # test that single-bin result is Quantity with correct units
        rauc = elephant.signal_processing.rauc(
            self.test_signal1)
        self.assertTrue(isinstance(rauc, pq.Quantity))
        self.assertEqual(
            rauc.units,
            self.test_signal1.units * self.test_signal1.times.units)

        # test that multi-bin result is AnalogSignal with correct units
        rauc_arr = elephant.signal_processing.rauc(
            self.test_signal1, bin_duration=1 * pq.s)
        self.assertTrue(isinstance(rauc_arr, neo.AnalogSignal))
        self.assertEqual(
            rauc_arr.units,
            self.test_signal1.units * self.test_signal1.times.units)

    def test_rauc_times_without_overextending_bin(self):
        """Test rauc returns correct times when signal is binned evenly"""

        bin_duration = 1 * pq.s  # results in all bin centers < original t_stop
        rauc_arr = elephant.signal_processing.rauc(
            self.test_signal1, bin_duration=bin_duration)
        self.assertTrue(isinstance(rauc_arr, neo.AnalogSignal))

        # test that sampling period is correct
        self.assertEqual(rauc_arr.sampling_period, bin_duration)

        # test that all times are correct
        target_times = np.arange(self.tmin,
                                 self.tmax,
                                 bin_duration.magnitude) \
            * bin_duration.units + bin_duration / 2
        assert_array_almost_equal(rauc_arr.times, target_times)

        # test that t_start and t_stop are correct
        self.assertEqual(rauc_arr.t_start, target_times[0])
        assert_array_almost_equal(
            rauc_arr.t_stop,
            target_times[-1] + bin_duration)

    def test_rauc_times_with_overextending_bin(self):
        """Test rauc returns correct times when signal is NOT binned evenly"""

        bin_duration = 0.99 * pq.s  # results in one bin center > original t_stop
        rauc_arr = elephant.signal_processing.rauc(
            self.test_signal1, bin_duration=bin_duration)
        self.assertTrue(isinstance(rauc_arr, neo.AnalogSignal))

        # test that sampling period is correct
        self.assertEqual(rauc_arr.sampling_period, bin_duration)

        # test that all times are correct
        target_times = np.arange(self.tmin,
                                 self.tmax,
                                 bin_duration.magnitude) \
            * bin_duration.units + bin_duration / 2
        assert_array_almost_equal(rauc_arr.times, target_times)

        # test that t_start and t_stop are correct
        self.assertEqual(rauc_arr.t_start, target_times[0])
        assert_array_almost_equal(
            rauc_arr.t_stop,
            target_times[-1] + bin_duration)

    def test_rauc_values_one_bin(self):
        """Test rauc returns correct values when there is just one bin"""
        rauc1 = elephant.signal_processing.rauc(
            self.test_signal1)
        rauc2 = elephant.signal_processing.rauc(
            self.test_signal2)
        self.assertTrue(isinstance(rauc1, pq.Quantity))
        self.assertTrue(isinstance(rauc2, pq.Quantity))

        # single channel
        assert_array_almost_equal(
            rauc1.magnitude,
            np.array([6.36517679]))

        # multi channel
        assert_array_almost_equal(
            rauc2.magnitude,
            np.array([6.36517679, 6.36617364]))

    def test_rauc_values_multi_bin(self):
        """Test rauc returns correct values when there are multiple bins"""
        rauc_arr1 = elephant.signal_processing.rauc(
            self.test_signal1, bin_duration=0.99 * pq.s)
        rauc_arr2 = elephant.signal_processing.rauc(
            self.test_signal2, bin_duration=0.99 * pq.s)
        self.assertTrue(isinstance(rauc_arr1, neo.AnalogSignal))
        self.assertTrue(isinstance(rauc_arr2, neo.AnalogSignal))

        # single channel
        assert_array_almost_equal(rauc_arr1.magnitude, np.array([
            [0.62562647],
            [0.62567202],
            [0.62576076],
            [0.62589236],
            [0.62606628],
            [0.62628184],
            [0.62653819],
            [0.62683432],
            [0.62716907],
            [0.62754110],
            [0.09304862]]))

        # multi channel
        assert_array_almost_equal(rauc_arr2.magnitude, np.array([
            [0.62562647, 0.63623770],
            [0.62567202, 0.63554830],
            [0.62576076, 0.63486313],
            [0.62589236, 0.63418488],
            [0.62606628, 0.63351623],
            [0.62628184, 0.63285983],
            [0.62653819, 0.63221825],
            [0.62683432, 0.63159403],
            [0.62716907, 0.63098964],
            [0.62754110, 0.63040747],
            [0.09304862, 0.03039579]]))

    def test_rauc_mean_baseline(self):
        """Test rauc returns correct values when baseline='mean' is given"""
        rauc1 = elephant.signal_processing.rauc(
            self.test_signal1, baseline='mean')
        rauc2 = elephant.signal_processing.rauc(
            self.test_signal2, baseline='mean')
        self.assertTrue(isinstance(rauc1, pq.Quantity))
        self.assertTrue(isinstance(rauc2, pq.Quantity))

        # single channel
        assert_array_almost_equal(
            rauc1.magnitude,
            np.array([6.36517679]))

        # multi channel
        assert_array_almost_equal(
            rauc2.magnitude,
            np.array([6.36517679, 6.36617364]))

    def test_rauc_median_baseline(self):
        """Test rauc returns correct values when baseline='median' is given"""
        rauc1 = elephant.signal_processing.rauc(
            self.test_signal1, baseline='median')
        rauc2 = elephant.signal_processing.rauc(
            self.test_signal2, baseline='median')
        self.assertTrue(isinstance(rauc1, pq.Quantity))
        self.assertTrue(isinstance(rauc2, pq.Quantity))

        # single channel
        assert_array_almost_equal(
            rauc1.magnitude,
            np.array([6.36517679]))

        # multi channel
        assert_array_almost_equal(
            rauc2.magnitude,
            np.array([6.36517679, 6.36617364]))

    def test_rauc_arbitrary_baseline(self):
        """Test rauc returns correct values when arbitrary baseline is given"""
        rauc1 = elephant.signal_processing.rauc(
            self.test_signal1, baseline=0.123 * pq.mV)
        rauc2 = elephant.signal_processing.rauc(
            self.test_signal2, baseline=0.123 * pq.mV)
        self.assertTrue(isinstance(rauc1, pq.Quantity))
        self.assertTrue(isinstance(rauc2, pq.Quantity))

        # single channel
        assert_array_almost_equal(
            rauc1.magnitude,
            np.array([6.41354725]))

        # multi channel
        assert_array_almost_equal(
            rauc2.magnitude,
            np.array([6.41354725, 6.41429810]))

    def test_rauc_time_slice(self):
        """Test rauc returns correct values when t_start, t_stop are given"""
        rauc1 = elephant.signal_processing.rauc(
            self.test_signal1, t_start=0.123 * pq.s, t_stop=0.456 * pq.s)
        rauc2 = elephant.signal_processing.rauc(
            self.test_signal2, t_start=0.123 * pq.s, t_stop=0.456 * pq.s)
        self.assertTrue(isinstance(rauc1, pq.Quantity))
        self.assertTrue(isinstance(rauc2, pq.Quantity))

        # single channel
        assert_array_almost_equal(
            rauc1.magnitude,
            np.array([0.16279006]))

        # multi channel
        assert_array_almost_equal(
            rauc2.magnitude,
            np.array([0.16279006, 0.26677944]))


if __name__ == '__main__':
    unittest.main()
