# -*- coding: utf-8 -*-
"""
Unit tests for the signal_processing module.

:copyright: Copyright 2014-2016 by the Elephant team, see AUTHORS.txt.
:license: Modified BSD, see LICENSE.txt for details.
"""
from __future__ import division, print_function

import unittest

import neo
import numpy as np
import scipy.signal as spsig
import scipy.stats
from numpy.testing.utils import assert_array_almost_equal
import quantities as pq

import elephant.signal_processing

from numpy.ma.testutils import assert_array_equal, assert_allclose


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
        '''
        Test z-score on a single AnalogSignal, asking to return a
        duplicate.
        '''
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
        '''
        Test z-score on a single AnalogSignal, asking for an inplace
        operation.
        '''
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
        '''
        Test z-score on a single AnalogSignal with multiple dimensions, asking
        to return a duplicate.
        '''
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

    def test_zscore_single_multidim_inplace(self):
        '''
        Test z-score on a single AnalogSignal with multiple dimensions, asking
        for an inplace operation.
        '''
        signal = neo.AnalogSignal(
            np.vstack([self.test_seq1, self.test_seq2]), units='mV',
            t_start=0. * pq.ms, sampling_rate=1000. * pq.Hz, dtype=float)

        m = np.mean(signal.magnitude, axis=0, keepdims=True)
        s = np.std(signal.magnitude, axis=0, keepdims=True)
        target = (signal.magnitude - m) / s

        assert_array_almost_equal(
            elephant.signal_processing.zscore(
                signal, inplace=True).magnitude, target, decimal=9)

        # Assert original signal is overwritten
        self.assertEqual(signal[0, 0].magnitude, target[0, 0])

    def test_zscore_single_dup_int(self):
        '''
        Test if the z-score is correctly calculated even if the input is an
        AnalogSignal of type int, asking for a duplicate (duplicate should
        be of type float).
        '''
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
        '''
        Test if the z-score is correctly calculated even if the input is an
        AnalogSignal of type int, asking for an inplace operation.
        '''
        signal = neo.AnalogSignal(
            self.test_seq1, units='mV',
            t_start=0. * pq.ms, sampling_rate=1000. * pq.Hz, dtype=int)

        m = np.mean(self.test_seq1)
        s = np.std(self.test_seq1)
        target = (self.test_seq1 - m) / s

        assert_array_almost_equal(
            elephant.signal_processing.zscore(signal, inplace=True).magnitude,
            target.reshape(-1, 1).astype(int), decimal=9)

        # Assert original signal is overwritten
        self.assertEqual(signal[0].magnitude, target.astype(int)[0])

    def test_zscore_list_dup(self):
        '''
        Test zscore on a list of AnalogSignal objects, asking to return a
        duplicate.
        '''
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
        '''
        Test zscore on a list of AnalogSignal objects, asking for an
        inplace operation.
        '''
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
        # generate white noise AnalogSignal
        noise = neo.AnalogSignal(
            np.random.normal(size=5000),
            sampling_rate=1000 * pq.Hz, units='mV')

        # test if the filter performance is as well with filftunc=lfilter as
        # with filtfunc=filtfilt (i.e. default option)
        kwds = {'signal': noise, 'highpass_freq': 250.0 * pq.Hz,
                'lowpass_freq': None, 'filter_function': 'filtfilt'}
        filtered_noise = elephant.signal_processing.butter(**kwds)
        _, psd_filtfilt = spsig.welch(
            filtered_noise.T, nperseg=1024, fs=1000.0, detrend=lambda x: x)

        kwds['filter_function'] = 'lfilter'
        filtered_noise = elephant.signal_processing.butter(**kwds)
        _, psd_lfilter = spsig.welch(
            filtered_noise.T, nperseg=1024, fs=1000.0, detrend=lambda x: x)

        self.assertAlmostEqual(psd_filtfilt[0, 0], psd_lfilter[0, 0])

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
            noise_np, 400.0, 100.0, fs=1000.0)
        self.assertTrue(isinstance(filtered_noise_np, np.ndarray))
        self.assertFalse(isinstance(filtered_noise_np, pq.quantity.Quantity))
        self.assertFalse(isinstance(filtered_noise_np, neo.AnalogSignal))
        self.assertEqual(filtered_noise_np.shape, noise_np.shape)

        # check input as Quantity array
        filtered_noise_pq = elephant.signal_processing.butter(
            noise_pq, 400.0 * pq.Hz, 100.0 * pq.Hz, fs=1000.0)
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
            noise, 250.0, fs=1000.0)
        filtered_noise_transposed = elephant.signal_processing.butter(
            noise.T, 250.0, fs=1000.0, axis=0)
        self.assertTrue(np.all(filtered_noise == filtered_noise_transposed.T))

    def test_butter_multidim_input(self):
        noise_pq = np.random.normal(size=(4, 5000)) * pq.mV
        noise_neo = neo.AnalogSignal(
            noise_pq.T, sampling_rate=1000.0 * pq.Hz)
        noise_neo1d = neo.AnalogSignal(
            noise_pq[0], sampling_rate=1000.0 * pq.Hz)
        filtered_noise_pq = elephant.signal_processing.butter(
            noise_pq, 250.0, fs=1000.0)
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

        self.long_signals = neo.AnalogSignal(
            sigs.T, units='mV',
            t_start=0. * pq.ms,
            sampling_rate=(len(time) / (time[-1] - time[0])).rescale(pq.Hz),
            dtype=float)

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
            self.long_signals, N='nextpow')
        self.assertEquals(np.shape(output), true_shape)
        self.assertEqual(output.units, pq.dimensionless)
        output = elephant.signal_processing.hilbert(
            self.long_signals, N=16384)
        self.assertEquals(np.shape(output), true_shape)
        self.assertEqual(output.units, pq.dimensionless)

    def test_hilbert_theoretical_long_signals(self):
        """
        Tests the output of the hilbert function with regard to amplitude and
        phase of long test signals
        """
        # Performing test using all pad types
        for padding in ['nextpow', 'none', 16384]:

            h = elephant.signal_processing.hilbert(
                self.long_signals, N=padding)

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
                self.one_period, N=padding)

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


if __name__ == '__main__':
    unittest.main()
