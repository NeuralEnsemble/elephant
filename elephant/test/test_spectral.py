# -*- coding: utf-8 -*-
"""
Unit tests for the spectral module.

:copyright: Copyright 2014-2020 by the Elephant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
"""

import unittest

import numpy as np
import scipy.signal as spsig
import quantities as pq
import neo.core as n
from numpy.testing import assert_array_almost_equal, assert_array_equal

import elephant.spectral


class WelchPSDTestCase(unittest.TestCase):
    def test_welch_psd_errors(self):
        # generate a dummy data
        data = n.AnalogSignal(np.zeros(5000), sampling_period=0.001 * pq.s,
                              units='mV')

        # check for invalid parameter values
        # - length of segments
        self.assertRaises(ValueError, elephant.spectral.welch_psd, data,
                          len_seg=0)
        self.assertRaises(ValueError, elephant.spectral.welch_psd, data,
                          len_seg=data.shape[0] * 2)
        # - number of segments
        self.assertRaises(ValueError, elephant.spectral.welch_psd, data,
                          num_seg=0)
        self.assertRaises(ValueError, elephant.spectral.welch_psd, data,
                          num_seg=data.shape[0] * 2)
        # - frequency resolution
        self.assertRaises(ValueError, elephant.spectral.welch_psd, data,
                          freq_res=-1)
        self.assertRaises(ValueError, elephant.spectral.welch_psd, data,
                          freq_res=data.sampling_rate / (data.shape[0] + 1))
        # - overlap
        self.assertRaises(ValueError, elephant.spectral.welch_psd, data,
                          overlap=-1.0)
        self.assertRaises(ValueError, elephant.spectral.welch_psd, data,
                          overlap=1.1)

    def test_welch_psd_behavior(self):
        # generate data by adding white noise and a sinusoid
        data_length = 5000
        sampling_period = 0.001
        signal_freq = 100.0
        noise = np.random.normal(size=data_length)
        signal = [np.sin(2 * np.pi * signal_freq * t)
                  for t in np.arange(0, data_length * sampling_period,
                                     sampling_period)]
        data = n.AnalogSignal(np.array(signal + noise),
                              sampling_period=sampling_period * pq.s,
                              units='mV')

        # consistency between different ways of specifying segment length
        freqs1, psd1 = elephant.spectral.welch_psd(
            data, len_segment=data_length // 5, overlap=0)
        freqs2, psd2 = elephant.spectral.welch_psd(
            data, n_segments=5, overlap=0)
        self.assertTrue((psd1 == psd2).all() and (freqs1 == freqs2).all())

        # frequency resolution and consistency with data
        freq_res = 1.0 * pq.Hz
        freqs, psd = elephant.spectral.welch_psd(
            data, frequency_resolution=freq_res)
        self.assertAlmostEqual(freq_res, freqs[1] - freqs[0])
        self.assertEqual(freqs[psd.argmax()], signal_freq)
        freqs_np, psd_np = elephant.spectral.welch_psd(
            data.magnitude.flatten(), fs=1 / sampling_period,
            frequency_resolution=freq_res)
        self.assertTrue((freqs == freqs_np).all() and (psd == psd_np).all())

        # check of scipy.signal.welch() parameters
        params = {'window': 'hamming', 'nfft': 1024, 'detrend': 'linear',
                  'return_onesided': False, 'scaling': 'spectrum'}
        for key, val in params.items():
            freqs, psd = elephant.spectral.welch_psd(
                data, len_segment=1000, overlap=0, **{key: val})
            freqs_spsig, psd_spsig = spsig.welch(np.rollaxis(data, 0, len(
                data.shape)), fs=1 / sampling_period, nperseg=1000,
                                                 noverlap=0, **{key: val})
            self.assertTrue(
                (freqs == freqs_spsig).all() and (
                    psd == psd_spsig).all())

        # - generate multidimensional data for check of parameter `axis`
        num_channel = 4
        data_length = 5000
        data_multidim = np.random.normal(size=(num_channel, data_length))
        freqs, psd = elephant.spectral.welch_psd(data_multidim)
        freqs_T, psd_T = elephant.spectral.welch_psd(data_multidim.T, axis=0)
        self.assertTrue(np.all(freqs == freqs_T))
        self.assertTrue(np.all(psd == psd_T.T))

    def test_welch_psd_input_types(self):
        # generate a test data
        sampling_period = 0.001
        data = n.AnalogSignal(np.array(np.random.normal(size=5000)),
                              sampling_period=sampling_period * pq.s,
                              units='mV')

        # outputs from AnalogSignal input are of Quantity type (standard usage)
        freqs_neo, psd_neo = elephant.spectral.welch_psd(data)
        self.assertTrue(isinstance(freqs_neo, pq.quantity.Quantity))
        self.assertTrue(isinstance(psd_neo, pq.quantity.Quantity))

        # outputs from Quantity array input are of Quantity type
        freqs_pq, psd_pq = elephant.spectral.welch_psd(
            data.magnitude.flatten() * data.units, fs=1 / sampling_period)
        self.assertTrue(isinstance(freqs_pq, pq.quantity.Quantity))
        self.assertTrue(isinstance(psd_pq, pq.quantity.Quantity))

        # outputs from Numpy ndarray input are NOT of Quantity type
        freqs_np, psd_np = elephant.spectral.welch_psd(
            data.magnitude.flatten(), fs=1 / sampling_period)
        self.assertFalse(isinstance(freqs_np, pq.quantity.Quantity))
        self.assertFalse(isinstance(psd_np, pq.quantity.Quantity))

        # check if the results from different input types are identical
        self.assertTrue(
            (freqs_neo == freqs_pq).all() and (
                psd_neo == psd_pq).all())
        self.assertTrue(
            (freqs_neo == freqs_np).all() and (
                psd_neo == psd_np).all())

    def test_welch_psd_multidim_input(self):
        # generate multidimensional data
        num_channel = 4
        data_length = 5000
        sampling_period = 0.001
        noise = np.random.normal(size=(num_channel, data_length))
        data_np = np.array(noise)
        # Since row-column order in AnalogSignal is different from the
        # conventional one, `data_np` needs to be transposed when its used to
        # define an AnalogSignal
        data_neo = n.AnalogSignal(data_np.T,
                                  sampling_period=sampling_period * pq.s,
                                  units='mV')
        data_neo_1dim = n.AnalogSignal(data_np[0],
                                       sampling_period=sampling_period * pq.s,
                                       units='mV')

        # check if the results from different input types are identical
        freqs_np, psd_np = elephant.spectral.welch_psd(data_np,
                                                       fs=1 / sampling_period)
        freqs_neo, psd_neo = elephant.spectral.welch_psd(data_neo)
        freqs_neo_1dim, psd_neo_1dim = elephant.spectral.welch_psd(
            data_neo_1dim)
        self.assertTrue(np.all(freqs_np == freqs_neo))
        self.assertTrue(np.all(psd_np == psd_neo))
        self.assertTrue(np.all(psd_neo_1dim == psd_neo[0]))


class WelchCohereTestCase(unittest.TestCase):
    def test_welch_cohere_errors(self):
        # generate a dummy data
        x = n.AnalogSignal(np.zeros(5000), sampling_period=0.001 * pq.s,
                           units='mV')
        y = n.AnalogSignal(np.zeros(5000), sampling_period=0.001 * pq.s,
                           units='mV')

        # check for invalid parameter values
        # - length of segments
        self.assertRaises(ValueError, elephant.spectral.welch_coherence, x, y,
                          len_seg=0)
        self.assertRaises(ValueError, elephant.spectral.welch_coherence, x, y,
                          len_seg=x.shape[0] * 2)
        # - number of segments
        self.assertRaises(ValueError, elephant.spectral.welch_coherence, x, y,
                          num_seg=0)
        self.assertRaises(ValueError, elephant.spectral.welch_coherence, x, y,
                          num_seg=x.shape[0] * 2)
        # - frequency resolution
        self.assertRaises(ValueError, elephant.spectral.welch_coherence, x, y,
                          freq_res=-1)
        self.assertRaises(ValueError, elephant.spectral.welch_coherence, x, y,
                          freq_res=x.sampling_rate / (x.shape[0] + 1))
        # - overlap
        self.assertRaises(ValueError, elephant.spectral.welch_coherence, x, y,
                          overlap=-1.0)
        self.assertRaises(ValueError, elephant.spectral.welch_coherence, x, y,
                          overlap=1.1)

    def test_welch_cohere_behavior(self):
        # generate data by adding white noise and a sinusoid
        data_length = 5000
        sampling_period = 0.001
        signal_freq = 100.0
        noise1 = np.random.normal(size=data_length) * 0.01
        noise2 = np.random.normal(size=data_length) * 0.01
        signal1 = [np.cos(2 * np.pi * signal_freq * t)
                   for t in np.arange(0, data_length * sampling_period,
                                      sampling_period)]
        signal2 = [np.sin(2 * np.pi * signal_freq * t)
                   for t in np.arange(0, data_length * sampling_period,
                                      sampling_period)]
        x = n.AnalogSignal(np.array(signal1 + noise1), units='mV',
                           sampling_period=sampling_period * pq.s)
        y = n.AnalogSignal(np.array(signal2 + noise2), units='mV',
                           sampling_period=sampling_period * pq.s)

        # consistency between different ways of specifying segment length
        freqs1, coherency1, phase_lag1 = elephant.spectral.welch_coherence(
            x, y, len_segment=data_length // 5, overlap=0)
        freqs2, coherency2, phase_lag2 = elephant.spectral.welch_coherence(
            x, y, n_segments=5, overlap=0)
        self.assertTrue((coherency1 == coherency2).all() and
                        (phase_lag1 == phase_lag2).all() and
                        (freqs1 == freqs2).all())

        # frequency resolution and consistency with data
        freq_res = 1.0 * pq.Hz
        freqs, coherency, phase_lag = elephant.spectral.welch_coherence(
            x, y, frequency_resolution=freq_res)
        self.assertAlmostEqual(freq_res, freqs[1] - freqs[0])
        self.assertAlmostEqual(freqs[coherency.argmax()], signal_freq,
                               places=2)
        self.assertAlmostEqual(phase_lag[coherency.argmax()], -np.pi / 2,
                               places=2)
        freqs_np, coherency_np, phase_lag_np =\
            elephant.spectral.welch_coherence(x.magnitude.flatten(),
                                              y.magnitude.flatten(),
                                              fs=1 / sampling_period,
                                              frequency_resolution=freq_res)
        assert_array_equal(freqs.simplified.magnitude, freqs_np)
        assert_array_equal(coherency[:, 0], coherency_np)
        assert_array_equal(phase_lag[:, 0], phase_lag_np)

        # - check the behavior of parameter `axis` using multidimensional data
        num_channel = 4
        data_length = 5000
        x_multidim = np.random.normal(size=(num_channel, data_length))
        y_multidim = np.random.normal(size=(num_channel, data_length))
        freqs, coherency, phase_lag =\
            elephant.spectral.welch_coherence(x_multidim, y_multidim)
        freqs_T, coherency_T, phase_lag_T = elephant.spectral.welch_coherence(
            x_multidim.T, y_multidim.T, axis=0)
        assert_array_equal(freqs, freqs_T)
        assert_array_equal(coherency, coherency_T.T)
        assert_array_equal(phase_lag, phase_lag_T.T)

    def test_welch_cohere_input_types(self):
        # generate a test data
        sampling_period = 0.001
        x = n.AnalogSignal(np.array(np.random.normal(size=5000)),
                           sampling_period=sampling_period * pq.s,
                           units='mV')
        y = n.AnalogSignal(np.array(np.random.normal(size=5000)),
                           sampling_period=sampling_period * pq.s,
                           units='mV')

        # outputs from AnalogSignal input are of Quantity type
        # (standard usage)
        freqs_neo, coherency_neo, phase_lag_neo =\
            elephant.spectral.welch_coherence(x, y)
        self.assertTrue(isinstance(freqs_neo, pq.quantity.Quantity))
        self.assertTrue(isinstance(phase_lag_neo, pq.quantity.Quantity))

        # outputs from Quantity array input are of Quantity type
        freqs_pq, coherency_pq, phase_lag_pq = elephant.spectral\
            .welch_coherence(x.magnitude.flatten() * x.units,
                             y.magnitude.flatten() * y.units,
                             fs=1 / sampling_period)
        self.assertTrue(isinstance(freqs_pq, pq.quantity.Quantity))
        self.assertTrue(isinstance(phase_lag_pq, pq.quantity.Quantity))

        # outputs from Numpy ndarray input are NOT of Quantity type
        freqs_np, coherency_np, phase_lag_np = elephant.spectral\
            .welch_coherence(x.magnitude.flatten(),
                             y.magnitude.flatten(),
                             fs=1 / sampling_period)
        self.assertFalse(isinstance(freqs_np, pq.quantity.Quantity))
        self.assertFalse(isinstance(phase_lag_np, pq.quantity.Quantity))

        # check if the results from different input types are identical
        self.assertTrue((freqs_neo == freqs_pq).all() and
                        (coherency_neo[:, 0] == coherency_pq).all() and
                        (phase_lag_neo[:, 0] == phase_lag_pq).all())
        self.assertTrue((freqs_neo == freqs_np).all() and
                        (coherency_neo[:, 0] == coherency_np).all() and
                        (phase_lag_neo[:, 0] == phase_lag_np).all())

    def test_welch_cohere_multidim_input(self):
        # generate multidimensional data
        num_channel = 4
        data_length = 5000
        sampling_period = 0.001
        x_np = np.array(np.random.normal(size=(num_channel, data_length)))
        y_np = np.array(np.random.normal(size=(num_channel, data_length)))
        # Since row-column order in AnalogSignal is different from the
        # convention in NumPy/SciPy, `data_np` needs to be transposed when its
        # used to define an AnalogSignal
        x_neo = n.AnalogSignal(x_np.T, units='mV',
                               sampling_period=sampling_period * pq.s)
        y_neo = n.AnalogSignal(y_np.T, units='mV',
                               sampling_period=sampling_period * pq.s)
        x_neo_1dim = n.AnalogSignal(x_np[0], units='mV',
                                    sampling_period=sampling_period * pq.s)
        y_neo_1dim = n.AnalogSignal(y_np[0], units='mV',
                                    sampling_period=sampling_period * pq.s)

        # check if the results from different input types are identical
        freqs_np, coherency_np, phase_lag_np = elephant.spectral\
            .welch_coherence(x_np, y_np, fs=1 / sampling_period)
        freqs_neo, coherency_neo, phase_lag_neo =\
            elephant.spectral.welch_coherence(x_neo, y_neo)
        freqs_neo_1dim, coherency_neo_1dim, phase_lag_neo_1dim =\
            elephant.spectral.welch_coherence(x_neo_1dim, y_neo_1dim)
        self.assertTrue(np.all(freqs_np == freqs_neo))
        self.assertTrue(np.all(coherency_np.T == coherency_neo))
        self.assertTrue(np.all(phase_lag_np.T == phase_lag_neo))
        self.assertTrue(
            np.all(coherency_neo_1dim[:, 0] == coherency_neo[:, 0]))
        self.assertTrue(
            np.all(phase_lag_neo_1dim[:, 0] == phase_lag_neo[:, 0]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
