# -*- coding: utf-8 -*-
"""
docstring goes here.
:copyright: Copyright 2015 by the Elephant team, see AUTHORS.txt.
:license: Modified BSD, see LICENSE.txt for details.
"""

import unittest

import numpy as np
import scipy.signal as spsig
import quantities as pq
import neo.core as n

import elephant.spectral


class WelchPSDTestCase(unittest.TestCase):
    def test_welch_psd_errors(self):
        # generate a dummy data
        data = n.AnalogSignalArray(np.zeros(5000), sampling_period=0.001*pq.s,
                              units='mV')

        # check for invalid parameter values
        # - length of segments
        self.assertRaises(ValueError, elephant.spectral.welch_psd, data,
                          len_seg=0)
        self.assertRaises(ValueError, elephant.spectral.welch_psd, data,
                          len_seg=data.shape[-1] * 2)
        # - number of segments
        self.assertRaises(ValueError, elephant.spectral.welch_psd, data,
                          num_seg=0)
        self.assertRaises(ValueError, elephant.spectral.welch_psd, data,
                          num_seg=data.shape[-1] * 2)
        # - frequency resolution
        self.assertRaises(ValueError, elephant.spectral.welch_psd, data,
                          freq_res=-1)
        self.assertRaises(ValueError, elephant.spectral.welch_psd, data,
                          freq_res=data.sampling_rate/(data.shape[-1]+1))
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
        signal = [np.sin(2*np.pi*signal_freq*t)
                  for t in np.arange(0, data_length*sampling_period,
                                     sampling_period)]
        data = n.AnalogSignalArray(np.array(signal+noise),
                                      sampling_period=sampling_period*pq.s,
                                      units='mV')

        # consistency between different ways of specifying segment length
        freqs1, psd1 = elephant.spectral.welch_psd(data, len_seg=data_length//5, overlap=0)
        freqs2, psd2 = elephant.spectral.welch_psd(data, num_seg=5, overlap=0)
        self.assertTrue(np.all((psd1==psd2, freqs1==freqs2)))

        # frequency resolution and consistency with data
        freq_res = 1.0 * pq.Hz
        freqs, psd = elephant.spectral.welch_psd(data, freq_res=freq_res)
        self.assertAlmostEqual(freq_res, freqs[1]-freqs[0])
        self.assertEqual(freqs[psd.argmax()], signal_freq)
        freqs_np, psd_np = elephant.spectral.welch_psd(data.magnitude, fs=1/sampling_period, freq_res=freq_res)
        self.assertTrue(np.all((freqs==freqs_np, psd==psd_np)))

        # check of scipy.signal.welch() parameters
        params = {'window': 'hamming', 'nfft': 1024, 'detrend': 'linear',
                  'return_onesided': False, 'scaling': 'spectrum'}
        for key, val in params.items():
            freqs, psd = elephant.spectral.welch_psd(data, len_seg=1000, overlap=0, **{key: val})
            freqs_spsig, psd_spsig = spsig.welch(data, fs=1/sampling_period, nperseg=1000, noverlap=0, **{key: val})
            self.assertTrue(np.all((freqs==freqs_spsig, psd==psd_spsig)))

        # - generate multidimensional data for check of parameter `axis`
        num_channel = 4
        data_length = 5000
        data_multidim = np.random.normal(size=(num_channel, data_length))
        freqs, psd = elephant.spectral.welch_psd(data_multidim)
        freqs_T, psd_T = elephant.spectral.welch_psd(data_multidim.T, axis=0)
        self.assertTrue(np.all(freqs==freqs_T))
        self.assertTrue(np.all(psd==psd_T.T))

    def test_welch_psd_input_types(self):
        # generate a test data
        sampling_period = 0.001
        data = n.AnalogSignalArray(np.array(np.random.normal(size=5000)),
                                   sampling_period=sampling_period*pq.s,
                                   units='mV')

        # outputs from AnalogSignalArray input are of Quantity type (standard usage)
        freqs_neo, psd_neo = elephant.spectral.welch_psd(data)
        self.assertTrue(isinstance(freqs_neo, pq.quantity.Quantity))
        self.assertTrue(isinstance(psd_neo, pq.quantity.Quantity))

        # outputs from Quantity array input are of Quantity type
        freqs_pq, psd_pq = elephant.spectral.welch_psd(data.magnitude*data.units, fs=1/sampling_period)
        self.assertTrue(isinstance(freqs_pq, pq.quantity.Quantity))
        self.assertTrue(isinstance(psd_pq, pq.quantity.Quantity))

        # outputs from Numpy ndarray input are NOT of Quantity type
        freqs_np, psd_np = elephant.spectral.welch_psd(data.magnitude, fs=1/sampling_period)
        self.assertFalse(isinstance(freqs_np, pq.quantity.Quantity))
        self.assertFalse(isinstance(psd_np, pq.quantity.Quantity))

        # check if the results from different input types are identical
        self.assertTrue(np.all((freqs_neo==freqs_pq, psd_neo==psd_pq)))
        self.assertTrue(np.all((freqs_neo==freqs_np, psd_neo==psd_np)))

    def test_welch_psd_multidim_input(self):
        # generate multidimensional data
        num_channel = 4
        data_length = 5000
        sampling_period = 0.001
        noise = np.random.normal(size=(num_channel, data_length))
        data_np = np.array(noise)
        # Since row-column order in AnalogSignalArray is different from the
        # conventional one, `data_np` needs to be transposed when its used to
        # define an AnalogSignalArray
        data_neo = n.AnalogSignalArray(data_np.T,
                                       sampling_period=sampling_period*pq.s,
                                       units='mV')
        data_neo_1dim = n.AnalogSignalArray(data_np[0],
                                       sampling_period=sampling_period*pq.s,
                                       units='mV')

        # check if the results from different input types are identical
        freqs_np, psd_np = elephant.spectral.welch_psd(data_np,
                                                     fs=1/sampling_period)
        freqs_neo, psd_neo = elephant.spectral.welch_psd(data_neo)
        freqs_neo_1dim, psd_neo_1dim = elephant.spectral.welch_psd(data_neo_1dim)
        self.assertTrue(np.all(freqs_np==freqs_neo))
        self.assertTrue(np.all(psd_np==psd_neo))
        self.assertTrue(np.all(psd_neo_1dim==psd_neo[0]))


class WelchCohereTestCase(unittest.TestCase):
    def test_welch_cohere_errors(self):
        # generate a dummy data
        x = n.AnalogSignalArray(np.zeros(5000), sampling_period=0.001*pq.s,
            units='mV')
        y = n.AnalogSignalArray(np.zeros(5000), sampling_period=0.001*pq.s,
            units='mV')

        # check for invalid parameter values
        # - length of segments
        self.assertRaises(ValueError, elephant.spectral.welch_cohere, x, y,
            len_seg=0)
        self.assertRaises(ValueError, elephant.spectral.welch_cohere, x, y,
            len_seg=x.shape[-1] * 2)
        # - number of segments
        self.assertRaises(ValueError, elephant.spectral.welch_cohere, x, y,
            num_seg=0)
        self.assertRaises(ValueError, elephant.spectral.welch_cohere, x, y,
            num_seg=x.shape[-1] * 2)
        # - frequency resolution
        self.assertRaises(ValueError, elephant.spectral.welch_cohere, x, y,
            freq_res=-1)
        self.assertRaises(ValueError, elephant.spectral.welch_cohere, x, y,
            freq_res=x.sampling_rate/(x.shape[-1]+1))
        # - overlap
        self.assertRaises(ValueError, elephant.spectral.welch_cohere, x, y,
            overlap=-1.0)
        self.assertRaises(ValueError, elephant.spectral.welch_cohere, x, y,
            overlap=1.1)

    def test_welch_cohere_behavior(self):
        # generate data by adding white noise and a sinusoid
        data_length = 5000
        sampling_period = 0.001
        signal_freq = 100.0
        noise1 = np.random.normal(size=data_length) * 0.01
        noise2 = np.random.normal(size=data_length) * 0.01
        signal1 = [np.cos(2*np.pi*signal_freq*t)
                  for t in np.arange(0, data_length*sampling_period,
                sampling_period)]
        signal2 = [np.sin(2*np.pi*signal_freq*t)
                   for t in np.arange(0, data_length*sampling_period,
                sampling_period)]
        x = n.AnalogSignalArray(np.array(signal1+noise1), units='mV',
            sampling_period=sampling_period*pq.s)
        y = n.AnalogSignalArray(np.array(signal2+noise2), units='mV',
            sampling_period=sampling_period*pq.s)

        # consistency between different ways of specifying segment length
        freqs1, coherency1, phase_lag1 = elephant.spectral.welch_cohere(x, y,
            len_seg=data_length//5, overlap=0)
        freqs2, coherency2, phase_lag2 = elephant.spectral.welch_cohere(x, y,
            num_seg=5, overlap=0)
        self.assertTrue(np.all((coherency1==coherency2,
                                phase_lag1==phase_lag2,
                                freqs1==freqs2)))

        # frequency resolution and consistency with data
        freq_res = 1.0 * pq.Hz
        freqs, coherency, phase_lag = elephant.spectral.welch_cohere(x, y,
            freq_res=freq_res)
        self.assertAlmostEqual(freq_res, freqs[1]-freqs[0])
        self.assertAlmostEqual(freqs[coherency.argmax()], signal_freq,
            places=2)
        self.assertAlmostEqual(phase_lag[coherency.argmax()], np.pi/2,
            places=2)
        freqs_np, coherency_np, phase_lag_np =\
            elephant.spectral.welch_cohere(x.magnitude, y.magnitude,
                fs=1/sampling_period, freq_res=freq_res)
        self.assertTrue(np.all((freqs==freqs_np,
                                coherency==coherency_np,
                                phase_lag==phase_lag_np)))

        # - check the behavior of parameter `axis` using multidimensional data
        num_channel = 4
        data_length = 5000
        x_multidim = np.random.normal(size=(num_channel, data_length))
        y_multidim = np.random.normal(size=(num_channel, data_length))
        freqs, coherency, phase_lag =\
            elephant.spectral.welch_cohere(x_multidim, y_multidim)
        freqs_T, coherency_T, phase_lag_T =\
            elephant.spectral.welch_cohere(x_multidim.T, y_multidim.T, axis=0)
        self.assertTrue(np.all(freqs==freqs_T))
        self.assertTrue(np.all(coherency==coherency_T.T))
        self.assertTrue(np.all(phase_lag==phase_lag_T.T))

    def test_welch_cohere_input_types(self):
        # generate a test data
        sampling_period = 0.001
        x = n.AnalogSignalArray(np.array(np.random.normal(size=5000)),
            sampling_period=sampling_period*pq.s,
            units='mV')
        y = n.AnalogSignalArray(np.array(np.random.normal(size=5000)),
            sampling_period=sampling_period*pq.s,
            units='mV')

        # outputs from AnalogSignalArray input are of Quantity type
        # (standard usage)
        freqs_neo, coherency_neo, phase_lag_neo =\
            elephant.spectral.welch_cohere(x, y)
        self.assertTrue(isinstance(freqs_neo, pq.quantity.Quantity))
        self.assertTrue(isinstance(phase_lag_neo, pq.quantity.Quantity))

        # outputs from Quantity array input are of Quantity type
        freqs_pq, coherency_pq, phase_lag_pq =\
            elephant.spectral.welch_cohere(x.magnitude*x.units,
                y.magnitude*y.units, fs=1/sampling_period)
        self.assertTrue(isinstance(freqs_pq, pq.quantity.Quantity))
        self.assertTrue(isinstance(phase_lag_pq, pq.quantity.Quantity))

        # outputs from Numpy ndarray input are NOT of Quantity type
        freqs_np, coherency_np, phase_lag_np =\
            elephant.spectral.welch_cohere(x.magnitude, y.magnitude,
                fs=1/sampling_period)
        self.assertFalse(isinstance(freqs_np, pq.quantity.Quantity))
        self.assertFalse(isinstance(phase_lag_np, pq.quantity.Quantity))

        # check if the results from different input types are identical
        self.assertTrue(np.all((freqs_neo==freqs_pq,
                                coherency_neo==coherency_pq,
                                phase_lag_neo==phase_lag_pq)))
        self.assertTrue(np.all((freqs_neo==freqs_np,
                                coherency_neo==coherency_np,
                                phase_lag_neo==phase_lag_np)))

    def test_welch_cohere_multidim_input(self):
        # generate multidimensional data
        num_channel = 4
        data_length = 5000
        sampling_period = 0.001
        x_np = np.array(np.random.normal(size=(num_channel, data_length)))
        y_np = np.array(np.random.normal(size=(num_channel, data_length)))
        # Since row-column order in AnalogSignalArray is different from the
        # convention in NumPy/SciPy, `data_np` needs to be transposed when its
        # used to define an AnalogSignalArray
        x_neo = n.AnalogSignalArray(x_np.T, units='mV',
            sampling_period=sampling_period*pq.s)
        y_neo = n.AnalogSignalArray(y_np.T, units='mV',
            sampling_period=sampling_period*pq.s)
        x_neo_1dim = n.AnalogSignalArray(x_np[0], units='mV',
            sampling_period=sampling_period*pq.s)
        y_neo_1dim = n.AnalogSignalArray(y_np[0], units='mV',
            sampling_period=sampling_period*pq.s)

        # check if the results from different input types are identical
        freqs_np, coherency_np, phase_lag_np =\
            elephant.spectral.welch_cohere(x_np, y_np, fs=1/sampling_period)
        freqs_neo, coherency_neo, phase_lag_neo =\
            elephant.spectral.welch_cohere(x_neo, y_neo)
        freqs_neo_1dim, coherency_neo_1dim, phase_lag_neo_1dim =\
            elephant.spectral.welch_cohere(x_neo_1dim, y_neo_1dim)
        self.assertTrue(np.all(freqs_np==freqs_neo))
        self.assertTrue(np.all(coherency_np.T==coherency_neo))
        self.assertTrue(np.all(phase_lag_np.T==phase_lag_neo))
        self.assertTrue(np.all(coherency_neo_1dim==coherency_neo[:, 0]))
        self.assertTrue(np.all(phase_lag_neo_1dim==phase_lag_neo[:, 0]))


def suite():
    suite = unittest.makeSuite(WelchPSDTestCase, 'test')
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())