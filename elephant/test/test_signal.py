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

import elephant.signal


class WelchPSDTestCase(unittest.TestCase):
    def test_welch_psd_errors(self):
        # generate a dummy data
        data = n.AnalogSignalArray(np.zeros(5000), sampling_period=0.001*pq.s,
                              units='mV')

        # check for invalid parameter values
        # - length of segments
        self.assertRaises(ValueError, elephant.signal.welch_psd, data,
                          len_seg=0)
        self.assertRaises(ValueError, elephant.signal.welch_psd, data,
                          len_seg=data.shape[-1] * 2)
        # - number of segments
        self.assertRaises(ValueError, elephant.signal.welch_psd, data,
                          num_seg=0)
        self.assertRaises(ValueError, elephant.signal.welch_psd, data,
                          num_seg=data.shape[-1] * 2)
        # - frequency resolution
        self.assertRaises(ValueError, elephant.signal.welch_psd, data,
                          freq_res=-1)
        self.assertRaises(ValueError, elephant.signal.welch_psd, data,
                          freq_res=data.sampling_rate/(data.shape[-1]+1))
        # - overlap
        self.assertRaises(ValueError, elephant.signal.welch_psd, data,
                          overlap=-1.0)
        self.assertRaises(ValueError, elephant.signal.welch_psd, data,
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
        data_multidim = n.AnalogSignalArray(np.array([signal+noise] * 5),
                                   sampling_period=sampling_period*pq.s,
                                   units='mV')

        # consistency between different ways of specifying segment length
        freqs1, psd1 = elephant.signal.welch_psd(data, len_seg=data_length/5, overlap=0)
        freqs2, psd2 = elephant.signal.welch_psd(data, num_seg=5, overlap=0)
        self.assertTrue(np.all((psd1==psd2, freqs1==freqs2)))

        # frequency resolution and consistency with data
        freq_res = 1.0 * pq.Hz
        freqs, psd = elephant.signal.welch_psd(data, freq_res=freq_res)
        self.assertAlmostEqual(freq_res, freqs[1]-freqs[0])
        self.assertEqual(freqs[psd.argmax()], signal_freq)
        freqs_np, psd_np = elephant.signal.welch_psd(data.magnitude, fs=1/sampling_period, freq_res=freq_res)
        self.assertTrue(np.all((freqs==freqs_np, psd==psd_np)))

        # multi-dimensional return value for multi-dimensional input array
        freqs, psds = elephant.signal.welch_psd(data_multidim)
        self.assertTrue(psds.shape[:-1] == data_multidim.shape[:-1])

        # check of scipy.signal.welch() parameters
        params = {'window': 'hamming', 'nfft': 1024, 'detrend': 'linear',
                  'return_onesided': False, 'scaling': 'spectrum'}
        for key, val in params.items():
            freq, psd = elephant.signal.welch_psd(data, len_seg=1000, overlap=0, **{key: val})
            freq_spsig, psd_spsig = spsig.welch(data, fs=1/sampling_period, nperseg=1000, noverlap=0, **{key: val})
            self.assertTrue(np.all((freq==freq_spsig, psd==psd_spsig)))

        freqs, psds = elephant.signal.welch_psd(data_multidim)
        freqs_spsig, psds_spsig = elephant.signal.welch_psd(data_multidim.T, axis=0)
        self.assertTrue(np.all((freq==freq_spsig, psd==psd_spsig)))

    def test_welch_psd_input_types(self):
        # generate a test data
        sampling_period = 0.001
        data = n.AnalogSignalArray(np.array(np.random.normal(size=5000)),
                                   sampling_period=sampling_period*pq.s,
                                   units='mV')

        # outputs from AnalogSignalArray input are of Quantity type (standard usage)
        freqs_neo, psd_neo = elephant.signal.welch_psd(data)
        self.assertTrue(isinstance(freqs_neo, pq.quantity.Quantity))
        self.assertTrue(isinstance(psd_neo, pq.quantity.Quantity))

        # outputs from Quantity array input are of Quantity type
        freqs_pq, psd_pq = elephant.signal.welch_psd(data.magnitude*data.units, fs=1/sampling_period)
        self.assertTrue(np.all((freqs_neo==freqs_pq, psd_neo==psd_pq)))
        self.assertTrue(isinstance(freqs_pq, pq.quantity.Quantity))
        self.assertTrue(isinstance(psd_pq, pq.quantity.Quantity))

        # outputs from Numpy ndarray input are NOT of Quantity type
        freqs_np, psd_np = elephant.signal.welch_psd(data.magnitude, fs=1/sampling_period)
        self.assertTrue(np.all((freqs_neo==freqs_np, psd_neo==psd_np)))
        self.assertFalse(isinstance(freqs_np, pq.quantity.Quantity))
        self.assertFalse(isinstance(psd_np, pq.quantity.Quantity))


def suite():
    suite = unittest.makeSuite(WelchPSDTestCase, 'test')
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())