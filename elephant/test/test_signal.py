# -*- coding: utf-8 -*-
"""
docstring goes here.
:copyright: Copyright 2015 by the Elephant team, see AUTHORS.txt.
:license: Modified BSD, see LICENSE.txt for details.
"""

import unittest

import numpy as np
import quantities as pq
import neo.core as n

import elephant.signal


class WelchPSDTestCase(unittest.TestCase):
    def test_welchpsd_errors(self):
        # generate a dummy data
        data = n.AnalogSignal(np.zeros(5000), sampling_period=0.001*pq.s,
                              units='mV')

        # check for unsupported parameter
        self.assertRaises(ValueError, elephant.signal.welchpsd,
                          data, unsupported_parameter=1)

        # check for invalid parameter values
        # - length of segments
        self.assertRaises(ValueError, elephant.signal.welchpsd, data,
                          len_seg=0)
        self.assertRaises(ValueError, elephant.signal.welchpsd, data,
                          len_seg=data.shape[-1] * 2)
        # - number of segments
        self.assertRaises(ValueError, elephant.signal.welchpsd, data,
                          num_seg=0)
        self.assertRaises(ValueError, elephant.signal.welchpsd, data,
                          num_seg=data.shape[-1] * 2)
        # - frequency resolution
        self.assertRaises(ValueError, elephant.signal.welchpsd, data,
                          freq_res=-1)
        self.assertRaises(ValueError, elephant.signal.welchpsd, data,
                          freq_res=data.sampling_rate/(data.shape[-1]+1))
        # - overlap
        self.assertRaises(ValueError, elephant.signal.welchpsd, data,
                          overlap=-1.0)
        self.assertRaises(ValueError, elephant.signal.welchpsd, data,
                          overlap=1.1)

    def test_welchpsd_behavior(self):
        # generate data by adding white noise and a sinusoid
        data_length = 5000
        sampling_period = 0.001
        signal_freq = 100.0
        noise = np.random.normal(size=data_length)
        signal = [np.sin(2*np.pi*signal_freq*t)
                  for t in np.arange(0, data_length*sampling_period,
                                     sampling_period)]
        data = n.AnalogSignal(signal+noise,
                              sampling_period=sampling_period*pq.s, units='mV')
        dataarr = n.AnalogSignalArray(np.array([signal+noise,] * 5),
                                      sampling_period=sampling_period*pq.s,
                                      units='mV')

        # consistency between different ways of specifying segment length
        psd1, freqs1 = elephant.signal.welchpsd(data, len_seg=1000, overlap=0)
        psd2, freqs2 = elephant.signal.welchpsd(data, num_seg=5, overlap=0)
        self.assertTrue(np.all(psd1 == psd2))
        self.assertTrue(np.all(freqs1 == freqs2))

        # frequency resolution and consistency with data
        freq_res = 1.0 * pq.Hz
        psd, freqs = elephant.signal.welchpsd(data, freq_res=freq_res)
        self.assertAlmostEqual(freq_res, freqs[1]-freqs[0])
        self.assertEqual(freqs[psd.argmax()], signal_freq)

        # same results for AnalogSignal and AnalogSignalArray
        psd, freqs = elephant.signal.welchpsd(data)
        psds, freqs = elephant.signal.welchpsd(dataarr)
        for psd_from_arr in psds:
            self.assertTrue(np.all(psd == psd_from_arr))


def suite():
    suite = unittest.makeSuite(WelchPSDTestCase, 'test')
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())