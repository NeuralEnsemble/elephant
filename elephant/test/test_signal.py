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
import neo
import neo.core as n

import elephant.signal_processing


class ButterTestCase(unittest.TestCase):
    def test_butter_filter_type(self):
        """
        Test if correct type of filtering is performed according to how cut-off
        frequencies are given
        """
        # generate white noise AnalogSignalArray
        noise = n.AnalogSignalArray(np.random.normal(size=5000),
                               sampling_rate=1000 * pq.Hz, units='mV')

        # test high-pass filtering: power at the lowest frequency
        # should be almost zero
        # Note: the default detrend function of scipy.signal.welch() seems to
        # cause artificial finite power at the lowest frequencies. Here I avoid
        # this by using an identity function for detrending
        filtered_noise = elephant.signal_processing.butter(noise, 250.0 * pq.Hz, None)
        _, psd = spsig.welch(filtered_noise, nperseg=1024, fs=1000.0,
                             detrend=lambda x: x)
        self.assertAlmostEqual(psd[0], 0)

        # test low-pass filtering: power at the highest frequency
        # should be almost zero
        filtered_noise = elephant.signal_processing.butter(noise, None, 250.0 * pq.Hz)
        _, psd = spsig.welch(filtered_noise, nperseg=1024, fs=1000.0)
        self.assertAlmostEqual(psd[-1], 0)

        # test band-pass filtering: power at the lowest and highest frequencies
        # should be almost zero
        filtered_noise = elephant.signal_processing.butter(noise, 200.0 * pq.Hz,
                                                300.0 * pq.Hz)
        _, psd = spsig.welch(filtered_noise, nperseg=1024, fs=1000.0,
                             detrend=lambda x: x)
        self.assertAlmostEqual(psd[0], 0)
        self.assertAlmostEqual(psd[-1], 0)

        # test band-stop filtering: power at the intermediate frequency
        # should be almost zero
        filtered_noise = elephant.signal_processing.butter(noise, 400.0 * pq.Hz,
                                                100.0 * pq.Hz)
        _, psd = spsig.welch(filtered_noise, nperseg=1024, fs=1000.0)
        self.assertAlmostEqual(psd[256], 0)

    def test_butter_filter_function(self):
        # generate white noise AnalogSignalArray
        noise = n.AnalogSignalArray(np.random.normal(size=5000),
                               sampling_rate=1000 * pq.Hz, units='mV')

        # test if the filter performance is as well with filftunc=lfilter as
        # with filtfunc=filtfilt (i.e. default option)
        kwds = {'signal': noise, 'highpass_freq': 250.0 * pq.Hz,
                'lowpass_freq': None, 'filter_function': 'filtfilt'}
        filtered_noise = elephant.signal_processing.butter(**kwds)
        _, psd_filtfilt = spsig.welch(filtered_noise, nperseg=1024, fs=1000.0, detrend=lambda x: x)

        kwds['filter_function'] = 'lfilter'
        filtered_noise = elephant.signal_processing.butter(**kwds)
        _, psd_lfilter = spsig.welch(filtered_noise, nperseg=1024, fs=1000.0, detrend=lambda x: x)

        self.assertAlmostEqual(psd_filtfilt[0], psd_lfilter[0])

    def test_butter_invalid_filter_function(self):
        # generate a dummy AnalogSignalArray
        anasig_dummy = n.AnalogSignalArray(np.zeros(5000),
                                      sampling_rate=1000 * pq.Hz, units='mV')
        # test exception upon invalid filtfunc string
        kwds = {'signal': anasig_dummy, 'highpass_freq': 250.0 * pq.Hz,
                'filter_function': 'invalid_filter'}
        self.assertRaises(ValueError, elephant.signal_processing.butter, **kwds)

    def test_butter_missing_cutoff_freqs(self):
        # generate a dummy AnalogSignalArray
        anasig_dummy = n.AnalogSignalArray(np.zeros(5000),
                                      sampling_rate=1000 * pq.Hz, units='mV')
        # test a case where no cut-off frequencies are given
        kwds = {'signal': anasig_dummy, 'highpass_freq': None,
                'lowpass_freq': None}
        self.assertRaises(ValueError, elephant.signal_processing.butter, **kwds)

    def test_butter_input_types(self):
        # generate white noise data of different types
        noise_np = np.random.normal(size=(4, 5000))
        noise_pq = noise_np * pq.mV
        noise = n.AnalogSignalArray(noise_pq, sampling_rate=1000.0*pq.Hz)

        # check input as NumPy ndarray
        filtered_noise_np = elephant.signal_processing.butter(noise_np, 400.0,
                                                           100.0, fs=1000.0)
        self.assertTrue(isinstance(filtered_noise_np, np.ndarray))
        self.assertFalse(isinstance(filtered_noise_np, pq.quantity.Quantity))
        self.assertFalse(isinstance(filtered_noise_np, neo.AnalogSignalArray))
        self.assertEqual(filtered_noise_np.shape, noise_np.shape)

        # check input as Quantity array
        filtered_noise_pq = elephant.signal_processing.butter(noise_pq,
                                                           400.0*pq.Hz,
                                                           100.0*pq.Hz,
                                                           fs=1000.0)
        self.assertTrue(isinstance(filtered_noise_pq, pq.quantity.Quantity))
        self.assertFalse(isinstance(filtered_noise_pq, neo.AnalogSignalArray))
        self.assertEqual(filtered_noise_pq.shape, noise_pq.shape)

        # check input as neo AnalogSignalArray
        filtered_noise = elephant.signal_processing.butter(noise,
                                                           400.0 * pq.Hz,
                                                           100.0 * pq.Hz)
        self.assertTrue(isinstance(filtered_noise, neo.AnalogSignalArray))
        self.assertEqual(filtered_noise.shape, noise.shape)

        # check if the results from different input types are identical
        self.assertTrue(np.all(filtered_noise_pq.magnitude==filtered_noise_np))
        self.assertTrue(np.all(filtered_noise.magnitude==filtered_noise_np))


    def test_butter_axis(self):
        noise = np.random.normal(size=(4, 5000))
        filtered_noise = elephant.signal_processing.butter(noise, 250.0,
                                                           fs=1000.0)
        filtered_noise_transposed = elephant.signal_processing.butter(noise.T,
                                                             250.0,
                                                             fs=1000.0,
                                                             axis=0)
        self.assertTrue(np.all(filtered_noise==filtered_noise_transposed.T))

def suite():
    suite = unittest.makeSuite(ButterTestCase, 'test')
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())
