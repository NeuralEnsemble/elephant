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


class ButterTestCase(unittest.TestCase):
    def test_butter_filter_type(self):
        """
        Test if correct type of filtering is performed according to how cut-off
        frequencies are given
        """
        # generate white noise AnalogSignal
        noise = n.AnalogSignal(np.random.normal(size=5000),
                               sampling_rate=1000 * pq.Hz, units='mV')

        # test high-pass filtering: power at the lowest frequency
        # should be almost zero
        # Note: the default detrend function of scipy.signal.welch() seems to
        # cause artificial finite power at the lowest frequencies. Here I avoid
        # this by using an identity function for detrending
        filtered_noise = elephant.signal.butter(noise, 250.0 * pq.Hz, None)
        _, psd = spsig.welch(filtered_noise, nperseg=1024, fs=1000.0,
                             detrend=lambda x: x)
        self.assertAlmostEqual(psd[0], 0)

        # test low-pass filtering: power at the highest frequency
        # should be almost zero
        filtered_noise = elephant.signal.butter(noise, None, 250.0 * pq.Hz)
        _, psd = spsig.welch(filtered_noise, nperseg=1024, fs=1000.0)
        self.assertAlmostEqual(psd[-1], 0)

        # test band-pass filtering: power at the lowest and highest frequencies
        # should be almost zero
        filtered_noise = elephant.signal.butter(noise, 200.0 * pq.Hz,
                                                300.0 * pq.Hz)
        _, psd = spsig.welch(filtered_noise, nperseg=1024, fs=1000.0,
                             detrend=lambda x: x)
        self.assertAlmostEqual(psd[0], 0)
        self.assertAlmostEqual(psd[-1], 0)

        # test band-stop filtering: power at the intermediate frequency
        # should be almost zero
        filtered_noise = elephant.signal.butter(noise, 400.0 * pq.Hz,
                                                100.0 * pq.Hz)
        _, psd = spsig.welch(filtered_noise, nperseg=1024, fs=1000.0)
        self.assertAlmostEqual(psd[256], 0)

    def test_butter_filter_function(self):
        # generate white noise AnalogSignal
        noise = n.AnalogSignal(np.random.normal(size=5000),
                               sampling_rate=1000 * pq.Hz, units='mV')

        # test if the filter performance is as well with filftunc=lfilter as
        # with filtfunc=filtfilt (i.e. default option)
        kwds = {'anasig': noise, 'highpassfreq': 250.0 * pq.Hz,
                'lowpassfreq': None, 'filtfunc': 'filtfilt'}
        filtered_noise = elephant.signal.butter(**kwds)
        _, psd_filtfilt = spsig.welch(filtered_noise, nperseg=1024, fs=1000.0, detrend=lambda x: x)

        kwds['filtfunc'] = 'lfilter'
        filtered_noise = elephant.signal.butter(**kwds)
        _, psd_lfilter = spsig.welch(filtered_noise, nperseg=1024, fs=1000.0, detrend=lambda x: x)

        self.assertAlmostEqual(psd_filtfilt[0], psd_lfilter[0])

    def test_butter_invalid_filter_function(self):
        # generate a dummy AnalogSignal
        anasig_dummy = n.AnalogSignal(np.zeros(5000),
                                      sampling_rate=1000 * pq.Hz, units='mV')
        # test exception upon invalid filtfunc string
        kwds = {'anasig': anasig_dummy, 'highpassfreq': 250.0 * pq.Hz,
                'filtfunc': 'invalid_filter'}
        self.assertRaises(ValueError, elephant.signal.butter, **kwds)

    def test_butter_missing_cutoff_freqs(self):
        # generate a dummy AnalogSignal
        anasig_dummy = n.AnalogSignal(np.zeros(5000),
                                      sampling_rate=1000 * pq.Hz, units='mV')
        # test a case where no cut-off frequencies are given
        kwds = {'anasig': anasig_dummy, 'highpassfreq': None,
                'lowpassfreq': None}
        self.assertRaises(ValueError, elephant.signal.butter, **kwds)



def suite():
    suite = unittest.makeSuite(ButterTestCase, 'test')
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())
