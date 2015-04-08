# -*- coding: utf-8 -*-
"""
Tests for the function sta.spike_triggered_average

:copyright: Copyright 2015 by the Elephant team, see AUTHORS.txt.
:license: Modified BSD, see LICENSE.txt for details.
"""

import unittest
import math
import numpy as np
from numpy.testing import assert_array_equal
from numpy.testing.utils import assert_array_almost_equal
import neo
from neo import AnalogSignalArray, SpikeTrain
import quantities as pq
from quantities import ms, mV, Hz
import elephant.sta as sta
import warnings

class sta_TestCase(unittest.TestCase):

    def setUp(self):
        self.asiga0 = AnalogSignalArray(np.array([
            np.sin(np.arange(0, 20 * math.pi, 0.1))]).T, 
            units='mV', sampling_rate=10 / ms)
        self.asiga1 = AnalogSignalArray(np.array([
            np.sin(np.arange(0, 20 * math.pi, 0.1)), 
            np.cos(np.arange(0, 20 * math.pi, 0.1))]).T, 
            units='mV', sampling_rate=10 / ms)
        self.asiga2 = AnalogSignalArray(np.array([
            np.sin(np.arange(0, 20 * math.pi, 0.1)), 
            np.cos(np.arange(0, 20 * math.pi, 0.1)), 
            np.tan(np.arange(0, 20 * math.pi, 0.1))]).T, 
            units='mV', sampling_rate=10 / ms)
        self.st0 = SpikeTrain(
            [9 * math.pi, 10 * math.pi, 11 * math.pi, 12 * math.pi], 
            units='ms', t_stop=self.asiga0.t_stop)
        self.lst = [SpikeTrain(
            [9 * math.pi, 10 * math.pi, 11 * math.pi, 12 * math.pi], 
            units='ms', t_stop=self.asiga1.t_stop), 
            SpikeTrain([30, 35, 40], units='ms', t_stop=self.asiga1.t_stop)]

    #***********************************************************************
    #************************ Test for typical values **********************

    def test_spike_triggered_average_with_n_spikes_on_constant_function(self):
        '''Signal should average to the input'''
        const = 13.8
        x = const * np.ones(201)
        asiga = AnalogSignalArray(
            np.array([x]).T, units='mV', sampling_rate=10 / ms)
        st = SpikeTrain([3, 5.6, 7, 7.1, 16, 16.3], units='ms', t_stop=20)
        window_starttime = -2 * ms
        window_endtime = 2 * ms
        STA = sta.spike_triggered_average(
            asiga, st, (window_starttime, window_endtime))
        a = int(((window_endtime - window_starttime) *
                asiga.sampling_rate).simplified)
        cutout = asiga[0: a]
        cutout.t_start = window_starttime
        assert_array_almost_equal(STA, cutout, 12)

    def test_spike_triggered_average_with_shifted_sin_wave(self):
        '''Signal should average to zero'''
        STA = sta.spike_triggered_average(
            self.asiga0, self.st0, (-4 * ms, 4 * ms))
        target = 5e-2 * mV
        self.assertEqual(np.abs(STA).max().dimensionality.simplified, 
                         pq.Quantity(1, "V").dimensionality.simplified)
        self.assertLess(np.abs(STA).max(), target)

    def test_only_one_spike(self):
        '''The output should be the same as the input'''
        x = np.arange(0, 20, 0.1)
        y = x**2
        sr = 10 / ms
        z = AnalogSignalArray(np.array([y]).T, units='mV', sampling_rate=sr)
        spiketime = 8 * ms
        spiketime_in_ms = int((spiketime / ms).simplified)
        st = SpikeTrain([spiketime_in_ms], units='ms', t_stop=20)
        window_starttime = -3 * ms
        window_endtime = 5 * ms
        STA = sta.spike_triggered_average(
            z, st, (window_starttime, window_endtime))
        cutout = z[int(((spiketime + window_starttime) * sr).simplified): 
            int(((spiketime + window_endtime) * sr).simplified)]
        cutout.t_start = window_starttime
        assert_array_equal(STA, cutout)

    def test_usage_of_spikes(self):
        st = SpikeTrain([16.5 * math.pi, 17.5 * math.pi, 
            18.5 * math.pi, 19.5 * math.pi], units='ms', t_stop=20 * math.pi)
        STA = sta.spike_triggered_average(
            self.asiga0, st, (-math.pi * ms, math.pi * ms))
        self.assertEqual(STA.annotations['used_spikes'], 3)
        self.assertEqual(STA.annotations['unused_spikes'], 1)


    #***********************************************************************
    #**** Test for an invalid value, to check that the function raises *****
    #********* an exception or returns an error code ***********************

    def test_analog_signal_of_wrong_type(self):
        '''Analog signal given as list, but must be AnalogSignalArray'''
        asiga = [0, 1, 2, 3, 4]
        self.assertRaises(TypeError, sta.spike_triggered_average, 
            asiga, self.st0, (-2 * ms, 2 * ms))

    def test_spiketrain_of_list_type_in_wrong_sense(self):
        st = [10, 11, 12]
        self.assertRaises(TypeError, sta.spike_triggered_average, 
            self.asiga0, st, (1 * ms, 2 * ms))

    def test_spiketrain_of_nonlist_and_nonspiketrain_type(self):
        st = (10, 11, 12)
        self.assertRaises(TypeError, sta.spike_triggered_average, 
            self.asiga0, st, (1 * ms, 2 * ms))

    def test_forgotten_AnalogSignalArray_argument(self):
        self.assertRaises(TypeError, sta.spike_triggered_average, 
            self.st0, (-2 * ms, 2 * ms))

    def test_one_smaller_nrspiketrains_smaller_nranalogsignals(self):
        '''Number of spiketrains between 1 and number of analogsignals'''
        self.assertRaises(ValueError, sta.spike_triggered_average, 
            self.asiga2, self.lst, (-2 * ms, 2 * ms))

    def test_more_spiketrains_than_analogsignals_forbidden(self):
        self.assertRaises(ValueError, sta.spike_triggered_average, 
            self.asiga0, self.lst, (-2 * ms, 2 * ms))

    def test_spike_earlier_than_analogsignal(self):
        st = SpikeTrain([-1 * math.pi, 2 * math.pi],
            units='ms', t_start=-2 * math.pi, t_stop=20 * math.pi)
        self.assertRaises(ValueError, sta.spike_triggered_average, 
            self.asiga0, st, (-2 * ms, 2 * ms))

    def test_spike_later_than_analogsignal(self):
        st = SpikeTrain(
            [math.pi, 21 * math.pi], units='ms', t_stop=25 * math.pi)
        self.assertRaises(ValueError, sta.spike_triggered_average, 
            self.asiga0, st, (-2 * ms, 2 * ms))

    def test_impossible_window(self):
        self.assertRaises(ValueError, sta.spike_triggered_average, 
            self.asiga0, self.st0, (-2 * ms, -5 * ms))

    def test_window_larger_than_signal(self):
        self.assertRaises(ValueError, sta.spike_triggered_average,
            self.asiga0, self.st0, (-15 * math.pi * ms, 15 * math.pi * ms))

    def test_wrong_window_starttime_unit(self):
        self.assertRaises(TypeError, sta.spike_triggered_average, 
            self.asiga0, self.st0, (-2 * mV, 2 * ms))

    def test_wrong_window_endtime_unit(self):
        self.assertRaises(TypeError, sta.spike_triggered_average, 
            self.asiga0, self.st0, (-2 * ms, 2 * Hz))

    def test_window_borders_as_complex_numbers(self):
        self.assertRaises(TypeError, sta.spike_triggered_average, self.asiga0,
            self.st0, ((-2 * math.pi + 3j) * ms, (2 * math.pi + 3j) * ms))

    #***********************************************************************
    #**** Test for an empty value (where the argument is a list, array, ****
    #********* vector or other container datatype). ************************

    def test_empty_analogsignal(self):
        asiga = AnalogSignalArray([], units='mV', sampling_rate=10 / ms)
        st = SpikeTrain([5], units='ms', t_stop=10)
        self.assertRaises(ValueError, sta.spike_triggered_average, 
            asiga, st, (-1 * ms, 1 * ms))

    def test_one_spiketrain_empty(self):
        '''Test for one empty SpikeTrain, but existing spikes in other'''
        st = [SpikeTrain(
            [9 * math.pi, 10 * math.pi, 11 * math.pi, 12 * math.pi], 
            units='ms', t_stop=self.asiga1.t_stop), 
            SpikeTrain([], units='ms', t_stop=self.asiga1.t_stop)]
        STA = sta.spike_triggered_average(self.asiga1, st, (-1 * ms, 1 * ms))
        cmp_array = AnalogSignalArray(np.array([np.zeros(20, dtype=float)]).T,
            units='mV', sampling_rate=10 / ms)
        cmp_array = cmp_array / 0.
        cmp_array.t_start = -1 * ms
        assert_array_equal(STA[:, 1], cmp_array[:, 0])

    def test_all_spiketrains_empty(self):
        st = SpikeTrain([], units='ms', t_stop=self.asiga1.t_stop)
        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            # Trigger warnings.
            STA = sta.spike_triggered_average(
                self.asiga1, st, (-1 * ms, 1 * ms))
            self.assertEqual("No spike at all was either found or used "
                             "for averaging", str(w[-1].message))
            nan_array = np.empty(20)
            nan_array.fill(np.nan)
            cmp_array = AnalogSignalArray(np.array([nan_array, nan_array]).T,
                units='mV', sampling_rate=10 / ms)
            assert_array_equal(STA, cmp_array)


if __name__ == '__main__':
    unittest.main()
