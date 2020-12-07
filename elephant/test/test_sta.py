# -*- coding: utf-8 -*-
"""
Tests for the function sta module

:copyright: Copyright 2014-2020 by the Elephant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
"""

import unittest
import math
import numpy as np
import scipy
from numpy.testing import assert_array_equal
from numpy.testing.utils import assert_array_almost_equal
import neo
from neo import AnalogSignal, SpikeTrain
from elephant.conversion import BinnedSpikeTrain
import quantities as pq
from quantities import ms, mV, Hz
import elephant.sta as sta
import warnings


class sta_TestCase(unittest.TestCase):

    def setUp(self):
        self.asiga0 = AnalogSignal(np.array([
            np.sin(np.arange(0, 20 * math.pi, 0.1))]).T,
            units='mV', sampling_rate=10 / ms)
        self.asiga1 = AnalogSignal(np.array([
            np.sin(np.arange(0, 20 * math.pi, 0.1)),
            np.cos(np.arange(0, 20 * math.pi, 0.1))]).T,
            units='mV', sampling_rate=10 / ms)
        self.asiga2 = AnalogSignal(np.array([
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

    # ***********************************************************************
    # ************************ Test for typical values **********************

    def test_spike_triggered_average_with_n_spikes_on_constant_function(self):
        """Signal should average to the input"""
        const = 13.8
        x = const * np.ones(201)
        asiga = AnalogSignal(
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
        """Signal should average to zero"""
        STA = sta.spike_triggered_average(
            self.asiga0, self.st0, (-4 * ms, 4 * ms))
        target = 5e-2 * mV
        self.assertEqual(np.abs(STA).max().dimensionality.simplified,
                         pq.Quantity(1, "V").dimensionality.simplified)
        self.assertLess(np.abs(STA).max(), target)

    def test_only_one_spike(self):
        """The output should be the same as the input"""
        x = np.arange(0, 20, 0.1)
        y = x**2
        sr = 10 / ms
        z = AnalogSignal(np.array([y]).T, units='mV', sampling_rate=sr)
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
        st = SpikeTrain([16.5 * math.pi,
                         17.5 * math.pi,
                         18.5 * math.pi,
                         19.5 * math.pi],
                        units='ms',
                        t_stop=20 * math.pi)
        STA = sta.spike_triggered_average(
            self.asiga0, st, (-math.pi * ms, math.pi * ms))
        self.assertEqual(STA.annotations['used_spikes'], 3)
        self.assertEqual(STA.annotations['unused_spikes'], 1)

    # ***********************************************************************
    # **** Test for an invalid value, to check that the function raises *****
    # ********* an exception or returns an error code ***********************

    def test_analog_signal_of_wrong_type(self):
        """Analog signal given as list, but must be AnalogSignal"""
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

    def test_forgotten_AnalogSignal_argument(self):
        self.assertRaises(TypeError, sta.spike_triggered_average,
                          self.st0, (-2 * ms, 2 * ms))

    def test_one_smaller_nrspiketrains_smaller_nranalogsignals(self):
        """Number of spiketrains between 1 and number of analogsignals"""
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
        self.assertRaises(
            ValueError,
            sta.spike_triggered_average,
            self.asiga0,
            self.st0,
            (-15 * math.pi * ms,
             15 * math.pi * ms))

    def test_wrong_window_starttime_unit(self):
        self.assertRaises(TypeError, sta.spike_triggered_average,
                          self.asiga0, self.st0, (-2 * mV, 2 * ms))

    def test_wrong_window_endtime_unit(self):
        self.assertRaises(TypeError, sta.spike_triggered_average,
                          self.asiga0, self.st0, (-2 * ms, 2 * Hz))

    def test_window_borders_as_complex_numbers(self):
        self.assertRaises(
            TypeError,
            sta.spike_triggered_average,
            self.asiga0,
            self.st0,
            ((-2 * math.pi + 3j) * ms,
             (2 * math.pi + 3j) * ms))

    # ***********************************************************************
    # **** Test for an empty value (where the argument is a list, array, ****
    # ********* vector or other container datatype). ************************

    def test_empty_analogsignal(self):
        asiga = AnalogSignal([], units='mV', sampling_rate=10 / ms)
        st = SpikeTrain([5], units='ms', t_stop=10)
        self.assertRaises(ValueError, sta.spike_triggered_average,
                          asiga, st, (-1 * ms, 1 * ms))

    def test_one_spiketrain_empty(self):
        """Test for one empty SpikeTrain, but existing spikes in other"""
        st = [SpikeTrain(
            [9 * math.pi, 10 * math.pi, 11 * math.pi, 12 * math.pi],
            units='ms', t_stop=self.asiga1.t_stop),
            SpikeTrain([], units='ms', t_stop=self.asiga1.t_stop)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            """
            Ignore the RuntimeWarning: invalid value encountered in true_divide
            new_signal = f(other, *args) for the empty SpikeTrain.
            """
            STA = sta.spike_triggered_average(self.asiga1,
                                              spiketrains=st,
                                              window=(-1 * ms, 1 * ms))
        assert np.isnan(STA.magnitude[:, 1]).all()

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
            cmp_array = AnalogSignal(np.array([nan_array, nan_array]).T,
                                     units='mV', sampling_rate=10 / ms)
            assert_array_equal(STA.magnitude, cmp_array.magnitude)


# =========================================================================
# Tests for new scipy verison (with scipy.signal.coherence)
# =========================================================================

@unittest.skipIf(not hasattr(scipy.signal, 'coherence'), "Please update scipy "
                 "to a version >= 0.16")
class sfc_TestCase_new_scipy(unittest.TestCase):

    def setUp(self):
        # standard testsignals
        tlen0 = 100 * pq.s
        f0 = 20. * pq.Hz
        fs0 = 1 * pq.ms
        t0 = np.arange(
            0, tlen0.rescale(pq.s).magnitude,
            fs0.rescale(pq.s).magnitude) * pq.s
        self.anasig0 = AnalogSignal(
            np.sin(2 * np.pi * (f0 * t0).simplified.magnitude),
            units=pq.mV, t_start=0 * pq.ms, sampling_period=fs0)
        self.st0 = SpikeTrain(
            np.arange(0, tlen0.rescale(pq.ms).magnitude, 50) * pq.ms,
            t_start=0 * pq.ms, t_stop=tlen0)
        self.bst0 = BinnedSpikeTrain(self.st0, bin_size=fs0)

        # shortened analogsignals
        self.anasig1 = self.anasig0.time_slice(1 * pq.s, None)
        self.anasig2 = self.anasig0.time_slice(None, 99 * pq.s)

        # increased sampling frequency
        fs1 = 0.1 * pq.ms
        self.anasig3 = AnalogSignal(
            np.sin(2 * np.pi * (f0 * t0).simplified.magnitude),
            units=pq.mV, t_start=0 * pq.ms, sampling_period=fs1)
        self.bst1 = BinnedSpikeTrain(
            self.st0.time_slice(self.anasig3.t_start, self.anasig3.t_stop),
            bin_size=fs1)

        # analogsignal containing multiple traces
        self.anasig4 = AnalogSignal(
            np.array([
                np.sin(2 * np.pi * (f0 * t0).simplified.magnitude),
                np.sin(4 * np.pi * (f0 * t0).simplified.magnitude)]).
            transpose(),
            units=pq.mV, t_start=0 * pq.ms, sampling_period=fs0)

        # shortened spike train
        self.st3 = SpikeTrain(
            np.arange(
                (tlen0.rescale(pq.ms).magnitude * .25),
                (tlen0.rescale(pq.ms).magnitude * .75), 50) * pq.ms,
            t_start=0 * pq.ms, t_stop=tlen0)
        self.bst3 = BinnedSpikeTrain(self.st3, bin_size=fs0)

        self.st4 = SpikeTrain(np.arange(
            (tlen0.rescale(pq.ms).magnitude * .25),
            (tlen0.rescale(pq.ms).magnitude * .75), 50) * pq.ms,
            t_start=5 * fs0, t_stop=tlen0 - 5 * fs0)
        self.bst4 = BinnedSpikeTrain(self.st4, bin_size=fs0)

        # spike train with incompatible bin_size
        self.bst5 = BinnedSpikeTrain(self.st3, bin_size=fs0 * 2.)

        # spike train with same bin_size as the analog signal, but with
        # bin edges not aligned to the time axis of the analog signal
        self.bst6 = BinnedSpikeTrain(
            self.st3,
            bin_size=fs0,
            t_start=4.5 * fs0,
            t_stop=tlen0 - 4.5 * fs0)

    # =========================================================================
    # Tests for correct input handling
    # =========================================================================

    def test_wrong_input_type(self):
        self.assertRaises(TypeError,
                          sta.spike_field_coherence,
                          np.array([1, 2, 3]), self.bst0)
        self.assertRaises(TypeError,
                          sta.spike_field_coherence,
                          self.anasig0, [1, 2, 3])
        self.assertRaises(ValueError,
                          sta.spike_field_coherence,
                          self.anasig0.duplicate_with_new_data([]), self.bst0)

    def test_start_stop_times_out_of_range(self):
        self.assertRaises(ValueError,
                          sta.spike_field_coherence,
                          self.anasig1, self.bst0)

        self.assertRaises(ValueError,
                          sta.spike_field_coherence,
                          self.anasig2, self.bst0)

    def test_non_matching_input_binning(self):
        self.assertRaises(ValueError,
                          sta.spike_field_coherence,
                          self.anasig0, self.bst1)

    def test_incompatible_spiketrain_analogsignal(self):
        # These spike trains have incompatible binning (bin_size or alignment
        # to time axis of analog signal)
        self.assertRaises(ValueError,
                          sta.spike_field_coherence,
                          self.anasig0, self.bst5)
        self.assertRaises(ValueError,
                          sta.spike_field_coherence,
                          self.anasig0, self.bst6)

    def test_signal_dimensions(self):
        # single analogsignal trace and single spike train
        s_single, f_single = sta.spike_field_coherence(self.anasig0, self.bst0)

        self.assertEqual(len(f_single.shape), 1)
        self.assertEqual(len(s_single.shape), 2)

        # multiple analogsignal traces and single spike train
        s_multi, f_multi = sta.spike_field_coherence(self.anasig4, self.bst0)

        self.assertEqual(len(f_multi.shape), 1)
        self.assertEqual(len(s_multi.shape), 2)

        # frequencies are identical since same sampling frequency was used
        # in both cases and data length is the same
        assert_array_equal(f_single, f_multi)
        # coherences of s_single and first signal in s_multi are identical,
        # since first analogsignal trace in anasig4 is same as in anasig0
        assert_array_equal(s_single[:, 0], s_multi[:, 0])

    def test_non_binned_spiketrain_input(self):
        s, f = sta.spike_field_coherence(self.anasig0, self.st0)

        f_ind = np.where(f >= 19.)[0][0]
        max_ind = np.argmax(s[1:]) + 1

        self.assertEqual(f_ind, max_ind)
        self.assertAlmostEqual(s[f_ind], 1., delta=0.01)

    # =========================================================================
    # Tests for correct return values
    # =========================================================================

    def test_spike_field_coherence_perfect_coherence(self):
        # check for detection of 20Hz peak in anasig0/bst0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            """
            When the spiketrain is a vector with zero values, ignore the
            warning RuntimeWarning: invalid value encountered in true_divide
              Cxy = np.abs(Pxy)**2 / Pxx / Pyy.
            """
            s, f = sta.spike_field_coherence(
                self.anasig0, self.bst0, window='boxcar')

        f_ind = np.where(f >= 19.)[0][0]
        max_ind = np.argmax(s[1:]) + 1

        self.assertEqual(f_ind, max_ind)
        self.assertAlmostEqual(s[f_ind], 1., delta=0.01)

    def test_output_frequencies(self):
        nfft = 256
        _, f = sta.spike_field_coherence(self.anasig3, self.bst1, nfft=nfft)

        # check number of frequency samples
        self.assertEqual(len(f), nfft / 2 + 1)

        f_max = self.anasig3.sampling_rate.rescale('Hz').magnitude / 2
        f_ground_truth = np.linspace(start=0,
                                     stop=f_max,
                                     num=nfft // 2 + 1) * pq.Hz

        # check values of frequency samples
        assert_array_almost_equal(f, f_ground_truth)

    def test_short_spiketrain(self):
        # this spike train has the same length as anasig0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            """
            When the spiketrain is a vector with zero values, ignore the
            warning RuntimeWarning: invalid value encountered in true_divide
              Cxy = np.abs(Pxy)**2 / Pxx / Pyy.
            """
            s1, f1 = sta.spike_field_coherence(
                self.anasig0, self.bst3, window='boxcar')

            # this spike train has the same spikes as above,
            # but it's shorter than anasig0
            s2, f2 = sta.spike_field_coherence(
                self.anasig0, self.bst4, window='boxcar')

        # the results above should be the same, nevertheless
        assert_array_equal(s1.magnitude, s2.magnitude)
        assert_array_equal(f1.magnitude, f2.magnitude)


# =========================================================================
# Tests for old scipy verison (without scipy.signal.coherence)
# =========================================================================

@unittest.skipIf(hasattr(scipy.signal, 'coherence'), 'Applies only for old '
                                                     'scipy versions (<0.16)')
class sfc_TestCase_old_scipy(unittest.TestCase):

    def setUp(self):
        # standard testsignals
        tlen0 = 100 * pq.s
        f0 = 20. * pq.Hz
        fs0 = 1 * pq.ms
        t0 = np.arange(
            0, tlen0.rescale(pq.s).magnitude,
            fs0.rescale(pq.s).magnitude) * pq.s
        self.anasig0 = AnalogSignal(
            np.sin(2 * np.pi * (f0 * t0).simplified.magnitude),
            units=pq.mV, t_start=0 * pq.ms, sampling_period=fs0)
        self.st0 = SpikeTrain(
            np.arange(0, tlen0.rescale(pq.ms).magnitude, 50) * pq.ms,
            t_start=0 * pq.ms, t_stop=tlen0)
        self.bst0 = BinnedSpikeTrain(self.st0, bin_size=fs0)

        def test_old_scipy_version(self):
            self.assertRaises(AttributeError, sta.spike_field_coherence,
                              self.anasig0, self.bst0)


if __name__ == '__main__':
    unittest.main()
