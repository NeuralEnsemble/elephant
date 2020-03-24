"""
Unit tests for the waveform_feature module.

:copyright: Copyright 2014-2019 by the Elephant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
"""

from __future__ import division

import sys
import unittest

import neo
import numpy as np
import quantities as pq

from elephant import waveform_features

python_version_major = sys.version_info.major


class WaveformWidthTestCase(unittest.TestCase):
    def setUp(self):
        self.waveform = [29., 42., 41., 18., 24., 28., 34., 34., 9.,
                         -31., -100., -145., -125., -88., -48., -18., 14., 36.,
                         30., 33., -4., -25., -3., 30., 51., 47., 70.,
                         76., 78., 57., 53., 49., 22., 15., 88., 109.,
                         79., 68.]
        self.target_width = 24

    def test_list(self):
        width = waveform_features.waveform_width(self.waveform)
        self.assertEqual(width, self.target_width)

    def test_np_array(self):
        waveform = np.asarray(self.waveform)
        width = waveform_features.waveform_width(waveform)
        self.assertEqual(width, self.target_width)

    def test_pq_quantity(self):
        waveform = np.asarray(self.waveform) * pq.mV
        width = waveform_features.waveform_width(waveform)
        self.assertEqual(width, self.target_width)

    def test_np_array_2d(self):
        waveform = np.asarray(self.waveform)
        waveform = np.vstack([waveform, waveform])
        self.assertRaises(ValueError, waveform_features.waveform_width,
                          waveform)

    def test_empty_list(self):
        self.assertRaises(ValueError, waveform_features.waveform_width, [])

    def test_cutoff(self):
        size = 10
        waveform = np.arange(size, dtype=float)
        for cutoff in (-1, 1):
            # outside of [0, 1) range
            self.assertRaises(ValueError, waveform_features.waveform_width,
                              waveform, cutoff=cutoff)
        for cutoff in np.linspace(0., 1., num=size, endpoint=False):
            width = waveform_features.waveform_width(waveform, cutoff=cutoff)
            self.assertEqual(width, size - 1)


class WaveformSignalToNoiseRatioTestCase(unittest.TestCase):
    def setUp(self):
        self.spiketrain_without_waveforms = neo.SpikeTrain(
            [0.5, 0.7, 1.2, 2.3, 4.3, 5.5, 6.7] * pq.s, t_stop=10.0 * pq.s)
        self.spiketrain_with_zero_waveforms = neo.SpikeTrain(
            [0.5, 0.7, 1.2, 2.3, 4.3, 5.5, 6.7] * pq.s, t_stop=10.0 * pq.s)
        self.spiketrain_with_zero_waveforms.waveforms = \
            np.zeros((10, 1, 10)) * pq.uV

        self.spiketrain_with_waveforms = neo.SpikeTrain(
            [0.5, 6.7] * pq.s, t_stop=10.0 * pq.s)
        self.spiketrain_with_waveforms.waveforms = \
            np.arange(20).reshape((2, 1, 10)) * pq.uV

    def test_without_wavefroms(self):
        self.assertRaises(ValueError, waveform_features.waveform_snr,
                          self.spiketrain_without_waveforms)

    @unittest.skipUnless(python_version_major == 3, "assertWarns requires 3.2")
    def test_with_zero_waveforms(self):
        with self.assertWarns(UserWarning):
            # expect np.nan result when spiketrain noise is zero.
            result = waveform_features.waveform_snr(
                self.spiketrain_with_zero_waveforms)
        self.assertTrue(np.isnan(result))

    def test_with_waveforms(self):
        target_value = 0.9
        result = waveform_features.waveform_snr(
            self.spiketrain_with_waveforms)
        self.assertEqual(result, target_value)


if __name__ == '__main__':
    unittest.main()
