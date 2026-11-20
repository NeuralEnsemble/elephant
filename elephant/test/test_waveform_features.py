"""
Unit tests for the waveform_feature module.

:copyright: Copyright 2014-2026 by the Elephant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
"""

from __future__ import division

import unittest

import numpy as np
import quantities as pq
from numpy.testing import assert_array_almost_equal

from elephant import waveform_features


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
    def test_zero_waveforms(self):
        zero_waveforms = [np.zeros((5, 10)),
                          np.zeros((5, 1, 10)),
                          np.zeros((5, 3, 10))]
        for zero_wf in zero_waveforms:
            with self.assertWarns(UserWarning):
                # expect np.nan result when waveform noise is zero.
                result = waveform_features.waveform_snr(zero_wf)
            self.assertTrue(np.all(np.isnan(result)))

    def test_waveforms_arange_single_spiketrain(self):
        target_snr = 0.9
        waveforms = np.arange(20).reshape((2, 1, 10))
        snr_float = waveform_features.waveform_snr(waveforms)
        self.assertIsInstance(snr_float, float)
        self.assertEqual(snr_float, target_snr)
        self.assertEqual(waveform_features.waveform_snr(np.squeeze(waveforms)),
                         target_snr)

    def test_single_time_sample_preserves_channels(self):
        """
        Intent: Keep the waveform and channel axes for one-sample waveforms.
        Condition: Multiple waveforms with one or three channels, one sample.
        Expected: Zero peak-to-trough range and zero SNR in every channel.
        """
        for shape in ((2, 1), (2, 1, 1), (2, 3, 1)):
            with self.subTest(shape=shape):
                waveforms = np.arange(np.prod(shape)).reshape(shape)
                result = waveform_features.waveform_snr(waveforms)
                expected = np.zeros(3) if shape == (2, 3, 1) else 0.0
                np.testing.assert_array_equal(result, expected)

    def test_single_waveform_preserves_channels(self):
        """
        Intent: Do not measure channel differences as spike-to-spike noise.
        Condition: One nonconstant waveform per channel, so noise is zero.
        Expected: Infinite SNR for each channel, not a cross-channel ratio.
        """
        for shape in ((1, 4), (1, 1, 4), (1, 3, 4)):
            with self.subTest(shape=shape), np.errstate(divide='ignore'):
                waveforms = np.arange(np.prod(shape)).reshape(shape)
                result = waveform_features.waveform_snr(waveforms)
                expected = np.full(3, np.inf) if shape == (1, 3, 4) else np.inf
                np.testing.assert_array_equal(result, expected)

    def test_invalid_dimensions(self):
        for shape in ((), (4,), (2, 2, 2, 2)):
            with self.subTest(shape=shape):
                with self.assertRaisesRegex(ValueError,
                                            'two or three dimensions'):
                    waveform_features.waveform_snr(np.zeros(shape))

    def test_quantity_single_sample(self):
        waveforms = np.arange(6).reshape(2, 3, 1) * pq.mV
        np.testing.assert_array_equal(
            waveform_features.waveform_snr(waveforms), np.zeros(3))

    def test_waveforms_arange_multiple_spiketrains(self):
        target_snr = [0.3, 0.3, 0.3]
        waveforms = np.arange(60).reshape((2, 3, 10))
        snr_arr = waveform_features.waveform_snr(waveforms)
        self.assertIsInstance(snr_arr, np.ndarray)
        assert_array_almost_equal(snr_arr, target_snr)


if __name__ == '__main__':
    unittest.main()
