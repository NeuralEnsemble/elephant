"""
Unit tests for the waveform_feature module.

:copyright: Copyright 2014-2019 by the Elephant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
"""

from __future__ import division

import unittest
import neo
import numpy as np
import quantities as pq

from elephant import waveform_features


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

    def test_with_zero_waveforms(self):
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
