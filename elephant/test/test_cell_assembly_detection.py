"""
Unit test for cell_assembly_detection
"""

import unittest
import numpy as np
from numpy.testing import assert_array_equal
import neo
import quantities as pq
import elephant.conversion as conv
import elephant.cell_assembly_detection as cad


class CadTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        # Parameters
        cls.bin_size = bin_size = 1 * pq.ms
        cls.max_lag = 10

        # Input parameters

        # Number of pattern occurrences
        n_occ1 = 150
        n_occ2 = 170
        n_occ3 = 210

        # Pattern lags
        lags1 = [0, 0.001]
        lags2 = [0, 0.002]
        lags3 = [0, 0.003]

        # Output pattern lags
        cls.output_lags1 = output_lags1 = [0, 1]
        cls.output_lags2 = output_lags2 = [0, 2]
        cls.output_lags3 = output_lags3 = [0, 3]

        # Length of the spiketrain
        t_start = 0
        t_stop = 1

        # Patterns times
        np.random.seed(1)
        patt1_times = neo.SpikeTrain(
            np.random.uniform(0, 1 - max(lags1), n_occ1) * pq.s,
            t_start=0 * pq.s, t_stop=1 * pq.s)
        patt2_times = neo.SpikeTrain(
            np.random.uniform(0, 1 - max(lags2), n_occ2) * pq.s,
            t_start=0 * pq.s, t_stop=1 * pq.s)
        patt3_times = neo.SpikeTrain(
            np.random.uniform(0, 1 - max(lags3), n_occ3) * pq.s,
            t_start=0 * pq.s, t_stop=1 * pq.s)

        # Patterns
        patt1 = [patt1_times] + [neo.SpikeTrain(
            patt1_times + lag * pq.s, t_start=t_start * pq.s,
            t_stop=t_stop * pq.s) for lag in lags1]
        patt2 = [patt2_times] + [neo.SpikeTrain(
            patt2_times + lag * pq.s, t_start=t_start * pq.s,
            t_stop=t_stop * pq.s) for lag in lags2]
        patt3 = [patt3_times] + [neo.SpikeTrain(
            patt3_times + lag * pq.s, t_start=t_start * pq.s,
            t_stop=t_stop * pq.s) for lag in lags3]

        # Binning spiketrains
        cls.bin_patt1 = conv.BinnedSpikeTrain(patt1, bin_size=bin_size)

        # Data
        cls.msip = conv.BinnedSpikeTrain(patt1 + patt2 + patt3,
                                         bin_size=bin_size)

        # Expected results
        n_spk1 = len(lags1) + 1
        n_spk2 = len(lags2) + 1
        n_spk3 = len(lags3) + 1
        cls.elements1 = range(n_spk1)
        cls.elements2 = range(n_spk2)
        cls.elements3 = range(n_spk3)
        cls.elements_msip = [range(n_spk1),
                             range(n_spk1, n_spk1 + n_spk2),
                             range(n_spk1 + n_spk2, n_spk1 + n_spk2 + n_spk3)]

        occ1 = np.unique(conv.BinnedSpikeTrain(patt1_times, bin_size
                                               ).spike_indices[0])
        cls.occ1 = occ1
        occ2 = np.unique(conv.BinnedSpikeTrain(patt2_times, bin_size
                                               ).spike_indices[0])
        cls.occ2 = occ2
        occ3 = np.unique(conv.BinnedSpikeTrain(patt3_times, bin_size
                                               ).spike_indices[0])
        cls.occ3 = occ3

        cls.occ_msip = [list(occ1), list(occ2), list(occ3)]
        cls.lags_msip = [output_lags1, output_lags2, output_lags3]

    # test for single pattern injection input
    def test_cad_single_sip(self):
        # collecting cad output
        output_single = cad.cell_assembly_detection(
            binned_spiketrain=self.bin_patt1, max_lag=self.max_lag)
        # check neurons in the pattern
        assert_array_equal(sorted(output_single[0]['neurons']),
                           self.elements1)
        # check the occurrences time of the pattern
        assert_array_equal(output_single[0]['times'],
                           self.occ1 * self.bin_size)
        # check the lags
        assert_array_equal(sorted(output_single[0]['lags']) * pq.s,
                           self.output_lags1 * self.bin_size)

    # test with multiple (3) patterns injected in the data
    def test_cad_msip(self):
        # collecting cad output
        output_msip = cad.cell_assembly_detection(
            binned_spiketrain=self.msip, max_lag=self.max_lag)
        for i, out in enumerate(output_msip):
            with self.subTest(i=i):
                assert_array_equal(out['times'],
                                   self.occ_msip[i] * self.bin_size)
                assert_array_equal(sorted(out['lags']) * pq.s,
                                   self.lags_msip[i] * self.bin_size)
                assert_array_equal(sorted(out['neurons']),
                                   self.elements_msip[i])

    # test the errors raised
    def test_cad_raise_error(self):
        # test error data input format
        self.assertRaises(TypeError, cad.cell_assembly_detection,
                          binned_spiketrain=[[1, 2, 3], [3, 4, 5]],
                          max_lag=self.max_lag)
        # test error significance level
        self.assertRaises(ValueError, cad.cell_assembly_detection,
                          binned_spiketrain=conv.BinnedSpikeTrain(
                              [neo.SpikeTrain([1, 2, 3] * pq.s,
                                              t_stop=5 * pq.s),
                               neo.SpikeTrain([3, 4, 5] * pq.s,
                                              t_stop=5 * pq.s)],
                              bin_size=self.bin_size),
                          max_lag=self.max_lag,
                          alpha=-3)
        # test error minimum number of occurrences
        self.assertRaises(ValueError, cad.cell_assembly_detection,
                          binned_spiketrain=conv.BinnedSpikeTrain(
                              [neo.SpikeTrain([1, 2, 3] * pq.s,
                                              t_stop=5 * pq.s),
                               neo.SpikeTrain([3, 4, 5] * pq.s,
                                              t_stop=5 * pq.s)],
                              bin_size=self.bin_size),
                          max_lag=self.max_lag,
                          min_occurrences=-1)
        # test error minimum number of spikes in a pattern
        self.assertRaises(ValueError, cad.cell_assembly_detection,
                          binned_spiketrain=conv.BinnedSpikeTrain(
                              [neo.SpikeTrain([1, 2, 3] * pq.s,
                                              t_stop=5 * pq.s),
                               neo.SpikeTrain([3, 4, 5] * pq.s,
                                              t_stop=5 * pq.s)],
                              bin_size=self.bin_size),
                          max_lag=self.max_lag,
                          max_spikes=1)
        # test error chunk size for variance computation
        self.assertRaises(ValueError, cad.cell_assembly_detection,
                          binned_spiketrain=conv.BinnedSpikeTrain(
                              [neo.SpikeTrain([1, 2, 3] * pq.s,
                                              t_stop=5 * pq.s),
                               neo.SpikeTrain([3, 4, 5] * pq.s,
                                              t_stop=5 * pq.s)],
                              bin_size=self.bin_size),
                          max_lag=self.max_lag,
                          size_chunks=1)
        # test error maximum lag
        self.assertRaises(ValueError, cad.cell_assembly_detection,
                          binned_spiketrain=conv.BinnedSpikeTrain(
                              [neo.SpikeTrain([1, 2, 3] * pq.s,
                                              t_stop=5 * pq.s),
                               neo.SpikeTrain([3, 4, 5] * pq.s,
                                              t_stop=5 * pq.s)],
                              bin_size=self.bin_size),
                          max_lag=1)
        # test error minimum length spike train
        self.assertRaises(ValueError, cad.cell_assembly_detection,
                          binned_spiketrain=conv.BinnedSpikeTrain(
                              [neo.SpikeTrain([1, 2, 3] * pq.ms,
                                              t_stop=6 * pq.ms),
                               neo.SpikeTrain([3, 4, 5] * pq.ms,
                                              t_stop=6 * pq.ms)],
                              bin_size=1 * pq.ms),
                          max_lag=self.max_lag)


if __name__ == "__main__":
    unittest.main(verbosity=2)
