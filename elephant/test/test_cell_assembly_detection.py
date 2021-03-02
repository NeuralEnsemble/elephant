
"""
Unit test for cell_assembly_detection
"""

import unittest
import numpy as np
from numpy.testing.utils import assert_array_equal
import neo
import quantities as pq
import elephant.conversion as conv
import elephant.cell_assembly_detection as cad


class CadTestCase(unittest.TestCase):

    def setUp(self):

        # Parameters
        self.bin_size = 1 * pq.ms
        self.alpha = 0.05
        self.size_chunks = 100
        self.max_lag = 10
        self.reference_lag = 2
        self.min_occ = 1
        self.max_spikes = np.inf
        self.significance_pruning = True
        self.subgroup_pruning = True
        self.flag_mypruning = False

        # Input parameters

        # Number of pattern occurrences
        self.n_occ1 = 150
        self.n_occ2 = 170
        self.n_occ3 = 210

        # Pattern lags
        self.lags1 = [0, 0.001]
        self.lags2 = [0, 0.002]
        self.lags3 = [0, 0.003]

        # Output pattern lags
        self.output_lags1 = [0, 1]
        self.output_lags2 = [0, 2]
        self.output_lags3 = [0, 3]

        # Length of the spiketrain
        self.t_start = 0
        self.t_stop = 1

        # Patterns times
        np.random.seed(1)
        self.patt1_times = neo.SpikeTrain(
            np.random.uniform(0, 1 - max(self.lags1), self.n_occ1) * pq.s,
            t_start=0 * pq.s, t_stop=1 * pq.s)
        self.patt2_times = neo.SpikeTrain(
            np.random.uniform(0, 1 - max(self.lags2), self.n_occ2) * pq.s,
            t_start=0 * pq.s, t_stop=1 * pq.s)
        self.patt3_times = neo.SpikeTrain(
            np.random.uniform(0, 1 - max(self.lags3), self.n_occ3) * pq.s,
            t_start=0 * pq.s, t_stop=1 * pq.s)

        # Patterns
        self.patt1 = [self.patt1_times] + [neo.SpikeTrain(
            self.patt1_times + l * pq.s, t_start=self.t_start * pq.s,
            t_stop=self.t_stop * pq.s) for l in self.lags1]
        self.patt2 = [self.patt2_times] + [neo.SpikeTrain(
            self.patt2_times + l * pq.s, t_start=self.t_start * pq.s,
            t_stop=self.t_stop * pq.s) for l in self.lags2]
        self.patt3 = [self.patt3_times] + [neo.SpikeTrain(
            self.patt3_times + l * pq.s, t_start=self.t_start * pq.s,
            t_stop=self.t_stop * pq.s) for l in self.lags3]

        # Binning spiketrains
        self.bin_patt1 = conv.BinnedSpikeTrain(self.patt1,
                                               bin_size=self.bin_size)

        # Data
        self.msip = self.patt1 + self.patt2 + self.patt3
        self.msip = conv.BinnedSpikeTrain(self.msip, bin_size=self.bin_size)

        # Expected results
        self.n_spk1 = len(self.lags1) + 1
        self.n_spk2 = len(self.lags2) + 1
        self.n_spk3 = len(self.lags3) + 1
        self.elements1 = range(self.n_spk1)
        self.elements2 = range(self.n_spk2)
        self.elements3 = range(self.n_spk3)
        self.elements_msip = [
            self.elements1, range(self.n_spk1, self.n_spk1 + self.n_spk2),
            range(self.n_spk1 + self.n_spk2,
                  self.n_spk1 + self.n_spk2 + self.n_spk3)]
        self.occ1 = np.unique(conv.BinnedSpikeTrain(
            self.patt1_times, self.bin_size).spike_indices[0])
        self.occ2 = np.unique(conv.BinnedSpikeTrain(
            self.patt2_times, self.bin_size).spike_indices[0])
        self.occ3 = np.unique(conv.BinnedSpikeTrain(
            self.patt3_times, self.bin_size).spike_indices[0])
        self.occ_msip = [list(self.occ1), list(self.occ2), list(self.occ3)]
        self.lags_msip = [self.output_lags1,
                          self.output_lags2,
                          self.output_lags3]

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
            assert_array_equal(out['times'], self.occ_msip[i] * self.bin_size)
            assert_array_equal(sorted(out['lags']) * pq.s,
                               self.lags_msip[i] * self.bin_size)
            assert_array_equal(sorted(out['neurons']), self.elements_msip[i])

    # test the errors raised
    def test_cad_raise_error(self):
        # test error data input format
        self.assertRaises(TypeError, cad.cell_assembly_detection,
                          data=[[1, 2, 3], [3, 4, 5]],
                          maxlag=self.max_lag)
        # test error significance level
        self.assertRaises(ValueError, cad.cell_assembly_detection,
                          data=conv.BinnedSpikeTrain(
                              [neo.SpikeTrain([1, 2, 3] * pq.s,
                                              t_stop=5 * pq.s),
                               neo.SpikeTrain([3, 4, 5] * pq.s,
                                              t_stop=5 * pq.s)],
                              bin_size=self.bin_size),
                          maxlag=self.max_lag,
                          alpha=-3)
        # test error minimum number of occurrences
        self.assertRaises(ValueError, cad.cell_assembly_detection,
                          data=conv.BinnedSpikeTrain(
                              [neo.SpikeTrain([1, 2, 3] * pq.s,
                                              t_stop=5 * pq.s),
                               neo.SpikeTrain([3, 4, 5] * pq.s,
                                              t_stop=5 * pq.s)],
                              bin_size=self.bin_size),
                          maxlag=self.max_lag,
                          min_occ=-1)
        # test error minimum number of spikes in a pattern
        self.assertRaises(ValueError, cad.cell_assembly_detection,
                          data=conv.BinnedSpikeTrain(
                              [neo.SpikeTrain([1, 2, 3] * pq.s,
                                              t_stop=5 * pq.s),
                               neo.SpikeTrain([3, 4, 5] * pq.s,
                                              t_stop=5 * pq.s)],
                              bin_size=self.bin_size),
                          maxlag=self.max_lag,
                          max_spikes=1)
        # test error chunk size for variance computation
        self.assertRaises(ValueError, cad.cell_assembly_detection,
                          data=conv.BinnedSpikeTrain(
                              [neo.SpikeTrain([1, 2, 3] * pq.s,
                                              t_stop=5 * pq.s),
                               neo.SpikeTrain([3, 4, 5] * pq.s,
                                              t_stop=5 * pq.s)],
                              bin_size=self.bin_size),
                          maxlag=self.max_lag,
                          size_chunks=1)
        # test error maximum lag
        self.assertRaises(ValueError, cad.cell_assembly_detection,
                          data=conv.BinnedSpikeTrain(
                              [neo.SpikeTrain([1, 2, 3] * pq.s,
                                              t_stop=5 * pq.s),
                               neo.SpikeTrain([3, 4, 5] * pq.s,
                                              t_stop=5 * pq.s)],
                              bin_size=self.bin_size),
                          maxlag=1)
        # test error minimum length spike train
        self.assertRaises(ValueError, cad.cell_assembly_detection,
                          data=conv.BinnedSpikeTrain(
                              [neo.SpikeTrain([1, 2, 3] * pq.ms,
                                              t_stop=6 * pq.ms),
                               neo.SpikeTrain([3, 4, 5] * pq.ms,
                                              t_stop=6 * pq.ms)],
                              bin_size=1 * pq.ms),
                          maxlag=self.max_lag)


def suite():
    suite = unittest.makeSuite(CadTestCase, 'test')
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())
