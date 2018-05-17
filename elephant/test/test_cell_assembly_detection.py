
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
        self.binsize = 1*pq.ms
        self.alph = 0.05
        self.size_chunks = 100
        self.maxlag = 10
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
            t_start=0*pq.s, t_stop=1*pq.s)
        self.patt2_times = neo.SpikeTrain(
            np.random.uniform(0, 1 - max(self.lags2), self.n_occ2) * pq.s,
            t_start=0*pq.s, t_stop=1*pq.s)
        self.patt3_times = neo.SpikeTrain(
            np.random.uniform(0, 1 - max(self.lags3), self.n_occ3) * pq.s,
            t_start=0*pq.s, t_stop=1*pq.s)

        # Patterns
        self.patt1 = [self.patt1_times] + [neo.SpikeTrain(
            self.patt1_times+l * pq.s, t_start=self.t_start * pq.s,
            t_stop=self.t_stop * pq.s) for l in self.lags1]
        self.patt2 = [self.patt2_times] + [neo.SpikeTrain(
            self.patt2_times+l * pq.s,  t_start=self.t_start * pq.s,
            t_stop=self.t_stop * pq.s) for l in self.lags2]
        self.patt3 = [self.patt3_times] + [neo.SpikeTrain(
            self.patt3_times+l * pq.s,  t_start=self.t_start * pq.s,
            t_stop=self.t_stop * pq.s) for l in self.lags3]

        # Binning spiketrains
        self.bin_patt1 = conv.BinnedSpikeTrain(self.patt1,
                                               binsize=self.binsize)

        # Data
        self.msip = self.patt1 + self.patt2 + self.patt3
        self.msip = conv.BinnedSpikeTrain(self.msip, binsize=self.binsize)

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
            self.patt1_times, self.binsize).spike_indices[0])
        self.occ2 = np.unique(conv.BinnedSpikeTrain(
            self.patt2_times, self.binsize).spike_indices[0])
        self.occ3 = np.unique(conv.BinnedSpikeTrain(
            self.patt3_times, self.binsize).spike_indices[0])
        self.occ_msip = [list(self.occ1), list(self.occ2), list(self.occ3)]
        self.lags_msip = [self.output_lags1,
                          self.output_lags2,
                          self.output_lags3]

    def test_cad_single_sip(self):
        output_single = cad.\
            cell_assembly_detection(data=self.bin_patt1, maxlag=self.maxlag)

        assert_array_equal(sorted(output_single[0]['neurons']),
                           self.elements1)
        assert_array_equal(output_single[0]['times'],
                           self.occ1)
        assert_array_equal(sorted(output_single[0]['lags']),
                           self.output_lags1)

    def test_cad_msip(self):
        output_msip = cad.\
            cell_assembly_detection(data=self.msip, maxlag=self.maxlag)

        elements_msip = []
        occ_msip = []
        lags_msip = []
        for out in output_msip:
            elements_msip.append(out['neurons'])
            occ_msip.append(out['times'])
            lags_msip.append(list(out['lags']))
        elements_msip = sorted(elements_msip, key=lambda d: len(d))
        occ_msip = sorted(occ_msip, key=lambda d: len(d))
        lags_msip = sorted(lags_msip, key=lambda d: len(d))
        elements_msip = [sorted(e) for e in elements_msip]

        assert_array_equal(elements_msip, self.elements_msip)

        assert_array_equal(occ_msip[0], self.occ_msip[0])
        assert_array_equal(occ_msip[1], self.occ_msip[1])
        assert_array_equal(occ_msip[2], self.occ_msip[2])
        lags_msip = [sorted(e) for e in lags_msip]
        assert_array_equal(lags_msip, self.lags_msip)

    def test_cad_raise_error(self):
        self.assertRaises(TypeError, cad.cell_assembly_detection,
                          data=[[1, 2, 3], [3, 4, 5]],
                          maxlag=self.maxlag)
        self.assertRaises(ValueError, cad.cell_assembly_detection,
                          data=conv.BinnedSpikeTrain(
                              [neo.SpikeTrain([1, 2, 3]*pq.s, t_stop=5*pq.s),
                               neo.SpikeTrain([3, 4, 5]*pq.s, t_stop=5*pq.s)],
                              binsize=self.binsize),
                          maxlag=self.maxlag,
                          alph=-3)
        self.assertRaises(ValueError, cad.cell_assembly_detection,
                          data=conv.BinnedSpikeTrain(
                              [neo.SpikeTrain([1, 2, 3]*pq.s, t_stop=5*pq.s),
                               neo.SpikeTrain([3, 4, 5]*pq.s, t_stop=5*pq.s)],
                              binsize=self.binsize),
                          maxlag=self.maxlag,
                          min_occ=-1)
        self.assertRaises(ValueError, cad.cell_assembly_detection,
                          data=conv.BinnedSpikeTrain(
                              [neo.SpikeTrain([1, 2, 3]*pq.s, t_stop=5*pq.s),
                               neo.SpikeTrain([3, 4, 5]*pq.s, t_stop=5*pq.s)],
                              binsize=self.binsize),
                          maxlag=self.maxlag,
                          max_spikes=1)
        self.assertRaises(ValueError, cad.cell_assembly_detection,
                          data=conv.BinnedSpikeTrain(
                              [neo.SpikeTrain([1, 2, 3]*pq.s, t_stop=5*pq.s),
                               neo.SpikeTrain([3, 4, 5]*pq.s, t_stop=5*pq.s)],
                              binsize=self.binsize),
                          maxlag=self.maxlag,
                          size_chunks=1)
        self.assertRaises(ValueError, cad.cell_assembly_detection,
                          data=conv.BinnedSpikeTrain(
                              [neo.SpikeTrain([1, 2, 3]*pq.s, t_stop=5*pq.s),
                               neo.SpikeTrain([3, 4, 5]*pq.s, t_stop=5*pq.s)],
                              binsize=self.binsize),
                          maxlag=1)
        self.assertRaises(ValueError, cad.cell_assembly_detection,
                          data=conv.BinnedSpikeTrain(
                              [neo.SpikeTrain([1, 2, 3]*pq.ms, t_stop=6*pq.ms),
                               neo.SpikeTrain([3, 4, 5]*pq.ms,
                                              t_stop=6*pq.ms)],
                              binsize=1*pq.ms),
                          maxlag=self.maxlag)


def suite():
    suite = unittest.makeSuite(CadTestCase, 'test')
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())
