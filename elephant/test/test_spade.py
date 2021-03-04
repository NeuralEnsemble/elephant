"""
Unit tests for the spade module.

:copyright: Copyright 2014-2020 by the Elephant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
"""
from __future__ import division

import unittest
import random

import neo
import numpy as np
import quantities as pq
from numpy.testing.utils import assert_array_equal

import elephant.conversion as conv
import elephant.spade as spade
import elephant.spike_train_generation as stg

try:
    import statsmodels
    HAVE_STATSMODELS = True
except ImportError:
    HAVE_STATSMODELS = False

HAVE_FIM = spade.HAVE_FIM


class SpadeTestCase(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        # Spade parameters
        self.bin_size = 1 * pq.ms
        self.winlen = 10
        self.n_subset = 10
        self.n_surr = 10
        self.alpha = 0.05
        self.stability_thresh = [0.1, 0.1]
        self.psr_param = [0, 0, 0]
        self.min_occ = 4
        self.min_spikes = 4
        self.max_occ = 4
        self.max_spikes = 4
        self.min_neu = 4
        # Test data parameters
        # CPP parameters
        self.n_neu = 100
        self.amplitude = [0] * self.n_neu + [1]
        self.cpp = stg.cpp(
            rate=3 * pq.Hz,
            amplitude_distribution=self.amplitude,
            t_stop=5 * pq.s)
        # Number of patterns' occurrences
        self.n_occ1 = 10
        self.n_occ2 = 12
        self.n_occ3 = 15
        # Patterns lags
        self.lags1 = [2]
        self.lags2 = [1, 3]
        self.lags3 = [1, 2, 4, 5, 7]
        # Length of the spiketrain
        self.t_stop = 3000
        # Patterns times
        self.patt1_times = neo.SpikeTrain(
            np.arange(
                0, 1000, 1000 // self.n_occ1) *
            pq.ms, t_stop=self.t_stop * pq.ms)
        self.patt2_times = neo.SpikeTrain(
            np.arange(
                1000, 2000, 1000 // self.n_occ2)[:-1] *
            pq.ms, t_stop=self.t_stop * pq.ms)
        self.patt3_times = neo.SpikeTrain(
            np.arange(
                2000, 3000, 1000 // self.n_occ3)[:-1] *
            pq.ms, t_stop=self.t_stop * pq.ms)
        # Patterns
        self.patt1 = [self.patt1_times] + [neo.SpikeTrain(
            self.patt1_times.view(pq.Quantity) + lag * pq.ms,
            t_stop=self.t_stop * pq.ms) for lag in self.lags1]
        self.patt2 = [self.patt2_times] + [neo.SpikeTrain(
            self.patt2_times.view(pq.Quantity) + lag * pq.ms,
            t_stop=self.t_stop * pq.ms) for lag in self.lags2]
        self.patt3 = [self.patt3_times] + [neo.SpikeTrain(
            self.patt3_times.view(pq.Quantity) + lag * pq.ms,
            t_stop=self.t_stop * pq.ms) for lag in self.lags3]
        # Data
        self.msip = self.patt1 + self.patt2 + self.patt3
        # Expected results
        self.n_spk1 = len(self.lags1) + 1
        self.n_spk2 = len(self.lags2) + 1
        self.n_spk3 = len(self.lags3) + 1
        self.elements1 = list(range(self.n_spk1))
        self.elements2 = list(range(self.n_spk2))
        self.elements3 = list(range(self.n_spk3))
        self.elements_msip = [
            self.elements1,
            list(
                range(
                    self.n_spk1,
                    self.n_spk1 +
                    self.n_spk2)),
            list(
                range(
                    self.n_spk1 +
                    self.n_spk2,
                    self.n_spk1 +
                    self.n_spk2 +
                    self.n_spk3))]
        self.occ1 = np.unique(conv.BinnedSpikeTrain(
            self.patt1_times, self.bin_size).spike_indices[0])
        self.occ2 = np.unique(conv.BinnedSpikeTrain(
            self.patt2_times, self.bin_size).spike_indices[0])
        self.occ3 = np.unique(conv.BinnedSpikeTrain(
            self.patt3_times, self.bin_size).spike_indices[0])
        self.occ_msip = [
            list(self.occ1), list(self.occ2), list(self.occ3)]
        self.lags_msip = [self.lags1, self.lags2, self.lags3]
        self.patt_psr = self.patt3 + [self.patt3[-1][:3]]

    # Testing cpp
    @unittest.skipUnless(HAVE_FIM, "Time consuming with pythonic FIM")
    def test_spade_cpp(self):
        output_cpp = spade.spade(self.cpp, self.bin_size, 1,
                                 approx_stab_pars=dict(
                                     n_subsets=self.n_subset,
                                     stability_thresh=self.stability_thresh),
                                 n_surr=self.n_surr, alpha=self.alpha,
                                 psr_param=self.psr_param,
                                 stat_corr='no',
                                 output_format='patterns')['patterns']
        elements_cpp = []
        lags_cpp = []
        # collecting spade output
        for out in output_cpp:
            elements_cpp.append(sorted(out['neurons']))
            lags_cpp.append(list(out['lags'].magnitude))
        # check neurons in the patterns
        assert_array_equal(elements_cpp, [range(self.n_neu)])
        # check the lags
        assert_array_equal(lags_cpp, [np.array([0] * (self.n_neu - 1))])

    # Testing spectrum cpp
    def test_spade_spectrum_cpp(self):
        # Computing Spectrum
        spectrum_cpp = spade.concepts_mining(self.cpp, self.bin_size,
                                             1, report='#')[0]
        # Check spectrum
        assert_array_equal(
            spectrum_cpp,
            [(len(self.cpp),
              np.sum(conv.BinnedSpikeTrain(
                  self.cpp[0], self.bin_size).to_bool_array()), 1)])

    # Testing with multiple patterns input
    def test_spade_msip(self):
        output_msip = spade.spade(self.msip, self.bin_size, self.winlen,
                                  approx_stab_pars=dict(
                                      n_subsets=self.n_subset,
                                      stability_thresh=self.stability_thresh),
                                  n_surr=self.n_surr, alpha=self.alpha,
                                  psr_param=self.psr_param,
                                  stat_corr='no',
                                  output_format='patterns')['patterns']
        elements_msip = []
        occ_msip = []
        lags_msip = []
        # collecting spade output
        for out in output_msip:
            elements_msip.append(out['neurons'])
            occ_msip.append(list(out['times'].magnitude))
            lags_msip.append(list(out['lags'].magnitude))
        elements_msip = sorted(elements_msip, key=len)
        occ_msip = sorted(occ_msip, key=len)
        lags_msip = sorted(lags_msip, key=len)
        # check neurons in the patterns
        assert_array_equal(elements_msip, self.elements_msip)
        # check the occurrences time of the patters
        assert_array_equal(occ_msip, self.occ_msip)
        # check the lags
        assert_array_equal(lags_msip, self.lags_msip)

    def test_parameters(self):
        """
        Test under different configuration of parameters than the default one
        """
        # test min_spikes parameter
        with self.assertWarns(UserWarning):
            # n_surr=0 and alpha=0.05 spawns expected UserWarning
            output_msip_min_spikes = spade.spade(
                self.msip,
                self.bin_size,
                self.winlen,
                min_spikes=self.min_spikes,
                approx_stab_pars=dict(n_subsets=self.n_subset),
                n_surr=0,
                alpha=self.alpha,
                psr_param=self.psr_param,
                stat_corr='no',
                output_format='patterns')['patterns']
        # collecting spade output
        elements_msip_min_spikes = []
        for out in output_msip_min_spikes:
            elements_msip_min_spikes.append(out['neurons'])
        elements_msip_min_spikes = sorted(
            elements_msip_min_spikes, key=len)
        lags_msip_min_spikes = []
        for out in output_msip_min_spikes:
            lags_msip_min_spikes.append(list(out['lags'].magnitude))
            pvalue = out['pvalue']
        lags_msip_min_spikes = sorted(
            lags_msip_min_spikes, key=len)
        # check the lags
        assert_array_equal(lags_msip_min_spikes, [
            l for l in self.lags_msip if len(l) + 1 >= self.min_spikes])
        # check the neurons in the patterns
        assert_array_equal(elements_msip_min_spikes, [
            el for el in self.elements_msip if len(el) >= self.min_neu and len(
                el) >= self.min_spikes])
        # check that the p-values assigned are equal to -1 (n_surr=0)
        assert_array_equal(-1, pvalue)

        # test min_occ parameter
        output_msip_min_occ = spade.spade(
            self.msip,
            self.bin_size,
            self.winlen,
            min_occ=self.min_occ,
            approx_stab_pars=dict(
                n_subsets=self.n_subset),
            n_surr=self.n_surr,
            alpha=self.alpha,
            psr_param=self.psr_param,
            stat_corr='no',
            output_format='patterns')['patterns']
        # collect spade output
        occ_msip_min_occ = []
        for out in output_msip_min_occ:
            occ_msip_min_occ.append(list(out['times'].magnitude))
        occ_msip_min_occ = sorted(occ_msip_min_occ, key=len)
        # test occurrences time
        assert_array_equal(occ_msip_min_occ, [
            occ for occ in self.occ_msip if len(occ) >= self.min_occ])

        # test max_spikes parameter
        output_msip_max_spikes = spade.spade(
            self.msip,
            self.bin_size,
            self.winlen,
            max_spikes=self.max_spikes,
            approx_stab_pars=dict(
                n_subsets=self.n_subset),
            n_surr=self.n_surr,
            alpha=self.alpha,
            psr_param=self.psr_param,
            stat_corr='no',
            output_format='patterns')['patterns']
        # collecting spade output
        elements_msip_max_spikes = []
        for out in output_msip_max_spikes:
            elements_msip_max_spikes.append(out['neurons'])
        elements_msip_max_spikes = sorted(
            elements_msip_max_spikes, key=len)
        lags_msip_max_spikes = []
        for out in output_msip_max_spikes:
            lags_msip_max_spikes.append(list(out['lags'].magnitude))
        lags_msip_max_spikes = sorted(
            lags_msip_max_spikes, key=len)
        # check the lags
        assert_array_equal(
            [len(lags) < self.max_spikes
             for lags in lags_msip_max_spikes],
            [True] * len(lags_msip_max_spikes))

        # test max_occ parameter
        output_msip_max_occ = spade.spade(
            self.msip,
            self.bin_size,
            self.winlen,
            max_occ=self.max_occ,
            approx_stab_pars=dict(
                n_subsets=self.n_subset),
            n_surr=self.n_surr,
            alpha=self.alpha,
            psr_param=self.psr_param,
            stat_corr='no',
            output_format='patterns')['patterns']
        # collect spade output
        occ_msip_max_occ = []
        for out in output_msip_max_occ:
            occ_msip_max_occ.append(list(out['times'].magnitude))
        occ_msip_max_occ = sorted(occ_msip_max_occ, key=len)
        # test occurrences time
        assert_array_equal(occ_msip_max_occ, [
            occ for occ in self.occ_msip if len(occ) <= self.max_occ])

    # test to compare the python and the C implementation of FIM
    # skip this test if C code not available
    @unittest.skipIf(not HAVE_FIM, 'Requires fim.so')
    def test_fpgrowth_fca(self):
        print("fim.so is found.")
        binary_matrix = conv.BinnedSpikeTrain(
            self.patt1, self.bin_size).to_sparse_bool_array().tocoo()
        context, transactions, rel_matrix = spade._build_context(
            binary_matrix, self.winlen)
        # mining the data with python fast_fca
        mining_results_fpg = spade._fpgrowth(
            transactions,
            rel_matrix=rel_matrix)
        # mining the data with C fim
        mining_results_ffca = spade._fast_fca(context)

        # testing that the outputs are identical
        assert_array_equal(sorted(mining_results_ffca[0][0]), sorted(
            mining_results_fpg[0][0]))
        assert_array_equal(sorted(mining_results_ffca[0][1]), sorted(
            mining_results_fpg[0][1]))

    # Tests 3d spectrum
    # Testing with multiple patterns input
    def test_spade_msip_3d(self):
        output_msip = spade.spade(self.msip, self.bin_size, self.winlen,
                                  approx_stab_pars=dict(
                                      n_subsets=self.n_subset,
                                      stability_thresh=self.stability_thresh),
                                  n_surr=self.n_surr, spectrum='3d#',
                                  alpha=self.alpha, psr_param=self.psr_param,
                                  stat_corr='no',
                                  output_format='patterns')['patterns']
        elements_msip = []
        occ_msip = []
        lags_msip = []
        # collecting spade output
        for out in output_msip:
            elements_msip.append(out['neurons'])
            occ_msip.append(list(out['times'].magnitude))
            lags_msip.append(list(out['lags'].magnitude))
        elements_msip = sorted(elements_msip, key=len)
        occ_msip = sorted(occ_msip, key=len)
        lags_msip = sorted(lags_msip, key=len)
        # check neurons in the patterns
        assert_array_equal(elements_msip, self.elements_msip)
        # check the occurrences time of the patters
        assert_array_equal(occ_msip, self.occ_msip)
        # check the lags
        assert_array_equal(lags_msip, self.lags_msip)

    # test under different configuration of parameters than the default one
    def test_parameters_3d(self):
        # test min_spikes parameter
        output_msip_min_spikes = spade.spade(
            self.msip,
            self.bin_size,
            self.winlen,
            min_spikes=self.min_spikes,
            approx_stab_pars=dict(
                n_subsets=self.n_subset),
            n_surr=self.n_surr,
            spectrum='3d#',
            alpha=self.alpha,
            psr_param=self.psr_param,
            stat_corr='no',
            output_format='patterns')['patterns']
        # collecting spade output
        elements_msip_min_spikes = []
        for out in output_msip_min_spikes:
            elements_msip_min_spikes.append(out['neurons'])
        elements_msip_min_spikes = sorted(
            elements_msip_min_spikes, key=len)
        lags_msip_min_spikes = []
        for out in output_msip_min_spikes:
            lags_msip_min_spikes.append(list(out['lags'].magnitude))
        lags_msip_min_spikes = sorted(
            lags_msip_min_spikes, key=len)
        # check the lags
        assert_array_equal(lags_msip_min_spikes, [
            l for l in self.lags_msip if len(l) + 1 >= self.min_spikes])
        # check the neurons in the patterns
        assert_array_equal(elements_msip_min_spikes, [
            el for el in self.elements_msip if len(el) >= self.min_neu and len(
                el) >= self.min_spikes])

        # test min_occ parameter
        output_msip_min_occ = spade.spade(
            self.msip,
            self.bin_size,
            self.winlen,
            min_occ=self.min_occ,
            approx_stab_pars=dict(
                n_subsets=self.n_subset),
            n_surr=self.n_surr,
            spectrum='3d#',
            alpha=self.alpha,
            psr_param=self.psr_param,
            stat_corr='no',
            output_format='patterns')['patterns']
        # collect spade output
        occ_msip_min_occ = []
        for out in output_msip_min_occ:
            occ_msip_min_occ.append(list(out['times'].magnitude))
        occ_msip_min_occ = sorted(occ_msip_min_occ, key=len)
        # test occurrences time
        assert_array_equal(occ_msip_min_occ, [
            occ for occ in self.occ_msip if len(occ) >= self.min_occ])

    # Test computation spectrum
    def test_spectrum(self):
        # test 2d spectrum
        spectrum = spade.concepts_mining(self.patt1, self.bin_size,
                                         self.winlen, report='#')[0]
        # test 3d spectrum
        assert_array_equal(spectrum, [[len(self.lags1) + 1, self.n_occ1, 1]])
        spectrum_3d = spade.concepts_mining(self.patt1, self.bin_size,
                                            self.winlen, report='3d#')[0]
        assert_array_equal(spectrum_3d, [
            [len(self.lags1) + 1, self.n_occ1, max(self.lags1), 1]])

    def test_spade_raise_error(self):
        # Test list not using neo.Spiketrain
        self.assertRaises(TypeError, spade.spade, [
            [1, 2, 3], [3, 4, 5]], 1 * pq.ms, 4, stat_corr='no')
        self.assertRaises(TypeError, spade.concepts_mining, [
            [1, 2, 3], [3, 4, 5]], 1 * pq.ms, 4)
        # Test neo.Spiketrain with different t_stop
        self.assertRaises(
            ValueError, spade.spade,
            [neo.SpikeTrain([1, 2, 3] * pq.s, t_stop=5 * pq.s),
             neo.SpikeTrain([3, 4, 5] * pq.s, t_stop=6 * pq.s)],
            1 * pq.ms, 4, stat_corr='no')
        # Test bin_size not pq.Quantity
        self.assertRaises(
            TypeError, spade.spade,
            [neo.SpikeTrain([1, 2, 3] * pq.s, t_stop=6 * pq.s),
             neo.SpikeTrain([3, 4, 5] * pq.s, t_stop=6 * pq.s)],
            bin_size=1., winlen=4, stat_corr='no')
        # Test winlen not int
        self.assertRaises(
            TypeError, spade.spade,
            [neo.SpikeTrain([1, 2, 3] * pq.s, t_stop=6 * pq.s),
             neo.SpikeTrain([3, 4, 5] * pq.s, t_stop=6 * pq.s)],
            bin_size=1.*pq.ms, winlen=4.1, stat_corr='no')
        # Test min_spikes not int
        self.assertRaises(
            TypeError, spade.spade,
            [neo.SpikeTrain([1, 2, 3] * pq.s, t_stop=6 * pq.s),
             neo.SpikeTrain([3, 4, 5] * pq.s, t_stop=6 * pq.s)],
            bin_size=1.*pq.ms, winlen=4, min_spikes=3.4, stat_corr='no')
        # Test min_occ not int
        self.assertRaises(
            TypeError, spade.spade,
            [neo.SpikeTrain([1, 2, 3] * pq.s, t_stop=6 * pq.s),
             neo.SpikeTrain([3, 4, 5] * pq.s, t_stop=6 * pq.s)],
            bin_size=1.*pq.ms, winlen=4, min_occ=3.4, stat_corr='no')
        # Test max_spikes not int
        self.assertRaises(
            TypeError, spade.spade,
            [neo.SpikeTrain([1, 2, 3] * pq.s, t_stop=6 * pq.s),
             neo.SpikeTrain([3, 4, 5] * pq.s, t_stop=6 * pq.s)],
            bin_size=1.*pq.ms, winlen=4, max_spikes=3.4, stat_corr='no')
        # Test max_occ not int
        self.assertRaises(
            TypeError, spade.spade,
            [neo.SpikeTrain([1, 2, 3] * pq.s, t_stop=6 * pq.s),
             neo.SpikeTrain([3, 4, 5] * pq.s, t_stop=6 * pq.s)],
            bin_size=1.*pq.ms, winlen=4, max_occ=3.4, stat_corr='no')
        # Test min_neu not int
        self.assertRaises(
            TypeError, spade.spade,
            [neo.SpikeTrain([1, 2, 3] * pq.s, t_stop=6 * pq.s),
             neo.SpikeTrain([3, 4, 5] * pq.s, t_stop=6 * pq.s)],
            bin_size=1.*pq.ms, winlen=4, min_neu=3.4, stat_corr='no')
        # Test wrong stability params
        self.assertRaises(
            ValueError, spade.spade,
            [neo.SpikeTrain([1, 2, 3] * pq.s, t_stop=6 * pq.s),
             neo.SpikeTrain([3, 4, 5] * pq.s, t_stop=6 * pq.s)],
            bin_size=1.*pq.ms, winlen=4, approx_stab_pars={'wrong key': 0},
            stat_corr='no')
        # Test n_surr not int
        self.assertRaises(
            TypeError, spade.spade,
            [neo.SpikeTrain([1, 2, 3] * pq.s, t_stop=6 * pq.s),
             neo.SpikeTrain([3, 4, 5] * pq.s, t_stop=6 * pq.s)],
            bin_size=1.*pq.ms, winlen=4, n_surr=3.4, stat_corr='no')
        # Test dither not pq.Quantity
        self.assertRaises(
            TypeError, spade.spade,
            [neo.SpikeTrain([1, 2, 3] * pq.s, t_stop=6 * pq.s),
             neo.SpikeTrain([3, 4, 5] * pq.s, t_stop=6 * pq.s)],
            bin_size=1.*pq.ms, winlen=4, n_surr=100, alpha=0.05,
            dither=15., stat_corr='no')
        # Test wrong alpha
        self.assertRaises(
            TypeError, spade.spade,
            [neo.SpikeTrain([1, 2, 3] * pq.s, t_stop=6 * pq.s),
             neo.SpikeTrain([3, 4, 5] * pq.s, t_stop=6 * pq.s)],
            bin_size=1.*pq.ms, winlen=4, n_surr=100, alpha='5 %',
            dither=15.*pq.ms, stat_corr='no')
        # Test wrong statistical correction
        self.assertRaises(
            ValueError, spade.spade,
            [neo.SpikeTrain([1, 2, 3] * pq.s, t_stop=6 * pq.s),
             neo.SpikeTrain([3, 4, 5] * pq.s, t_stop=6 * pq.s)],
            bin_size=1.*pq.ms, winlen=4, n_surr=100, alpha=0.05,
            dither=15.*pq.ms, stat_corr='wrong correction')
        # Test wrong psr_params
        self.assertRaises(
            TypeError, spade.spade,
            [neo.SpikeTrain([1, 2, 3] * pq.s, t_stop=6 * pq.s),
             neo.SpikeTrain([3, 4, 5] * pq.s, t_stop=6 * pq.s)],
            bin_size=1.*pq.ms, winlen=4, n_surr=100, alpha=0.05,
            dither=15.*pq.ms, stat_corr='no', psr_param=(2.5, 3.4, 2.1))
        # Test wrong psr_params
        self.assertRaises(
            TypeError, spade.spade,
            [neo.SpikeTrain([1, 2, 3] * pq.s, t_stop=6 * pq.s),
             neo.SpikeTrain([3, 4, 5] * pq.s, t_stop=6 * pq.s)],
            bin_size=1.*pq.ms, winlen=4, n_surr=100, alpha=0.05,
            dither=15.*pq.ms, stat_corr='no', psr_param=3.1)
        # Test output format
        self.assertRaises(
            ValueError, spade.spade,
            [neo.SpikeTrain([1, 2, 3] * pq.s, t_stop=6 * pq.s),
             neo.SpikeTrain([3, 4, 5] * pq.s, t_stop=6 * pq.s)],
            bin_size=1.*pq.ms, winlen=4, n_surr=100, alpha=0.05,
            dither=15.*pq.ms, stat_corr='no', output_format='wrong_output')
        # Test wrong spectrum parameter
        self.assertRaises(
            ValueError, spade.spade,
            [neo.SpikeTrain([1, 2, 3] * pq.s, t_stop=6 * pq.s),
             neo.SpikeTrain([3, 4, 5] * pq.s, t_stop=6 * pq.s)],
            1 * pq.ms, 4, n_surr=1, stat_corr='no',
            spectrum='invalid_key')
        self.assertRaises(
            ValueError, spade.concepts_mining,
            [neo.SpikeTrain([1, 2, 3] * pq.s, t_stop=6 * pq.s),
             neo.SpikeTrain([3, 4, 5] * pq.s, t_stop=6 * pq.s)],
            1 * pq.ms, 4, report='invalid_key')
        self.assertRaises(
            ValueError, spade.pvalue_spectrum,
            [neo.SpikeTrain([1, 2, 3] * pq.s, t_stop=6 * pq.s),
             neo.SpikeTrain([3, 4, 5] * pq.s, t_stop=6 * pq.s)],
            1 * pq.ms, 4, dither=10*pq.ms, n_surr=1,
            spectrum='invalid_key')
        # Test negative minimum number of spikes
        self.assertRaises(
            ValueError, spade.spade,
            [neo.SpikeTrain([1, 2, 3] * pq.s, t_stop=5 * pq.s),
             neo.SpikeTrain([3, 4, 5] * pq.s, t_stop=5 * pq.s)],
            1 * pq.ms, 4, min_neu=-3, stat_corr='no')
        # Test wrong dither method
        self.assertRaises(
            ValueError, spade.spade,
            [neo.SpikeTrain([1, 2, 3] * pq.s, t_stop=5 * pq.s),
             neo.SpikeTrain([3, 4, 5] * pq.s, t_stop=5 * pq.s)],
            1 * pq.ms, 4, surr_method='invalid_key', stat_corr='no')
        # Test negative number of surrogates
        self.assertRaises(
            ValueError, spade.pvalue_spectrum,
            [neo.SpikeTrain([1, 2, 3] * pq.s, t_stop=5 * pq.s),
             neo.SpikeTrain([3, 4, 5] * pq.s, t_stop=5 * pq.s)],
            1 * pq.ms, 4, dither=10*pq.ms, n_surr=100,
            surr_method='invalid_key')
        # Test negative number of surrogates
        self.assertRaises(
            ValueError, spade.pvalue_spectrum,
            [neo.SpikeTrain([1, 2, 3] * pq.s, t_stop=5 * pq.s),
             neo.SpikeTrain([3, 4, 5] * pq.s, t_stop=5 * pq.s)],
            1 * pq.ms, 4, 3 * pq.ms, n_surr=-3)
        # Test wrong correction parameter
        self.assertRaises(ValueError, spade.test_signature_significance,
                          pv_spec=((2, 3, 0.2), (2, 4, 0.1)),
                          concepts=([[(2, 3), (1, 2, 3)]]),
                          alpha=0.01,
                          winlen=1,
                          corr='invalid_key')
        # Test negative number of subset for stability
        self.assertRaises(ValueError, spade.approximate_stability, (),
                          np.array([]), n_subsets=-3)

    def test_pattern_set_reduction(self):
        winlen = 6
        # intent(concept1) is a superset of intent(concept2)
        # extent(concept1) is a subset of extent(concept2)
        # intent(concept2) is a subset of intent(concept3)
        #     when taking into account the shift due to the window positions
        # intent(concept1) has a non-empty intersection with intent(concept3)
        #     when taking into account the shift due to the window positions
        # intent(concept4) is disjoint from all others
        concept1 = ((12, 19, 26), (2, 10, 18))
        concept2 = ((12, 19), (2, 10, 18, 26))
        concept3 = ((0, 7, 14, 21), (0, 8))
        concept4 = ((1, 6), (0, 8))

        # reject concept2 using min_occ
        # make sure to keep concept1 by setting k_superset_filtering = 1
        concepts = spade.pattern_set_reduction([concept1, concept2],
                                               ns_signatures=[],
                                               winlen=winlen, spectrum='#',
                                               h_subset_filtering=0, min_occ=2,
                                               k_superset_filtering=1)
        self.assertEqual(concepts, [concept1])

        # keep concept2 by increasing h_subset_filtering
        concepts = spade.pattern_set_reduction([concept1, concept2],
                                               ns_signatures=[],
                                               winlen=winlen, spectrum='#',
                                               h_subset_filtering=2, min_occ=2,
                                               k_superset_filtering=1)
        self.assertEqual(concepts, [concept1, concept2])

        # reject concept1 using min_spikes
        concepts = spade.pattern_set_reduction([concept1, concept2],
                                               ns_signatures=[],
                                               winlen=winlen, spectrum='#',
                                               h_subset_filtering=2,
                                               min_spikes=2,
                                               k_superset_filtering=0)
        self.assertEqual(concepts, [concept2])

        # reject concept2 using ns_signatures
        concepts = spade.pattern_set_reduction([concept1, concept2],
                                               ns_signatures=[(2, 2)],
                                               winlen=winlen, spectrum='#',
                                               h_subset_filtering=1, min_occ=2,
                                               k_superset_filtering=1)
        self.assertEqual(concepts, [concept1])

        # reject concept1 using ns_signatures
        # make sure to keep concept2 by increasing h_subset_filtering
        concepts = spade.pattern_set_reduction([concept1, concept2],
                                               ns_signatures=[(2, 3)],
                                               winlen=winlen, spectrum='#',
                                               h_subset_filtering=3,
                                               min_spikes=2,
                                               min_occ=2,
                                               k_superset_filtering=1)
        self.assertEqual(concepts, [concept2])

        # reject concept2 using the covered spikes criterion
        concepts = spade.pattern_set_reduction([concept1, concept2],
                                               ns_signatures=[(2, 2)],
                                               winlen=winlen, spectrum='#',
                                               h_subset_filtering=0,
                                               min_occ=2,
                                               k_superset_filtering=0,
                                               l_covered_spikes=0)
        self.assertEqual(concepts, [concept1])

        # reject concept1 using superset filtering
        # (case with non-empty intersection but no superset)
        concepts = spade.pattern_set_reduction([concept1, concept3],
                                               ns_signatures=[], min_spikes=2,
                                               winlen=winlen, spectrum='#',
                                               k_superset_filtering=0)
        self.assertEqual(concepts, [concept3])

        # keep concept1 by increasing k_superset_filtering
        concepts = spade.pattern_set_reduction([concept1, concept3],
                                               ns_signatures=[], min_spikes=2,
                                               winlen=winlen, spectrum='#',
                                               k_superset_filtering=1)
        self.assertEqual(concepts, [concept1, concept3])

        # reject concept3 using ns_signatures
        concepts = spade.pattern_set_reduction([concept1, concept3],
                                               ns_signatures=[(3, 2)],
                                               min_spikes=2,
                                               winlen=winlen, spectrum='#',
                                               k_superset_filtering=1)
        self.assertEqual(concepts, [concept1])

        # reject concept3 using the covered spikes criterion
        concepts = spade.pattern_set_reduction([concept1, concept3],
                                               ns_signatures=[(3, 2), (2, 3)],
                                               min_spikes=2,
                                               winlen=winlen, spectrum='#',
                                               k_superset_filtering=1,
                                               l_covered_spikes=0)
        self.assertEqual(concepts, [concept1])

        # check that two concepts with disjoint intents are both kept
        concepts = spade.pattern_set_reduction([concept3, concept4],
                                               ns_signatures=[],
                                               winlen=winlen, spectrum='#')
        self.assertEqual(concepts, [concept3, concept4])

    @unittest.skipUnless(HAVE_STATSMODELS,
                         "'fdr_bh' stat corr requires statsmodels")
    def test_signature_significance_fdr_bh_corr(self):
        """
        A typical corr='fdr_bh' scenario, that requires statsmodels.
        """
        sig_spectrum = spade.test_signature_significance(
            pv_spec=((2, 3, 0.2), (2, 4, 0.05)),
            concepts=([[(2, 3), (1, 2, 3)],
                       [(2, 4), (1, 2, 3, 4)]]),
            alpha=0.15, winlen=1, corr='fdr_bh')
        self.assertEqual(sig_spectrum, [(2., 3., False), (2., 4., True)])

    def test_different_surrogate_method(self):
        np.random.seed(0)
        random.seed(0)
        spiketrains = [stg.homogeneous_poisson_process(rate=20*pq.Hz)
                       for _ in range(2)]
        surr_methods = ('dither_spikes', 'joint_isi_dithering',
                        'bin_shuffling',
                        'dither_spikes_with_refractory_period')
        pv_specs = {'dither_spikes': [[2, 2, 0.8], [2, 3, 0.2]],
                    'joint_isi_dithering': [[2, 2, 0.8]],
                    'bin_shuffling': [[2, 2, 1.0], [2, 3, 0.2]],
                    'dither_spikes_with_refractory_period':
                        [[2, 2, 0.8]]}
        for surr_method in surr_methods:
            pv_spec = spade.pvalue_spectrum(
                spiketrains, bin_size=self.bin_size,
                winlen=self.winlen, dither=15*pq.ms,
                n_surr=5, surr_method=surr_method)
            self.assertEqual(pv_spec, pv_specs[surr_method])


def suite():
    suite = unittest.makeSuite(SpadeTestCase, 'test')
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())
