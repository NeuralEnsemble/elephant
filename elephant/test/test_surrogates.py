# -*- coding: utf-8 -*-
"""
docstring goes here.

:copyright: Copyright 2014 by the Elephant team, see AUTHORS.txt.
:license: Modified BSD, see LICENSE.txt for details.
"""

import unittest
import elephant.surrogates as surr
import numpy as np
import quantities as pq
import neo


class SurrogatesTestCase(unittest.TestCase):

    def test_spike_dithering_output_format(self):

        st = neo.SpikeTrain([90, 150, 180, 350] * pq.ms, t_stop=500 * pq.ms)

        nr_surr = 2
        dither = 10 * pq.ms
        surrs = surr.spike_dithering(st, dither=dither, n=nr_surr)

        self.assertEqual(type(surrs), list)
        self.assertEqual(len(surrs), nr_surr)

        for surrog in surrs:
            self.assertEqual(type(surrs[0]), neo.SpikeTrain)
            self.assertEqual(surrog.units, st.units)
            self.assertEqual(surrog.t_start, st.t_start)
            self.assertEqual(surrog.t_stop, st.t_stop)
            self.assertEqual(len(surrog), len(st))

    def test_spike_dithering_empty_train(self):

        st = neo.SpikeTrain([] * pq.ms, t_stop=500 * pq.ms)

        dither = 10 * pq.ms
        surrog = surr.spike_dithering(st, dither=dither, n=1)[0]
        self.assertEqual(len(surrog), 0)

    def test_spike_dithering_output_decimals(self):

        st = neo.SpikeTrain([90, 150, 180, 350] * pq.ms, t_stop=500 * pq.ms)

        nr_surr = 2
        dither = 10 * pq.ms
        surrs = surr.spike_dithering(st, dither=dither, decimals=3, n=nr_surr)

        for surrog in surrs:
            for i in range(len(surrog)):
                self.assertNotEqual(surrog[i] - int(surrog[i]) * pq.ms,
                                    surrog[i] - surrog[i])

    def test_spike_dithering_false_edges(self):

        st = neo.SpikeTrain([90, 150, 180, 350] * pq.ms, t_stop=500 * pq.ms)

        nr_surr = 2
        dither = 10 * pq.ms
        surrs = surr.spike_dithering(st, dither=dither, n=nr_surr, edges=False)

        for surrog in surrs:
            for i in range(len(surrog)):
                self.assertLessEqual(surrog[i], st.t_stop)

    def test_spike_time_rand_output_format(self):

        st = neo.SpikeTrain([90, 150, 180, 350] * pq.ms, t_stop=500 * pq.ms)

        nr_surr = 2
        surrs = surr.spike_time_rand(st, n=nr_surr)

        self.assertEqual(type(surrs), list)
        self.assertEqual(len(surrs), nr_surr)

        for surrog in surrs:
            self.assertEqual(type(surrs[0]), neo.SpikeTrain)
            self.assertEqual(surrog.units, st.units)
            self.assertEqual(surrog.t_start, st.t_start)
            self.assertEqual(surrog.t_stop, st.t_stop)
            self.assertEqual(len(surrog), len(st))

    def test_spike_time_rand_empty_train(self):

        st = neo.SpikeTrain([] * pq.ms, t_stop=500 * pq.ms)

        surrog = surr.spike_time_rand(st, n=1)[0]
        self.assertEqual(len(surrog), 0)

    def test_spike_time_rand_output_decimals(self):
        st = neo.SpikeTrain([90, 150, 180, 350] * pq.ms, t_stop=500 * pq.ms)

        nr_surr = 2
        surrs = surr.spike_time_rand(st, n=nr_surr, decimals=3)

        for surrog in surrs:
            for i in range(len(surrog)):
                self.assertNotEqual(surrog[i] - int(surrog[i]) * pq.ms,
                                    surrog[i] - surrog[i])

    def test_isi_shuffling_output_format(self):

        st = neo.SpikeTrain([90, 150, 180, 350] * pq.ms, t_stop=500 * pq.ms)

        nr_surr = 2
        surrs = surr.isi_shuffling(st, n=nr_surr)

        self.assertEqual(type(surrs), list)
        self.assertEqual(len(surrs), nr_surr)

        for surrog in surrs:
            self.assertEqual(type(surrs[0]), neo.SpikeTrain)
            self.assertEqual(surrog.units, st.units)
            self.assertEqual(surrog.t_start, st.t_start)
            self.assertEqual(surrog.t_stop, st.t_stop)
            self.assertEqual(len(surrog), len(st))

    def test_isi_shuffling_empty_train(self):

        st = neo.SpikeTrain([] * pq.ms, t_stop=500 * pq.ms)

        surrog = surr.isi_shuffling(st, n=1)[0]
        self.assertEqual(len(surrog), 0)

    def test_isi_shuffling_same_isis(self):

        st = neo.SpikeTrain([90, 150, 180, 350] * pq.ms, t_stop=500 * pq.ms)

        surrog = surr.isi_shuffling(st, n=1)[0]

        st_pq = st.view(pq.Quantity)
        surr_pq = surrog.view(pq.Quantity)

        isi0_orig = st[0] - st.t_start
        ISIs_orig = np.sort([isi0_orig] + [isi for isi in np.diff(st_pq)])

        isi0_surr = surrog[0] - surrog.t_start
        ISIs_surr = np.sort([isi0_surr] + [isi for isi in np.diff(surr_pq)])

        self.assertTrue(np.all(ISIs_orig == ISIs_surr))

    def test_isi_shuffling_output_decimals(self):

        st = neo.SpikeTrain([90, 150, 180, 350] * pq.ms, t_stop=500 * pq.ms)

        surrog = surr.isi_shuffling(st, n=1, decimals=95)[0]

        st_pq = st.view(pq.Quantity)
        surr_pq = surrog.view(pq.Quantity)

        isi0_orig = st[0] - st.t_start
        ISIs_orig = np.sort([isi0_orig] + [isi for isi in np.diff(st_pq)])

        isi0_surr = surrog[0] - surrog.t_start
        ISIs_surr = np.sort([isi0_surr] + [isi for isi in np.diff(surr_pq)])

        self.assertTrue(np.all(ISIs_orig == ISIs_surr))

    def test_train_shifting_output_format(self):

        st = neo.SpikeTrain([90, 150, 180, 350] * pq.ms, t_stop=500 * pq.ms)

        nr_surr = 2
        shift = 10 * pq.ms
        surrs = surr.train_shifting(st, shift=shift, n=nr_surr)

        self.assertEqual(type(surrs), list)
        self.assertEqual(len(surrs), nr_surr)

        for surrog in surrs:
            self.assertEqual(type(surrs[0]), neo.SpikeTrain)
            self.assertEqual(surrog.units, st.units)
            self.assertEqual(surrog.t_start, st.t_start)
            self.assertEqual(surrog.t_stop, st.t_stop)
            self.assertEqual(len(surrog), len(st))

    def test_train_shifting_empty_train(self):

        st = neo.SpikeTrain([] * pq.ms, t_stop=500 * pq.ms)

        shift = 10 * pq.ms
        surrog = surr.train_shifting(st, shift=shift, n=1)[0]
        self.assertEqual(len(surrog), 0)

    def test_train_shifting_output_decimals(self):
        st = neo.SpikeTrain([90, 150, 180, 350] * pq.ms, t_stop=500 * pq.ms)

        nr_surr = 2
        shift = 10 * pq.ms
        surrs = surr.train_shifting(st, shift=shift, n=nr_surr, decimals=3)

        for surrog in surrs:
            for i in range(len(surrog)):
                self.assertNotEqual(surrog[i] - int(surrog[i]) * pq.ms,
                                    surrog[i] - surrog[i])

    def test_train_shifting_false_edges(self):

        st = neo.SpikeTrain([90, 150, 180, 350] * pq.ms, t_stop=500 * pq.ms)

        nr_surr = 2
        shift = 10 * pq.ms
        surrs = surr.train_shifting(st, shift=shift, n=nr_surr, edges=False)

        for surrog in surrs:
            for i in range(len(surrog)):
                self.assertLessEqual(surrog[i], st.t_stop)

    def test_spike_jittering_output_format(self):

        st = neo.SpikeTrain([90, 150, 180, 350] * pq.ms, t_stop=500 * pq.ms)

        nr_surr = 2
        binsize = 100 * pq.ms
        surrs = surr.spike_jittering(st, binsize=binsize, n=nr_surr)

        self.assertEqual(type(surrs), list)
        self.assertEqual(len(surrs), nr_surr)

        for surrog in surrs:
            self.assertEqual(type(surrs[0]), neo.SpikeTrain)
            self.assertEqual(surrog.units, st.units)
            self.assertEqual(surrog.t_start, st.t_start)
            self.assertEqual(surrog.t_stop, st.t_stop)
            self.assertEqual(len(surrog), len(st))

    def test_spike_jittering_empty_train(self):

        st = neo.SpikeTrain([] * pq.ms, t_stop=500 * pq.ms)

        binsize = 75 * pq.ms
        surrog = surr.spike_jittering(st, binsize=binsize, n=1)[0]
        self.assertEqual(len(surrog), 0)

    def test_spike_jittering_same_bins(self):

        st = neo.SpikeTrain([90, 150, 180, 350] * pq.ms, t_stop=500 * pq.ms)

        binsize = 100 * pq.ms
        surrog = surr.spike_jittering(st, binsize=binsize, n=1)[0]

        bin_ids_orig = np.array((st.view(pq.Quantity) / binsize).rescale(
            pq.dimensionless).magnitude, dtype=int)
        bin_ids_surr = np.array((surrog.view(pq.Quantity) / binsize).rescale(
            pq.dimensionless).magnitude, dtype=int)
        self.assertTrue(np.all(bin_ids_orig == bin_ids_surr))

        # Bug encountered when the original and surrogate trains have
        # different number of spikes
        self.assertEqual(len(st), len(surrog))

    def test_spike_jittering_unequal_binsize(self):

        st = neo.SpikeTrain([90, 150, 180, 480] * pq.ms, t_stop=500 * pq.ms)

        binsize = 75 * pq.ms
        surrog = surr.spike_jittering(st, binsize=binsize, n=1)[0]

        bin_ids_orig = np.array((st.view(pq.Quantity) / binsize).rescale(
            pq.dimensionless).magnitude, dtype=int)
        bin_ids_surr = np.array((surrog.view(pq.Quantity) / binsize).rescale(
            pq.dimensionless).magnitude, dtype=int)

        self.assertTrue(np.all(bin_ids_orig == bin_ids_surr))

    def test_surr_method(self):

        st = neo.SpikeTrain([90, 150, 180, 350] * pq.ms, t_stop=500 * pq.ms)
        nr_surr = 2
        surrs = surr.surrogates(st, dt=3*pq.ms, n=nr_surr,
                                surr_method='isi_shuffling', edges=False)

        self.assertRaises(ValueError, surr.surrogates, st, n=1,
                          surr_method='spike_shifting',
                          dt=None, decimals=None, edges=True)
        self.assertTrue(len(surrs) == nr_surr)

        nr_surr2 = 4
        surrs2 = surr.surrogates(st, dt=5*pq.ms, n=nr_surr2,
                                 surr_method='train_shifting', edges=True)

        for surrog in surrs:
            self.assertTrue(isinstance(surrs[0], neo.SpikeTrain))
            self.assertEqual(surrog.units, st.units)
            self.assertEqual(surrog.t_start, st.t_start)
            self.assertEqual(surrog.t_stop, st.t_stop)
            self.assertEqual(len(surrog), len(st))
        self.assertTrue(len(surrs) == nr_surr)

        for surrog in surrs2:
            self.assertTrue(isinstance(surrs2[0], neo.SpikeTrain))
            self.assertEqual(surrog.units, st.units)
            self.assertEqual(surrog.t_start, st.t_start)
            self.assertEqual(surrog.t_stop, st.t_stop)
            self.assertEqual(len(surrog), len(st))
        self.assertTrue(len(surrs2) == nr_surr2)


def suite():
    suite = unittest.makeSuite(SurrogatesTestCase, 'test')
    return suite

if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())