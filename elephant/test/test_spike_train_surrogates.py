# -*- coding: utf-8 -*-
"""
unittests for spike_train_surrogates module.

:copyright: Copyright 2014-2016 by the Elephant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
"""

import unittest
import elephant.spike_train_surrogates as surr
import elephant.spike_train_generation as stg
import numpy as np
from numpy.testing import assert_array_almost_equal
import quantities as pq
import neo
import random


class SurrogatesTestCase(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        random.seed(0)

    def test_dither_spikes_output_format(self):

        st = neo.SpikeTrain([90, 150, 180, 350] * pq.ms, t_stop=5 * pq.s)
        st.t_stop = .5 * pq.s
        nr_surr = 2
        dither = 10 * pq.ms
        surrs = surr.dither_spikes(st, dither=dither, n=nr_surr)

        self.assertIsInstance(surrs, list)
        self.assertEqual(len(surrs), nr_surr)

        for surrog in surrs:
            self.assertIsInstance(surrs[0], neo.SpikeTrain)
            self.assertEqual(surrog.units, st.units)
            self.assertEqual(surrog.t_start, st.t_start)
            self.assertEqual(surrog.t_stop, st.t_stop)
            self.assertEqual(len(surrog), len(st))

    def test_dither_spikes_empty_train(self):

        st = neo.SpikeTrain([] * pq.ms, t_stop=500 * pq.ms)

        dither = 10 * pq.ms
        surrog = surr.dither_spikes(st, dither=dither, n=1)[0]
        self.assertEqual(len(surrog), 0)

    def test_dither_spikes_output_decimals(self):

        st = neo.SpikeTrain([90, 150, 180, 350] * pq.ms, t_stop=500 * pq.ms)

        nr_surr = 2
        dither = 10 * pq.ms
        np.random.seed(42)
        surrs = surr.dither_spikes(st, dither=dither, decimals=3, n=nr_surr)

        np.random.seed(42)
        dither_values = np.random.random_sample((nr_surr, len(st)))
        expected_non_dithered = np.sum(dither_values == 0)

        observed_non_dithered = 0
        for surrog in surrs:
            for i in range(len(surrog)):
                if surrog[i] - int(surrog[i]) * pq.ms == surrog[i] - surrog[i]:
                    observed_non_dithered += 1

        self.assertEqual(observed_non_dithered, expected_non_dithered)

    def test_dither_spikes_false_edges(self):

        st = neo.SpikeTrain([90, 150, 180, 350] * pq.ms, t_stop=500 * pq.ms)

        nr_surr = 2
        dither = 10 * pq.ms
        surrs = surr.dither_spikes(st, dither=dither, n=nr_surr, edges=False)

        for surrog in surrs:
            for i in range(len(surrog)):
                self.assertLessEqual(surrog[i], st.t_stop)

    def test_randomise_spikes_output_format(self):

        st = neo.SpikeTrain([90, 150, 180, 350] * pq.ms, t_stop=500 * pq.ms)

        nr_surr = 2
        surrs = surr.randomise_spikes(st, n=nr_surr)

        self.assertIsInstance(surrs, list)
        self.assertEqual(len(surrs), nr_surr)

        for surrog in surrs:
            self.assertIsInstance(surrs[0], neo.SpikeTrain)
            self.assertEqual(surrog.units, st.units)
            self.assertEqual(surrog.t_start, st.t_start)
            self.assertEqual(surrog.t_stop, st.t_stop)
            self.assertEqual(len(surrog), len(st))

    def test_randomise_spikes_empty_train(self):

        st = neo.SpikeTrain([] * pq.ms, t_stop=500 * pq.ms)

        surrog = surr.randomise_spikes(st, n=1)[0]
        self.assertEqual(len(surrog), 0)

    def test_randomise_spikes_output_decimals(self):
        st = neo.SpikeTrain([90, 150, 180, 350] * pq.ms, t_stop=500 * pq.ms)

        nr_surr = 2
        surrs = surr.randomise_spikes(st, n=nr_surr, decimals=3)

        for surrog in surrs:
            for i in range(len(surrog)):
                self.assertNotEqual(surrog[i] - int(surrog[i]) * pq.ms,
                                    surrog[i] - surrog[i])

    def test_shuffle_isis_output_format(self):

        st = neo.SpikeTrain([90, 150, 180, 350] * pq.ms, t_stop=500 * pq.ms)

        nr_surr = 2
        surrs = surr.shuffle_isis(st, n=nr_surr)

        self.assertIsInstance(surrs, list)
        self.assertEqual(len(surrs), nr_surr)

        for surrog in surrs:
            self.assertIsInstance(surrs[0], neo.SpikeTrain)
            self.assertEqual(surrog.units, st.units)
            self.assertEqual(surrog.t_start, st.t_start)
            self.assertEqual(surrog.t_stop, st.t_stop)
            self.assertEqual(len(surrog), len(st))

    def test_shuffle_isis_empty_train(self):

        st = neo.SpikeTrain([] * pq.ms, t_stop=500 * pq.ms)

        surrog = surr.shuffle_isis(st, n=1)[0]
        self.assertEqual(len(surrog), 0)

    def test_shuffle_isis_same_isis(self):

        st = neo.SpikeTrain([90, 150, 180, 350] * pq.ms, t_stop=500 * pq.ms)

        surrog = surr.shuffle_isis(st, n=1)[0]

        st_pq = st.view(pq.Quantity)
        surr_pq = surrog.view(pq.Quantity)

        isi0_orig = st[0] - st.t_start
        ISIs_orig = np.sort([isi0_orig] + [isi for isi in np.diff(st_pq)])

        isi0_surr = surrog[0] - surrog.t_start
        ISIs_surr = np.sort([isi0_surr] + [isi for isi in np.diff(surr_pq)])

        self.assertTrue(np.all(ISIs_orig == ISIs_surr))

    def test_shuffle_isis_output_decimals(self):

        st = neo.SpikeTrain([90, 150, 180, 350] * pq.ms, t_stop=500 * pq.ms)

        surrog = surr.shuffle_isis(st, n=1, decimals=95)[0]

        st_pq = st.view(pq.Quantity)
        surr_pq = surrog.view(pq.Quantity)

        isi0_orig = st[0] - st.t_start
        ISIs_orig = np.sort([isi0_orig] + [isi for isi in np.diff(st_pq)])

        isi0_surr = surrog[0] - surrog.t_start
        ISIs_surr = np.sort([isi0_surr] + [isi for isi in np.diff(surr_pq)])

        self.assertTrue(np.all(ISIs_orig == ISIs_surr))

    def test_dither_spike_train_output_format(self):

        st = neo.SpikeTrain([90, 150, 180, 350] * pq.ms, t_stop=500 * pq.ms)

        nr_surr = 2
        shift = 10 * pq.ms
        surrs = surr.dither_spike_train(st, shift=shift, n=nr_surr)

        self.assertIsInstance(surrs, list)
        self.assertEqual(len(surrs), nr_surr)

        for surrog in surrs:
            self.assertIsInstance(surrs[0], neo.SpikeTrain)
            self.assertEqual(surrog.units, st.units)
            self.assertEqual(surrog.t_start, st.t_start)
            self.assertEqual(surrog.t_stop, st.t_stop)
            self.assertEqual(len(surrog), len(st))

    def test_dither_spike_train_empty_train(self):

        st = neo.SpikeTrain([] * pq.ms, t_stop=500 * pq.ms)

        shift = 10 * pq.ms
        surrog = surr.dither_spike_train(st, shift=shift, n=1)[0]
        self.assertEqual(len(surrog), 0)

    def test_dither_spike_train_output_decimals(self):
        st = neo.SpikeTrain([90, 150, 180, 350] * pq.ms, t_stop=500 * pq.ms)

        nr_surr = 2
        shift = 10 * pq.ms
        surrs = surr.dither_spike_train(st, shift=shift, n=nr_surr, decimals=3)

        for surrog in surrs:
            for i in range(len(surrog)):
                self.assertNotEqual(surrog[i] - int(surrog[i]) * pq.ms,
                                    surrog[i] - surrog[i])

    def test_dither_spike_train_false_edges(self):

        st = neo.SpikeTrain([90, 150, 180, 350] * pq.ms, t_stop=500 * pq.ms)

        nr_surr = 2
        shift = 10 * pq.ms
        surrs = surr.dither_spike_train(
            st, shift=shift, n=nr_surr, edges=False)

        for surrog in surrs:
            for i in range(len(surrog)):
                self.assertLessEqual(surrog[i], st.t_stop)

    def test_jitter_spikes_output_format(self):

        st = neo.SpikeTrain([90, 150, 180, 350] * pq.ms, t_stop=500 * pq.ms)

        nr_surr = 2
        binsize = 100 * pq.ms
        surrs = surr.jitter_spikes(st, binsize=binsize, n=nr_surr)

        self.assertIsInstance(surrs, list)
        self.assertEqual(len(surrs), nr_surr)

        for surrog in surrs:
            self.assertIsInstance(surrs[0], neo.SpikeTrain)
            self.assertEqual(surrog.units, st.units)
            self.assertEqual(surrog.t_start, st.t_start)
            self.assertEqual(surrog.t_stop, st.t_stop)
            self.assertEqual(len(surrog), len(st))

    def test_jitter_spikes_empty_train(self):

        st = neo.SpikeTrain([] * pq.ms, t_stop=500 * pq.ms)

        binsize = 75 * pq.ms
        surrog = surr.jitter_spikes(st, binsize=binsize, n=1)[0]
        self.assertEqual(len(surrog), 0)

    def test_jitter_spikes_same_bins(self):

        st = neo.SpikeTrain([90, 150, 180, 350] * pq.ms, t_stop=500 * pq.ms)

        binsize = 100 * pq.ms
        surrog = surr.jitter_spikes(st, binsize=binsize, n=1)[0]

        bin_ids_orig = np.array((st.view(pq.Quantity) / binsize).rescale(
            pq.dimensionless).magnitude, dtype=int)
        bin_ids_surr = np.array((surrog.view(pq.Quantity) / binsize).rescale(
            pq.dimensionless).magnitude, dtype=int)
        self.assertTrue(np.all(bin_ids_orig == bin_ids_surr))

        # Bug encountered when the original and surrogate trains have
        # different number of spikes
        self.assertEqual(len(st), len(surrog))

    def test_jitter_spikes_unequal_binsize(self):

        st = neo.SpikeTrain([90, 150, 180, 480] * pq.ms, t_stop=500 * pq.ms)

        binsize = 75 * pq.ms
        surrog = surr.jitter_spikes(st, binsize=binsize, n=1)[0]

        bin_ids_orig = np.array((st.view(pq.Quantity) / binsize).rescale(
            pq.dimensionless).magnitude, dtype=int)
        bin_ids_surr = np.array((surrog.view(pq.Quantity) / binsize).rescale(
            pq.dimensionless).magnitude, dtype=int)

        self.assertTrue(np.all(bin_ids_orig == bin_ids_surr))

    def test_surr_method(self):

        st = neo.SpikeTrain([90, 150, 180, 350] * pq.ms, t_stop=500 * pq.ms)
        nr_surr = 2
        surrs = surr.surrogates(st, dt=3 * pq.ms, n=nr_surr,
                                surr_method='shuffle_isis', edges=False)

        self.assertRaises(AttributeError, surr.surrogates, st, n=1,
                          surr_method='spike_shifting',
                          dt=None, decimals=None, edges=True)
        self.assertTrue(len(surrs) == nr_surr)

        nr_surr2 = 4
        surrs2 = surr.surrogates(st, dt=5 * pq.ms, n=nr_surr2,
                                 surr_method='dither_spike_train', edges=True)

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

    def test_joint_isi_dithering_format(self):

        rate = 100.*pq.Hz
        t_stop = 1.*pq.s
        st = stg.homogeneous_poisson_process(rate, t_stop=t_stop)
        n_surr = 2
        dither = 10 * pq.ms

        # Test fast version
        joint_isi_instance = surr.JointISI(st, dither=dither)
        surrs = joint_isi_instance.dithering(n_surrogates=n_surr)

        self.assertIsInstance(surrs, list)
        self.assertEqual(len(surrs), n_surr)
        self.assertEqual(joint_isi_instance.method, 'fast')

        for surrog in surrs:
            self.assertIsInstance(surrog, neo.SpikeTrain)
            self.assertEqual(surrog.units, st.units)
            self.assertEqual(surrog.t_start, st.t_start)
            self.assertEqual(surrog.t_stop, st.t_stop)
            self.assertEqual(len(surrog), len(st))

        # Test window_version
        joint_isi_instance = surr.JointISI(st,
                                           method='window',
                                           dither=2*dither,
                                           num_bins=50)
        surrs = joint_isi_instance.dithering(n_surrogates=n_surr)

        self.assertIsInstance(surrs, list)
        self.assertEqual(len(surrs), n_surr)
        self.assertEqual(joint_isi_instance.method, 'window')

        for surrog in surrs:
            self.assertIsInstance(surrog, neo.SpikeTrain)
            self.assertEqual(surrog.units, st.units)
            self.assertEqual(surrog.t_start, st.t_start)
            self.assertEqual(surrog.t_stop, st.t_stop)
            self.assertEqual(len(surrog), len(st))

        # Test surrogate methods wrapper
        surrs = surr.surrogates(
            st, n=n_surr, surr_method='joint_isi_dithering')
        self.assertIsInstance(surrs, list)
        self.assertEqual(len(surrs), n_surr)

        for surrog in surrs:
            self.assertIsInstance(surrog, neo.SpikeTrain)
            self.assertEqual(surrog.units, st.units)
            self.assertEqual(surrog.t_start, st.t_start)
            self.assertEqual(surrog.t_stop, st.t_stop)
            self.assertEqual(len(surrog), len(st))

    def test_joint_isi_dithering_empty_train(self):
        st = neo.SpikeTrain([] * pq.ms, t_stop=500 * pq.ms)
        surrog = surr.JointISI(st).dithering()[0]
        self.assertEqual(len(surrog), 0)

    def test_joint_isi_dithering_output(self):
        st = stg.homogeneous_poisson_process(
            rate=100. * pq.Hz,
            refractory_period=3 * pq.ms,
            t_stop=0.1 * pq.s)
        surrog_st = surr.JointISI(st).dithering()[0]
        ground_truth = [0.005571, 0.018363, 0.026825, 0.036336, 0.045193,
                        0.05146, 0.058489, 0.078053]
        assert_array_almost_equal(surrog_st.magnitude, ground_truth)


def suite():
    suite = unittest.makeSuite(SurrogatesTestCase, 'test')
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())
