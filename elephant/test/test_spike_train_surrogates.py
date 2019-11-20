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
import quantities as pq
import neo

np.random.seed(0)


class SurrogatesTestCase(unittest.TestCase):

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

        self.assertRaises(ValueError, surr.surrogates, st, n=1,
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
        joint_isi_instance = surr.JointISI(st, n_surr=n_surr, dither=dither)
        surrs = joint_isi_instance.dithering()

        self.assertIsInstance(surrs, list)
        self.assertEqual(len(surrs), n_surr)
        self.assertEqual(joint_isi_instance._method, 'fast')

        for surrog in surrs:
            self.assertIsInstance(surrog, neo.SpikeTrain)
            self.assertEqual(surrog.units, st.units)
            self.assertEqual(surrog.t_start, st.t_start)
            self.assertEqual(surrog.t_stop, st.t_stop)
            self.assertEqual(len(surrog), len(st))

        # Test window_version
        joint_isi_instance.update_parameters(method='window',
                                             dither=2*dither,
                                             num_bins=50)
        surrs = joint_isi_instance.dithering()

        self.assertIsInstance(surrs, list)
        self.assertEqual(len(surrs), n_surr)
        self.assertEqual(joint_isi_instance._method, 'window')

        for surrog in surrs:
            self.assertIsInstance(surrog, neo.SpikeTrain)
            self.assertEqual(surrog.units, st.units)
            self.assertEqual(surrog.t_start, st.t_start)
            self.assertEqual(surrog.t_stop, st.t_stop)
            self.assertEqual(len(surrog), len(st))

        # Test function wrapper
        surrs = surr.joint_isi_dithering(st, n=n_surr, dither=dither)
        self.assertIsInstance(surrs, list)
        self.assertEqual(len(surrs), n_surr)

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

    def test_update_class_parameters(self):
        rate = 100. * pq.Hz
        t_stop = 1. * pq.s
        np.random.seed(0)
        st = stg.homogeneous_poisson_process(rate, t_stop=t_stop)
        n_surr = 2
        dither = 10 * pq.ms

        joint_isi_instance = surr.JointISI(st, n_surr=n_surr, dither=dither)

        old_jisih = joint_isi_instance._jisih

        # Set Minimal number of spikes higher than spikes in spiketrain (107)
        joint_isi_instance.update_parameters(min_spikes=200)
        self.assertEqual(joint_isi_instance._to_less_spikes, True)

        new_n_surr = 4
        new_dither = 15. * pq.ms
        new_truncation_limit = 120 * pq.ms
        new_num_bins = 120
        new_sigma = 5. * pq.ms
        new_alternate = 2
        new_use_sqrt = True
        new_cutoff = False
        new_expected_refr_period = 0.0001 * pq.ms
        unit = st.units

        joint_isi_instance.update_parameters(
            min_spikes=4,
            n_surr=new_n_surr,
            dither=new_dither,
            truncation_limit=new_truncation_limit,
            num_bins=new_num_bins,
            sigma=new_sigma,
            alternate=new_alternate,
            use_sqrt=new_use_sqrt,
            cutoff=new_cutoff,
            expected_refr_period=new_expected_refr_period)

        self.assertEqual(joint_isi_instance._to_less_spikes, False)
        self.assertEqual(joint_isi_instance._n_surr, new_n_surr)
        self.assertEqual(
            joint_isi_instance._dither,
            new_dither.rescale(unit).magnitude)
        self.assertEqual(
            joint_isi_instance._truncation_limit,
            new_truncation_limit.rescale(unit).magnitude)
        self.assertEqual(joint_isi_instance._num_bins, new_num_bins)
        self.assertEqual(
            joint_isi_instance._bin_width,
            new_truncation_limit.rescale(unit).magnitude/new_num_bins)
        self.assertEqual(
            joint_isi_instance._sigma,
            new_sigma.rescale(unit).magnitude)
        self.assertEqual(joint_isi_instance._sampling_rhythm, new_alternate+1)
        self.assertEqual(joint_isi_instance._use_sqrt, new_use_sqrt)
        self.assertEqual(joint_isi_instance._cutoff, new_cutoff)
        self.assertEqual(
            joint_isi_instance._refr_period,
            new_expected_refr_period.rescale(unit).magnitude)

        self.assertNotEqual(
            old_jisih.shape[0],
            joint_isi_instance._jisih.shape[0])

        # Check that a square root is applied to the Joint-ISI histogram.
        joint_isi_instance.update_parameters(sigma=0.*pq.ms)
        self.assertEqual(joint_isi_instance._sigma, 0.)
        jisih_with_sqrt = joint_isi_instance._jisih

        joint_isi_instance.update_parameters(use_sqrt=False)

        jisih_without_sqrt = joint_isi_instance._jisih

        epsilon = 0.05
        self.assertEqual(
            np.all(np.abs(jisih_with_sqrt - np.sqrt(jisih_without_sqrt))
                   < epsilon),
            True)

        # Check if it sets the refr period to the least isi.
        high_expected_refr_period = 1.*pq.s
        joint_isi_instance.update_parameters(
            expected_refr_period=high_expected_refr_period)

        self.assertNotEqual(joint_isi_instance._refr_period,
                            high_expected_refr_period.rescale(unit).magnitude)
        self.assertEqual(joint_isi_instance._refr_period,
                         np.min(joint_isi_instance._isi))

        with self.assertRaises(ValueError):
            joint_isi_instance.update_parameters(method='non existing method')

        # do this error check also for initializing the calss

        with self.assertRaises(ValueError):
            surr.JointISI(st, n_surr=n_surr, dither=dither,
                          method='non existing method')


def suite():
    suite = unittest.makeSuite(SurrogatesTestCase, 'test')
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())
