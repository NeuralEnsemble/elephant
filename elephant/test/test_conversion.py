# -*- coding: utf-8 -*-
"""
Unit tests for the conversion module.

:copyright: Copyright 2014-2020 by the Elephant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
"""

import unittest

import neo
import numpy as np
import quantities as pq
from numpy.testing import (assert_array_almost_equal, assert_array_equal)

import elephant.conversion as cv
from elephant.utils import get_common_start_stop_times
from elephant.spike_train_generation import homogeneous_poisson_process


def get_nearest(times, time):
    return (np.abs(times - time)).argmin()


class binarize_TestCase(unittest.TestCase):
    def setUp(self):
        self.test_array_1d = np.array([1.23, 0.3, 0.87, 0.56])

    def test_binarize_with_spiketrain_exact(self):
        st = neo.SpikeTrain(self.test_array_1d, units='ms',
                            t_stop=10.0, sampling_rate=100)
        times = np.arange(0, 10. + .01, .01)
        target = np.zeros_like(times).astype('bool')
        for time in self.test_array_1d:
            target[get_nearest(times, time)] = True
        times = pq.Quantity(times, units='ms')

        res, tres = cv.binarize(st, return_times=True)
        assert_array_almost_equal(res, target, decimal=9)
        assert_array_almost_equal(tres, times, decimal=9)

    def test_binarize_with_spiketrain_exact_set_ends(self):
        st = neo.SpikeTrain(self.test_array_1d, units='ms',
                            t_stop=10.0, sampling_rate=100)
        times = np.arange(5., 10. + .01, .01)
        target = np.zeros_like(times).astype('bool')
        times = pq.Quantity(times, units='ms')

        res, tres = cv.binarize(st, return_times=True, t_start=5., t_stop=10.)
        assert_array_almost_equal(res, target, decimal=9)
        assert_array_almost_equal(tres, times, decimal=9)

    def test_binarize_with_spiketrain_round(self):
        st = neo.SpikeTrain(self.test_array_1d, units='ms',
                            t_stop=10.0, sampling_rate=10.0)
        times = np.arange(0, 10. + .1, .1)
        target = np.zeros_like(times).astype('bool')
        for time in np.round(self.test_array_1d, 1):
            target[get_nearest(times, time)] = True
        times = pq.Quantity(times, units='ms')

        res, tres = cv.binarize(st, return_times=True)
        assert_array_almost_equal(res, target, decimal=9)
        assert_array_almost_equal(tres, times, decimal=9)

    def test_binarize_with_quantities_exact(self):
        st = pq.Quantity(self.test_array_1d, units='ms')
        times = np.arange(0, 1.23 + .01, .01)
        target = np.zeros_like(times).astype('bool')
        for time in self.test_array_1d:
            target[get_nearest(times, time)] = True
        times = pq.Quantity(times, units='ms')

        res, tres = cv.binarize(st, return_times=True,
                                sampling_rate=100. * pq.kHz)
        assert_array_almost_equal(res, target, decimal=9)
        assert_array_almost_equal(tres, times, decimal=9)

    def test_binarize_with_quantities_exact_set_ends(self):
        st = pq.Quantity(self.test_array_1d, units='ms')
        times = np.arange(0, 10. + .01, .01)
        target = np.zeros_like(times).astype('bool')
        for time in self.test_array_1d:
            target[get_nearest(times, time)] = True
        times = pq.Quantity(times, units='ms')

        res, tres = cv.binarize(st, return_times=True, t_stop=10.,
                                sampling_rate=100. * pq.kHz)
        assert_array_almost_equal(res, target, decimal=9)
        assert_array_almost_equal(tres, times, decimal=9)

    def test_binarize_with_quantities_round_set_ends(self):
        st = pq.Quantity(self.test_array_1d, units='ms')
        times = np.arange(5., 10. + .1, .1)
        target = np.zeros_like(times).astype('bool')
        times = pq.Quantity(times, units='ms')

        res, tres = cv.binarize(st, return_times=True, t_start=5., t_stop=10.,
                                sampling_rate=10. * pq.kHz)
        assert_array_almost_equal(res, target, decimal=9)
        assert_array_almost_equal(tres, times, decimal=9)

    def test_binarize_with_plain_array_exact(self):
        st = self.test_array_1d
        times = np.arange(0, 1.23 + .01, .01)
        target = np.zeros_like(times).astype('bool')
        for time in self.test_array_1d:
            target[get_nearest(times, time)] = True

        res, tres = cv.binarize(st, return_times=True, sampling_rate=100)
        assert_array_almost_equal(res, target, decimal=9)
        assert_array_almost_equal(tres, times, decimal=9)

    def test_binarize_with_plain_array_exact_set_ends(self):
        st = self.test_array_1d
        times = np.arange(0, 10. + .01, .01)
        target = np.zeros_like(times).astype('bool')
        for time in self.test_array_1d:
            target[get_nearest(times, time)] = True

        res, tres = cv.binarize(st, return_times=True, t_stop=10.,
                                sampling_rate=100.)
        assert_array_almost_equal(res, target, decimal=9)
        assert_array_almost_equal(tres, times, decimal=9)

    def test_binarize_no_time(self):
        st = self.test_array_1d
        times = np.arange(0, 1.23 + .01, .01)
        target = np.zeros_like(times).astype('bool')
        for time in self.test_array_1d:
            target[get_nearest(times, time)] = True

        res0, tres = cv.binarize(st, return_times=True, sampling_rate=100)
        res1 = cv.binarize(st, return_times=False, sampling_rate=100)
        res2 = cv.binarize(st, sampling_rate=100)
        assert_array_almost_equal(res0, res1, decimal=9)
        assert_array_almost_equal(res0, res2, decimal=9)

    def test_binariz_rate_with_plain_array_and_units_typeerror(self):
        st = self.test_array_1d
        self.assertRaises(TypeError, cv.binarize, st,
                          t_start=pq.Quantity(0, 'ms'),
                          sampling_rate=10.)
        self.assertRaises(TypeError, cv.binarize, st,
                          t_stop=pq.Quantity(10, 'ms'),
                          sampling_rate=10.)
        self.assertRaises(TypeError, cv.binarize, st,
                          t_start=pq.Quantity(0, 'ms'),
                          t_stop=pq.Quantity(10, 'ms'),
                          sampling_rate=10.)
        self.assertRaises(TypeError, cv.binarize, st,
                          t_start=pq.Quantity(0, 'ms'),
                          t_stop=10.,
                          sampling_rate=10.)
        self.assertRaises(TypeError, cv.binarize, st,
                          t_start=0.,
                          t_stop=pq.Quantity(10, 'ms'),
                          sampling_rate=10.)
        self.assertRaises(TypeError, cv.binarize, st,
                          sampling_rate=10. * pq.Hz)

    def test_binariz_without_sampling_rate_valueerror(self):
        st0 = self.test_array_1d
        st1 = pq.Quantity(st0, 'ms')
        self.assertRaises(ValueError, cv.binarize, st0)
        self.assertRaises(ValueError, cv.binarize, st0,
                          t_start=0)
        self.assertRaises(ValueError, cv.binarize, st0,
                          t_stop=10)
        self.assertRaises(ValueError, cv.binarize, st0,
                          t_start=0, t_stop=10)
        self.assertRaises(ValueError, cv.binarize, st1,
                          t_start=pq.Quantity(0, 'ms'), t_stop=10.)
        self.assertRaises(ValueError, cv.binarize, st1,
                          t_start=0., t_stop=pq.Quantity(10, 'ms'))
        self.assertRaises(ValueError, cv.binarize, st1)

    def test_bin_edges_empty_binned_spiketrain(self):
        st = neo.SpikeTrain(times=np.array([2.5]) * pq.s, t_start=0 * pq.s,
                            t_stop=3 * pq.s)
        with self.assertWarns(UserWarning):
            bst = cv.BinnedSpikeTrain(st, bin_size=2 * pq.s, t_start=0 * pq.s,
                                      t_stop=3 * pq.s)
        assert_array_equal(bst.bin_edges, [0., 2.] * pq.s)
        assert_array_equal(bst.spike_indices, [[]])  # no binned spikes
        self.assertEqual(bst.get_num_of_spikes(), 0)


class BinnedSpikeTrainTestCase(unittest.TestCase):
    def setUp(self):
        self.spiketrain_a = neo.SpikeTrain(
            [0.5, 0.7, 1.2, 3.1, 4.3, 5.5, 6.7] * pq.s, t_stop=10.0 * pq.s)
        self.spiketrain_b = neo.SpikeTrain(
            [0.1, 0.7, 1.2, 2.2, 4.3, 5.5, 8.0] * pq.s, t_stop=10.0 * pq.s)
        self.bin_size = 1 * pq.s
        self.tolerance = 1e-8

    def test_binarize(self):
        spiketrains = [self.spiketrain_a, self.spiketrain_b,
                       self.spiketrain_a, self.spiketrain_b]
        for sparse_format in ("csr", "csc"):
            bst = cv.BinnedSpikeTrain(spiketrains=spiketrains,
                                      bin_size=self.bin_size,
                                      sparse_format=sparse_format)
            bst_bin = bst.binarize(copy=True)
            bst_copy = bst.copy()
            assert_array_equal(bst_bin.to_array(), bst.to_bool_array())
            bst_copy.sparse_matrix.data[:] = 1
            self.assertEqual(bst_bin, bst_copy)

    def test_slice(self):
        spiketrains = [self.spiketrain_a, self.spiketrain_b,
                       self.spiketrain_a, self.spiketrain_b]
        bst = cv.BinnedSpikeTrain(spiketrains=spiketrains,
                                  bin_size=self.bin_size)
        self.assertEqual(bst[:, :], bst)
        self.assertEqual(bst[1:], cv.BinnedSpikeTrain(spiketrains[1:],
                                                      bin_size=self.bin_size))
        self.assertEqual(bst[:, :4], bst.time_slice(t_stop=4 * pq.s))
        self.assertEqual(bst[:, 1:-1], cv.BinnedSpikeTrain(
            spiketrains, bin_size=self.bin_size,
            t_start=1 * pq.s, t_stop=9 * pq.s
        ))
        self.assertEqual(bst[0, 0], cv.BinnedSpikeTrain(
            neo.SpikeTrain([0.5, 0.7], t_stop=1, units='s'),
            bin_size=self.bin_size
        ))

        # 2-seconds stride: leave [0..1, 2..3, 4..5, 6..7] interval
        self.assertEqual(bst[0, ::2], cv.BinnedSpikeTrain(
            neo.SpikeTrain([0.5, 0.7, 4.3, 6.7], t_stop=10, units='s'),
            bin_size=2 * self.bin_size
        ))

        bst_copy = bst.copy()
        bst_copy[:] = 1
        assert_array_equal(bst_copy.sparse_matrix.todense(), 1)

    def test_time_slice(self):
        spiketrains = [self.spiketrain_a, self.spiketrain_b]
        bst = cv.BinnedSpikeTrain(spiketrains=spiketrains,
                                  bin_size=self.bin_size)
        bst_equal = bst.time_slice(t_start=bst.t_start - 5 * pq.s,
                                   t_stop=bst.t_stop + 5 * pq.s)
        self.assertEqual(bst_equal, bst)
        bst_same = bst.time_slice(t_start=None, t_stop=None)
        self.assertIs(bst_same, bst)
        bst_copy = bst.time_slice(t_start=None, t_stop=None, copy=True)
        self.assertIsNot(bst_copy, bst)
        self.assertEqual(bst_copy, bst)
        bst_empty = bst.time_slice(t_start=0.2 * pq.s, t_stop=0.3 * pq.s)
        self.assertEqual(bst_empty.n_bins, 0)
        t_range = np.arange(0, 10, self.bin_size.item()) * pq.s
        for i, t_start in enumerate(t_range[:-1]):
            for t_stop in t_range[i + 1:]:
                bst_ij = bst.time_slice(t_start=t_start, t_stop=t_stop)
                bst_ij2 = bst_ij.time_slice(t_start=t_start, t_stop=t_stop)
                self.assertEqual(bst_ij2, bst_ij)
                self.assertEqual(bst_ij2.tolerance, bst.tolerance)
                sts = [st.time_slice(t_start=t_start, t_stop=t_stop)
                       for st in spiketrains]
                bst_ref = cv.BinnedSpikeTrain(sts, bin_size=self.bin_size)
                self.assertEqual(bst_ij, bst_ref)

        # invalid input: not a quantity
        self.assertRaises(TypeError, bst.time_slice, t_start=2)

    def test_to_spike_trains(self):
        np.random.seed(1)
        spiketrains = [homogeneous_poisson_process(rate=10 * pq.Hz,
                                                   t_start=-1 * pq.s,
                                                   t_stop=10 * pq.s)]
        for sparse_format in ("csr", "csc"):
            bst1 = cv.BinnedSpikeTrain(
                spiketrains=[self.spiketrain_a, self.spiketrain_b],
                bin_size=self.bin_size, sparse_format=sparse_format
            )
            bst2 = cv.BinnedSpikeTrain(spiketrains=spiketrains,
                                       bin_size=300 * pq.ms,
                                       sparse_format=sparse_format)
            for bst in (bst1, bst2):
                for spikes in ("random", "left", "center"):
                    spiketrains_gen = bst.to_spike_trains(spikes=spikes,
                                                          annotate_bins=True)
                    for st, indices in zip(spiketrains_gen, bst.spike_indices):
                        # check sorted
                        self.assertTrue((np.diff(st.magnitude) > 0).all())
                        assert_array_equal(st.array_annotations['bins'],
                                           indices)
                        self.assertEqual(st.annotations['bin_size'],
                                         bst.bin_size)
                        self.assertEqual(st.t_start, bst.t_start)
                        self.assertEqual(st.t_stop, bst.t_stop)
                    bst_same = cv.BinnedSpikeTrain(spiketrains_gen,
                                                   bin_size=bst.bin_size,
                                                   sparse_format=sparse_format)
                    self.assertEqual(bst_same, bst)

                # invalid mode
                self.assertRaises(ValueError, bst.to_spike_trains,
                                  spikes='right')

    def test_get_num_of_spikes(self):
        spiketrains = [self.spiketrain_a, self.spiketrain_b]
        for spiketrain in spiketrains:
            binned = cv.BinnedSpikeTrain(spiketrain, n_bins=10,
                                         bin_size=1 * pq.s, t_start=0 * pq.s)
            self.assertEqual(binned.get_num_of_spikes(),
                             len(binned.spike_indices[0]))
        for sparse_format in ("csr", "csc"):
            binned_matrix = cv.BinnedSpikeTrain(spiketrains, n_bins=10,
                                                bin_size=1 * pq.s,
                                                sparse_format=sparse_format)
            n_spikes_per_row = binned_matrix.get_num_of_spikes(axis=1)
            n_spikes_per_row_from_indices = list(
                map(len, binned_matrix.spike_indices))
            assert_array_equal(n_spikes_per_row, n_spikes_per_row_from_indices)
            self.assertEqual(binned_matrix.get_num_of_spikes(),
                             sum(n_spikes_per_row_from_indices))

    def test_binned_spiketrain_sparse(self):
        a = neo.SpikeTrain([1.7, 1.8, 4.3] * pq.s, t_stop=10.0 * pq.s)
        b = neo.SpikeTrain([1.7, 1.8, 4.3] * pq.s, t_stop=10.0 * pq.s)
        bin_size = 1 * pq.s
        nbins = 10
        x = cv.BinnedSpikeTrain([a, b], n_bins=nbins, bin_size=bin_size,
                                t_start=0 * pq.s)
        x_sparse = [2, 1, 2, 1]
        assert_array_equal(x.sparse_matrix.data, x_sparse)
        assert_array_equal(x.spike_indices, [[1, 1, 4], [1, 1, 4]])

    def test_binned_spiketrain_shape(self):
        a = self.spiketrain_a
        x = cv.BinnedSpikeTrain(a, n_bins=10,
                                bin_size=self.bin_size,
                                t_start=0 * pq.s)
        x_bool = cv.BinnedSpikeTrain(a, n_bins=10, bin_size=self.bin_size,
                                     t_start=0 * pq.s)
        self.assertEqual(x.to_array().shape, (1, 10))
        self.assertEqual(x_bool.to_bool_array().shape, (1, 10))

    # shape of the matrix for a list of spike trains
    def test_binned_spiketrain_shape_list(self):
        a = self.spiketrain_a
        b = self.spiketrain_b
        c = [a, b]
        nbins = 5
        x = cv.BinnedSpikeTrain(c, n_bins=nbins, t_start=0 * pq.s,
                                t_stop=10.0 * pq.s)
        x_bool = cv.BinnedSpikeTrain(c, n_bins=nbins, t_start=0 * pq.s,
                                     t_stop=10.0 * pq.s)
        self.assertEqual(x.to_array().shape, (2, 5))
        self.assertEqual(x_bool.to_bool_array().shape, (2, 5))

    def test_binned_spiketrain_neg_times(self):
        a = neo.SpikeTrain(
            [-6.5, 0.5, 0.7, 1.2, 3.1, 4.3, 5.5, 6.7] * pq.s,
            t_start=-6.5 * pq.s, t_stop=10.0 * pq.s)
        bin_size = self.bin_size
        nbins = 16
        x = cv.BinnedSpikeTrain(a, n_bins=nbins, bin_size=bin_size,
                                t_start=-6.5 * pq.s)
        y = [[1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0]]
        assert_array_equal(x.to_bool_array(), y)

    def test_binned_spiketrain_neg_times_list(self):
        a = neo.SpikeTrain(
            [-6.5, 0.5, 0.7, 1.2, 3.1, 4.3, 5.5, 6.7] * pq.s,
            t_start=-7 * pq.s, t_stop=7 * pq.s)
        b = neo.SpikeTrain(
            [-0.1, -0.7, 1.2, 2.2, 4.3, 5.5, 8.0] * pq.s,
            t_start=-1 * pq.s, t_stop=8 * pq.s)
        spiketrains = [a, b]

        # not the same t_start and t_stop
        self.assertRaises(ValueError, cv.BinnedSpikeTrain,
                          spiketrains=spiketrains,
                          bin_size=self.bin_size)
        t_start, t_stop = get_common_start_stop_times(spiketrains)
        self.assertEqual(t_start, -1 * pq.s)
        self.assertEqual(t_stop, 7 * pq.s)
        x_bool = cv.BinnedSpikeTrain(spiketrains, bin_size=self.bin_size,
                                     t_start=t_start, t_stop=t_stop)
        y_bool = [[0, 1, 1, 0, 1, 1, 1, 1],
                  [1, 0, 1, 1, 0, 1, 1, 0]]

        assert_array_equal(x_bool.to_bool_array(), y_bool)

    # checking spike_indices(f) and matrix(m) for 1 spiketrain
    def test_binned_spiketrain_indices(self):
        a = self.spiketrain_a
        bin_size = self.bin_size
        nbins = 10
        x = cv.BinnedSpikeTrain(a, n_bins=nbins, bin_size=bin_size,
                                t_start=0 * pq.s)
        x_bool = cv.BinnedSpikeTrain(a, n_bins=nbins, bin_size=bin_size,
                                     t_start=0 * pq.s)
        y_matrix = [[2., 1., 0., 1., 1., 1., 1., 0., 0., 0.]]
        y_bool_matrix = [[1., 1., 0., 1., 1., 1., 1., 0., 0., 0.]]
        assert_array_equal(x.to_array(), y_matrix)
        assert_array_equal(x_bool.to_bool_array(), y_bool_matrix)
        s = x_bool.to_sparse_bool_array()[
            x_bool.to_sparse_bool_array().nonzero()]
        assert_array_equal(s, [[True] * 6])

    def test_binned_spiketrain_list(self):
        a = self.spiketrain_a
        b = self.spiketrain_b

        bin_size = self.bin_size
        nbins = 10
        c = [a, b]
        x = cv.BinnedSpikeTrain(c, n_bins=nbins, bin_size=bin_size,
                                t_start=0 * pq.s)
        x_bool = cv.BinnedSpikeTrain(c, n_bins=nbins, bin_size=bin_size,
                                     t_start=0 * pq.s)
        y_matrix = [[2, 1, 0, 1, 1, 1, 1, 0, 0, 0],
                    [2, 1, 1, 0, 1, 1, 0, 0, 1, 0]]
        y_matrix_bool = [[1, 1, 0, 1, 1, 1, 1, 0, 0, 0],
                         [1, 1, 1, 0, 1, 1, 0, 0, 1, 0]]
        assert_array_equal(x.to_array(), y_matrix)
        assert_array_equal(x_bool.to_bool_array(), y_matrix_bool)

    # t_stop is None
    def test_binned_spiketrain_list_t_stop(self):
        a = self.spiketrain_a
        b = self.spiketrain_b
        c = [a, b]
        bin_size = self.bin_size
        nbins = 10
        x = cv.BinnedSpikeTrain(c, n_bins=nbins, bin_size=bin_size,
                                t_start=0 * pq.s,
                                t_stop=None)
        x_bool = cv.BinnedSpikeTrain(c, n_bins=nbins, bin_size=bin_size,
                                     t_start=0 * pq.s)
        self.assertEqual(x.t_stop, 10 * pq.s)
        self.assertEqual(x_bool.t_stop, 10 * pq.s)

    # Test number of bins
    def test_binned_spiketrain_list_numbins(self):
        a = self.spiketrain_a
        b = self.spiketrain_b
        c = [a, b]
        bin_size = 1 * pq.s
        x = cv.BinnedSpikeTrain(c, bin_size=bin_size, t_start=0 * pq.s,
                                t_stop=10. * pq.s)
        x_bool = cv.BinnedSpikeTrain(c, bin_size=bin_size, t_start=0 * pq.s,
                                     t_stop=10. * pq.s)
        self.assertEqual(x.n_bins, 10)
        self.assertEqual(x_bool.n_bins, 10)

    def test_binned_spiketrain_matrix(self):
        # Init
        a = self.spiketrain_a
        b = self.spiketrain_b
        x_bool_a = cv.BinnedSpikeTrain(a, bin_size=pq.s, t_start=0 * pq.s,
                                       t_stop=10. * pq.s)
        x_bool_b = cv.BinnedSpikeTrain(b, bin_size=pq.s, t_start=0 * pq.s,
                                       t_stop=10. * pq.s)

        # Assumed results
        y_matrix_a = [[2, 1, 0, 1, 1, 1, 1, 0, 0, 0]]
        y_matrix_bool_a = [[1, 1, 0, 1, 1, 1, 1, 0, 0, 0]]
        y_matrix_bool_b = [[1, 1, 1, 0, 1, 1, 0, 0, 1, 0]]

        # Asserts
        assert_array_equal(x_bool_a.to_bool_array(), y_matrix_bool_a)
        assert_array_equal(x_bool_b.to_bool_array(), y_matrix_bool_b)
        assert_array_equal(x_bool_a.to_array(), y_matrix_a)

    # Test if t_start is calculated correctly
    def test_binned_spiketrain_parameter_calc_tstart(self):
        x = cv.BinnedSpikeTrain(self.spiketrain_a, bin_size=1 * pq.s,
                                n_bins=10, t_stop=10. * pq.s)
        self.assertEqual(x.t_start, 0. * pq.s)
        self.assertEqual(x.t_stop, 10. * pq.s)
        self.assertEqual(x.bin_size, 1 * pq.s)
        self.assertEqual(x.n_bins, 10)

    # Test if error raises when type of n_bins is not an integer
    def test_binned_spiketrain_n_bins_not_int(self):
        a = self.spiketrain_a
        self.assertRaises(ValueError, cv.BinnedSpikeTrain, a, bin_size=pq.s,
                          n_bins=1.4, t_start=0 * pq.s,
                          t_stop=10. * pq.s)

    def test_to_array(self):
        x = cv.BinnedSpikeTrain(self.spiketrain_a, bin_size=1 * pq.s,
                                n_bins=10, t_stop=10. * pq.s)
        arr_float = x.to_array(dtype=np.float32)
        assert_array_equal(arr_float, x.to_array().astype(np.float32))

    # Test if error is raised when providing insufficient number of
    # parameters
    def test_binned_spiketrain_insufficient_arguments(self):
        a = self.spiketrain_a
        self.assertRaises(ValueError, cv.BinnedSpikeTrain, a)
        self.assertRaises(
            ValueError,
            cv.BinnedSpikeTrain,
            a,
            bin_size=1 * pq.s,
            t_start=0 * pq.s,
            t_stop=0 * pq.s)

    def test_different_input_types(self):
        a = self.spiketrain_a
        q = [1, 2, 3] * pq.s
        self.assertRaises(ValueError, cv.BinnedSpikeTrain,
                          spiketrains=[a, q], bin_size=pq.s)

    def test_get_start_stop(self):
        a = self.spiketrain_a
        b = neo.SpikeTrain(
            [-0.1, -0.7, 1.2, 2.2, 4.3, 5.5, 8.0] * pq.s,
            t_start=-1 * pq.s, t_stop=8 * pq.s)
        start, stop = get_common_start_stop_times(a)
        self.assertEqual(start, a.t_start)
        self.assertEqual(stop, a.t_stop)
        start, stop = get_common_start_stop_times([a, b])
        self.assertEqual(start, a.t_start)
        self.assertEqual(stop, b.t_stop)

    def test_consistency_errors(self):
        a = self.spiketrain_a
        b = neo.SpikeTrain([-2, -1] * pq.s, t_start=-2 * pq.s,
                           t_stop=-1 * pq.s)
        self.assertRaises(TypeError, cv.BinnedSpikeTrain, [a, b], t_start=5,
                          t_stop=0, bin_size=pq.s, n_bins=10)

        b = neo.SpikeTrain([-7, -8, -9] * pq.s, t_start=-9 * pq.s,
                           t_stop=-7 * pq.s)
        self.assertRaises(TypeError, cv.BinnedSpikeTrain, b, t_start=None,
                          t_stop=10, bin_size=pq.s, n_bins=10)
        self.assertRaises(ValueError, cv.BinnedSpikeTrain, a, t_start=0 * pq.s,
                          t_stop=10 * pq.s, bin_size=3 * pq.s, n_bins=10)

        b = neo.SpikeTrain([-4, -2, 0, 1] * pq.s, t_start=-4 * pq.s,
                           t_stop=1 * pq.s)
        self.assertRaises(
            TypeError,
            cv.BinnedSpikeTrain,
            b,
            bin_size=-2 * pq.s,
            t_start=-4 * pq.s,
            t_stop=0 * pq.s)

    # Test edges
    def test_binned_spiketrain_bin_edges(self):
        a = self.spiketrain_a
        x = cv.BinnedSpikeTrain(a, bin_size=1 * pq.s, n_bins=10,
                                t_stop=10. * pq.s)
        # Test all edges
        assert_array_equal(x.bin_edges, [float(i) for i in range(11)])

        # Test center edges
        assert_array_equal(x.bin_centers, np.arange(0, 10) + 0.5)

    # Test for different units but same times
    def test_binned_spiketrain_different_units(self):
        a = self.spiketrain_a
        b = a.rescale(pq.ms)
        bin_size = 1 * pq.s
        xa = cv.BinnedSpikeTrain(a, bin_size=bin_size)
        xb = cv.BinnedSpikeTrain(b, bin_size=bin_size.rescale(pq.ms))
        assert_array_equal(xa.to_array(), xb.to_array())
        assert_array_equal(xa.to_bool_array(), xb.to_bool_array())
        assert_array_equal(xa.sparse_matrix.data,
                           xb.sparse_matrix.data)
        assert_array_equal(xa.bin_edges, xb.bin_edges)

    def test_binary_to_binned_matrix(self):
        a = [[1, 0, 0, 0], [0, 1, 1, 0]]
        x = cv.BinnedSpikeTrain(a, t_start=0 * pq.s, t_stop=5 * pq.s)
        # Check for correctness with different init params
        assert_array_equal(x.to_array(), a)
        assert_array_equal(x.to_bool_array(), a)
        self.assertEqual(x.n_bins, 4)
        self.assertEqual(x.bin_size, 1.25 * pq.s)

        x = cv.BinnedSpikeTrain(a, t_start=1 * pq.s, bin_size=2 * pq.s)
        assert_array_equal(x.to_array(), a)
        assert_array_equal(x.to_bool_array(), a)
        self.assertEqual(x.t_stop, 9 * pq.s)

        x = cv.BinnedSpikeTrain(a, t_stop=9 * pq.s, bin_size=2 * pq.s)
        self.assertEqual(x.t_start, 1 * pq.s)

        # Raise error
        self.assertRaises(ValueError, cv.BinnedSpikeTrain, a,
                          t_start=5 * pq.s, t_stop=0 * pq.s, bin_size=pq.s,
                          n_bins=10)
        self.assertRaises(ValueError, cv.BinnedSpikeTrain, a, t_start=0 * pq.s,
                          t_stop=10 * pq.s, bin_size=3 * pq.s, n_bins=10)
        self.assertRaises(ValueError, cv.BinnedSpikeTrain, a,
                          bin_size=-2 * pq.s, t_start=-4 * pq.s,
                          t_stop=0 * pq.s)

        # Check binary property
        self.assertTrue(x.is_binary)

    def test_binned_to_binned(self):
        a = self.spiketrain_a
        x = cv.BinnedSpikeTrain(a, bin_size=1 * pq.s).to_array()
        y = cv.BinnedSpikeTrain(x, bin_size=1 * pq.s, t_start=0 * pq.s)
        assert_array_equal(y.to_array(), x)

        # test with a list
        x = cv.BinnedSpikeTrain([[0, 1, 2, 3]], bin_size=1 * pq.s,
                                t_stop=3 * pq.s).to_array()
        y = cv.BinnedSpikeTrain(x, bin_size=1 * pq.s, t_start=0 * pq.s)
        assert_array_equal(y.to_array(), x)

        # test with a numpy array
        a = np.array([[0, 1, 2, 3], [1, 2, 2.5, 3]])
        x = cv.BinnedSpikeTrain(a, bin_size=1 * pq.s,
                                t_stop=3 * pq.s).to_array()
        y = cv.BinnedSpikeTrain(x, bin_size=1 * pq.s, t_start=0 * pq.s)
        assert_array_equal(y.to_array(), x)

        # Check binary property
        self.assertFalse(y.is_binary)

        # Raise Errors
        # give a strangely shaped matrix as input (not MxN)
        a = np.array([[0, 1, 2, 3], [1, 2, 3]], dtype=object)
        self.assertRaises(ValueError, cv.BinnedSpikeTrain, a, t_start=0 * pq.s,
                          bin_size=1 * pq.s)
        # Give no t_start or t_stop
        a = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])
        self.assertRaises(ValueError, cv.BinnedSpikeTrain, a,
                          bin_size=1 * pq.s)
        # Input format not supported
        a = np.array(([0, 1, 2], [0, 1, 2, 3, 4]), dtype=object)
        self.assertRaises(ValueError, cv.BinnedSpikeTrain, a,
                          bin_size=1 * pq.s)

    def test_binnend_spiketrain_different_input_units(self):
        train = neo.SpikeTrain(times=np.array([1.001, 1.002, 1.005]) * pq.s,
                               t_start=1 * pq.s, t_stop=1.01 * pq.s)
        bst = cv.BinnedSpikeTrain(train,
                                  t_start=1 * pq.s, t_stop=1.01 * pq.s,
                                  bin_size=1 * pq.ms)
        self.assertEqual(bst.units, pq.s)
        target_edges = np.array([1000, 1001, 1002, 1003, 1004, 1005, 1006,
                                 1007, 1008, 1009, 1010], dtype=np.float
                                ) * pq.ms
        target_centers = np.array(
            [1000.5, 1001.5, 1002.5, 1003.5, 1004.5, 1005.5, 1006.5, 1007.5,
             1008.5, 1009.5], dtype=np.float) * pq.ms
        assert_array_almost_equal(bst.bin_edges, target_edges)
        assert_array_almost_equal(bst.bin_centers, target_centers)

        bst = cv.BinnedSpikeTrain(train,
                                  t_start=1 * pq.s, t_stop=1010 * pq.ms,
                                  bin_size=1 * pq.ms)
        self.assertEqual(bst.units, pq.s)
        assert_array_almost_equal(bst.bin_edges, target_edges)
        assert_array_almost_equal(bst.bin_centers, target_centers)

    def test_rescale(self):
        train = neo.SpikeTrain(times=np.array([1.001, 1.002, 1.005]) * pq.s,
                               t_start=1 * pq.s, t_stop=1.01 * pq.s)
        bst = cv.BinnedSpikeTrain(train, t_start=1 * pq.s,
                                  t_stop=1.01 * pq.s,
                                  bin_size=1 * pq.ms)
        self.assertEqual(bst.units, pq.s)
        self.assertEqual(bst._t_start, 1)  # 1 s
        self.assertEqual(bst._t_stop, 1.01)  # 1.01 s
        self.assertEqual(bst._bin_size, 0.001)  # 0.001 s

        bst.rescale(units='ms')
        self.assertEqual(bst.units, pq.ms)
        self.assertEqual(bst._t_start, 1000)  # 1 s
        self.assertEqual(bst._t_stop, 1010)  # 1.01 s
        self.assertEqual(bst._bin_size, 1)  # 0.001 s

    def test_repr(self):
        train = neo.SpikeTrain(times=np.array([1.001, 1.002, 1.005]) * pq.s,
                               t_start=1 * pq.s, t_stop=1.01 * pq.s)
        bst = cv.BinnedSpikeTrain(train, t_start=1 * pq.s,
                                  t_stop=1.01 * pq.s,
                                  bin_size=1 * pq.ms)
        self.assertEqual(repr(bst), "BinnedSpikeTrain(t_start=1.0 s, "
                                    "t_stop=1.01 s, bin_size=0.001 s; "
                                    "shape=(1, 10), format=csr_matrix)")

    def test_binned_sparsity(self):
        train = neo.SpikeTrain(np.arange(10), t_stop=10 * pq.s, units=pq.s)
        bst = cv.BinnedSpikeTrain(train, n_bins=100)
        self.assertAlmostEqual(bst.sparsity, 0.1)

    # Test fix for rounding errors
    def test_binned_spiketrain_rounding(self):
        train = neo.SpikeTrain(times=np.arange(120000) / 30000. * pq.s,
                               t_start=0 * pq.s, t_stop=4 * pq.s)
        with self.assertWarns(UserWarning):
            bst = cv.BinnedSpikeTrain(train,
                                      t_start=0 * pq.s, t_stop=4 * pq.s,
                                      bin_size=1. / 30000. * pq.s)
        assert_array_equal(bst.to_array().nonzero()[1],
                           np.arange(120000))


if __name__ == '__main__':
    unittest.main()
