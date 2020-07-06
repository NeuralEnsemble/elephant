# -*- coding: utf-8 -*-
"""
Unit tests for the conversion module.

:copyright: Copyright 2014-2016 by the Elephant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
"""

import sys
import unittest

import neo
import numpy as np
import quantities as pq
from numpy.testing.utils import (assert_array_almost_equal,
                                 assert_array_equal)

import elephant.conversion as cv

python_version_major = sys.version_info.major


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

    @unittest.skipUnless(python_version_major == 3, "assertWarns requires 3.2")
    def test_bin_edges(self):
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

    def test_get_num_of_spikes(self):
        spiketrains = [self.spiketrain_a, self.spiketrain_b]
        for spiketrain in spiketrains:
            binned = cv.BinnedSpikeTrain(spiketrain, n_bins=10,
                                         bin_size=1 * pq.s, t_start=0 * pq.s)
            self.assertEqual(binned.get_num_of_spikes(),
                             len(binned.spike_indices[0]))
        binned_matrix = cv.BinnedSpikeTrain(spiketrains, n_bins=10,
                                            bin_size=1 * pq.s)
        n_spikes_per_row = binned_matrix.get_num_of_spikes(axis=1)
        n_spikes_per_row_from_indices = list(map(len,
                                                 binned_matrix.spike_indices))
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
        s = x.to_sparse_array()
        self.assertTrue(np.array_equal(s.data, x_sparse))
        assert_array_equal(x.spike_indices, [[1, 1, 4], [1, 1, 4]])

    def test_binned_spiketrain_shape(self):
        a = self.spiketrain_a
        x = cv.BinnedSpikeTrain(a, n_bins=10,
                                bin_size=self.bin_size,
                                t_start=0 * pq.s)
        x_bool = cv.BinnedSpikeTrain(a, n_bins=10, bin_size=self.bin_size,
                                     t_start=0 * pq.s)
        self.assertTrue(x.to_array().shape == (1, 10))
        self.assertTrue(x_bool.to_bool_array().shape == (1, 10))

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
        self.assertTrue(x.to_array().shape == (2, 5))
        self.assertTrue(x_bool.to_bool_array().shape == (2, 5))

    def test_binned_spiketrain_neg_times(self):
        a = neo.SpikeTrain(
            [-6.5, 0.5, 0.7, 1.2, 3.1, 4.3, 5.5, 6.7] * pq.s,
            t_start=-6.5 * pq.s, t_stop=10.0 * pq.s)
        bin_size = self.bin_size
        nbins = 16
        x = cv.BinnedSpikeTrain(a, n_bins=nbins, bin_size=bin_size,
                                t_start=-6.5 * pq.s)
        y = [
            np.array([1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0])]
        self.assertTrue(np.array_equal(x.to_bool_array(), y))

    @unittest.skipUnless(python_version_major == 3, "assertWarns requires 3.2")
    def test_binned_spiketrain_neg_times_list(self):
        a = neo.SpikeTrain(
            [-6.5, 0.5, 0.7, 1.2, 3.1, 4.3, 5.5, 6.7] * pq.s,
            t_start=-7 * pq.s, t_stop=7 * pq.s)
        b = neo.SpikeTrain(
            [-0.1, -0.7, 1.2, 2.2, 4.3, 5.5, 8.0] * pq.s,
            t_start=-1 * pq.s, t_stop=8 * pq.s)
        c = [a, b]

        bin_size = self.bin_size
        with self.assertWarns(UserWarning):
            x_bool = cv.BinnedSpikeTrain(c, bin_size=bin_size)
        y_bool = [[0, 1, 1, 0, 1, 1, 1, 1],
                  [1, 0, 1, 1, 0, 1, 1, 0]]

        self.assertTrue(
            np.array_equal(x_bool.to_bool_array(), y_bool))

    # checking spike_indices(f) and matrix(m) for 1 spiketrain
    def test_binned_spiketrain_indices(self):
        a = self.spiketrain_a
        bin_size = self.bin_size
        nbins = 10
        x = cv.BinnedSpikeTrain(a, n_bins=nbins, bin_size=bin_size,
                                t_start=0 * pq.s)
        x_bool = cv.BinnedSpikeTrain(a, n_bins=nbins, bin_size=bin_size,
                                     t_start=0 * pq.s)
        y_matrix = [
            np.array([2., 1., 0., 1., 1., 1., 1., 0., 0., 0.])]
        y_bool_matrix = [
            np.array([1., 1., 0., 1., 1., 1., 1., 0., 0., 0.])]
        self.assertTrue(
            np.array_equal(x.to_array(),
                           y_matrix))
        self.assertTrue(
            np.array_equal(x_bool.to_bool_array(), y_bool_matrix))
        self.assertTrue(
            np.array_equal(x_bool.to_bool_array(), y_bool_matrix))
        s = x_bool.to_sparse_bool_array()[
            x_bool.to_sparse_bool_array().nonzero()]
        self.assertTrue(np.array_equal(s, [[True] * 6]))

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
        y_matrix = np.array(
            [[2, 1, 0, 1, 1, 1, 1, 0, 0, 0],
             [2, 1, 1, 0, 1, 1, 0, 0, 1, 0]])
        y_matrix_bool = np.array(
            [[1, 1, 0, 1, 1, 1, 1, 0, 0, 0],
             [1, 1, 1, 0, 1, 1, 0, 0, 1, 0]])
        self.assertTrue(
            np.array_equal(x.to_array(),
                           y_matrix))
        self.assertTrue(
            np.array_equal(x_bool.to_bool_array(), y_matrix_bool))

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
        self.assertTrue(x.t_stop == 10 * pq.s)
        self.assertTrue(x_bool.t_stop == 10 * pq.s)

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
        self.assertTrue(x.n_bins == 10)
        self.assertTrue(x_bool.n_bins == 10)

    def test_binned_spiketrain_matrix(self):
        # Init
        a = self.spiketrain_a
        b = self.spiketrain_b
        x_bool_a = cv.BinnedSpikeTrain(a, bin_size=pq.s, t_start=0 * pq.s,
                                       t_stop=10. * pq.s)
        x_bool_b = cv.BinnedSpikeTrain(b, bin_size=pq.s, t_start=0 * pq.s,
                                       t_stop=10. * pq.s)

        # Assumed results
        y_matrix_a = [
            np.array([2, 1, 0, 1, 1, 1, 1, 0, 0, 0])]
        y_matrix_bool_a = [np.array([1, 1, 0, 1, 1, 1, 1, 0, 0, 0])]
        y_matrix_bool_b = [np.array([1, 1, 1, 0, 1, 1, 0, 0, 1, 0])]

        # Asserts
        self.assertTrue(
            np.array_equal(x_bool_a.to_bool_array(), y_matrix_bool_a))
        self.assertTrue(np.array_equal(x_bool_b.to_bool_array(),
                                       y_matrix_bool_b))
        self.assertTrue(
            np.array_equal(x_bool_a.to_array(), y_matrix_a))

    def test_binned_spiketrain_matrix_storing(self):
        a = self.spiketrain_a
        b = self.spiketrain_b

        x_bool = cv.BinnedSpikeTrain(a, bin_size=pq.s, t_start=0 * pq.s,
                                     t_stop=10. * pq.s)
        x = cv.BinnedSpikeTrain(b, bin_size=pq.s, t_start=0 * pq.s,
                                t_stop=10. * pq.s)
        # Store Matrix in variable
        matrix_bool = x_bool.to_bool_array()
        matrix = x.to_array(store_array=True)

        # Check if same matrix
        self.assertTrue(np.array_equal(x._mat_u,
                                       matrix))
        # Get the stored matrix using method
        self.assertTrue(
            np.array_equal(x_bool.to_bool_array(),
                           matrix_bool))
        self.assertTrue(
            np.array_equal(x.to_array(),
                           matrix))

        # Test storing of sparse mat
        sparse_bool = x_bool.to_sparse_bool_array()
        self.assertTrue(np.array_equal(
            sparse_bool.toarray(),
            x_bool.to_sparse_bool_array().toarray()))

        # New class without calculating the matrix
        x = cv.BinnedSpikeTrain(b, bin_size=pq.s, t_start=0 * pq.s,
                                t_stop=10. * pq.s)
        # No matrix calculated, should be None
        self.assertEqual(x._mat_u, None)
        # Test with stored matrix
        self.assertFalse(np.array_equal(x, matrix))

    # Test matrix removing
    def test_binned_spiketrain_remove_matrix(self):
        a = self.spiketrain_a
        x = cv.BinnedSpikeTrain(a, bin_size=1 * pq.s, n_bins=10,
                                t_stop=10. * pq.s)
        # Store
        x.to_array(store_array=True)
        # Remove
        x.remove_stored_array()
        # Assert matrix is not stored
        self.assertIsNone(x._mat_u)

    # Test if t_start is calculated correctly
    def test_binned_spiketrain_parameter_calc_tstart(self):
        a = self.spiketrain_a
        x = cv.BinnedSpikeTrain(a, bin_size=1 * pq.s, n_bins=10,
                                t_stop=10. * pq.s)
        self.assertEqual(x.t_start, 0. * pq.s)
        self.assertEqual(x.t_stop, 10. * pq.s)
        self.assertEqual(x.bin_size, 1 * pq.s)
        self.assertEqual(x.n_bins, 10)

    # Test if error raises when type of n_bins is not an integer
    def test_binned_spiketrain_numbins_type_error(self):
        a = self.spiketrain_a
        self.assertRaises(TypeError, cv.BinnedSpikeTrain, a, bin_size=pq.s,
                          n_bins=1.4, t_start=0 * pq.s,
                          t_stop=10. * pq.s)

    # Test if error is raised when providing insufficient number of
    # parameters
    def test_binned_spiketrain_insufficient_arguments(self):
        a = self.spiketrain_a
        self.assertRaises(AttributeError, cv.BinnedSpikeTrain, a)
        self.assertRaises(
            ValueError,
            cv.BinnedSpikeTrain,
            a,
            bin_size=1 * pq.s,
            t_start=0 * pq.s,
            t_stop=0 * pq.s)

    def test_calc_attributes_error(self):
        self.assertRaises(ValueError, cv._calc_number_of_bins,
                          1, 1 * pq.s, 0 * pq.s, self.tolerance)
        self.assertRaises(ValueError, cv._calc_bin_size,
                          1, 1 * pq.s, 0 * pq.s)

    def test_different_input_types(self):
        a = self.spiketrain_a
        q = [1, 2, 3] * pq.s
        self.assertRaises(
            TypeError, cv.BinnedSpikeTrain, [
                a, q], bin_size=pq.s)

    def test_get_start_stop(self):
        a = self.spiketrain_a
        b = neo.SpikeTrain(
            [-0.1, -0.7, 1.2, 2.2, 4.3, 5.5, 8.0] * pq.s,
            t_start=-1 * pq.s, t_stop=8 * pq.s)
        start, stop = cv._get_start_stop_from_input(a)
        self.assertEqual(start, a.t_start)
        self.assertEqual(stop, a.t_stop)
        start, stop = cv._get_start_stop_from_input([a, b])
        self.assertEqual(start, a.t_start)
        self.assertEqual(stop, b.t_stop)

    def test_consistency_errors(self):
        a = self.spiketrain_a
        b = neo.SpikeTrain([-2, -1] * pq.s, t_start=-2 * pq.s,
                           t_stop=-1 * pq.s)
        self.assertRaises(ValueError, cv.BinnedSpikeTrain, [a, b], t_start=5,
                          t_stop=0, bin_size=pq.s, n_bins=10)

        b = neo.SpikeTrain([-7, -8, -9] * pq.s, t_start=-9 * pq.s,
                           t_stop=-7 * pq.s)
        self.assertRaises(ValueError, cv.BinnedSpikeTrain, b, t_start=0,
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
        edges = [float(i) for i in range(11)]
        self.assertTrue(np.array_equal(x.bin_edges, edges))

        # Test left edges
        edges = [float(i) for i in range(10)]
        self.assertTrue(np.array_equal(x.bin_edges[:-1], edges))

        # Test right edges
        edges = [float(i) for i in range(1, 11)]
        self.assertTrue(np.array_equal(x.bin_edges[1:], edges))

        # Test center edges
        edges = np.arange(0, 10) + 0.5
        self.assertTrue(np.array_equal(x.bin_centers, edges))

    # Test for different units but same times
    def test_binned_spiketrain_different_units(self):
        a = self.spiketrain_a
        b = a.rescale(pq.ms)
        bin_size = 1 * pq.s
        xa = cv.BinnedSpikeTrain(a, bin_size=bin_size)
        xb = cv.BinnedSpikeTrain(b, bin_size=bin_size.rescale(pq.ms))
        self.assertTrue(
            np.array_equal(xa.to_bool_array(), xb.to_bool_array()))
        self.assertTrue(
            np.array_equal(xa.to_sparse_array().data,
                           xb.to_sparse_array().data))
        self.assertTrue(
            np.array_equal(xa.bin_edges[:-1],
                           xb.bin_edges[:-1].rescale(bin_size.units)))

    def test_binary_to_binned_matrix(self):
        a = [[1, 0, 0, 0], [0, 1, 1, 0]]
        x = cv.BinnedSpikeTrain(a, t_start=0 * pq.s, t_stop=5 * pq.s)
        # Check for correctness with different init params
        self.assertTrue(np.array_equal(a, x.to_bool_array()))
        self.assertTrue(np.array_equal(np.array(a), x.to_bool_array()))
        self.assertTrue(np.array_equal(a, x.to_bool_array()))
        self.assertEqual(x.n_bins, 4)
        self.assertEqual(x.bin_size, 1.25 * pq.s)

        x = cv.BinnedSpikeTrain(a, t_start=1 * pq.s, bin_size=2 * pq.s)
        self.assertTrue(np.array_equal(a, x.to_bool_array()))
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
        self.assertTrue(np.array_equal(x, y.to_array()))

        # test with a list
        x = cv.BinnedSpikeTrain([[0, 1, 2, 3]], bin_size=1 * pq.s,
                                t_stop=3 * pq.s).to_array()
        y = cv.BinnedSpikeTrain(x, bin_size=1 * pq.s, t_start=0 * pq.s)
        self.assertTrue(np.array_equal(x, y.to_array()))

        # test with a numpy array
        a = np.array([[0, 1, 2, 3], [1, 2, 2.5, 3]])
        x = cv.BinnedSpikeTrain(a, bin_size=1 * pq.s,
                                t_stop=3 * pq.s).to_array()
        y = cv.BinnedSpikeTrain(x, bin_size=1 * pq.s, t_start=0 * pq.s)
        self.assertTrue(np.array_equal(x, y.to_array()))

        # Check binary property
        self.assertFalse(y.is_binary)

        # Raise Errors
        # give a strangely shaped matrix as input (not MxN), which should
        # produce a TypeError
        a = np.array([[0, 1, 2, 3], [1, 2, 3]])
        self.assertRaises(TypeError, cv.BinnedSpikeTrain, a, t_start=0 * pq.s,
                          bin_size=1 * pq.s)
        # Give no t_start or t_stop
        a = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])
        self.assertRaises(AttributeError, cv.BinnedSpikeTrain, a,
                          bin_size=1 * pq.s)
        # Input format not supported
        a = np.array(([0, 1, 2], [0, 1, 2, 3, 4]))
        self.assertRaises(TypeError, cv.BinnedSpikeTrain, a, bin_size=1 * pq.s)

    def test_binnend_spiketrain_rescaling(self):
        train = neo.SpikeTrain(times=np.array([1.001, 1.002, 1.005]) * pq.s,
                               t_start=1 * pq.s, t_stop=1.01 * pq.s)
        bst = cv.BinnedSpikeTrain(train,
                                  t_start=1 * pq.s, t_stop=1.01 * pq.s,
                                  bin_size=1 * pq.ms)
        target_edges = np.array([1000, 1001, 1002, 1003, 1004, 1005, 1006,
                                 1007, 1008, 1009, 1010], dtype=np.float)
        target_centers = np.array(
            [1000.5, 1001.5, 1002.5, 1003.5, 1004.5, 1005.5, 1006.5, 1007.5,
             1008.5, 1009.5], dtype=np.float)
        self.assertTrue(np.allclose(bst.bin_edges.magnitude, target_edges))
        self.assertTrue(np.allclose(bst.bin_centers.magnitude, target_centers))
        self.assertTrue(bst.bin_centers.units == pq.ms)
        self.assertTrue(bst.bin_edges.units == pq.ms)
        bst = cv.BinnedSpikeTrain(train,
                                  t_start=1 * pq.s, t_stop=1010 * pq.ms,
                                  bin_size=1 * pq.ms)
        self.assertTrue(np.allclose(bst.bin_edges.magnitude, target_edges))
        self.assertTrue(np.allclose(bst.bin_centers.magnitude, target_centers))
        self.assertTrue(bst.bin_centers.units == pq.ms)
        self.assertTrue(bst.bin_edges.units == pq.ms)

    def test_binned_sparsity(self):
        train = neo.SpikeTrain(np.arange(10), t_stop=10 * pq.s, units=pq.s)
        bst = cv.BinnedSpikeTrain(train, n_bins=100)
        self.assertAlmostEqual(bst.sparsity, 0.1)

    # Test fix for rounding errors
    @unittest.skipUnless(python_version_major == 3, "assertWarns requires 3.2")
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
