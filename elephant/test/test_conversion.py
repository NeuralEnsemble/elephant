# -*- coding: utf-8 -*-
"""
Unit tests for the conversion module.

:copyright: Copyright 2014-2016 by the Elephant team, see AUTHORS.txt.
:license: Modified BSD, see LICENSE.txt for details.
"""

import unittest

import neo
import numpy as np
from numpy.testing.utils import assert_array_almost_equal
import quantities as pq

import elephant.conversion as cv


def get_nearest(times, time):
    return (np.abs(times-time)).argmin()


class binarize_TestCase(unittest.TestCase):
    def setUp(self):
        self.test_array_1d = np.array([1.23, 0.3, 0.87, 0.56])

    def test_binarize_with_spiketrain_exact(self):
        st = neo.SpikeTrain(self.test_array_1d, units='ms',
                            t_stop=10.0, sampling_rate=100)
        times = np.arange(0, 10.+.01, .01)
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
        times = np.arange(5., 10.+.01, .01)
        target = np.zeros_like(times).astype('bool')
        times = pq.Quantity(times, units='ms')

        res, tres = cv.binarize(st, return_times=True, t_start=5., t_stop=10.)
        assert_array_almost_equal(res, target, decimal=9)
        assert_array_almost_equal(tres, times, decimal=9)

    def test_binarize_with_spiketrain_round(self):
        st = neo.SpikeTrain(self.test_array_1d, units='ms',
                            t_stop=10.0, sampling_rate=10.0)
        times = np.arange(0, 10.+.1, .1)
        target = np.zeros_like(times).astype('bool')
        for time in np.round(self.test_array_1d, 1):
            target[get_nearest(times, time)] = True
        times = pq.Quantity(times, units='ms')

        res, tres = cv.binarize(st, return_times=True)
        assert_array_almost_equal(res, target, decimal=9)
        assert_array_almost_equal(tres, times, decimal=9)

    def test_binarize_with_quantities_exact(self):
        st = pq.Quantity(self.test_array_1d, units='ms')
        times = np.arange(0, 1.23+.01, .01)
        target = np.zeros_like(times).astype('bool')
        for time in self.test_array_1d:
            target[get_nearest(times, time)] = True
        times = pq.Quantity(times, units='ms')

        res, tres = cv.binarize(st, return_times=True,
                                sampling_rate=100.*pq.kHz)
        assert_array_almost_equal(res, target, decimal=9)
        assert_array_almost_equal(tres, times, decimal=9)

    def test_binarize_with_quantities_exact_set_ends(self):
        st = pq.Quantity(self.test_array_1d, units='ms')
        times = np.arange(0, 10.+.01, .01)
        target = np.zeros_like(times).astype('bool')
        for time in self.test_array_1d:
            target[get_nearest(times, time)] = True
        times = pq.Quantity(times, units='ms')

        res, tres = cv.binarize(st, return_times=True, t_stop=10.,
                                sampling_rate=100.*pq.kHz)
        assert_array_almost_equal(res, target, decimal=9)
        assert_array_almost_equal(tres, times, decimal=9)

    def test_binarize_with_quantities_round_set_ends(self):
        st = pq.Quantity(self.test_array_1d, units='ms')
        times = np.arange(5., 10.+.1, .1)
        target = np.zeros_like(times).astype('bool')
        times = pq.Quantity(times, units='ms')

        res, tres = cv.binarize(st, return_times=True, t_start=5., t_stop=10.,
                                sampling_rate=10.*pq.kHz)
        assert_array_almost_equal(res, target, decimal=9)
        assert_array_almost_equal(tres, times, decimal=9)

    def test_binarize_with_plain_array_exact(self):
        st = self.test_array_1d
        times = np.arange(0, 1.23+.01, .01)
        target = np.zeros_like(times).astype('bool')
        for time in self.test_array_1d:
            target[get_nearest(times, time)] = True

        res, tres = cv.binarize(st, return_times=True, sampling_rate=100)
        assert_array_almost_equal(res, target, decimal=9)
        assert_array_almost_equal(tres, times, decimal=9)

    def test_binarize_with_plain_array_exact_set_ends(self):
        st = self.test_array_1d
        times = np.arange(0, 10.+.01, .01)
        target = np.zeros_like(times).astype('bool')
        for time in self.test_array_1d:
            target[get_nearest(times, time)] = True

        res, tres = cv.binarize(st, return_times=True, t_stop=10., sampling_rate=100.)
        assert_array_almost_equal(res, target, decimal=9)
        assert_array_almost_equal(tres, times, decimal=9)

    def test_binarize_no_time(self):
        st = self.test_array_1d
        times = np.arange(0, 1.23+.01, .01)
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
                          sampling_rate=10.*pq.Hz)

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


class TimeHistogramTestCase(unittest.TestCase):
    def setUp(self):
        self.spiketrain_a = neo.SpikeTrain(
            [0.5, 0.7, 1.2, 3.1, 4.3, 5.5, 6.7] * pq.s, t_stop=10.0 * pq.s)
        self.spiketrain_b = neo.SpikeTrain(
            [0.1, 0.7, 1.2, 2.2, 4.3, 5.5, 8.0] * pq.s, t_stop=10.0 * pq.s)
        self.binsize = 1 * pq.s

    def tearDown(self):
        self.spiketrain_a = None
        del self.spiketrain_a
        self.spiketrain_b = None
        del self.spiketrain_b

    def test_binned_spiketrain_sparse(self):
        a = neo.SpikeTrain([1.7, 1.8, 4.3] * pq.s, t_stop=10.0 * pq.s)
        b = neo.SpikeTrain([1.7, 1.8, 4.3] * pq.s, t_stop=10.0 * pq.s)
        binsize = 1 * pq.s
        nbins = 10
        x = cv.BinnedSpikeTrain([a, b], num_bins=nbins, binsize=binsize,
                                t_start=0 * pq.s)
        x_sparse = [2, 1, 2, 1]
        s = x.to_sparse_array()
        self.assertTrue(np.array_equal(s.data, x_sparse))
        self.assertTrue(
            np.array_equal(x.spike_indices, [[1, 1, 4], [1, 1, 4]]))

    def test_binned_spiketrain_shape(self):
        a = self.spiketrain_a
        x = cv.BinnedSpikeTrain(a, num_bins=10,
                                binsize=self.binsize,
                                t_start=0 * pq.s)
        x_bool = cv.BinnedSpikeTrain(a, num_bins=10, binsize=self.binsize,
                                     t_start=0 * pq.s)
        self.assertTrue(x.to_array().shape == (1, 10))
        self.assertTrue(x_bool.to_bool_array().shape == (1, 10))

    # shape of the matrix for a list of spike trains
    def test_binned_spiketrain_shape_list(self):
        a = self.spiketrain_a
        b = self.spiketrain_b
        c = [a, b]
        nbins = 5
        x = cv.BinnedSpikeTrain(c, num_bins=nbins, t_start=0 * pq.s,
                                t_stop=10.0 * pq.s)
        x_bool = cv.BinnedSpikeTrain(c, num_bins=nbins, t_start=0 * pq.s,
                                     t_stop=10.0 * pq.s)
        self.assertTrue(x.to_array().shape == (2, 5))
        self.assertTrue(x_bool.to_bool_array().shape == (2, 5))

    def test_binned_spiketrain_neg_times(self):
        a = neo.SpikeTrain(
            [-6.5, 0.5, 0.7, 1.2, 3.1, 4.3, 5.5, 6.7] * pq.s,
            t_start=-6.5 * pq.s, t_stop=10.0 * pq.s)
        binsize = self.binsize
        nbins = 16
        x = cv.BinnedSpikeTrain(a, num_bins=nbins, binsize=binsize,
                                t_start=-6.5 * pq.s)
        y = [
            np.array([1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0])]
        self.assertTrue(np.array_equal(x.to_bool_array(), y))

    def test_binned_spiketrain_neg_times_list(self):
        a = neo.SpikeTrain(
            [-6.5, 0.5, 0.7, 1.2, 3.1, 4.3, 5.5, 6.7] * pq.s,
            t_start=-7 * pq.s, t_stop=7 * pq.s)
        b = neo.SpikeTrain(
            [-0.1, -0.7, 1.2, 2.2, 4.3, 5.5, 8.0] * pq.s,
            t_start=-1 * pq.s, t_stop=8 * pq.s)
        c = [a, b]

        binsize = self.binsize
        x_bool = cv.BinnedSpikeTrain(c, binsize=binsize)
        y_bool = [[0, 1, 1, 0, 1, 1, 1, 1],
                     [1, 0, 1, 1, 0, 1, 1, 0]]

        self.assertTrue(
            np.array_equal(x_bool.to_bool_array(), y_bool))

    # checking spike_indices(f) and matrix(m) for 1 spiketrain
    def test_binned_spiketrain_indices(self):
        a = self.spiketrain_a
        binsize = self.binsize
        nbins = 10
        x = cv.BinnedSpikeTrain(a, num_bins=nbins, binsize=binsize,
                                t_start=0 * pq.s)
        x_bool = cv.BinnedSpikeTrain(a, num_bins=nbins, binsize=binsize,
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
        self.assertTrue(np.array_equal(s, [[True]*6]))

    def test_binned_spiketrain_list(self):
        a = self.spiketrain_a
        b = self.spiketrain_b

        binsize = self.binsize
        nbins = 10
        c = [a, b]
        x = cv.BinnedSpikeTrain(c, num_bins=nbins, binsize=binsize,
                                t_start=0 * pq.s)
        x_bool = cv.BinnedSpikeTrain(c, num_bins=nbins, binsize=binsize,
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
        binsize = self.binsize
        nbins = 10
        x = cv.BinnedSpikeTrain(c, num_bins=nbins, binsize=binsize,
                                t_start=0 * pq.s,
                                t_stop=None)
        x_bool = cv.BinnedSpikeTrain(c, num_bins=nbins, binsize=binsize,
                                     t_start=0 * pq.s)
        self.assertTrue(x.t_stop == 10 * pq.s)
        self.assertTrue(x_bool.t_stop == 10 * pq.s)

    # Test number of bins
    def test_binned_spiketrain_list_numbins(self):
        a = self.spiketrain_a
        b = self.spiketrain_b
        c = [a, b]
        binsize = 1 * pq.s
        x = cv.BinnedSpikeTrain(c, binsize=binsize, t_start=0 * pq.s,
                                t_stop=10. * pq.s)
        x_bool = cv.BinnedSpikeTrain(c, binsize=binsize, t_start=0 * pq.s,
                                     t_stop=10. * pq.s)
        self.assertTrue(x.num_bins == 10)
        self.assertTrue(x_bool.num_bins == 10)

    def test_binned_spiketrain_matrix(self):
        # Init
        a = self.spiketrain_a
        b = self.spiketrain_b
        x_bool_a = cv.BinnedSpikeTrain(a, binsize=pq.s, t_start=0 * pq.s,
                                       t_stop=10. * pq.s)
        x_bool_b = cv.BinnedSpikeTrain(b, binsize=pq.s, t_start=0 * pq.s,
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

        x_bool = cv.BinnedSpikeTrain(a, binsize=pq.s, t_start=0 * pq.s,
                                     t_stop=10. * pq.s)
        x = cv.BinnedSpikeTrain(b, binsize=pq.s, t_start=0 * pq.s,
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
        self.assertTrue(np.array_equal(sparse_bool.toarray(),
                                       x_bool.to_sparse_bool_array().toarray()))

        # New class without calculating the matrix
        x = cv.BinnedSpikeTrain(b, binsize=pq.s, t_start=0 * pq.s,
                                t_stop=10. * pq.s)
        # No matrix calculated, should be None
        self.assertEqual(x._mat_u, None)
        # Test with stored matrix
        self.assertFalse(np.array_equal(x, matrix))

    # Test matrix removing
    def test_binned_spiketrain_remove_matrix(self):
        a = self.spiketrain_a
        x = cv.BinnedSpikeTrain(a, binsize=1 * pq.s, num_bins=10,
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
        x = cv.BinnedSpikeTrain(a, binsize=1 * pq.s, num_bins=10,
                                t_stop=10. * pq.s)
        self.assertEqual(x.t_start, 0. * pq.s)
        self.assertEqual(x.t_stop, 10. * pq.s)
        self.assertEqual(x.binsize, 1 * pq.s)
        self.assertEqual(x.num_bins, 10)

    # Test if error raises when type of num_bins is not an integer
    def test_binned_spiketrain_numbins_type_error(self):
        a = self.spiketrain_a
        self.assertRaises(TypeError, cv.BinnedSpikeTrain, a, binsize=pq.s,
                          num_bins=1.4, t_start=0 * pq.s,
                          t_stop=10. * pq.s)

    # Test if error is raised when providing insufficient number of
    # parameters
    def test_binned_spiketrain_insufficient_arguments(self):
        a = self.spiketrain_a
        self.assertRaises(AttributeError, cv.BinnedSpikeTrain, a)
        self.assertRaises(ValueError, cv.BinnedSpikeTrain, a, binsize=1 * pq.s,
                          t_start=0 * pq.s, t_stop=0 * pq.s)

    def test_calc_attributes_error(self):
        self.assertRaises(ValueError, cv._calc_num_bins, 1, 1 * pq.s, 0 * pq.s)
        self.assertRaises(ValueError, cv._calc_binsize, 1, 1 * pq.s, 0 * pq.s)

    def test_different_input_types(self):
        a = self.spiketrain_a
        q = [1, 2, 3] * pq.s
        self.assertRaises(TypeError, cv.BinnedSpikeTrain, [a, q], binsize=pq.s)

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
                          t_stop=0, binsize=pq.s, num_bins=10)

        b = neo.SpikeTrain([-7, -8, -9] * pq.s, t_start=-9 * pq.s,
                           t_stop=-7 * pq.s)
        self.assertRaises(ValueError, cv.BinnedSpikeTrain, b, t_start=0,
                          t_stop=10, binsize=pq.s, num_bins=10)
        self.assertRaises(ValueError, cv.BinnedSpikeTrain, a, t_start=0 * pq.s,
                          t_stop=10 * pq.s, binsize=3 * pq.s, num_bins=10)

        b = neo.SpikeTrain([-4, -2, 0, 1] * pq.s, t_start=-4 * pq.s,
                           t_stop=1 * pq.s)
        self.assertRaises(TypeError, cv.BinnedSpikeTrain, b, binsize=-2*pq.s,
                          t_start=-4 * pq.s, t_stop=0 * pq.s)

    # Test edges
    def test_binned_spiketrain_bin_edges(self):
        a = self.spiketrain_a
        x = cv.BinnedSpikeTrain(a, binsize=1 * pq.s, num_bins=10,
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
        binsize = 1 * pq.s
        xa = cv.BinnedSpikeTrain(a, binsize=binsize)
        xb = cv.BinnedSpikeTrain(b, binsize=binsize.rescale(pq.ms))
        self.assertTrue(
            np.array_equal(xa.to_bool_array(), xb.to_bool_array()))
        self.assertTrue(
            np.array_equal(xa.to_sparse_array().data,
                           xb.to_sparse_array().data))
        self.assertTrue(
            np.array_equal(xa.bin_edges[:-1],
                           xb.bin_edges[:-1].rescale(binsize.units)))


if __name__ == '__main__':
    unittest.main()