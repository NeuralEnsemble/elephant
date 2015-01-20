# -*- coding: utf-8 -*-
"""
docstring goes here.

:copyright: Copyright 2014 by the Elephant team, see AUTHORS.txt.
:license: Modified BSD, see LICENSE.txt for details.
"""

import unittest

import neo
import numpy as np
from numpy.testing.utils import assert_array_almost_equal
import quantities as pq

import elephant.conversion as cv


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


class BinnedTestCase(unittest.TestCase):
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

    def test_binned_sparse(self):
        a = neo.SpikeTrain([1.7, 1.8, 4.3] * pq.s, t_stop=10.0 * pq.s)
        b = neo.SpikeTrain([1.7, 1.8, 4.3] * pq.s, t_stop=10.0 * pq.s)
        binsize = 1 * pq.s
        nbins = 10
        x = cv.Binned([a, b], num_bins=nbins, binsize=binsize,
                      t_start=0 * pq.s)
        x_sparse = [2, 1, 2, 1]
        s = x.sparse_mat_unclip
        self.assertTrue(np.array_equal(s.data, x_sparse))
        self.assertTrue(
            np.array_equal(x.spike_indices, [[1, 1, 4], [1, 1, 4]]))

    def test_binned_shape(self):
        a = self.spiketrain_a
        x_unclipped = cv.Binned(a, num_bins=10,
                                binsize=self.binsize,
                                t_start=0 * pq.s)
        x_clipped = cv.Binned(a, num_bins=10, binsize=self.binsize,
                              t_start=0 * pq.s)
        self.assertTrue(x_unclipped.mat_unclipped.shape == (1, 10))
        self.assertTrue(x_clipped.mat_clipped.shape == (1, 10))

    # shape of the matrix for a list of spike trains
    def test_binned_shape_list(self):
        a = self.spiketrain_a
        b = self.spiketrain_b
        c = [a, b]
        nbins = 5
        x_unclipped = cv.Binned(c, num_bins=nbins, t_start=0 * pq.s,
                                t_stop=10.0 * pq.s)
        x_clipped = cv.Binned(c, num_bins=nbins, t_start=0 * pq.s,
                              t_stop=10.0 * pq.s)
        self.assertTrue(x_unclipped.mat_unclipped.shape == (2, 5))
        self.assertTrue(x_clipped.mat_clipped.shape == (2, 5))

    def test_binned_neg_times(self):
        a = neo.SpikeTrain([-6.5, 0.5, 0.7, 1.2, 3.1, 4.3, 5.5, 6.7] * pq.s,
                           t_start=-6.5 * pq.s, t_stop=10.0 * pq.s)
        binsize = self.binsize
        nbins = 16
        x = cv.Binned(a, num_bins=nbins, binsize=binsize,
                      t_start=-6.5 * pq.s)
        y = [np.array([1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0])]
        self.assertTrue(np.array_equal(x.mat_clipped, y))

    def test_binned_neg_times_list(self):
        a = neo.SpikeTrain([-6.5, 0.5, 0.7, 1.2, 3.1, 4.3, 5.5, 6.7] * pq.s,
                           t_start=-7 * pq.s, t_stop=7 * pq.s)
        b = neo.SpikeTrain([-0.1, -0.7, 1.2, 2.2, 4.3, 5.5, 8.0] * pq.s,
                           t_start=-1 * pq.s, t_stop=8 * pq.s)
        c = [a, b]

        binsize = self.binsize
        x_clipped = cv.Binned(c, binsize=binsize)
        y_clipped = [[0, 1, 1, 0, 1, 1, 1, 1],
                     [1, 0, 1, 1, 0, 1, 1, 0]]

        self.assertTrue(np.array_equal(x_clipped.mat_clipped, y_clipped))

    # checking spike_indices(f) and matrix(m) for 1 spiketrain with clip(c) and
    # without clip(u)
    def test_binned_fmcu(self):
        a = self.spiketrain_a
        binsize = self.binsize
        nbins = 10
        x_unclipped = cv.Binned(a, num_bins=nbins, binsize=binsize,
                                t_start=0 * pq.s)
        x_clipped = cv.Binned(a, num_bins=nbins, binsize=binsize,
                              t_start=0 * pq.s)
        y_matrix_unclipped = [
            np.array([2., 1., 0., 1., 1., 1., 1., 0., 0., 0.])]
        y_matrix_clipped = [np.array([1., 1., 0., 1., 1., 1., 1., 0., 0., 0.])]
        self.assertTrue(
            np.array_equal(x_unclipped.mat_unclipped, y_matrix_unclipped))
        self.assertTrue(
            np.array_equal(x_clipped.mat_clipped, y_matrix_clipped))
        self.assertTrue(
            np.array_equal(x_clipped.mat_clipped, y_matrix_clipped))
        s = x_clipped.sparse_mat_clip[x_clipped.sparse_mat_clip.nonzero()]
        self.assertTrue(np.array_equal(s, [[1, 1, 1, 1, 1, 1]]))

    def test_binned_fmcu_list(self):
        a = self.spiketrain_a
        b = self.spiketrain_b

        binsize = self.binsize
        nbins = 10
        c = [a, b]
        x_unclipped = cv.Binned(c, num_bins=nbins, binsize=binsize,
                                t_start=0 * pq.s)
        x_clipped = cv.Binned(c, num_bins=nbins, binsize=binsize,
                              t_start=0 * pq.s)
        y_matrix_unclipped = np.array(
            [[2, 1, 0, 1, 1, 1, 1, 0, 0, 0], [2, 1, 1, 0, 1, 1, 0, 0, 1, 0]])
        y_matrix_clipped = np.array(
            [[1, 1, 0, 1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 1, 1, 0, 0, 1, 0]])
        self.assertTrue(
            np.array_equal(x_unclipped.mat_unclipped, y_matrix_unclipped))
        self.assertTrue(
            np.array_equal(x_clipped.mat_clipped, y_matrix_clipped))

    # t_stop is None
    def test_binned_list_t_stop(self):
        a = self.spiketrain_a
        b = self.spiketrain_b
        c = [a, b]
        binsize = self.binsize
        nbins = 10
        x = cv.Binned(c, num_bins=nbins, binsize=binsize, t_start=0 * pq.s,
                      t_stop=None)
        x_clipped = cv.Binned(c, num_bins=nbins, binsize=binsize,
                              t_start=0 * pq.s)
        self.assertTrue(x.t_stop == 10 * pq.s)
        self.assertTrue(x_clipped.t_stop == 10 * pq.s)

    # Test number of bins
    def test_binned_list_numbins(self):
        a = self.spiketrain_a
        b = self.spiketrain_b
        c = [a, b]
        binsize = 1 * pq.s
        x_unclipped = cv.Binned(c, binsize=binsize, t_start=0 * pq.s,
                                t_stop=10. * pq.s)
        x_clipped = cv.Binned(c, binsize=binsize, t_start=0 * pq.s,
                              t_stop=10. * pq.s)
        self.assertTrue(x_unclipped.num_bins == 10)
        self.assertTrue(x_clipped.num_bins == 10)

    def test_binned_matrix(self):
        # Init
        a = self.spiketrain_a
        b = self.spiketrain_b
        x_clipped_a = cv.Binned(a, binsize=pq.s, t_start=0 * pq.s,
                                t_stop=10. * pq.s)
        x_clipped_b = cv.Binned(b, binsize=pq.s, t_start=0 * pq.s,
                                t_stop=10. * pq.s, store_mat=True)

        # Assumed results
        y_matrix_unclipped_a = [np.array([2, 1, 0, 1, 1, 1, 1, 0, 0, 0])]
        y_matrix_clipped_a = [np.array([1, 1, 0, 1, 1, 1, 1, 0, 0, 0])]
        y_matrix_clipped_b = [np.array([1, 1, 1, 0, 1, 1, 0, 0, 1, 0])]
        y_clip_add = [np.array(
            [1, 1, 1, 1, 1, 1, 1, 0, 1, 0])]  # matrix of the clipped addition
        y_uclip_add = [np.array([4, 2, 1, 1, 2, 2, 1, 0, 1, 0])]
        # Asserts
        self.assertTrue(
            np.array_equal(x_clipped_a.mat_clipped, y_matrix_clipped_a))
        self.assertTrue(np.array_equal(x_clipped_b.mat_clipped,
                                       y_matrix_clipped_b))
        self.assertTrue(
            np.array_equal(x_clipped_a.mat_unclipped,
                           y_matrix_unclipped_a))

    def test_binned_matrix_storing(self):
        a = self.spiketrain_a
        b = self.spiketrain_b

        x_clipped = cv.Binned(a, binsize=pq.s, t_start=0 * pq.s,
                              t_stop=10. * pq.s, store_mat=True)
        x_unclipped = cv.Binned(b, binsize=pq.s, t_start=0 * pq.s,
                                t_stop=10. * pq.s, store_mat=True)
        # Store Matrix in variable
        matrix_clipped = x_clipped.mat_clipped
        matrix_unclipped = x_unclipped.mat_unclipped

        # Check for boolean
        self.assertEqual(x_unclipped.store_mat_u, True)
        # Check if same matrix
        self.assertTrue(np.array_equal(x_unclipped.mat_u,
                                       matrix_unclipped))
        # Get the stored matrix using method
        self.assertTrue(
            np.array_equal(x_clipped.mat_clipped,
                           matrix_clipped))
        self.assertTrue(
            np.array_equal(x_unclipped.mat_unclipped,
                           matrix_unclipped))
        x_unclipped.store_mat_unclipped()

        # Test storing of sparse mat
        sparse_clip = x_clipped.sparse_mat_clip
        x_clipped.store_sparse_mat_clip()
        self.assertTrue(np.array_equal(sparse_clip.toarray(),
                                       x_clipped.sparse_mat_clip.toarray()))

        # New class without calculating the matrix
        x_clipped = cv.Binned(a, binsize=pq.s, t_start=0 * pq.s,
                              t_stop=10. * pq.s, store_mat=True)
        x_unclipped = cv.Binned(b, binsize=pq.s, t_start=0 * pq.s,
                                t_stop=10. * pq.s, store_mat=True)
        # No matrix calculated, should be None
        self.assertEqual(x_unclipped.mat_u, None)
        # Test with stored matrix
        self.assertFalse(np.array_equal(x_unclipped, matrix_unclipped))

    # Test if t_start is calculated correctly
    def test_binned_parameter_calc_tstart(self):
        a = self.spiketrain_a
        x = cv.Binned(a, binsize=1 * pq.s, num_bins=10,
                      t_stop=10. * pq.s)
        self.assertEqual(x.t_start, 0. * pq.s)
        self.assertEqual(x.t_stop, 10. * pq.s)
        self.assertEqual(x.binsize, 1 * pq.s)
        self.assertEqual(x.num_bins, 10)

    # Test if error raises when type of num_bins is not an integer
    def test_binned_numbins_type_error(self):
        a = self.spiketrain_a
        self.assertRaises(TypeError, cv.Binned, a, binsize=pq.s,
                          num_bins=1.4, t_start=0 * pq.s, t_stop=10. * pq.s)

    # Test if error is raised when providing insufficient number of parameter
    def test_binned_insufficient_arguments(self):
        a = self.spiketrain_a
        self.assertRaises(AttributeError, cv.Binned, a)

    # Test edges
    def test_binned_bin_edges(self):
        a = self.spiketrain_a
        x = cv.Binned(a, binsize=1 * pq.s, num_bins=10,
                      t_stop=10. * pq.s)
        # Test all edges
        edges = [float(i) for i in range(11)]
        self.assertTrue(np.array_equal(x.edges, edges))

        # Test left edges
        edges = [float(i) for i in range(10)]
        self.assertTrue(np.array_equal(x.left_edges, edges))

        # Test right edges
        edges = [float(i) for i in range(1, 11)]
        self.assertTrue(np.array_equal(x.right_edges, edges))

        # Test center edges
        edges = np.arange(0, 10) + 0.5
        self.assertTrue(np.array_equal(x.center_edges, edges))

    # Test for different units but same times
    def test_binned_different_units(self):
        a = self.spiketrain_a
        b = a.rescale(pq.ms)
        binsize = 1 * pq.s
        xa = cv.Binned(a, binsize=binsize)
        xb = cv.Binned(b, binsize=binsize.rescale(pq.ms))
        self.assertTrue(
            np.array_equal(xa.mat_clipped, xb.mat_clipped))
        self.assertTrue(
            np.array_equal(xa.sparse_mat_unclip.data,
                           xb.sparse_mat_unclip.data))
        self.assertTrue(
            np.array_equal(xa.left_edges,
                           xb.left_edges.rescale(binsize.units)))

if __name__ == '__main__':
    unittest.main()
