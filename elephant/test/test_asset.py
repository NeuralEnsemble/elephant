# -*- coding: utf-8 -*-
"""
Unit tests for the ASSET analysis.

:copyright: Copyright 2014-2016 by the Elephant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
"""

import unittest
import numpy as np
import scipy.spatial
import quantities as pq
import neo
import random

try:
    import sklearn
except ImportError:
    HAVE_SKLEARN = False
else:
    import elephant.asset as asset
    HAVE_SKLEARN = True
    stretchedmetric2d = asset._stretched_metric_2d


@unittest.skipUnless(HAVE_SKLEARN, 'requires sklearn')
class AssetTestCase(unittest.TestCase):

    def test_stretched_metric_2d_size(self):
        nr_points = 4
        x = np.arange(nr_points)
        D = stretchedmetric2d(x, x, stretch=1, ref_angle=45)
        self.assertEqual(D.shape, (nr_points, nr_points))

    def test_stretched_metric_2d_correct_stretching(self):
        x = (0, 1, 0)
        y = (0, 0, 1)
        stretch = 10
        ref_angle = 0
        D = stretchedmetric2d(x, y, stretch=stretch, ref_angle=ref_angle)
        self.assertEqual(D[0, 1], 1)
        self.assertEqual(D[0, 2], stretch)

    def test_stretched_metric_2d_symmetric(self):
        x = (1, 2, 2)
        y = (1, 2, 0)
        stretch = 10
        D = stretchedmetric2d(x, y, stretch=stretch, ref_angle=45)
        np.testing.assert_array_almost_equal(D, D.T, decimal=12)

    def test_stretched_metric_2d_equals_euclidean_if_stretch_1(self):
        x = np.arange(10)
        y = y = x ** 2 - 2 * x - 4
        # compute stretched distance matrix
        stretch = 1
        D = stretchedmetric2d(x, y, stretch=stretch, ref_angle=45)
        # Compute Euclidean distance matrix
        points = np.vstack([x, y]).T
        E = scipy.spatial.distance_matrix(points, points)
        # assert D == E
        np.testing.assert_array_almost_equal(D, E, decimal=12)

    def test_sse_difference(self):
        a = {(1, 2): set([1, 2, 3]), (3, 4): set([5, 6]), (6, 7): set([0, 1])}
        b = {(1, 2): set([1, 2, 5]), (5, 6): set([0, 2]), (6, 7): set([0, 1])}
        diff_ab_pixelwise = {(3, 4): set([5, 6])}
        diff_ba_pixelwise = {(5, 6): set([0, 2])}
        diff_ab_linkwise = {(1, 2): set([3]), (3, 4): set([5, 6])}
        diff_ba_linkwise = {(1, 2): set([5]), (5, 6): set([0, 2])}
        self.assertEqual(
            asset.sse_difference(a, b, 'pixelwise'), diff_ab_pixelwise)
        self.assertEqual(
            asset.sse_difference(b, a, 'pixelwise'), diff_ba_pixelwise)
        self.assertEqual(
            asset.sse_difference(a, b, 'linkwise'), diff_ab_linkwise)
        self.assertEqual(
            asset.sse_difference(b, a, 'linkwise'), diff_ba_linkwise)

    def test_sse_intersection(self):
        a = {(1, 2): set([1, 2, 3]), (3, 4): set([5, 6]), (6, 7): set([0, 1])}
        b = {(1, 2): set([1, 2, 5]), (5, 6): set([0, 2]), (6, 7): set([0, 1])}
        inters_ab_pixelwise = {(1, 2): set([1, 2, 3]), (6, 7): set([0, 1])}
        inters_ba_pixelwise = {(1, 2): set([1, 2, 5]), (6, 7): set([0, 1])}
        inters_ab_linkwise = {(1, 2): set([1, 2]), (6, 7): set([0, 1])}
        inters_ba_linkwise = {(1, 2): set([1, 2]), (6, 7): set([0, 1])}
        self.assertEqual(
            asset.sse_intersection(a, b, 'pixelwise'), inters_ab_pixelwise)
        self.assertEqual(
            asset.sse_intersection(b, a, 'pixelwise'), inters_ba_pixelwise)
        self.assertEqual(
            asset.sse_intersection(a, b, 'linkwise'), inters_ab_linkwise)
        self.assertEqual(
            asset.sse_intersection(b, a, 'linkwise'), inters_ba_linkwise)

    def test_sse_relations(self):
        a = {(1, 2): set([1, 2, 3]), (3, 4): set([5, 6]), (6, 7): set([0, 1])}
        b = {(1, 2): set([1, 2, 5]), (5, 6): set([0, 2]), (6, 7): set([0, 1])}
        c = {(5, 6): set([0, 2])}
        d = {(3, 4): set([0, 1]), (5, 6): set([0, 1, 2])}
        self.assertTrue(asset.sse_isequal({}, {}))
        self.assertTrue(asset.sse_isequal(a, a))
        self.assertFalse(asset.sse_isequal(b, c))
        self.assertTrue(asset.sse_isdisjoint(a, c))
        self.assertTrue(asset.sse_isdisjoint(a, d))
        self.assertFalse(asset.sse_isdisjoint(a, b))
        self.assertTrue(asset.sse_issub(c, b))
        self.assertTrue(asset.sse_issub(c, d))
        self.assertFalse(asset.sse_issub(a, b))
        self.assertTrue(asset.sse_issuper(b, c))
        self.assertTrue(asset.sse_issuper(d, c))
        self.assertFalse(asset.sse_issuper(a, b))
        self.assertTrue(asset.sse_overlap(a, b))
        self.assertFalse(asset.sse_overlap(c, d))

    def test_mask_matrix(self):
        mat1 = np.array([[0, 1], [1, 2]])
        mat2 = np.array([[2, 1], [1, 3]])
        mask_1_2 = asset.mask_matrices([mat1, mat2], [1, 2])
        mask_1_2_correct = np.array([[False, False], [False, True]])
        self.assertTrue(np.all(mask_1_2 == mask_1_2_correct))
        self.assertIsInstance(mask_1_2[0, 0], np.bool_)

    def test_cluster_matrix_entries(self):
        # test with symmetric matrix
        mat = np.array([[0, 0, 1, 0],
                        [0, 0, 0, 1],
                        [1, 0, 0, 0],
                        [0, 1, 0, 0]])
        clustered = asset.cluster_matrix_entries(
            mat, eps=1.5, min_neighbors=2, stretch=1)
        correct = np.array([[0, 0, 1, 0],
                            [0, 0, 0, 1],
                            [2, 0, 0, 0],
                            [0, 2, 0, 0]])
        np.testing.assert_array_equal(clustered, correct)

        # test with non-symmetric matrix
        mat = np.array([[0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [1, 0, 0, 1],
                        [0, 1, 0, 0]])
        clustered = asset.cluster_matrix_entries(
            mat, eps=1.5, min_neighbors=3, stretch=1)
        correct = np.array([[0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [-1, 0, 0, 1],
                            [0, -1, 0, 0]])
        np.testing.assert_array_equal(clustered, correct)

        # test with lowered min_neighbors
        mat = np.array([[0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [1, 0, 0, 1],
                        [0, 1, 0, 0]])
        clustered = asset.cluster_matrix_entries(
            mat, eps=1.5, min_neighbors=2, stretch=1)
        correct = np.array([[0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [2, 0, 0, 1],
                            [0, 2, 0, 0]])
        np.testing.assert_array_equal(clustered, correct)

    def test_intersection_matrix(self):
        st1 = neo.SpikeTrain([1, 2, 4]*pq.ms, t_stop=6*pq.ms)
        st2 = neo.SpikeTrain([1, 3, 4]*pq.ms, t_stop=6*pq.ms)
        st3 = neo.SpikeTrain([2, 5]*pq.ms, t_start=1*pq.ms, t_stop=6*pq.ms)
        binsize = 1 * pq.ms

        # Check that the routine works for correct input...
        # ...same t_start, t_stop on both time axes
        imat_1_2, xedges, yedges = asset.intersection_matrix(
            [st1, st2], binsize, t_stop_x=5*pq.ms, t_stop_y=5*pq.ms)
        trueimat_1_2 = np.array([[0.,  0.,  0.,  0.,  0.],
                                 [0.,  2.,  1.,  1.,  2.],
                                 [0.,  1.,  1.,  0.,  1.],
                                 [0.,  1.,  0.,  1.,  1.],
                                 [0.,  2.,  1.,  1.,  2.]])
        self.assertTrue(np.all(xedges == np.arange(6)*pq.ms))  # correct bins
        self.assertTrue(np.all(yedges == np.arange(6)*pq.ms))  # correct bins
        self.assertTrue(np.all(imat_1_2 == trueimat_1_2))  # correct matrix
        # ...different t_start, t_stop on the two time axes
        imat_1_2, xedges, yedges = asset.intersection_matrix(
            [st1, st2], binsize, t_start_y=6*pq.ms,
            spiketrains_y=[st + 6 * pq.ms for st in [st1, st2]],
            t_stop_x=5*pq.ms, t_stop_y=11*pq.ms)
        self.assertTrue(np.all(xedges == np.arange(6)*pq.ms))  # correct bins
        np.testing.assert_array_almost_equal(yedges, np.arange(6, 12)*pq.ms)
        self.assertTrue(np.all(imat_1_2 == trueimat_1_2))  # correct matrix

        # Check that errors are raised correctly...
        # ...for partially overlapping time intervals
        self.assertRaises(ValueError, asset.intersection_matrix,
                          spiketrains=[st1, st2], binsize=binsize,
                          t_start_y=1*pq.ms)
        # ...for different SpikeTrain's t_starts
        self.assertRaises(ValueError, asset.intersection_matrix,
                          spiketrains=[st1, st3], binsize=binsize)

    def test_integration(self):
        # define parameters
        np.random.seed(1)
        random.seed(1)
        size_group = 3
        size_sse = 3
        T = 60 * pq.ms
        binsize = 3 * pq.ms
        delay = 9 * pq.ms
        bins_between_sses = 3
        time_between_sses = 9 * pq.ms
        kernel_width = 9 * pq.ms
        jitter = 9 * pq.ms
        alpha = 0.9
        filter_shape = (5,1)
        nr_largest = 3
        eps = 3
        min_neighbors = 3
        stretch = 5
        n_surr = 20
        # ground gruth for pmats
        starting_bin_1 = int((delay/binsize).magnitude.item())
        starting_bin_2 = int((2 * delay/binsize +
                              time_between_sses/binsize).magnitude.item())
        indices_pmat_1 = np.arange(starting_bin_1, starting_bin_1 + size_sse)
        indices_pmat_2 = np.arange(starting_bin_2,
                                   starting_bin_2 + size_sse)
        indices_pmat = (np.concatenate((indices_pmat_1, indices_pmat_2)),
                        np.concatenate((indices_pmat_2, indices_pmat_1)))
        # generate spike trains
        spiketrains = [neo.SpikeTrain([index_spiketrain,
                                       index_spiketrain +
                                       size_sse +
                                       bins_between_sses] * binsize
                                      + delay + 1 * pq.ms,
                                      t_stop=T)
                       for index_group in np.arange(size_group)
                       for index_spiketrain in np.arange(size_sse)]
        # calculate probability matrix analytical
        pmat, imat, x_bins, y_bins = asset.probability_matrix_analytical(
            spiketrains,
            binsize=binsize,
            kernel_width=kernel_width)
        # calculate probability matrix montecarlo
        pmat_montecarlo, imat, x_bins, y_bins = \
            asset.probability_matrix_montecarlo(
            spiketrains,
            j=jitter,
            binsize=binsize,
            n_surr=n_surr,
            surr_method='dither_spikes')
        # test probability matrices
        np.testing.assert_array_equal(np.where(pmat > alpha), indices_pmat)
        np.testing.assert_array_equal(np.where(pmat_montecarlo > alpha),
                                      indices_pmat)
        # calculate joint probability matrix
        jmat = asset.joint_probability_matrix(pmat,
                                              filter_shape=filter_shape,
                                              nr_largest=nr_largest,
                                              verbose=True)
        # test joint probability matrix
        index_high_probabilities = (np.array([9,  9, 10, 10, 10, 11, 11]),
                                    np.array([3, 4, 3, 4, 5, 4, 5]))
        index_medium_probabilities = (np.array([8,  8,  9,  9,  9, 10, 10,
                                                 10, 11, 11, 11, 12, 12]),
                                      np.array([2, 3, 2, 3, 4, 3, 4, 5, 4, 5,
                                                6, 5, 6]))
        index_low_probabilities = (np.array([7,  8,  8,  9,  9,  9, 10, 10,
                                             10, 11, 11, 11, 12, 12, 12, 13,
                                             13]),
                                   np.array([2, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5,
                                             6, 5, 6, 7, 6, 7]))
        np.testing.assert_array_equal(np.where(jmat > 0.98),
                                      index_high_probabilities)
        np.testing.assert_array_equal(np.where(jmat > 0.9),
                                      index_medium_probabilities)
        np.testing.assert_array_equal(np.where(jmat > 0.8),
                                      index_low_probabilities)
        # test if all other entries are zeros
        mask_zeros = np.ones(jmat.shape, bool)
        mask_zeros[index_low_probabilities] = False
        self.assertTrue(np.all(jmat[mask_zeros] == 0))

        # calculate mask matrix and cluster matrix
        mmat = asset.mask_matrices([pmat, jmat], [alpha, alpha])
        cmat = asset.cluster_matrix_entries(mmat,
                                            eps=eps,
                                            min_neighbors=min_neighbors,
                                            stretch=stretch)

        # extract sses and test them
        sses = asset.extract_sse(spiketrains, binsize, cmat)
        expected_sses = {1: {(9, 3): {0, 3, 6}, (10, 4): {1, 4, 7},
                             (11, 5): {2, 5, 8}}}
        self.assertDictEqual(sses, expected_sses)


def suite():
    suite = unittest.makeSuite(AssetTestCase, 'test')
    return suite


def run():
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())


if __name__ == "__main__":
    unittest.main()
