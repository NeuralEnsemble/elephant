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

try:
    import sklearn
except ImportError:
    HAVE_SKLEARN = False
else:
    import elephant.asset as asset
    HAVE_SKLEARN = True
    stretchedmetric2d = asset._stretched_metric_2d
    cluster = asset.cluster_matrix_entries


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

    def test_cluster_correct(self):
        mat = np.zeros((6, 6))
        mat[[2, 4, 5], [0, 0, 1]] = 1
        mat_clustered = cluster(mat, eps=4, min=2, stretch=6)

        mat_correct = np.zeros((6, 6))
        mat_correct[[4, 5], [0, 1]] = 1
        mat_correct[2, 0] = -1
        np.testing.assert_array_equal(mat_clustered, mat_correct)

    def test_cluster_symmetric(self):
        x = [0, 1, 2, 5, 6, 7]
        y = [3, 4, 5, 1, 2, 3]
        mat = np.zeros((10, 10))
        mat[x, y] = 1
        mat = mat + mat.T
        # compute stretched distance matrix
        mat_clustered = cluster(mat, eps=4, min=2, stretch=6)
        mat_equals_m1 = (mat_clustered == -1)
        mat_equals_0 = (mat_clustered == 0)
        mat_larger_0 = (mat_clustered > 0)
        np.testing.assert_array_equal(mat_equals_m1, mat_equals_m1.T)
        np.testing.assert_array_equal(mat_equals_0, mat_equals_0.T)
        np.testing.assert_array_equal(mat_larger_0, mat_larger_0.T)

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
        mat = np.array([[False, False, True, False],
                        [False, True, False, False],
                        [True, False, False, True],
                        [False, False, True, False]])
        clustered1 = asset.cluster_matrix_entries(
            mat, eps=1.5, min=2, stretch=1)
        clustered2 = asset.cluster_matrix_entries(
            mat, eps=1.5, min=3, stretch=1)
        clustered1_correctA = np.array([[0, 0, 1, 0],
                                       [0, 1, 0, 0],
                                       [1, 0, 0, 2],
                                       [0, 0, 2, 0]])
        clustered1_correctB = np.array([[0, 0, 2, 0],
                                       [0, 2, 0, 0],
                                       [2, 0, 0, 1],
                                       [0, 0, 1, 0]])
        clustered2_correct = np.array([[0, 0, 1, 0],
                                       [0, 1, 0, 0],
                                       [1, 0, 0, -1],
                                       [0, 0, -1, 0]])
        self.assertTrue(np.all(clustered1 == clustered1_correctA) or
                        np.all(clustered1 == clustered1_correctB))
        self.assertTrue(np.all(clustered2 == clustered2_correct))

    def test_intersection_matrix(self):
        st1 = neo.SpikeTrain([1, 2, 4]*pq.ms, t_stop=6*pq.ms)
        st2 = neo.SpikeTrain([1, 3, 4]*pq.ms, t_stop=6*pq.ms)
        st3 = neo.SpikeTrain([2, 5]*pq.ms, t_start=1*pq.ms, t_stop=6*pq.ms)
        st4 = neo.SpikeTrain([1, 3, 6]*pq.ms, t_stop=8*pq.ms)
        binsize = 1 * pq.ms

        # Check that the routine works for correct input...
        # ...same t_start, t_stop on both time axes
        imat_1_2, xedges, yedges = asset.intersection_matrix(
            [st1, st2], binsize, dt=5*pq.ms)
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
            [st1, st2], binsize, t_start_y=1*pq.ms, dt=5*pq.ms)
        trueimat_1_2 = np.array([[0.,  0.,  0.,  0., 0.],
                                 [2.,  1.,  1.,  2., 0.],
                                 [1.,  1.,  0.,  1., 0.],
                                 [1.,  0.,  1.,  1., 0.],
                                 [2.,  1.,  1.,  2., 0.]])
        self.assertTrue(np.all(xedges == np.arange(6)*pq.ms))  # correct bins
        self.assertTrue(np.all(imat_1_2 == trueimat_1_2))  # correct matrix

        # Check that errors are raised correctly...
        # ...for dt too large compared to length of spike trains
        self.assertRaises(ValueError, asset.intersection_matrix,
                          spiketrains=[st1, st2], binsize=binsize, dt=8*pq.ms)
        # ...for different SpikeTrain's t_starts
        self.assertRaises(ValueError, asset.intersection_matrix,
                          spiketrains=[st1, st3], binsize=binsize, dt=8*pq.ms)
        # ...when the analysis is specified for a time span where the
        # spike trains are not defined (e.g. t_start_x < SpikeTrain.t_start)
        self.assertRaises(ValueError, asset.intersection_matrix,
                          spiketrains=[st1, st2], binsize=binsize, dt=8*pq.ms,
                          t_start_x=-2*pq.ms, t_start_y=-2*pq.ms)


def suite():
    suite = unittest.makeSuite(AssetTestCase, 'test')
    return suite


def run():
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())


if __name__ == "__main__":
    unittest.main()
