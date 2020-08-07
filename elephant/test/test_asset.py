# -*- coding: utf-8 -*-
"""
Unit tests for the ASSET analysis.

:copyright: Copyright 2014-2016 by the Elephant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
"""

import random
import unittest
import itertools

import neo
import numpy as np
import quantities as pq
import scipy.spatial
from numpy.testing import assert_array_almost_equal, assert_array_equal

from elephant import statistics, kernels
from elephant.spike_train_generation import homogeneous_poisson_process

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
        assert_array_almost_equal(D, D.T, decimal=12)

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
        assert_array_almost_equal(D, E, decimal=12)

    def test_sse_difference(self):
        a = {(1, 2): set([1, 2, 3]), (3, 4): set([5, 6]), (6, 7): set([0, 1])}
        b = {(1, 2): set([1, 2, 5]), (5, 6): set([0, 2]), (6, 7): set([0, 1])}
        diff_ab_pixelwise = {(3, 4): set([5, 6])}
        diff_ba_pixelwise = {(5, 6): set([0, 2])}
        diff_ab_linkwise = {(1, 2): set([3]), (3, 4): set([5, 6])}
        diff_ba_linkwise = {(1, 2): set([5]), (5, 6): set([0, 2])}
        self.assertEqual(
            asset.synchronous_events_difference(a, b, 'pixelwise'),
            diff_ab_pixelwise)
        self.assertEqual(
            asset.synchronous_events_difference(b, a, 'pixelwise'),
            diff_ba_pixelwise)
        self.assertEqual(
            asset.synchronous_events_difference(a, b, 'linkwise'),
            diff_ab_linkwise)
        self.assertEqual(
            asset.synchronous_events_difference(b, a, 'linkwise'),
            diff_ba_linkwise)

    def test_sse_intersection(self):
        a = {(1, 2): set([1, 2, 3]), (3, 4): set([5, 6]), (6, 7): set([0, 1])}
        b = {(1, 2): set([1, 2, 5]), (5, 6): set([0, 2]), (6, 7): set([0, 1])}
        inters_ab_pixelwise = {(1, 2): set([1, 2, 3]), (6, 7): set([0, 1])}
        inters_ba_pixelwise = {(1, 2): set([1, 2, 5]), (6, 7): set([0, 1])}
        inters_ab_linkwise = {(1, 2): set([1, 2]), (6, 7): set([0, 1])}
        inters_ba_linkwise = {(1, 2): set([1, 2]), (6, 7): set([0, 1])}
        self.assertEqual(
            asset.synchronous_events_intersection(a, b, 'pixelwise'),
            inters_ab_pixelwise)
        self.assertEqual(
            asset.synchronous_events_intersection(b, a, 'pixelwise'),
            inters_ba_pixelwise)
        self.assertEqual(
            asset.synchronous_events_intersection(a, b, 'linkwise'),
            inters_ab_linkwise)
        self.assertEqual(
            asset.synchronous_events_intersection(b, a, 'linkwise'),
            inters_ba_linkwise)

    def test_sse_relations(self):
        a = {(1, 2): set([1, 2, 3]), (3, 4): set([5, 6]), (6, 7): set([0, 1])}
        b = {(1, 2): set([1, 2, 5]), (5, 6): set([0, 2]), (6, 7): set([0, 1])}
        c = {(5, 6): set([0, 2])}
        d = {(3, 4): set([0, 1]), (5, 6): set([0, 1, 2])}
        self.assertTrue(asset.synchronous_events_identical({}, {}))
        self.assertTrue(asset.synchronous_events_identical(a, a))
        self.assertFalse(asset.synchronous_events_identical(b, c))
        self.assertTrue(asset.synchronous_events_no_overlap(a, c))
        self.assertTrue(asset.synchronous_events_no_overlap(a, d))
        self.assertFalse(asset.synchronous_events_no_overlap(a, b))
        self.assertFalse(asset.synchronous_events_no_overlap({}, {}))
        self.assertTrue(asset.synchronous_events_contained_in(c, b))
        self.assertTrue(asset.synchronous_events_contained_in(c, d))
        self.assertFalse(asset.synchronous_events_contained_in(a, d))
        self.assertFalse(asset.synchronous_events_contained_in(a, b))
        self.assertTrue(asset.synchronous_events_contains_all(b, c))
        self.assertTrue(asset.synchronous_events_contains_all(d, c))
        self.assertFalse(asset.synchronous_events_contains_all(a, b))
        self.assertTrue(asset.synchronous_events_overlap(a, b))
        self.assertFalse(asset.synchronous_events_overlap(c, d))

    def test_mask_matrix(self):
        mat1 = np.array([[0, 1], [1, 2]])
        mat2 = np.array([[2, 1], [1, 3]])

        mask_1_2 = asset.ASSET.mask_matrices([mat1, mat2], [1, 2])
        mask_1_2_correct = np.array([[False, False], [False, True]])
        self.assertTrue(np.all(mask_1_2 == mask_1_2_correct))
        self.assertIsInstance(mask_1_2[0, 0], np.bool_)

        self.assertRaises(ValueError, asset.ASSET.mask_matrices, [], [])
        self.assertRaises(ValueError, asset.ASSET.mask_matrices,
                          [np.arange(5)], [])

    def test_cluster_matrix_entries(self):
        # test with symmetric matrix
        mat = np.array([[0, 0, 1, 0],
                        [0, 0, 0, 1],
                        [1, 0, 0, 0],
                        [0, 1, 0, 0]])

        clustered = asset.ASSET.cluster_matrix_entries(
            mat, max_distance=1.5, min_neighbors=2, stretch=1)
        correct = np.array([[0, 0, 1, 0],
                            [0, 0, 0, 1],
                            [2, 0, 0, 0],
                            [0, 2, 0, 0]])
        assert_array_equal(clustered, correct)

        # test with non-symmetric matrix
        mat = np.array([[0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [1, 0, 0, 1],
                        [0, 1, 0, 0]])
        clustered = asset.ASSET.cluster_matrix_entries(
            mat, max_distance=1.5, min_neighbors=3, stretch=1)
        correct = np.array([[0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [-1, 0, 0, 1],
                            [0, -1, 0, 0]])
        assert_array_equal(clustered, correct)

        # test with lowered min_neighbors
        mat = np.array([[0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [1, 0, 0, 1],
                        [0, 1, 0, 0]])
        clustered = asset.ASSET.cluster_matrix_entries(
            mat, max_distance=1.5, min_neighbors=2, stretch=1)
        correct = np.array([[0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [2, 0, 0, 1],
                            [0, 2, 0, 0]])
        assert_array_equal(clustered, correct)

        mat = np.zeros((4, 4))
        clustered = asset.ASSET.cluster_matrix_entries(
            mat, max_distance=1.5, min_neighbors=2, stretch=1)
        correct = mat
        assert_array_equal(clustered, correct)

    def test_intersection_matrix(self):
        st1 = neo.SpikeTrain([1, 2, 4] * pq.ms, t_stop=6 * pq.ms)
        st2 = neo.SpikeTrain([1, 3, 4] * pq.ms, t_stop=6 * pq.ms)
        st3 = neo.SpikeTrain([2, 5] * pq.ms, t_start=1 * pq.ms,
                             t_stop=6 * pq.ms)
        bin_size = 1 * pq.ms

        asset_obj_same_t_start_stop = asset.ASSET(
            [st1, st2], bin_size=bin_size, t_stop_i=5 * pq.ms,
            t_stop_j=5 * pq.ms)

        # Check that the routine works for correct input...
        # ...same t_start, t_stop on both time axes
        imat_1_2 = asset_obj_same_t_start_stop.intersection_matrix()
        trueimat_1_2 = np.array([[0., 0., 0., 0., 0.],
                                 [0., 2., 1., 1., 2.],
                                 [0., 1., 1., 0., 1.],
                                 [0., 1., 0., 1., 1.],
                                 [0., 2., 1., 1., 2.]])
        assert_array_equal(asset_obj_same_t_start_stop.x_edges,
                           np.arange(6) * pq.ms)  # correct bins
        assert_array_equal(asset_obj_same_t_start_stop.y_edges,
                           np.arange(6) * pq.ms)  # correct bins
        assert_array_equal(imat_1_2, trueimat_1_2)  # correct matrix
        # ...different t_start, t_stop on the two time axes
        asset_obj_different_t_start_stop = asset.ASSET(
            [st1, st2], spiketrains_j=[st + 6 * pq.ms for st in [st1, st2]],
            bin_size=bin_size, t_start_j=6 * pq.ms, t_stop_i=5 * pq.ms,
            t_stop_j=11 * pq.ms)
        imat_1_2 = asset_obj_different_t_start_stop.intersection_matrix()
        assert_array_equal(asset_obj_different_t_start_stop.x_edges,
                           np.arange(6) * pq.ms)  # correct bins
        assert_array_equal(asset_obj_different_t_start_stop.y_edges,
                           np.arange(6, 12) * pq.ms)
        self.assertTrue(np.all(imat_1_2 == trueimat_1_2))  # correct matrix

        # test with norm=1
        imat_1_2 = asset_obj_same_t_start_stop.intersection_matrix(
            normalization='intersection')
        trueimat_1_2 = np.array([[0., 0., 0., 0., 0.],
                                 [0., 1., 1., 1., 1.],
                                 [0., 1., 1., 0., 1.],
                                 [0., 1., 0., 1., 1.],
                                 [0., 1., 1., 1., 1.]])
        assert_array_equal(imat_1_2, trueimat_1_2)

        # test with norm=2
        imat_1_2 = asset_obj_same_t_start_stop.intersection_matrix(
            normalization='mean')
        sq = np.sqrt(2) / 2.
        trueimat_1_2 = np.array([[0., 0., 0., 0., 0.],
                                 [0., 1., sq, sq, 1.],
                                 [0., sq, 1., 0., sq],
                                 [0., sq, 0., 1., sq],
                                 [0., 1., sq, sq, 1.]])
        assert_array_almost_equal(imat_1_2, trueimat_1_2)

        # test with norm=3
        imat_1_2 = asset_obj_same_t_start_stop.intersection_matrix(
            normalization='union')
        trueimat_1_2 = np.array([[0., 0., 0., 0., 0.],
                                 [0., 1., .5, .5, 1.],
                                 [0., .5, 1., 0., .5],
                                 [0., .5, 0., 1., .5],
                                 [0., 1., .5, .5, 1.]])
        assert_array_almost_equal(imat_1_2, trueimat_1_2)

        # Check that errors are raised correctly...
        # ...for partially overlapping time intervals
        self.assertRaises(ValueError, asset.ASSET,
                          spiketrains_i=[st1, st2], bin_size=bin_size,
                          t_start_j=1 * pq.ms)
        # ...for different SpikeTrain's t_starts
        self.assertRaises(ValueError, asset.ASSET,
                          spiketrains_i=[st1, st3], bin_size=bin_size)
        # ...for different SpikeTrain's t_stops
        self.assertRaises(ValueError, asset.ASSET,
                          spiketrains_i=[st1, st2], bin_size=bin_size,
                          t_stop_j=5 * pq.ms)

    def test_combinations_with_replacement(self):
        # Test that _combinations_with_replacement yields the same tuples
        # as in the original implementation with itertools.product(*lists)
        # and filtering by _wrong_order.

        def _wrong_order(a):
            if a[-1] > a[0]:
                return True
            for i in range(len(a) - 1):
                if a[i] < a[i + 1]:
                    return True
            return False

        for n in range(1, 15):
            for d in range(1, 6):
                lists = [range(j, n + 1) for j in range(d, 0, -1)]
                matrix_entries = list(
                    asset._combinations_with_replacement(n=n, d=d)
                )
                matrix_entries_correct = [
                    indices for indices in itertools.product(*lists)
                    if not _wrong_order(indices)
                ]
                it_todo = asset._num_iterations(n=n, d=d)
                self.assertEqual(matrix_entries, matrix_entries_correct)
                self.assertEqual(it_todo, len(matrix_entries_correct))


@unittest.skipUnless(HAVE_SKLEARN, 'requires sklearn')
class AssetTestIntegration(unittest.TestCase):
    def setUp(self):
        # common for all tests
        self.bin_size = 3 * pq.ms

    def test_probability_matrix_symmetric(self):
        np.random.seed(1)
        kernel_width = 9 * pq.ms
        rate = 50 * pq.Hz
        n_spiketrains = 50
        spiketrains = []
        spiketrains_copy = []
        for _ in range(n_spiketrains):
            st = homogeneous_poisson_process(rate, t_stop=100 * pq.ms)
            spiketrains.append(st)
            spiketrains_copy.append(st.copy())

        asset_obj = asset.ASSET(spiketrains, bin_size=self.bin_size)
        asset_obj_symmetric = asset.ASSET(spiketrains,
                                          spiketrains_j=spiketrains_copy,
                                          bin_size=self.bin_size)

        imat = asset_obj.intersection_matrix()
        pmat = asset_obj.probability_matrix_analytical(
            kernel_width=kernel_width)

        imat_symm = asset_obj_symmetric.intersection_matrix()
        pmat_symm = asset_obj_symmetric.probability_matrix_analytical(
            kernel_width=kernel_width)

        assert_array_almost_equal(pmat, pmat_symm)
        assert_array_almost_equal(imat, imat_symm)
        assert_array_almost_equal(asset_obj.x_edges,
                                  asset_obj_symmetric.x_edges)
        assert_array_almost_equal(asset_obj.y_edges,
                                  asset_obj_symmetric.y_edges)

    def _test_integration_subtest(self, spiketrains, spiketrains_y,
                                  indices_pmat, index_proba, expected_sses):
        # define parameters
        random.seed(1)
        kernel_width = 9 * pq.ms
        surrogate_dt = 9 * pq.ms
        alpha = 0.9
        filter_shape = (5, 1)
        nr_largest = 3
        max_distance = 3
        min_neighbors = 3
        stretch = 5
        n_surr = 20

        def _get_rates(_spiketrains):
            kernel_sigma = kernel_width / 2. / np.sqrt(3.)
            kernel = kernels.RectangularKernel(sigma=kernel_sigma)
            rates = [statistics.instantaneous_rate(
                st,
                kernel=kernel,
                sampling_period=1 * pq.ms)
                for st in _spiketrains]
            return rates

        asset_obj = asset.ASSET(spiketrains, spiketrains_y,
                                bin_size=self.bin_size)

        # calculate the intersection matrix
        imat = asset_obj.intersection_matrix()

        # calculate probability matrix analytical
        pmat = asset_obj.probability_matrix_analytical(
            imat,
            kernel_width=kernel_width)

        # check if pmat is the same when rates are provided
        pmat_as_rates = asset_obj.probability_matrix_analytical(
            imat,
            firing_rates_x=_get_rates(spiketrains),
            firing_rates_y=_get_rates(spiketrains_y))
        assert_array_almost_equal(pmat, pmat_as_rates)

        # calculate probability matrix montecarlo
        pmat_montecarlo = asset_obj.probability_matrix_montecarlo(
            n_surrogates=n_surr,
            imat=imat,
            surrogate_dt=surrogate_dt,
            surrogate_method='dither_spikes')

        # test probability matrices
        assert_array_equal(np.where(pmat > alpha), indices_pmat)
        assert_array_equal(np.where(pmat_montecarlo > alpha),
                           indices_pmat)
        # calculate joint probability matrix
        jmat = asset_obj.joint_probability_matrix(
            pmat,
            filter_shape=filter_shape,
            n_largest=nr_largest)
        # test joint probability matrix
        assert_array_equal(np.where(jmat > 0.98), index_proba['high'])
        assert_array_equal(np.where(jmat > 0.9), index_proba['medium'])
        assert_array_equal(np.where(jmat > 0.8), index_proba['low'])
        # test if all other entries are zeros
        mask_zeros = np.ones(jmat.shape, bool)
        mask_zeros[index_proba['low']] = False
        self.assertTrue(np.all(jmat[mask_zeros] == 0))

        # calculate mask matrix and cluster matrix
        mmat = asset_obj.mask_matrices([pmat, jmat], [alpha, alpha])
        cmat = asset_obj.cluster_matrix_entries(
            mmat,
            max_distance=max_distance,
            min_neighbors=min_neighbors,
            stretch=stretch)

        # extract sses and test them
        sses = asset_obj.extract_synchronous_events(cmat)
        self.assertDictEqual(sses, expected_sses)

    def test_integration(self):
        """
        The test is written according to the notebook (for developers only):
        https://github.com/INM-6/elephant-tutorials/blob/master/
        simple_test_asset.ipynb
        """
        # define parameters
        np.random.seed(1)
        size_group = 3
        size_sse = 3
        T = 60 * pq.ms
        delay = 9 * pq.ms
        bins_between_sses = 3
        time_between_sses = 9 * pq.ms
        # ground truth for pmats
        starting_bin_1 = int((delay / self.bin_size).magnitude.item())
        starting_bin_2 = int(
            (2 * delay / self.bin_size + time_between_sses / self.bin_size
             ).magnitude.item())
        indices_pmat_1 = np.arange(starting_bin_1, starting_bin_1 + size_sse)
        indices_pmat_2 = np.arange(starting_bin_2,
                                   starting_bin_2 + size_sse)
        indices_pmat = (np.concatenate((indices_pmat_1, indices_pmat_2)),
                        np.concatenate((indices_pmat_2, indices_pmat_1)))
        # generate spike trains
        spiketrains = [neo.SpikeTrain([index_spiketrain,
                                       index_spiketrain +
                                       size_sse +
                                       bins_between_sses] * self.bin_size
                                      + delay + 1 * pq.ms,
                                      t_stop=T)
                       for index_group in range(size_group)
                       for index_spiketrain in range(size_sse)]
        index_proba = {
            "high": (np.array([9, 9, 10, 10, 10, 11, 11]),
                     np.array([3, 4, 3, 4, 5, 4, 5])),
            "medium": (np.array([8, 8, 9, 9, 9, 10, 10,
                                 10, 11, 11, 11, 12, 12]),
                       np.array([2, 3, 2, 3, 4, 3, 4, 5, 4, 5, 6, 5, 6])),
            "low": (np.array([7, 8, 8, 9, 9, 9, 10, 10, 10,
                              11, 11, 11, 12, 12, 12, 13, 13]),
                    np.array([2, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5,
                              6, 5, 6, 7, 6, 7]))
        }
        expected_sses = {1: {(9, 3): {0, 3, 6}, (10, 4): {1, 4, 7},
                             (11, 5): {2, 5, 8}}}
        self._test_integration_subtest(spiketrains,
                                       spiketrains_y=spiketrains,
                                       indices_pmat=indices_pmat,
                                       index_proba=index_proba,
                                       expected_sses=expected_sses)

    def test_integration_nonsymmetric(self):
        # define parameters
        np.random.seed(1)
        random.seed(1)
        size_group = 3
        size_sse = 3
        delay = 18 * pq.ms
        T = 4 * delay + 2 * size_sse * self.bin_size
        time_between_sses = 2 * delay
        # ground truth for pmats
        starting_bin = int((delay / self.bin_size).magnitude.item())
        indices_pmat_1 = np.arange(starting_bin, starting_bin + size_sse)
        indices_pmat = (indices_pmat_1, indices_pmat_1)
        # generate spike trains
        spiketrains = [
            neo.SpikeTrain([index_spiketrain] * self.bin_size + delay,
                           t_start=0 * pq.ms,
                           t_stop=2 * delay + size_sse * self.bin_size)
            for index_group in range(size_group)
            for index_spiketrain in range(size_sse)]
        spiketrains_y = [
            neo.SpikeTrain([index_spiketrain] * self.bin_size + delay +
                           time_between_sses + size_sse * self.bin_size,
                           t_start=size_sse * self.bin_size + 2 * delay,
                           t_stop=T)
            for index_group in range(size_group)
            for index_spiketrain in range(size_sse)]
        index_proba = {
            "high": ([6, 6, 7, 7, 7, 8, 8],
                     [6, 7, 6, 7, 8, 7, 8]),
            "medium": ([5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9],
                       [5, 6, 5, 6, 7, 6, 7, 8, 7, 8, 9, 8, 9]),
            "low": ([4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7,
                     8, 8, 8, 9, 9, 9, 10, 10],
                    [4, 5, 4, 5, 6, 5, 6, 7, 6, 7, 8,
                     7, 8, 9, 8, 9, 10, 9, 10])
        }
        expected_sses = {1: {(6, 6): {0, 3, 6}, (7, 7): {1, 4, 7},
                             (8, 8): {2, 5, 8}}}
        self._test_integration_subtest(spiketrains,
                                       spiketrains_y=spiketrains_y,
                                       indices_pmat=indices_pmat,
                                       index_proba=index_proba,
                                       expected_sses=expected_sses)


def suite():
    suite = unittest.makeSuite(AssetTestCase, 'test')
    return suite


def run():
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())


if __name__ == "__main__":
    unittest.main()
