# -*- coding: utf-8 -*-
"""
Unit tests for the spike_train_correlation module.

:copyright: Copyright 2015 by the Elephant team, see AUTHORS.txt.
:license: Modified BSD, see LICENSE.txt for details.
"""

import unittest

import numpy as np
from numpy.testing.utils import assert_array_equal
from numpy.testing.utils import assert_array_almost_equal
import quantities as pq
import neo
import elephant.conversion as conv
import elephant.spike_train_generation as sgen
import elephant.spike_train_correlation as sc

class crosscorrelogram_TestCase(unittest.TestCase):
    def setUp(self):
        self.test_array_1d_0 = [
            1.3, 7.56, 15.87, 28.23, 30.9, 34.2, 38.2, 43.2]
        self.test_array_1d_1 = [
            1.02, 2.71, 18.82, 28.46, 28.79, 43.6]

        # Build spike trains
        self.st_0 = neo.SpikeTrain(
            self.test_array_1d_0, units='ms', t_stop=50.)
        self.st_1 = neo.SpikeTrain(
            self.test_array_1d_1, units='ms', t_stop=50.)
        self.st_2 = sgen.homogeneous_poisson_process(100 * pq.Hz, t_stop = 1000. * pq.ms)

        # And binned counterparts
        self.binned_st_0 = conv.BinnedSpikeTrain(self.st_0, 
                t_start=0 * pq.ms, t_stop=50. * pq.ms,
                binsize=1 * pq.ms)
        self.binned_st_1 = conv.BinnedSpikeTrain(self.st_1, 
                t_start=0 * pq.ms, t_stop=50. * pq.ms,
                binsize=1 * pq.ms)
        self.binned_st_2 = conv.BinnedSpikeTrain(self.st_2, 
                binsize=1 * pq.ms)

    def test_autocorrelogram_binned_train(self):
        '''
        Test calculation of auto-correlogram
        '''

        st = self.binned_st_0
        lags, xcorr = sc.crosscorrelogram(st, st, [-5 * pq.ms, 5 * pq.ms],
                chance_corrected=False)

        # check dimensions
        self.assertEqual(len(lags), len(xcorr))
        self.assertEqual(len(xcorr), 11)

        # check normalisation
        self.assertEqual(xcorr.max(), xcorr[5])
        self.assertAlmostEqual(xcorr[5], 1., 2)

    def test_chance_coincidences(self):
        '''
        Test cross-correlogram correction for chance coincidences.
        '''

        st = self.binned_st_2
        lags, xcorr = sc.crosscorrelogram(st, st, [-5 * pq.ms, 5 * pq.ms],
                chance_corrected=True)

        # remove peak due to autocorrelation
        xcorr[5] = 0.
        assert_array_almost_equal(xcorr, 0., 1)


    def test_check_inputs(self):
        '''
        Test if fails on malformed arguments
        '''





class covariance_TestCase(unittest.TestCase):

    def setUp(self):
        # These two arrays must be such that they do not have coincidences
        # spanning across two neighbor bins assuming ms bins [0,1),[1,2),...
        self.test_array_1d_0 = [
            1.3, 7.56, 15.87, 28.23, 30.9, 34.2, 38.2, 43.2]
        self.test_array_1d_1 = [
            1.02, 2.71, 18.82, 28.46, 28.79, 43.6]

        # Build spike trains
        self.st_0 = neo.SpikeTrain(
            self.test_array_1d_0, units='ms', t_stop=50.)
        self.st_1 = neo.SpikeTrain(
            self.test_array_1d_1, units='ms', t_stop=50.)

        # And binned counterparts
        self.binned_st = conv.BinnedSpikeTrain(
            [self.st_0, self.st_1], t_start=0 * pq.ms, t_stop=50. * pq.ms,
            binsize=1 * pq.ms)

    def test_covariance_binned(self):
        '''
        Test covariance between two binned spike trains.
        '''

        # Calculate clipped and unclipped
        res_clipped = sc.covariance(
            self.binned_st, binary=True)
        res_unclipped = sc.covariance(
            self.binned_st, binary=False)

        # Check dimensions
        self.assertEqual(len(res_clipped), 2)
        self.assertEqual(len(res_unclipped), 2)

        # Check result unclipped against result calculated from scratch for
        # the off-diagonal element
        mat = self.binned_st.to_array()
        mean_0 = np.mean(mat[0])
        mean_1 = np.mean(mat[1])
        target_from_scratch = \
            np.dot(mat[0] - mean_0, mat[1] - mean_1) / (len(mat[0]) - 1)

        # Check result unclipped against result calculated by numpy.corrcoef
        target_numpy = np.cov(mat)

        self.assertAlmostEqual(target_from_scratch, target_numpy[0][1])
        self.assertAlmostEqual(res_unclipped[0][1], target_from_scratch)
        self.assertAlmostEqual(res_unclipped[1][0], target_from_scratch)

        # Check result clipped against result calculated from scratch for
        # the off-diagonal elemant
        mat = self.binned_st.to_bool_array()
        mean_0 = np.mean(mat[0])
        mean_1 = np.mean(mat[1])
        target_from_scratch = \
            np.dot(mat[0] - mean_0, mat[1] - mean_1) / (len(mat[0]) - 1)

        # Check result unclipped against result calculated by numpy.corrcoef
        target_numpy = np.cov(mat)

        self.assertAlmostEqual(target_from_scratch, target_numpy[0][1])
        self.assertAlmostEqual(res_clipped[0][1], target_from_scratch)
        self.assertAlmostEqual(res_clipped[1][0], target_from_scratch)

    def test_covariance_binned_same_spiketrains(self):
        '''
        Test if the covariation between two identical binned spike
        trains evaluates to the expected 2x2 matrix.
        '''
        # Calculate correlation
        binned_st = conv.BinnedSpikeTrain(
            [self.st_0, self.st_0], t_start=0 * pq.ms, t_stop=50. * pq.ms,
            binsize=1 * pq.ms)
        target = sc.covariance(binned_st)

        # Check dimensions
        self.assertEqual(len(target), 2)
        # Check result
        assert_array_equal(target[0][0], target[1][1])

    def test_covariance_binned_short_input(self):
        '''
        Test if input list of only one binned spike train yields correct result
        that matches numpy.cov (covariance with itself)
        '''
        # Calculate correlation
        binned_st = conv.BinnedSpikeTrain(
            self.st_0, t_start=0 * pq.ms, t_stop=50. * pq.ms,
            binsize=1 * pq.ms)
        target = sc.covariance(binned_st)

        # Check result unclipped against result calculated by numpy.corrcoef
        mat = binned_st.to_bool_array()
        target_numpy = np.cov(mat)

        # Check result and dimensionality of result
        self.assertEqual(target.ndim, target_numpy.ndim)
        self.assertAlmostEqual(target, target_numpy)


class corrcoeff_TestCase(unittest.TestCase):

    def setUp(self):
        # These two arrays must be such that they do not have coincidences
        # spanning across two neighbor bins assuming ms bins [0,1),[1,2),...
        self.test_array_1d_0 = [
            1.3, 7.56, 15.87, 28.23, 30.9, 34.2, 38.2, 43.2]
        self.test_array_1d_1 = [
            1.02, 2.71, 18.82, 28.46, 28.79, 43.6]

        # Build spike trains
        self.st_0 = neo.SpikeTrain(
            self.test_array_1d_0, units='ms', t_stop=50.)
        self.st_1 = neo.SpikeTrain(
            self.test_array_1d_1, units='ms', t_stop=50.)

        # And binned counterparts
        self.binned_st = conv.BinnedSpikeTrain(
            [self.st_0, self.st_1], t_start=0 * pq.ms, t_stop=50. * pq.ms,
            binsize=1 * pq.ms)


    def test_corrcoef_binned(self):
        '''
        Test the correlation coefficient between two binned spike trains.
        '''

        # Calculate clipped and unclipped
        res_clipped = sc.corrcoef(
            self.binned_st, binary=True)
        res_unclipped = sc.corrcoef(
            self.binned_st, binary=False)

        # Check dimensions
        self.assertEqual(len(res_clipped), 2)
        self.assertEqual(len(res_unclipped), 2)

        # Check result unclipped against result calculated from scratch for
        # the off-diagonal element
        mat = self.binned_st.to_array()
        mean_0 = np.mean(mat[0])
        mean_1 = np.mean(mat[1])
        target_from_scratch = \
            np.dot(mat[0] - mean_0, mat[1] - mean_1) / \
            np.sqrt(
                np.dot(mat[0] - mean_0, mat[0] - mean_0) *
                np.dot(mat[1] - mean_1, mat[1] - mean_1))

        # Check result unclipped against result calculated by numpy.corrcoef
        target_numpy = np.corrcoef(mat)

        self.assertAlmostEqual(target_from_scratch, target_numpy[0][1])
        self.assertAlmostEqual(res_unclipped[0][1], target_from_scratch)
        self.assertAlmostEqual(res_unclipped[1][0], target_from_scratch)

        # Check result clipped against result calculated from scratch for
        # the off-diagonal elemant
        mat = self.binned_st.to_bool_array()
        mean_0 = np.mean(mat[0])
        mean_1 = np.mean(mat[1])
        target_from_scratch = \
            np.dot(mat[0] - mean_0, mat[1] - mean_1) / \
            np.sqrt(
                np.dot(mat[0] - mean_0, mat[0] - mean_0) *
                np.dot(mat[1] - mean_1, mat[1] - mean_1))

        # Check result unclipped against result calculated by numpy.corrcoef
        target_numpy = np.corrcoef(mat)

        self.assertAlmostEqual(target_from_scratch, target_numpy[0][1])
        self.assertAlmostEqual(res_clipped[0][1], target_from_scratch)
        self.assertAlmostEqual(res_clipped[1][0], target_from_scratch)

    def test_corrcoef_binned_same_spiketrains(self):
        '''
        Test if the correlation coefficient between two identical binned spike
        trains evaluates to a 2x2 matrix of ones.
        '''
        # Calculate correlation
        binned_st = conv.BinnedSpikeTrain(
            [self.st_0, self.st_0], t_start=0 * pq.ms, t_stop=50. * pq.ms,
            binsize=1 * pq.ms)
        target = sc.corrcoef(binned_st)

        # Check dimensions
        self.assertEqual(len(target), 2)
        # Check result
        assert_array_equal(target, 1.)

    def test_corrcoef_binned_short_input(self):
        '''
        Test if input list of one binned spike train yields 1.0.
        '''
        # Calculate correlation
        binned_st = conv.BinnedSpikeTrain(
            self.st_0, t_start=0 * pq.ms, t_stop=50. * pq.ms,
            binsize=1 * pq.ms)
        target = sc.corrcoef(binned_st)

        # Check result and dimensionality of result
        self.assertEqual(target.ndim, 0)
        self.assertEqual(target, 1.)


if __name__ == '__main__':
    unittest.main()
