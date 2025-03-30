# -*- coding: utf-8 -*-
"""
Unit tests for the spike_train_correlation module.

:copyright: Copyright 2014-2024 by the Elephant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
"""

import math
import unittest

import neo
from neo.io import NixIO
import numpy as np
import quantities as pq
from numpy.testing import assert_array_equal, assert_array_almost_equal

import elephant.conversion as conv
import elephant.spike_train_correlation as sc
from elephant.spike_train_generation import (
    StationaryPoissonProcess,
    StationaryGammaProcess,
)
import math
from elephant.datasets import download_datasets
from elephant.spike_train_generation import (
    homogeneous_poisson_process,
    homogeneous_gamma_process,
)


class CovarianceTestCase(unittest.TestCase):
    def setUp(self):
        # These two arrays must be such that they do not have coincidences
        # spanning across two neighbor bins assuming ms bins [0,1),[1,2),...
        self.test_array_1d_0 = [1.3, 7.56, 15.87, 28.23, 30.9, 34.2, 38.2, 43.2]
        self.test_array_1d_1 = [1.02, 2.71, 18.82, 28.46, 28.79, 43.6]

        # Build spike trains
        self.st_0 = neo.SpikeTrain(self.test_array_1d_0, units="ms", t_stop=50.0)
        self.st_1 = neo.SpikeTrain(self.test_array_1d_1, units="ms", t_stop=50.0)

        # And binned counterparts
        self.binned_st = conv.BinnedSpikeTrain(
            [self.st_0, self.st_1],
            t_start=0 * pq.ms,
            t_stop=50.0 * pq.ms,
            bin_size=1 * pq.ms,
        )

    def test_covariance_binned(self):
        """
        Test covariance between two binned spike trains.
        """

        # Calculate clipped and unclipped
        res_clipped = sc.covariance(self.binned_st, binary=True, fast=False)
        res_unclipped = sc.covariance(self.binned_st, binary=False, fast=False)

        # Check dimensions
        self.assertEqual(len(res_clipped), 2)
        self.assertEqual(len(res_unclipped), 2)

        # Check result unclipped against result calculated from scratch for
        # the off-diagonal element
        mat = self.binned_st.to_array()
        mean_0 = np.mean(mat[0])
        mean_1 = np.mean(mat[1])
        target_from_scratch = np.dot(mat[0] - mean_0, mat[1] - mean_1) / (
            len(mat[0]) - 1
        )

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
        target_from_scratch = np.dot(mat[0] - mean_0, mat[1] - mean_1) / (
            len(mat[0]) - 1
        )

        # Check result unclipped against result calculated by numpy.corrcoef
        target_numpy = np.cov(mat)

        self.assertAlmostEqual(target_from_scratch, target_numpy[0][1])
        self.assertAlmostEqual(res_clipped[0][1], target_from_scratch)
        self.assertAlmostEqual(res_clipped[1][0], target_from_scratch)

    def test_covariance_binned_same_spiketrains(self):
        """
        Test if the covariation between two identical binned spike
        trains evaluates to the expected 2x2 matrix.
        """
        # Calculate correlation
        binned_st = conv.BinnedSpikeTrain(
            [self.st_0, self.st_0],
            t_start=0 * pq.ms,
            t_stop=50.0 * pq.ms,
            bin_size=1 * pq.ms,
        )
        result = sc.covariance(binned_st, fast=False)

        # Check dimensions
        self.assertEqual(len(result), 2)
        # Check result
        assert_array_equal(result[0][0], result[1][1])

    def test_covariance_binned_short_input(self):
        """
        Test if input list of only one binned spike train yields correct result
        that matches numpy.cov (covariance with itself)
        """
        # Calculate correlation
        binned_st = conv.BinnedSpikeTrain(
            self.st_0, t_start=0 * pq.ms, t_stop=50.0 * pq.ms, bin_size=1 * pq.ms
        )
        result = sc.covariance(binned_st, binary=True, fast=False)

        # Check result unclipped against result calculated by numpy.corrcoef
        mat = binned_st.to_bool_array()
        target = np.cov(mat)

        # Check result and dimensionality of result
        self.assertEqual(result.ndim, target.ndim)
        assert_array_almost_equal(result, target)
        assert_array_almost_equal(
            target, sc.covariance(binned_st, binary=True, fast=True)
        )

    def test_covariance_fast_mode(self):
        np.random.seed(27)
        st = StationaryPoissonProcess(
            rate=10 * pq.Hz, t_stop=10 * pq.s
        ).generate_spiketrain()
        binned_st = conv.BinnedSpikeTrain(st, n_bins=10)
        assert_array_almost_equal(
            sc.covariance(binned_st, fast=False), sc.covariance(binned_st, fast=True)
        )


class CorrCoefTestCase(unittest.TestCase):
    def setUp(self):
        # These two arrays must be such that they do not have coincidences
        # spanning across two neighbor bins assuming ms bins [0,1),[1,2),...
        self.test_array_1d_0 = [1.3, 7.56, 15.87, 28.23, 30.9, 34.2, 38.2, 43.2]
        self.test_array_1d_1 = [1.02, 2.71, 18.82, 28.46, 28.79, 43.6]
        self.test_array_1d_2 = []

        # Build spike trains
        self.st_0 = neo.SpikeTrain(self.test_array_1d_0, units="ms", t_stop=50.0)
        self.st_1 = neo.SpikeTrain(self.test_array_1d_1, units="ms", t_stop=50.0)
        self.st_2 = neo.SpikeTrain(self.test_array_1d_2, units="ms", t_stop=50.0)

        # And binned counterparts
        self.binned_st = conv.BinnedSpikeTrain(
            [self.st_0, self.st_1],
            t_start=0 * pq.ms,
            t_stop=50.0 * pq.ms,
            bin_size=1 * pq.ms,
        )

    def test_corrcoef_binned(self):
        """
        Test the correlation coefficient between two binned spike trains.
        """

        # Calculate clipped and unclipped
        res_clipped = sc.correlation_coefficient(self.binned_st, binary=True)
        res_unclipped = sc.correlation_coefficient(self.binned_st, binary=False)

        # Check dimensions
        self.assertEqual(len(res_clipped), 2)
        self.assertEqual(len(res_unclipped), 2)

        # Check result unclipped against result calculated from scratch for
        # the off-diagonal element
        mat = self.binned_st.to_array()
        mean_0 = np.mean(mat[0])
        mean_1 = np.mean(mat[1])
        target_from_scratch = np.dot(mat[0] - mean_0, mat[1] - mean_1) / np.sqrt(
            np.dot(mat[0] - mean_0, mat[0] - mean_0)
            * np.dot(mat[1] - mean_1, mat[1] - mean_1)
        )

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
        target_from_scratch = np.dot(mat[0] - mean_0, mat[1] - mean_1) / np.sqrt(
            np.dot(mat[0] - mean_0, mat[0] - mean_0)
            * np.dot(mat[1] - mean_1, mat[1] - mean_1)
        )

        # Check result unclipped against result calculated by numpy.corrcoef
        target_numpy = np.corrcoef(mat)

        self.assertAlmostEqual(target_from_scratch, target_numpy[0][1])
        self.assertAlmostEqual(res_clipped[0][1], target_from_scratch)
        self.assertAlmostEqual(res_clipped[1][0], target_from_scratch)

    def test_corrcoef_binned_same_spiketrains(self):
        """
        Test if the correlation coefficient between two identical binned spike
        trains evaluates to a 2x2 matrix of ones.
        """
        # Calculate correlation
        binned_st = conv.BinnedSpikeTrain(
            [self.st_0, self.st_0],
            t_start=0 * pq.ms,
            t_stop=50.0 * pq.ms,
            bin_size=1 * pq.ms,
        )
        result = sc.correlation_coefficient(binned_st, fast=False)
        target = np.ones((2, 2))

        # Check dimensions
        self.assertEqual(len(result), 2)
        # Check result
        assert_array_almost_equal(result, target)
        assert_array_almost_equal(
            result, sc.correlation_coefficient(binned_st, fast=True)
        )

    def test_corrcoef_binned_short_input(self):
        """
        Test if input list of one binned spike train yields 1.0.
        """
        # Calculate correlation
        binned_st = conv.BinnedSpikeTrain(
            self.st_0, t_start=0 * pq.ms, t_stop=50.0 * pq.ms, bin_size=1 * pq.ms
        )
        result = sc.correlation_coefficient(binned_st, fast=False)
        target = np.array(1.0)

        # Check result and dimensionality of result
        self.assertEqual(result.ndim, 0)
        assert_array_almost_equal(result, target)
        assert_array_almost_equal(
            result, sc.correlation_coefficient(binned_st, fast=True)
        )

    def test_empty_spike_train(self):
        """
        Test whether a warning is yielded in case of empty spike train.
        Also check correctness of the output array.
        """
        # st_2 is empty
        binned_12 = conv.BinnedSpikeTrain([self.st_1, self.st_2], bin_size=1 * pq.ms)

        with self.assertWarns(UserWarning):
            result = sc.correlation_coefficient(binned_12, fast=False)

        # test for NaNs in the output array
        target = np.zeros((2, 2)) * np.NaN
        target[0, 0] = 1.0
        assert_array_almost_equal(result, target)

    def test_corrcoef_fast_mode(self):
        np.random.seed(27)
        st = StationaryPoissonProcess(
            rate=10 * pq.Hz, t_stop=10 * pq.s
        ).generate_spiketrain()
        binned_st = conv.BinnedSpikeTrain(st, n_bins=10)
        assert_array_almost_equal(
            sc.correlation_coefficient(binned_st, fast=False),
            sc.correlation_coefficient(binned_st, fast=True),
        )


class CrossCorrelationHistogramTest(unittest.TestCase):
    def setUp(self):
        # These two arrays must be such that they do not have coincidences
        # spanning across two neighbor bins assuming ms bins [0,1),[1,2),...
        self.test_array_1d_1 = [1.3, 7.56, 15.87, 28.23, 30.9, 34.2, 38.2, 43.2]
        self.test_array_1d_2 = [1.02, 2.71, 18.82, 28.46, 28.79, 43.6]

        # Build spike trains
        self.st_1 = neo.SpikeTrain(self.test_array_1d_1, units="ms", t_stop=50.0)
        self.st_2 = neo.SpikeTrain(self.test_array_1d_2, units="ms", t_stop=50.0)

        # And binned counterparts
        self.binned_st1 = conv.BinnedSpikeTrain(
            [self.st_1], t_start=0 * pq.ms, t_stop=50.0 * pq.ms, bin_size=1 * pq.ms
        )
        self.binned_st2 = conv.BinnedSpikeTrain(
            [self.st_2], t_start=0 * pq.ms, t_stop=50.0 * pq.ms, bin_size=1 * pq.ms
        )
        self.binned_sts = conv.BinnedSpikeTrain(
            [self.st_1, self.st_2],
            t_start=0 * pq.ms,
            t_stop=50.0 * pq.ms,
            bin_size=1 * pq.ms,
        )

        # Binned sts to check errors raising
        self.st_check_bin_size = conv.BinnedSpikeTrain(
            [self.st_1], t_start=0 * pq.ms, t_stop=50.0 * pq.ms, bin_size=5 * pq.ms
        )
        self.st_check_t_start = conv.BinnedSpikeTrain(
            [self.st_1], t_start=1 * pq.ms, t_stop=50.0 * pq.ms, bin_size=1 * pq.ms
        )
        self.st_check_t_stop = conv.BinnedSpikeTrain(
            [self.st_1], t_start=0 * pq.ms, t_stop=40.0 * pq.ms, bin_size=1 * pq.ms
        )
        self.st_check_dimension = conv.BinnedSpikeTrain(
            [self.st_1, self.st_2],
            t_start=0 * pq.ms,
            t_stop=50.0 * pq.ms,
            bin_size=1 * pq.ms,
        )

    def test_cross_correlation_histogram(self):
        """
        Test generic result of a cross-correlation histogram between two binned
        spike trains.
        """
        # Calculate CCH using Elephant (normal and binary version) with
        # mode equal to 'full' (whole spike trains are correlated)
        cch_clipped, bin_ids_clipped = sc.cross_correlation_histogram(
            self.binned_st1, self.binned_st2, window="full", binary=True
        )
        cch_unclipped, bin_ids_unclipped = sc.cross_correlation_histogram(
            self.binned_st1, self.binned_st2, window="full", binary=False
        )

        cch_clipped_mem, bin_ids_clipped_mem = sc.cross_correlation_histogram(
            self.binned_st1,
            self.binned_st2,
            window="full",
            binary=True,
            method="memory",
        )
        cch_unclipped_mem, bin_ids_unclipped_mem = sc.cross_correlation_histogram(
            self.binned_st1,
            self.binned_st2,
            window="full",
            binary=False,
            method="memory",
        )
        # Check consistency two methods
        assert_array_equal(
            np.squeeze(cch_clipped.magnitude), np.squeeze(cch_clipped_mem.magnitude)
        )
        assert_array_equal(
            np.squeeze(cch_clipped.times), np.squeeze(cch_clipped_mem.times)
        )
        assert_array_equal(
            np.squeeze(cch_unclipped.magnitude), np.squeeze(cch_unclipped_mem.magnitude)
        )
        assert_array_equal(
            np.squeeze(cch_unclipped.times), np.squeeze(cch_unclipped_mem.times)
        )
        assert_array_almost_equal(bin_ids_clipped, bin_ids_clipped_mem)
        assert_array_almost_equal(bin_ids_unclipped, bin_ids_unclipped_mem)

        # Check normal correlation Note: Use numpy correlate to verify result.
        # Note: numpy conventions for input array 1 and input array 2 are
        # swapped compared to Elephant!
        mat1 = self.binned_st1.to_array()[0]
        mat2 = self.binned_st2.to_array()[0]
        target_numpy = np.correlate(mat2, mat1, mode="full")
        assert_array_equal(target_numpy, np.squeeze(cch_unclipped.magnitude))

        # Check cross correlation function for several displacements tau
        # Note: Use Elephant corrcoeff to verify result
        tau = [-25.0, 0.0, 13.0]  # in ms
        for t in tau:
            # adjust t_start, t_stop to shift by tau
            t0 = np.min([self.st_1.t_start + t * pq.ms, self.st_2.t_start])
            t1 = np.max([self.st_1.t_stop + t * pq.ms, self.st_2.t_stop])
            st1 = neo.SpikeTrain(
                self.st_1.magnitude + t,
                units="ms",
                t_start=t0 * pq.ms,
                t_stop=t1 * pq.ms,
            )
            st2 = neo.SpikeTrain(
                self.st_2.magnitude, units="ms", t_start=t0 * pq.ms, t_stop=t1 * pq.ms
            )
            binned_sts = conv.BinnedSpikeTrain(
                [st1, st2], bin_size=1 * pq.ms, t_start=t0 * pq.ms, t_stop=t1 * pq.ms
            )
            # caluclate corrcoef
            corrcoef = sc.correlation_coefficient(binned_sts)[1, 0]

            # expand t_stop to have two spike trains with same length as st1,
            # st2
            st1 = neo.SpikeTrain(
                self.st_1.magnitude,
                units="ms",
                t_start=self.st_1.t_start,
                t_stop=self.st_1.t_stop + np.abs(t) * pq.ms,
            )
            st2 = neo.SpikeTrain(
                self.st_2.magnitude,
                units="ms",
                t_start=self.st_2.t_start,
                t_stop=self.st_2.t_stop + np.abs(t) * pq.ms,
            )
            binned_st1 = conv.BinnedSpikeTrain(
                st1,
                t_start=0 * pq.ms,
                t_stop=(50 + np.abs(t)) * pq.ms,
                bin_size=1 * pq.ms,
            )
            binned_st2 = conv.BinnedSpikeTrain(
                st2,
                t_start=0 * pq.ms,
                t_stop=(50 + np.abs(t)) * pq.ms,
                bin_size=1 * pq.ms,
            )
            # calculate CCHcoef and take value at t=tau
            CCHcoef, _ = sc.cch(
                binned_st1, binned_st2, cross_correlation_coefficient=True
            )
            left_edge = -binned_st1.n_bins + 1
            tau_bin = int(t / float(binned_st1.bin_size.magnitude))
            assert_array_almost_equal(corrcoef, CCHcoef[tau_bin - left_edge].magnitude)

        # Check correlation using binary spike trains
        mat1 = np.array(self.binned_st1.to_bool_array()[0], dtype=int)
        mat2 = np.array(self.binned_st2.to_bool_array()[0], dtype=int)
        target_numpy = np.correlate(mat2, mat1, mode="full")
        assert_array_equal(target_numpy, np.squeeze(cch_clipped.magnitude))

        # Check the time axis and bin IDs of the resulting AnalogSignal
        assert_array_almost_equal(
            (bin_ids_clipped - 0.5) * self.binned_st1.bin_size, cch_unclipped.times
        )
        assert_array_almost_equal(
            (bin_ids_clipped - 0.5) * self.binned_st1.bin_size, cch_clipped.times
        )

        # Calculate CCH using Elephant (normal and binary version) with
        # mode equal to 'valid' (only completely overlapping intervals of the
        # spike trains are correlated)
        cch_clipped, bin_ids_clipped = sc.cross_correlation_histogram(
            self.binned_st1, self.binned_st2, window="valid", binary=True
        )
        cch_unclipped, bin_ids_unclipped = sc.cross_correlation_histogram(
            self.binned_st1, self.binned_st2, window="valid", binary=False
        )
        cch_clipped_mem, bin_ids_clipped_mem = sc.cross_correlation_histogram(
            self.binned_st1,
            self.binned_st2,
            window="valid",
            binary=True,
            method="memory",
        )
        cch_unclipped_mem, bin_ids_unclipped_mem = sc.cross_correlation_histogram(
            self.binned_st1,
            self.binned_st2,
            window="valid",
            binary=False,
            method="memory",
        )

        # Check consistency two methods
        assert_array_equal(
            np.squeeze(cch_clipped.magnitude), np.squeeze(cch_clipped_mem.magnitude)
        )
        assert_array_equal(
            np.squeeze(cch_clipped.times), np.squeeze(cch_clipped_mem.times)
        )
        assert_array_equal(
            np.squeeze(cch_unclipped.magnitude), np.squeeze(cch_unclipped_mem.magnitude)
        )
        assert_array_equal(
            np.squeeze(cch_unclipped.times), np.squeeze(cch_unclipped_mem.times)
        )
        assert_array_equal(bin_ids_clipped, bin_ids_clipped_mem)
        assert_array_equal(bin_ids_unclipped, bin_ids_unclipped_mem)

        # Check normal correlation Note: Use numpy correlate to verify result.
        # Note: numpy conventions for input array 1 and input array 2 are
        # swapped compared to Elephant!
        mat1 = self.binned_st1.to_array()[0]
        mat2 = self.binned_st2.to_array()[0]
        target_numpy = np.correlate(mat2, mat1, mode="valid")
        assert_array_equal(target_numpy, np.squeeze(cch_unclipped.magnitude))

        # Check correlation using binary spike trains
        mat1 = np.array(self.binned_st1.to_bool_array()[0], dtype=int)
        mat2 = np.array(self.binned_st2.to_bool_array()[0], dtype=int)
        target_numpy = np.correlate(mat2, mat1, mode="valid")
        assert_array_equal(target_numpy, np.squeeze(cch_clipped.magnitude))

        # Check the time axis and bin IDs of the resulting AnalogSignal
        assert_array_equal(
            (bin_ids_clipped - 0.5) * self.binned_st1.bin_size, cch_unclipped.times
        )
        assert_array_equal(
            (bin_ids_clipped - 0.5) * self.binned_st1.bin_size, cch_clipped.times
        )

        # Check for wrong window parameter setting
        self.assertRaises(
            ValueError,
            sc.cross_correlation_histogram,
            self.binned_st1,
            self.binned_st2,
            window="dsaij",
        )
        self.assertRaises(
            ValueError,
            sc.cross_correlation_histogram,
            self.binned_st1,
            self.binned_st2,
            window="dsaij",
            method="memory",
        )

    def test_raising_error_wrong_inputs(self):
        """Check that an exception is thrown if the two spike trains are not
        fullfilling the requirement of the function"""
        # Check the bin_sizes are the same
        self.assertRaises(
            ValueError,
            sc.cross_correlation_histogram,
            self.binned_st1,
            self.st_check_bin_size,
        )
        # Check input are one dimensional
        self.assertRaises(
            ValueError,
            sc.cross_correlation_histogram,
            self.st_check_dimension,
            self.binned_st2,
        )
        self.assertRaises(
            ValueError,
            sc.cross_correlation_histogram,
            self.binned_st2,
            self.st_check_dimension,
        )

    def test_window(self):
        """Test if the window parameter is correctly interpreted."""
        cch_win, bin_ids = sc.cch(self.binned_st1, self.binned_st2, window=[-30, 30])
        cch_win_mem, bin_ids_mem = sc.cch(
            self.binned_st1, self.binned_st2, window=[-30, 30], method="memory"
        )

        self.assertEqual(len(bin_ids), cch_win.shape[0])
        assert_array_equal(bin_ids, np.arange(-30, 31, 1))
        assert_array_equal((bin_ids - 0.5) * self.binned_st1.bin_size, cch_win.times)

        assert_array_equal(bin_ids_mem, np.arange(-30, 31, 1))
        assert_array_equal(
            (bin_ids_mem - 0.5) * self.binned_st1.bin_size, cch_win.times
        )

        assert_array_equal(cch_win, cch_win_mem)
        cch_unclipped, _ = sc.cross_correlation_histogram(
            self.binned_st1, self.binned_st2, window="full", binary=False
        )
        assert_array_equal(cch_win, cch_unclipped[19:80])

        _, bin_ids = sc.cch(self.binned_st1, self.binned_st2, window=[20, 30])
        _, bin_ids_mem = sc.cch(
            self.binned_st1, self.binned_st2, window=[20, 30], method="memory"
        )

        assert_array_equal(bin_ids, np.arange(20, 31, 1))
        assert_array_equal(bin_ids_mem, np.arange(20, 31, 1))

        _, bin_ids = sc.cch(self.binned_st1, self.binned_st2, window=[-30, -20])

        _, bin_ids_mem = sc.cch(
            self.binned_st1, self.binned_st2, window=[-30, -20], method="memory"
        )

        assert_array_equal(bin_ids, np.arange(-30, -19, 1))
        assert_array_equal(bin_ids_mem, np.arange(-30, -19, 1))

        # Check for wrong assignments to the window parameter
        # Test for window longer than the total length of the spike trains
        self.assertRaises(
            ValueError,
            sc.cross_correlation_histogram,
            self.binned_st1,
            self.binned_st2,
            window=[-60, 50],
        )
        self.assertRaises(
            ValueError,
            sc.cross_correlation_histogram,
            self.binned_st1,
            self.binned_st2,
            window=[-50, 60],
        )
        # Test for no integer or wrong string in input
        self.assertRaises(
            ValueError,
            sc.cross_correlation_histogram,
            self.binned_st1,
            self.binned_st2,
            window=[-25.5, 25.5],
        )
        self.assertRaises(
            ValueError,
            sc.cross_correlation_histogram,
            self.binned_st1,
            self.binned_st2,
            window="test",
        )

    def test_border_correction(self):
        """Test if the border correction for bins at the edges is correctly
        performed"""

        # check that nothing changes for valid lags
        cch_valid, _ = sc.cross_correlation_histogram(
            self.binned_st1,
            self.binned_st2,
            window="full",
            border_correction=True,
            binary=False,
            kernel=None,
        )
        valid_lags = sc._CrossCorrHist.get_valid_lags(self.binned_st1, self.binned_st2)
        left_edge, right_edge = valid_lags[(0, -1),]
        cch_builder = sc._CrossCorrHist(
            self.binned_st1, self.binned_st2, window=(left_edge, right_edge)
        )
        cch_valid = cch_builder.correlate_speed(cch_mode="valid")
        cch_corrected = cch_builder.border_correction(cch_valid)

        np.testing.assert_array_equal(cch_valid, cch_corrected)

        # test the border correction for lags without full overlap
        cch_full, lags_full = sc.cross_correlation_histogram(
            self.binned_st1, self.binned_st2, window="full"
        )

        cch_full_corrected, _ = sc.cross_correlation_histogram(
            self.binned_st1, self.binned_st2, window="full", border_correction=True
        )

        n_bins_outside_window = np.min(
            np.abs(np.subtract.outer(lags_full, valid_lags)), axis=1
        )

        min_n_bins = min(self.binned_st1.n_bins, self.binned_st2.n_bins)

        border_correction = (cch_full_corrected / cch_full).magnitude.flatten()

        # exclude NaNs caused by zeros in the cch
        mask = np.logical_not(np.isnan(border_correction))

        np.testing.assert_array_almost_equal(
            border_correction[mask],
            (float(min_n_bins) / (min_n_bins - n_bins_outside_window))[mask],
        )

    def test_kernel(self):
        """Test if the smoothing kernel is correctly defined, and wheter it is
        applied properly."""
        smoothed_cch, _ = sc.cross_correlation_histogram(
            self.binned_st1, self.binned_st2, kernel=np.ones(3)
        )
        smoothed_cch_mem, _ = sc.cross_correlation_histogram(
            self.binned_st1, self.binned_st2, kernel=np.ones(3), method="memory"
        )

        cch, _ = sc.cross_correlation_histogram(
            self.binned_st1, self.binned_st2, kernel=None
        )
        cch_mem, _ = sc.cross_correlation_histogram(
            self.binned_st1, self.binned_st2, kernel=None, method="memory"
        )

        self.assertNotEqual(smoothed_cch.all, cch.all)
        self.assertNotEqual(smoothed_cch_mem.all, cch_mem.all)

        self.assertRaises(
            ValueError, sc.cch, self.binned_st1, self.binned_st2, kernel=np.ones(100)
        )
        self.assertRaises(
            ValueError,
            sc.cch,
            self.binned_st1,
            self.binned_st2,
            kernel=np.ones(100),
            method="memory",
        )

    def test_exist_alias(self):
        """
        Test if alias cch still exists.
        """
        self.assertEqual(sc.cross_correlation_histogram, sc.cch)

    def test_annotations(self):
        cch, _ = sc.cross_correlation_histogram(
            self.binned_st1, self.binned_st2, kernel=np.ones(3)
        )
        target_dict = dict(
            window="full",
            border_correction=False,
            binary=False,
            kernel=True,
            normalization="counts",
        )
        self.assertIn("cch_parameters", cch.annotations)
        self.assertEqual(cch.annotations["cch_parameters"], target_dict)


class CrossCorrelationHistDifferentTStartTStopTest(unittest.TestCase):
    def _run_sub_tests(self, st1, st2, lags_true):
        for window in ("valid", "full"):
            for method in ("speed", "memory"):
                with self.subTest(window=window, method=method):
                    bin_size = 1 * pq.s
                    st1_binned = conv.BinnedSpikeTrain(st1, bin_size=bin_size)
                    st2_binned = conv.BinnedSpikeTrain(st2, bin_size=bin_size)
                    left, right = lags_true[window][(0, -1),]
                    cch_window, lags_window = sc.cross_correlation_histogram(
                        st1_binned,
                        st2_binned,
                        window=(left, right),
                        method=method,
                    )
                    cch, lags = sc.cross_correlation_histogram(
                        st1_binned, st2_binned, window=window
                    )

                    # target cross correlation
                    cch_target = np.correlate(
                        st1_binned.to_array()[0], st2_binned.to_array()[0], mode=window
                    )

                    self.assertEqual(len(lags_window), cch_window.shape[0])
                    assert_array_almost_equal(cch.magnitude, cch_window.magnitude)
                    # the output is reversed since we cross-correlate
                    # st2 with st1 rather than st1 with st2 (numpy behavior)
                    assert_array_almost_equal(np.ravel(cch.magnitude), cch_target[::-1])
                    assert_array_equal(lags, lags_true[window])
                    assert_array_equal(lags, lags_window)

    def test_cross_correlation_histogram_valid_full_overlap(self):
        # ex. 1 in the source code
        st1 = neo.SpikeTrain([3.5, 4.5, 7.5] * pq.s, t_start=3 * pq.s, t_stop=8 * pq.s)
        st2 = neo.SpikeTrain(
            [1.5, 2.5, 4.5, 8.5, 9.5, 10.5] * pq.s, t_start=1 * pq.s, t_stop=13 * pq.s
        )
        lags_true = {
            "valid": np.arange(-2, 6, dtype=np.int32),
            "full": np.arange(-6, 10, dtype=np.int32),
        }
        self._run_sub_tests(st1, st2, lags_true)

    def test_cross_correlation_histogram_valid_partial_overlap(self):
        # ex. 2 in the source code
        st1 = neo.SpikeTrain(
            [2.5, 3.5, 4.5, 6.5] * pq.s, t_start=1 * pq.s, t_stop=7 * pq.s
        )
        st2 = neo.SpikeTrain(
            [3.5, 5.5, 6.5, 7.5, 8.5] * pq.s, t_start=2 * pq.s, t_stop=9 * pq.s
        )
        lags_true = {
            "valid": np.arange(1, 3, dtype=np.int32),
            "full": np.arange(-4, 8, dtype=np.int32),
        }
        self._run_sub_tests(st1, st2, lags_true)

    def test_cross_correlation_histogram_valid_no_overlap(self):
        st1 = neo.SpikeTrain(
            [2.5, 3.5, 4.5, 6.5] * pq.s, t_start=1 * pq.s, t_stop=7 * pq.s
        )
        st2 = neo.SpikeTrain(
            [3.5, 5.5, 6.5, 7.5, 8.5] * pq.s + 6 * pq.s,
            t_start=8 * pq.s,
            t_stop=15 * pq.s,
        )
        lags_true = {
            "valid": np.arange(7, 9, dtype=np.int32),
            "full": np.arange(2, 14, dtype=np.int32),
        }
        self._run_sub_tests(st1, st2, lags_true)

    def test_invalid_time_shift(self):
        # time shift of 0.4 s is not multiple of bin_size=1 s
        st1 = neo.SpikeTrain([2.5, 3.5] * pq.s, t_start=1 * pq.s, t_stop=7 * pq.s)
        st2 = neo.SpikeTrain([3.5, 5.5] * pq.s, t_start=1.4 * pq.s, t_stop=7.4 * pq.s)
        bin_size = 1 * pq.s
        st1_binned = conv.BinnedSpikeTrain(st1, bin_size=bin_size)
        st2_binned = conv.BinnedSpikeTrain(st2, bin_size=bin_size)
        self.assertRaises(
            ValueError, sc.cross_correlation_histogram, st1_binned, st2_binned
        )


class SpikeTimeTilingCoefficientTestCase(unittest.TestCase):
    def setUp(self):
        # These two arrays must be such that they do not have coincidences
        # spanning across two neighbor bins assuming ms bins [0,1),[1,2),...
        self.test_array_1d_1 = [1.3, 7.56, 15.87, 28.23, 30.9, 34.2, 38.2, 43.2]
        self.test_array_1d_2 = [1.02, 2.71, 18.82, 28.46, 28.79, 43.6]

        # Build spike trains
        self.st_1 = neo.SpikeTrain(self.test_array_1d_1, units="ms", t_stop=50.0)
        self.st_2 = neo.SpikeTrain(self.test_array_1d_2, units="ms", t_stop=50.0)

    def test_sttc_dt_smaller_zero(self):
        self.assertRaises(ValueError, sc.sttc, self.st_1, self.st_2, dt=0 * pq.s)
        self.assertRaises(ValueError, sc.sttc, self.st_1, self.st_2, dt=-1 * pq.ms)

    def test_sttc_different_t_stop(self):
        st_1 = neo.SpikeTrain([1], units="ms", t_stop=10.0)
        st_2 = neo.SpikeTrain([5], units="ms", t_stop=10.0)
        st_2.t_stop = 1 * pq.ms
        self.assertRaises(ValueError, sc.sttc, st_1, st_2)

    def test_sttc_different_t_start(self):
        st_1 = neo.SpikeTrain([1], units="ms", t_stop=10.0)
        st_2 = neo.SpikeTrain([5], units="ms", t_stop=10.0)
        st_2.t_start = 1 * pq.ms
        self.assertRaises(ValueError, sc.sttc, st_1, st_2)

    def test_sttc_different_units_dt(self):
        # test for result
        # target obtained with pencil and paper according to original paper.
        target = 0.495860165593
        self.assertAlmostEqual(target, sc.sttc(self.st_1, self.st_2, 0.005 * pq.s))

        # test for same result with dt given in ms
        self.assertAlmostEqual(target, sc.sttc(self.st_1, self.st_2, 5.0 * pq.ms))

    def test_sttc_different_units_spiketrains(self):
        st1 = neo.SpikeTrain([1], units="ms", t_stop=10.0)
        st2 = neo.SpikeTrain([5], units="s", t_stop=10.0)
        self.assertRaises(ValueError, sc.sttc, st1, st2)

    def test_sttc_not_enough_spiketrains(self):
        # test no spiketrains
        self.assertRaises(TypeError, sc.sttc, [], [])

        # test one spiketrain
        self.assertRaises(TypeError, sc.sttc, self.st_1, [])

    def test_sttc_one_spike(self):
        # test for one spike in a spiketrain
        st1 = neo.SpikeTrain([1], units="ms", t_stop=10.0)
        st2 = neo.SpikeTrain([5], units="ms", t_stop=10.0)
        self.assertEqual(sc.sttc(st1, st2), 1.0)
        self.assertTrue(bool(sc.sttc(st1, st2, 0.1 * pq.ms) < 0))

    def test_sttc_high_value_dt(self):
        # test for high value of dt
        self.assertEqual(sc.sttc(self.st_1, self.st_2, dt=5 * pq.s), 1.0)

    def test_sttc_edge_cases(self):
        # test for TA = PB = 1 but TB /= PA /= 1 and vice versa
        st2 = neo.SpikeTrain([5], units="ms", t_stop=10.0)
        st3 = neo.SpikeTrain([1, 5, 9], units="ms", t_stop=10.0)
        target2 = 1.0 / 3.0

        self.assertAlmostEqual(target2, sc.sttc(st3, st2, 0.003 * pq.s))
        self.assertAlmostEqual(target2, sc.sttc(st2, st3, 0.003 * pq.s))

    def test_sttc_unsorted_spiketimes(self):
        # regression test for issue  #563
        # https://github.com/NeuralEnsemble/elephant/issues/563
        spiketrain_E7 = neo.SpikeTrain(
            [
                1678.0,
                23786.3,
                34641.8,
                71520.7,
                73606.9,
                78383.3,
                97387.9,
                144313.4,
                4607.6,
                19275.1,
                152894.2,
                44240.1,
            ],
            units="ms",
            t_stop=300000 * pq.ms,
        )

        spiketrain_E3 = neo.SpikeTrain(
            [
                1678.0,
                23786.3,
                34641.8,
                71520.7,
                73606.9,
                78383.3,
                97387.9,
                144313.4,
                4607.6,
                19275.1,
                152894.2,
                44240.1,
            ],
            units="ms",
            t_stop=300000 * pq.ms,
        )
        sttc_unsorted_E7_E3 = sc.sttc(spiketrain_E7, spiketrain_E3, dt=0.10 * pq.s)
        self.assertAlmostEqual(sttc_unsorted_E7_E3, 1)
        spiketrain_E7.sort()
        spiketrain_E3.sort()
        sttc_sorted_E7_E3 = sc.sttc(spiketrain_E7, spiketrain_E3, dt=0.10 * pq.s)
        self.assertAlmostEqual(sttc_unsorted_E7_E3, sttc_sorted_E7_E3)

        spiketrain_E8 = neo.SpikeTrain(
            [
                20646.8,
                25875.1,
                26154.4,
                35121.0,
                55909.7,
                79164.8,
                110849.8,
                117484.1,
                3731.5,
                4213.9,
                119995.1,
                123748.1,
                171016.8,
                172989.0,
                185145.2,
                12043.5,
                185995.9,
                186740.1,
                12629.8,
                23394.3,
                34993.2,
            ],
            units="ms",
            t_stop=300000 * pq.ms,
        )

        spiketrain_B3 = neo.SpikeTrain(
            [
                10600.7,
                19699.6,
                22803.0,
                40769.3,
                121385.7,
                127402.9,
                130829.2,
                134363.8,
                1193.5,
                8012.7,
                142037.3,
                146628.2,
                165925.3,
                168489.3,
                175194.3,
                10339.8,
                178676.4,
                180807.2,
                201431.3,
                22231.1,
                38113.4,
            ],
            units="ms",
            t_stop=300000 * pq.ms,
        )

        self.assertTrue(sc.sttc(spiketrain_E8, spiketrain_B3, dt=0.10 * pq.s) < 1)

        sttc_unsorted_E8_B3 = sc.sttc(spiketrain_E8, spiketrain_B3, dt=0.10 * pq.s)
        spiketrain_E8.sort()
        spiketrain_B3.sort()
        sttc_sorted_E8_B3 = sc.sttc(spiketrain_E8, spiketrain_B3, dt=0.10 * pq.s)
        self.assertAlmostEqual(sttc_unsorted_E8_B3, sttc_sorted_E8_B3)

    def test_sttc_validation_test(self):
        """This test checks the results of elephants implementation of
        the spike time tiling coefficient against the results of the
        original c-implementation.
        The c-code and the test data is located at
        NeuralEnsemble/elephant-data/unittest/spike_train_correlation/
        spike_time_tiling_coefficient"""

        repo_path = (
            r"unittest/spike_train_correlation/spike_time_tiling_coefficient/data"  # noqa
        )

        files_to_download = [
            (
                "spike_time_tiling_coefficient_results.nix",
                "e3749d79046622494660a03e89950f51",
            )
        ]

        downloaded_files = {}
        for filename, checksum in files_to_download:
            downloaded_files[filename] = {
                "filename": filename,
                "path": download_datasets(
                    repo_path=f"{repo_path}/{filename}", checksum=checksum
                ),
            }

        reader = NixIO(
            downloaded_files["spike_time_tiling_coefficient_results.nix"]["path"],
            mode="ro",
        )
        test_data_block = reader.read()

        for segment in test_data_block[0].segments:
            spiketrain_i = segment.spiketrains[0]
            spiketrain_j = segment.spiketrains[1]
            dt = segment.annotations["dt"]
            sttc_result = segment.annotations["sttc_result"]
            self.assertAlmostEqual(sc.sttc(spiketrain_i, spiketrain_j, dt), sttc_result)

    def test_sttc_exist_alias(self):
        # Test if alias cch still exists.
        self.assertEqual(sc.spike_time_tiling_coefficient, sc.sttc)


class SpikeTrainTimescaleTestCase(unittest.TestCase):
    def test_timescale_calculation(self):
        """
        Test the timescale generation using an alpha-shaped ISI distribution,
        see [1, eq. 1.68]. This is equivalent to a homogeneous gamma process
        with alpha=2 and beta=2*nu where nu is the rate.

        For this process, the autocorrelation function is given by a sum of a
        delta peak and a (negative) exponential, see [1, eq. 1.69].
        The exponential decays with \tau_corr = 1 / (4*nu), thus this fixes
        timescale.

        [1] Lindner, B. (2009). A brief introduction to some simple stochastic
            processes. Stochastic Methods in Neuroscience, 1.
        """
        nu = 25 / pq.s
        T = 15 * pq.min
        bin_size = 1 * pq.ms
        timescale = 1 / (4 * nu)
        np.random.seed(35)

        for _ in range(10):
            spikes = StationaryGammaProcess(
                rate=2 * nu / 2, shape_factor=2, t_start=0 * pq.ms, t_stop=T
            ).generate_spiketrain()
            spikes_bin = conv.BinnedSpikeTrain(spikes, bin_size)
            timescale_i = sc.spike_train_timescale(spikes_bin, 10 * timescale)
            assert_array_almost_equal(timescale, timescale_i, decimal=3)

    def test_timescale_errors(self):
        spikes = neo.SpikeTrain([1, 5, 7, 8] * pq.ms, t_stop=10 * pq.ms)
        binsize = 1 * pq.ms
        spikes_bin = conv.BinnedSpikeTrain(spikes, binsize)

        # Tau max with no units
        tau_max = 1
        self.assertRaises(ValueError, sc.spike_train_timescale, spikes_bin, tau_max)

        # Tau max that is not a multiple of the binsize
        tau_max = 1.1 * pq.ms
        self.assertRaises(ValueError, sc.spike_train_timescale, spikes_bin, tau_max)

    def test_timescale_nan(self):
        st0 = neo.SpikeTrain([] * pq.ms, t_stop=10 * pq.ms)
        st1 = neo.SpikeTrain([1] * pq.ms, t_stop=10 * pq.ms)
        st2 = neo.SpikeTrain([1, 5] * pq.ms, t_stop=10 * pq.ms)
        st3 = neo.SpikeTrain([1, 5, 6] * pq.ms, t_stop=10 * pq.ms)
        st4 = neo.SpikeTrain([1, 5, 6, 9] * pq.ms, t_stop=10 * pq.ms)

        binsize = 1 * pq.ms
        tau_max = 1 * pq.ms

        for st in [st0, st1]:
            bst = conv.BinnedSpikeTrain(st, binsize)
            with self.assertWarns(UserWarning):
                timescale = sc.spike_train_timescale(bst, tau_max)
            self.assertTrue(math.isnan(timescale))

        for st in [st2, st3, st4]:
            bst = conv.BinnedSpikeTrain(st, binsize)
            timescale = sc.spike_train_timescale(bst, tau_max)
            self.assertFalse(math.isnan(timescale))


if __name__ == "__main__":
    unittest.main()
