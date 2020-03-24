# -*- coding: utf-8 -*-
"""
This modules provides functions to calculate correlations between spike trains.

:copyright: Copyright 2015-2016 by the Elephant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
"""
from __future__ import division, print_function, unicode_literals

import warnings

import neo
import numpy as np
import quantities as pq
import scipy.signal
from scipy import integrate

# The highest sparsity of the `BinnedSpikeTrain` matrix for which
# memory-efficient (sparse) implementation of `covariance()` is faster than
# with the corresponding numpy dense array.
_SPARSITY_MEMORY_EFFICIENT_THR = 0.1


class _CrossCorrHist(object):
    """
    Cross-correlation histogram for `BinnedSpikeTrain`s.
    This class is used inside :func:`cross_correlation_histogram` function
    and is not meant to be used outside of it.

    Parameters
    ----------
    binned_st1, binned_st2 : elephant.conversion.BinnedSpikeTrain
        Binned spike trains to cross-correlate. The two spike trains must
        have the same `t_start` and `t_stop`.
    window : list or tuple
        List of integers - (left_edge, right_edge).
        Refer to the docs of `cross_correlation_histogram()`.
    """

    def __init__(self, binned_st1, binned_st2, window):
        self.binned_st1 = binned_st1
        self.binned_st2 = binned_st2
        self.window = window

    @staticmethod
    def get_valid_lags(binned_st1, binned_st2):
        """
        Computes the lags at which the cross-correlation
        of the input spiketrains can be calculated with full
        overlap.

        Parameters
        ----------
        binned_st1, binned_st2 : elephant.conversion.BinnedSpikeTrain
            Binned spike trains to cross-correlate. The input spike trains can
            have any `t_start` and `t_stop`.

        Returns
        -------
        lags : np.ndarray
            Array of lags at which the cross-correlation can be computed
            at full overlap (valid mode).
        """

        binsize = binned_st1.binsize

        # see cross_correlation_histogram for the examples
        if binned_st1.num_bins < binned_st2.num_bins:
            # ex. 1) lags range: [-2, 5] ms
            # ex. 2) lags range: [1, 2] ms
            left_edge = (binned_st2.t_start -
                         binned_st1.t_start) / binsize
            right_edge = (binned_st2.t_stop -
                          binned_st1.t_stop) / binsize
        else:
            # ex. 3) lags range: [-1, 3] ms
            left_edge = (binned_st2.t_stop -
                         binned_st1.t_stop) / binsize
            right_edge = (binned_st2.t_start -
                          binned_st1.t_start) / binsize
        right_edge = int(right_edge.simplified.magnitude)
        left_edge = int(left_edge.simplified.magnitude)
        lags = np.arange(left_edge, right_edge + 1, dtype=np.int32)

        return lags

    def correlate_memory(self, cch_mode):
        """
        Slow, but memory-safe mode.

        Return
        -------
        cross_corr : np.ndarray
            Cross-correlation of `self.binned_st` and `self.binned_st2`.
        """
        binned_st1 = self.binned_st1
        binned_st2 = self.binned_st2

        st1_spmat = self.binned_st1._sparse_mat_u
        st2_spmat = self.binned_st2._sparse_mat_u
        left_edge, right_edge = self.window

        # extract the nonzero column indices of 1-d matrices
        st1_bin_idx_unique = st1_spmat.nonzero()[1]
        st2_bin_idx_unique = st2_spmat.nonzero()[1]

        # 'valid' mode requires bins correction due to the shift in t_starts
        # 'full' and 'pad' modes don't need this correction
        if cch_mode == "valid":
            if binned_st1.num_bins > binned_st2.num_bins:
                st2_bin_idx_unique += right_edge
            else:
                st2_bin_idx_unique += left_edge

        st1_spmat = st1_spmat.data
        st2_spmat = st2_spmat.data

        # Initialize the counts to an array of zeros,
        # and the bin IDs to integers
        # spanning the time axis
        nr_lags = right_edge - left_edge + 1
        cross_corr = np.zeros(nr_lags)

        # Compute the CCH at lags in left_edge,...,right_edge only
        for idx, i in enumerate(st1_bin_idx_unique):
            il = np.searchsorted(st2_bin_idx_unique, left_edge + i)
            ir = np.searchsorted(st2_bin_idx_unique,
                                 right_edge + i, side='right')
            timediff = st2_bin_idx_unique[il:ir] - i
            assert ((timediff >= left_edge) & (
                timediff <= right_edge)).all(), 'Not all the '
            'entries of cch lie in the window'
            cross_corr[timediff - left_edge] += (
                st1_spmat[idx] * st2_spmat[il:ir])
            st2_bin_idx_unique = st2_bin_idx_unique[il:]
            st2_spmat = st2_spmat[il:]
        return cross_corr

    def correlate_speed(self, cch_mode):
        """
        Fast, but might require a lot of memory.

        Parameters
        ----------
        cch_mode : str
            Cross-correlation mode.

        Returns
        -------
        cross_corr : np.ndarray
            Cross-correlation of `self.binned_st` and `self.binned_st2`.
        """
        # Retrieve the array of the binned spike trains
        st1_arr = self.binned_st1.to_array()[0]
        st2_arr = self.binned_st2.to_array()[0]
        left_edge, right_edge = self.window
        if cch_mode == 'pad':
            # Zero padding to stay between left_edge and right_edge
            pad_width = min(max(-left_edge, 0), max(right_edge, 0))
            st2_arr = np.pad(st2_arr, pad_width=pad_width, mode='constant')
            cch_mode = 'valid'
        # Cross correlate the spike trains
        cross_corr = scipy.signal.fftconvolve(st2_arr, st1_arr[::-1],
                                              mode=cch_mode)
        # convolution of integers is integers
        cross_corr = np.round(cross_corr)
        return cross_corr

    def border_correction(self, cross_corr):
        """
        Parameters
        ----------
        cross_corr : np.ndarray
            Cross-correlation array. The output of `self.correlate_speed()`
            or `self.correlate_memory()`.

        Returns
        -------
        np.ndarray
            Cross-correlation array with the border correction applied.
        """
        min_num_bins = min(self.binned_st1.num_bins, self.binned_st2.num_bins)
        left_edge, right_edge = self.window
        valid_lags = _CrossCorrHist.get_valid_lags(self.binned_st1,
                                                   self.binned_st2)
        lags_to_compute = np.arange(left_edge, right_edge + 1)
        outer_subtraction = np.subtract.outer(lags_to_compute, valid_lags)
        min_distance_from_window = np.abs(outer_subtraction).min(axis=1)
        n_values_fall_in_window = min_num_bins - min_distance_from_window
        correction = float(min_num_bins) / n_values_fall_in_window
        return cross_corr * correction

    def cross_corr_coef(self, cross_corr):
        """
        Normalizes the CCH to obtain the cross-correlation coefficient
        function, ranging from -1 to 1.

        Parameters
        ----------
        cross_corr : np.ndarray
            Cross-correlation array. The output of `self.correlate_speed()`
            or `self.correlate_memory()`.

        Notes
        -----
        See Notes in `cross_correlation_histogram()`.

        Returns
        -------
        np.ndarray
            Normalized cross-correlation array in range `[-1, 1]`.
        """
        max_num_bins = max(self.binned_st1.num_bins, self.binned_st2.num_bins)
        n_spikes1 = self.binned_st1.get_num_of_spikes()
        n_spikes2 = self.binned_st2.get_num_of_spikes()
        data1 = self.binned_st1._sparse_mat_u.data
        data2 = self.binned_st2._sparse_mat_u.data
        ii = data1.dot(data1)
        jj = data2.dot(data2)
        cov_mean = n_spikes1 * n_spikes2 / max_num_bins
        std_xy = np.sqrt((ii - n_spikes1 ** 2. / max_num_bins) * (
            jj - n_spikes2 ** 2. / max_num_bins))
        cross_corr_normalized = (cross_corr - cov_mean) / std_xy
        return cross_corr_normalized

    def kernel_smoothing(self, cross_corr, kernel):
        """
        Performs 1-d convolution with the `kernel`.

        Parameters
        ----------
        cross_corr : np.ndarray
            Cross-correlation array. The output of `self.correlate_speed()`
            or `self.correlate_memory()`.
        kernel : list
            1-d kernel.

        Returns
        -------
        np.ndarray
            Smoothed array.
        """
        left_edge, right_edge = self.window
        kern_len_max = abs(left_edge) + abs(right_edge) + 1
        # Define the kern for smoothing as an ndarray
        if len(kernel) > kern_len_max:
            raise ValueError(
                'The length of the kernel {} cannot be larger than the '
                'length {} of the resulting CCH.'.format(len(kernel),
                                                         kern_len_max))
        kernel = np.divide(kernel, kernel.sum())
        # Smooth the cross-correlation histogram with the kern
        return np.convolve(cross_corr, kernel, mode='same')


def covariance(binned_sts, binary=False, fast=True):
    r"""
    Calculate the NxN matrix of pairwise covariances between all combinations
    of N binned spike trains.

    For each pair of spike trains :math:`(i,j)`, the covariance :math:`C[i,j]`
    is obtained by binning :math:`i` and :math:`j` at the desired bin size. Let
    :math:`b_i` and :math:`b_j` denote the binned spike trains and
    :math:`\mu_i` and :math:`\mu_j` their respective averages. Then

    .. math::
         C[i,j] = <b_i-\mu_i, b_j-\mu_j> / (L-1)

    where `<., .>` is the scalar product of two vectors, and :math:`L` is the
    number of bins.

    For an input of N spike trains, an N x N matrix is returned containing the
    covariances for each combination of input spike trains.

    If binary is True, the binned spike trains are clipped to 0 or 1 before
    computing the covariance, so that the binned vectors :math:`b_i` and
    :math:`b_j` are binary.

    Parameters
    ----------
    binned_sts : (N, ) elephant.conversion.BinnedSpikeTrain
        A binned spike train containing the spike trains to be evaluated.
    binary : bool, optional
        If True, the spikes of a particular spike train falling in the same bin
        are counted as 1, resulting in binary binned vectors :math:`b_i`. If
        False, the binned vectors :math:`b_i` contain the spike counts per bin.
        Default: False.
    fast : bool, optional
        If `fast=True` and the sparsity of `binned_sts` is `> 0.1`, use
        `np.cov()`. Otherwise, use memory efficient implementation.
        See Notes [2].
        Default: True.

    Returns
    -------
    C : (N, N) np.ndarray
        The square matrix of covariances. The element :math:`C[i,j]=C[j,i]` is
        the covariance between `binned_sts[i]` and `binned_sts[j]`.

    Raises
    ------
    MemoryError
        When using `fast=True` and `binned_sts` shape is large.

    Warns
    --------
    UserWarning
        If at least one row in `binned_sts` is empty (has no spikes).

    See Also
    --------
    corrcoef : Pearson correlation coefficient

    Notes
    -----
    1. The spike trains in the binned structure are assumed to cover the
       complete time span `[t_start, t_stop)` of `binned_sts`.
    2. Using `fast=True` might lead to `MemoryError`. If it's the case,
       switch to `fast=False`.

    Examples
    --------
    Generate two Poisson spike trains

    >>> import neo
    >>> from quantities import s, Hz, ms
    >>> from elephant.spike_train_generation import homogeneous_poisson_process
    >>> from elephant.conversion import BinnedSpikeTrain
    >>> st1 = homogeneous_poisson_process(
    ...       rate=10.0*Hz, t_start=0.0*s, t_stop=10.0*s)
    >>> st2 = homogeneous_poisson_process(
    ...       rate=10.0*Hz, t_start=0.0*s, t_stop=10.0*s)
    >>> cov_matrix = covariance(BinnedSpikeTrain([st1, st2], binsize=5*ms))
    >>> print(cov_matrix[0, 1])
    -0.001668334167083546

    """
    if binary:
        binned_sts = binned_sts.binarize(copy=True)

    if fast and binned_sts.sparsity > _SPARSITY_MEMORY_EFFICIENT_THR:
        array = binned_sts.to_array()
        return np.cov(array)

    return _covariance_sparse(
        binned_sts, corrcoef_norm=False)


def corrcoef(binned_sts, binary=False, fast=True):
    r"""
    Calculate the NxN matrix of pairwise Pearson's correlation coefficients
    between all combinations of N binned spike trains.

    For each pair of spike trains :math:`(i,j)`, the correlation coefficient
    :math:`C[i,j]` is obtained by binning :math:`i` and :math:`j` at the
    desired bin size. Let :math:`b_i` and :math:`b_j` denote the binned spike
    trains and :math:`\mu_i` and :math:`\mu_j` their respective means. Then

    .. math::
         C[i,j] = <b_i-\mu_i, b_j-\mu_j> /
                  \sqrt{<b_i-\mu_i, b_i-\mu_i> \cdot <b_j-\mu_j, b_j-\mu_j>}

    where `<., .>` is the scalar product of two vectors.

    For an input of N spike trains, an N x N matrix is returned.
    Each entry in the matrix is a real number ranging between -1 (perfectly
    anti-correlated spike trains) and +1 (perfectly correlated spike trains).
    However, if k-th spike train is empty, k-th row and k-th column of the
    returned matrix are set to np.nan.

    If binary is True, the binned spike trains are clipped to 0 or 1 before
    computing the correlation coefficients, so that the binned vectors
    :math:`b_i` and :math:`b_j` are binary.

    Parameters
    ----------
    binned_sts : (N, ) elephant.conversion.BinnedSpikeTrain
        A binned spike train containing the spike trains to be evaluated.
    binary : bool, optional
        If True, two spikes of a particular spike train falling in the same bin
        are counted as 1, resulting in binary binned vectors :math:`b_i`. If
        False, the binned vectors :math:`b_i` contain the spike counts per bin.
        Default: False.
    fast : bool, optional
        If `fast=True` and the sparsity of `binned_sts` is `> 0.1`, use
        `np.corrcoef()`. Otherwise, use memory efficient implementation.
        See Notes[2]
        Default: True.

    Returns
    -------
    C : (N, N) np.ndarray
        The square matrix of correlation coefficients. The element
        :math:`C[i,j]=C[j,i]` is the Pearson's correlation coefficient between
        `binned_sts[i]` and `binned_sts[j]`. If `binned_sts` contains only one
        `neo.SpikeTrain`, C=1.0.

    Raises
    ------
    MemoryError
        When using `fast=True` and `binned_sts` shape is large.

    Warns
    --------
    UserWarning
        If at least one row in `binned_sts` is empty (has no spikes).

    See Also
    --------
    covariance

    Notes
    -----
    1. The spike trains in the binned structure are assumed to cover the
       complete time span `[t_start, t_stop)` of `binned_sts`.
    2. Using `fast=True` might lead to `MemoryError`. If it's the case,
       switch to `fast=False`.

    Examples
    --------
    Generate two Poisson spike trains

    >>> import neo
    >>> from quantities import s, Hz, ms
    >>> from elephant.spike_train_generation import homogeneous_poisson_process
    >>> from elephant.conversion import BinnedSpikeTrain
    >>> st1 = homogeneous_poisson_process(
    ...       rate=10.0*Hz, t_start=0.0*s, t_stop=10.0*s)
    >>> st2 = homogeneous_poisson_process(
    ...       rate=10.0*Hz, t_start=0.0*s, t_stop=10.0*s)
    >>> cc_matrix = corrcoef(BinnedSpikeTrain([st1, st2], binsize=5*ms))
    >>> print(cc_matrix[0, 1])
    0.015477320222075359

    """
    if binary:
        binned_sts = binned_sts.binarize(copy=True)

    if fast and binned_sts.sparsity > _SPARSITY_MEMORY_EFFICIENT_THR:
        array = binned_sts.to_array()
        return np.corrcoef(array)

    return _covariance_sparse(
        binned_sts, corrcoef_norm=True)


def _covariance_sparse(binned_sts, corrcoef_norm):
    r"""
    Memory efficient helper function for `covariance()` and `corrcoef()`
    that performs the complete calculation for either the covariance
    (`corrcoef_norm=False`) or correlation coefficient (`corrcoef_norm=True`).
    Both calculations differ only by the denominator.

    For any two `BinnedSpikeTrain`s :math:`\hat{b_x}` and :math:`\hat{b_y}`
    with mean :math:`\vec{\mu_x}` and :math:`\vec{mu_y}` respectively
    computes the dot product

    .. math::
        <\hat{b_x} - \vec{\mu_x}, \hat{b_y} - \vec{\mu_y}>_{ij} =
            (\hat{b_x} \cdot \hat{b_y}^T)_{ij} -
            \frac{(\vec{N_x}^T \cdot \vec{N_y})_{ij}}{L}

    where :math:`N_x^i = \sum_j{b_x^{ij}}` - the number of spikes in `i`th row
    of :math:`\hat{b_x}`, :math:`L` - the number of bins, and
    :math:`\vec{\mu_x} = \frac{\vec{N_x}}{L}`.

    Parameters
    ----------
    binned_sts : (N, ) elephant.conversion.BinnedSpikeTrain
        See `covariance()` or `corrcoef()`, respectively.
    corrcoef_norm : bool
        Use normalization factor for the correlation coefficient rather than
        for the covariance.

    Warns
    --------
    UserWarning
        If at least one row in `binned_sts` is empty (has no spikes).

    Returns
    -------
    (N, N) np.ndarray
        Pearson correlation or covariance matrix.
    """
    spmat = binned_sts._sparse_mat_u
    n_bins = binned_sts.num_bins

    # Check for empty spike trains
    n_spikes_per_row = spmat.sum(axis=1)
    if n_spikes_per_row.min() == 0:
        warnings.warn(
            'Detected empty spike trains (rows) in the argument binned_sts.')

    res = spmat.dot(spmat.T) - n_spikes_per_row * n_spikes_per_row.T / n_bins
    res = np.asarray(res)
    if corrcoef_norm:
        stdx = np.sqrt(res.diagonal())
        stdx = np.expand_dims(stdx, axis=0)
        res /= (stdx.T * stdx)
    else:
        res /= (n_bins - 1)
    res = np.squeeze(res)
    return res


def cross_correlation_histogram(
        binned_st1, binned_st2, window='full', border_correction=False,
        binary=False, kernel=None, method='speed', cross_corr_coef=False):
    """
    Computes the cross-correlation histogram (CCH) between two binned spike
    trains `binned_st1` and `binned_st2`.

    Parameters
    ----------
    binned_st1, binned_st2 : elephant.conversion.BinnedSpikeTrain
        Binned spike trains of lengths N and M to cross-correlate. The input
        spike trains can have any `t_start` and `t_stop`.
    window : {'valid', 'full'} or list of int, optional
        ‘full’: This returns the cross-correlation at each point of overlap,
                with an output shape of (N+M-1,). At the end-points of the
                cross-correlogram, the signals do not overlap completely, and
                boundary effects may be seen.
        ‘valid’: Mode valid returns output of length max(M, N) - min(M, N) + 1.
                 The cross-correlation product is only given for points where
                 the signals overlap completely.
                 Values outside the signal boundary have no effect.
        List of integers (min_lag, max_lag):
              The entries of window are two integers representing the left and
              right extremes (expressed as number of bins) where the
              cross-correlation is computed.
        Default: 'full'.
    border_correction : bool, optional
        whether to correct for the border effect. If True, the value of the
        CCH at bin :math:`b` (for :math:`b=-H,-H+1, ...,H`, where :math:`H` is
        the CCH half-length) is multiplied by the correction factor:

        .. math::
                            (H+1)/(H+1-|b|),

        which linearly corrects for loss of bins at the edges.
        Default: False.
    binary : bool, optional
        If True, spikes falling in the same bin are counted as a single spike;
        otherwise they are counted as different spikes.
        Default: False.
    kernel : np.ndarray or None, optional
        A one dimensional array containing a smoothing kernel applied
        to the resulting CCH. The length N of the kernel indicates the
        smoothing window. The smoothing window cannot be larger than the
        maximum lag of the CCH. The kernel is normalized to unit area before
        being applied to the resulting CCH. Popular choices for the kernel are
          * normalized boxcar kernel: `numpy.ones(N)`
          * hamming: `numpy.hamming(N)`
          * hanning: `numpy.hanning(N)`
          * bartlett: `numpy.bartlett(N)`
        If None, the CCH is not smoothed.
        Default: None.
    method : {'speed', 'memory'}, optional
        Defines the algorithm to use. "speed" uses `numpy.correlate` to
        calculate the correlation between two binned spike trains using a
        non-sparse data representation. Due to various optimizations, it is the
        fastest realization. In contrast, the option "memory" uses an own
        implementation to calculate the correlation based on sparse matrices,
        which is more memory efficient but slower than the "speed" option.
        Default: "speed".
    cross_corr_coef : bool, optional
        If True, a normalization is applied to the CCH to obtain the
        cross-correlation  coefficient function ranging from -1 to 1 according
        to Equation (5.10) in [1]_. See Notes.
        Default: False.

    Returns
    -------
    cch_result : neo.AnalogSignal
        Containing the cross-correlation histogram between `binned_st1` and
        `binned_st2`.

        Offset bins correspond to correlations at delays equivalent
        to the differences between the spike times of `binned_st1` and those of
        `binned_st2`: an entry at positive lag corresponds to a spike in
        `binned_st2` following a spike in `binned_st1` bins to the right, and
        an entry at negative lag corresponds to a spike in `binned_st1`
        following a spike in `binned_st2`.

        To illustrate this definition, consider two spike trains with the same
        `t_start` and `t_stop`:
        `binned_st1` ('reference neuron') : 0 0 0 0 1 0 0 0 0 0 0
        `binned_st2` ('target neuron')    : 0 0 0 0 0 0 0 1 0 0 0
        Here, the CCH will have an entry of `1` at `lag=+3`.

        Consistent with the definition of `neo.AnalogSignals`, the time axis
        represents the left bin borders of each histogram bin. For example,
        the time axis might be:
        `np.array([-2.5 -1.5 -0.5 0.5 1.5]) * ms`
    lags : np.ndarray
        Contains the IDs of the individual histogram bins, where the central
        bin has ID 0, bins to the left have negative IDs and bins to the right
        have positive IDs, e.g.,:
        `np.array([-3, -2, -1, 0, 1, 2, 3])`

    Notes
    -----
    1. The Eq. (5.10) in [1]_ is valid for binned spike trains with at most one
       spike per bin. For a general case, refer to the implementation of
       `_covariance_sparse()`.
    2. Alias: `cch`

    References
    ----------
    .. [1] "Analysis of parallel spike trains", 2010, Gruen & Rotter, Vol 7.

    Examples
    --------
    Plot the cross-correlation histogram between two Poisson spike trains

    >>> import elephant
    >>> import matplotlib.pyplot as plt
    >>> import quantities as pq

    >>> binned_st1 = elephant.conversion.BinnedSpikeTrain(
    ...        elephant.spike_train_generation.homogeneous_poisson_process(
    ...            10. * pq.Hz, t_start=0 * pq.ms, t_stop=5000 * pq.ms),
    ...        binsize=5. * pq.ms)
    >>> binned_st2 = elephant.conversion.BinnedSpikeTrain(
    ...        elephant.spike_train_generation.homogeneous_poisson_process(
    ...            10. * pq.Hz, t_start=0 * pq.ms, t_stop=5000 * pq.ms),
    ...        binsize=5. * pq.ms)

    >>> cc_hist = \
    ...    elephant.spike_train_correlation.cross_correlation_histogram(
    ...        binned_st1, binned_st2, window=[-30,30],
    ...        border_correction=False,
    ...        binary=False, kernel=None, method='memory')

    >>> plt.bar(left=cc_hist[0].times.magnitude,
    ...         height=cc_hist[0][:, 0].magnitude,
    ...         width=cc_hist[0].sampling_period.magnitude)
    >>> plt.xlabel('time (' + str(cc_hist[0].times.units) + ')')
    >>> plt.ylabel('cross-correlation histogram')
    >>> plt.axis('tight')
    >>> plt.show()

    """

    # Check that the spike trains are binned with the same temporal
    # resolution
    if binned_st1.matrix_rows != 1 or binned_st2.matrix_rows != 1:
        raise ValueError("Spike trains must be one dimensional")
    if not np.isclose(binned_st1.binsize.simplified.item(),
                      binned_st2.binsize.simplified.item()):
        raise ValueError("Bin sizes must be equal")

    binsize = binned_st1.binsize
    left_edge_min = -binned_st1.num_bins + 1
    right_edge_max = binned_st2.num_bins - 1

    t_lags_shift = (binned_st2.t_start - binned_st1.t_start) / binsize
    t_lags_shift = t_lags_shift.simplified.item()
    if not np.isclose(t_lags_shift, round(t_lags_shift)):
        # For example, if binsize=1 ms, binned_st1.t_start=0 ms, and
        # binned_st2.t_start=0.5 ms then there is a global shift in the
        # binning of the spike trains.
        raise ValueError(
            "Binned spiketrains time shift is not multiple of binsize")
    t_lags_shift = int(round(t_lags_shift))

    # In the examples below we fix st2 and "move" st1.
    # Zero-lag is equal to `max(st1.t_start, st2.t_start)`.
    # Binned spiketrains (t_start and t_stop) with binsize=1ms:
    # 1) st1=[3, 8] ms, st2=[1, 13] ms
    #    t_start_shift = -2 ms
    #    zero-lag is at 3 ms
    # 2) st1=[1, 7] ms, st2=[2, 9] ms
    #    t_start_shift = 1 ms
    #    zero-lag is at 2 ms
    # 3) st1=[1, 7] ms, st2=[4, 6] ms
    #    t_start_shift = 3 ms
    #    zero-lag is at 4 ms

    # Find left and right edges of unaligned (time-dropped) time signals
    if len(window) == 2 and np.issubdtype(type(window[0]), int) \
            and np.issubdtype(type(window[1]), int):
        # ex. 1) lags range: [w[0] - 2, w[1] - 2] ms
        # ex. 2) lags range: [w[0] + 1, w[1] + 1] ms
        # ex. 3) lags range: [w[0] + 3, w[0] + 3] ms
        if window[0] >= window[1]:
            raise ValueError(
                "Window's left edge ({left}) must be lower than the right "
                "edge ({right})".format(left=window[0], right=window[1]))
        left_edge, right_edge = np.subtract(window, t_lags_shift)
        if left_edge < left_edge_min or right_edge > right_edge_max:
            raise ValueError(
                "The window exceeds the length of the spike trains")
        lags = np.arange(window[0], window[1] + 1, dtype=np.int32)
        cch_mode = 'pad'
    elif window == 'full':
        # cch computed for all the possible entries
        # ex. 1) lags range: [-6, 9] ms
        # ex. 2) lags range: [-4, 7] ms
        # ex. 3) lags range: [-2, 4] ms
        left_edge = left_edge_min
        right_edge = right_edge_max
        lags = np.arange(left_edge + t_lags_shift,
                         right_edge + 1 + t_lags_shift, dtype=np.int32)
        cch_mode = window
    elif window == 'valid':
        lags = _CrossCorrHist.get_valid_lags(binned_st1, binned_st2)
        left_edge, right_edge = lags[(0, -1), ]
        cch_mode = window
    else:
        raise ValueError("Invalid window parameter")

    if binary:
        binned_st1 = binned_st1.binarize(copy=True)
        binned_st2 = binned_st2.binarize(copy=True)

    cch_builder = _CrossCorrHist(binned_st1, binned_st2,
                                 window=(left_edge, right_edge))
    if method == 'memory':
        cross_corr = cch_builder.correlate_memory(cch_mode=cch_mode)
    else:
        cross_corr = cch_builder.correlate_speed(cch_mode=cch_mode)

    if border_correction:
        if window == 'valid':
            warnings.warn(
                "Border correction does not have any effect in "
                "'valid' window mode since there are no border effects!")
        else:
            cross_corr = cch_builder.border_correction(cross_corr)
    if kernel is not None:
        cross_corr = cch_builder.kernel_smoothing(cross_corr, kernel=kernel)
    if cross_corr_coef:
        cross_corr = cch_builder.cross_corr_coef(cross_corr)

    # Transform the array count into an AnalogSignal
    cch_result = neo.AnalogSignal(
        signal=np.expand_dims(cross_corr, axis=1),
        units=pq.dimensionless,
        t_start=(lags[0] - 0.5) * binned_st1.binsize,
        sampling_period=binned_st1.binsize)
    return cch_result, lags


# Alias for common abbreviation
cch = cross_correlation_histogram


def spike_time_tiling_coefficient(spiketrain_1, spiketrain_2, dt=0.005 * pq.s):
    """
    Calculates the Spike Time Tiling Coefficient (STTC) as described in [1]_
    following their implementation in C.
    The STTC is a pairwise measure of correlation between spike trains.
    It has been proposed as a replacement for the correlation index as it
    presents several advantages (e.g. it's not confounded by firing rate,
    appropriately distinguishes lack of correlation from anti-correlation,
    periods of silence don't add to the correlation and it's sensitive to
    firing patterns).

    The STTC is calculated as follows:

    .. math::
        STTC = 1/2((PA - TB)/(1 - PA*TB) + (PB - TA)/(1 - PB*TA))

    Where `PA` is the proportion of spikes from train 1 that lie within
    `[-dt, +dt]` of any spike of train 2 divided by the total number of spikes
    in train 1, `PB` is the same proportion for the spikes in train 2;
    `TA` is the proportion of total recording time within `[-dt, +dt]` of any
    spike in train 1, TB is the same proportion for train 2.
    For :math:`TA = PB = 1`and for :math:`TB = PA = 1`
    the resulting :math:`0/0` is replaced with :math:`1`,
    since every spike from the train with :math:`T = 1` is within
    `[-dt, +dt]` of a spike of the other train.

    This is a Python implementation compatible with the elephant library of
    the original code by C. Cutts written in C and avaiable at:
    (https://github.com/CCutts/Detecting_pairwise_correlations_in_spike_trains/blob/master/spike_time_tiling_coefficient.c)

    Parameters
    ----------
    spiketrain_1, spiketrain_2: neo.SpikeTrain
        Spike trains to cross-correlate. They must have the same `t_start` and
        `t_stop`.
    dt: pq.Quantity.
        The synchronicity window is used for both: the quantification of the
        proportion of total recording time that lies `[-dt, +dt]` of each spike
        in each train and the proportion of spikes in `spiketrain_1` that lies
        `[-dt, +dt]` of any spike in `spiketrain_2`.
        Default : `0.005 * pq.s`

    Returns
    -------
    index:  float or np.nan
        The spike time tiling coefficient (STTC). Returns np.nan if any spike
        train is empty.

    References
    ----------
    .. [1] Cutts, C. S., & Eglen, S. J. (2014). Detecting Pairwise Correlations
           in Spike Trains: An Objective Comparison of Methods and Application
           to the Study of Retinal Waves. Journal of Neuroscience, 34(43),
           14288–14303.

    Notes
    -----
    Alias: `sttc`
    """

    def run_P(spiketrain_1, spiketrain_2):
        """
        Check every spike in train 1 to see if there's a spike in train 2
        within dt
        """
        N2 = len(spiketrain_2)

        # Search spikes of spiketrain_1 in spiketrain_2
        # ind will contain index of
        ind = np.searchsorted(spiketrain_2.times, spiketrain_1.times)

        # To prevent IndexErrors
        # If a spike of spiketrain_1 is after the last spike of spiketrain_2,
        # the index is N2, however spiketrain_2[N2] raises an IndexError.
        # By shifting this index, the spike of spiketrain_1 will be compared
        # to the last 2 spikes of spiketrain_2 (negligible overhead).
        # Note: Not necessary for index 0 that will be shifted to -1,
        # because spiketrain_2[-1] is valid (additional negligible comparison)
        ind[ind == N2] = N2 - 1

        # Compare to nearest spike in spiketrain_2 BEFORE spike in spiketrain_1
        close_left = np.abs(
            spiketrain_2.times[ind - 1] - spiketrain_1.times) <= dt
        # Compare to nearest spike in spiketrain_2 AFTER (or simultaneous)
        # spike in spiketrain_2
        close_right = np.abs(
            spiketrain_2.times[ind] - spiketrain_1.times) <= dt

        # spiketrain_2 spikes that are in [-dt, dt] range of spiketrain_1
        # spikes are counted only ONCE (as per original implementation)
        close = close_left + close_right

        # Count how many spikes in spiketrain_1 have a "partner" in
        # spiketrain_2
        return np.count_nonzero(close)

    def run_T(spiketrain):
        """
        Calculate the proportion of the total recording time 'tiled' by spikes.
        """
        N = len(spiketrain)
        time_A = 2 * N * dt  # maximum possible time

        if N == 1:  # for just one spike in train
            if spiketrain[0] - spiketrain.t_start < dt:
                time_A += -dt + spiketrain[0] - spiketrain.t_start
            if spiketrain[0] + dt > spiketrain.t_stop:
                time_A += -dt - spiketrain[0] + spiketrain.t_stop
        else:  # if more than one spike in train
            # Vectorized loop of spike time differences
            diff = np.diff(spiketrain)
            diff_overlap = diff[diff < 2 * dt]
            # Subtract overlap
            time_A += -2 * dt * len(diff_overlap) + np.sum(diff_overlap)

            # check if spikes are within dt of the start and/or end
            # if so subtract overlap of first and/or last spike
            if (spiketrain[0] - spiketrain.t_start) < dt:
                time_A += spiketrain[0] - dt - spiketrain.t_start

            if (spiketrain.t_stop - spiketrain[N - 1]) < dt:
                time_A += -spiketrain[-1] - dt + spiketrain.t_stop

        T = time_A / (spiketrain.t_stop - spiketrain.t_start)
        return T.simplified.item()  # enforce simplification, strip units

    N1 = len(spiketrain_1)
    N2 = len(spiketrain_2)

    if N1 == 0 or N2 == 0:
        index = np.nan
    else:
        TA = run_T(spiketrain_1)
        TB = run_T(spiketrain_2)
        PA = run_P(spiketrain_1, spiketrain_2)
        PA = PA / N1
        PB = run_P(spiketrain_2, spiketrain_1)
        PB = PB / N2
        # check if the P and T values are 1 to avoid division by zero
        # This only happens for TA = PB = 1 and/or TB = PA = 1,
        # which leads to 0/0 in the calculation of the index.
        # In those cases, every spike in the train with P = 1
        # is within dt of a spike in the other train,
        # so we set the respective (partial) index to 1.
        if PA * TB == 1:
            if PB * TA == 1:
                index = 1.
            else:
                index = 0.5 + 0.5 * (PB - TA) / (1 - PB * TA)
        elif PB * TA == 1:
            index = 0.5 + 0.5 * (PA - TB) / (1 - PA * TB)
        else:
            index = 0.5 * (PA - TB) / (1 - PA * TB) + 0.5 * (PB - TA) / (
                1 - PB * TA)
    return index


sttc = spike_time_tiling_coefficient


def spike_train_timescale(binned_st, tau_max):
    r"""
    Calculates the auto-correlation time of a binned spike train.
    Uses the definition of the auto-correlation time proposed in [[1]_,
    Eq. (6)]:

    .. math::
        \tau_\mathrm{corr} = \int_{-\tau_\mathrm{max}}^{\tau_\mathrm{max}}\
            \left[ \frac{\hat{C}(\tau)}{\hat{C}(0)} \right]^2 d\tau

    where :math:`\hat{C}(\tau) = C(\tau)-\nu\delta(\tau)` denotes
    the auto-correlation function excluding the Dirac delta at zero timelag.

    Parameters
    ----------
    binned_st : elephant.conversion.BinnedSpikeTrain
        A binned spike train containing the spike train to be evaluated.
    tau_max : pq.Quantity
        Maximal integration time of the auto-correlation function.

    Returns
    -------
    timescale : pq.Quantity
        The auto-correlation time of the binned spiketrain.

    Notes
    -----
    * :math:`\tau_\mathrm{max}` is a critical parameter: numerical estimates
      of the auto-correlation functions are inherently noisy. Due to the
      square in the definition above, this noise is integrated. Thus, it is
      necessary to introduce a cutoff for the numerical integration - this
      cutoff should be neither smaller than the true auto-correlation time
      nor much bigger.
    * The bin size of `binned_st` is another critical parameter as it defines
      the discretization of the integral :math:`d\tau`. If it is too big, the
      numerical approximation of the integral is inaccurate.

    References
    ----------
    .. [1] Wieland, S., Bernardi, D., Schwalger, T., & Lindner, B. (2015).
        Slow fluctuations in recurrent networks of spiking neurons.
        Physical Review E, 92(4), 040901.
    """
    binsize = binned_st.binsize
    if not (tau_max / binsize).simplified.units == pq.dimensionless:
        raise AssertionError("tau_max needs units of time")

    # safe casting of tau_max/binsize to integer
    tau_max_bins = int(np.round((tau_max / binsize).simplified.magnitude))
    if not np.isclose(tau_max.simplified.magnitude,
                      (tau_max_bins * binsize).simplified.magnitude):
        raise AssertionError("tau_max has to be a multiple of the binsize")

    cch_window = [-tau_max_bins, tau_max_bins]
    corrfct, bin_ids = cross_correlation_histogram(
        binned_st, binned_st, window=cch_window, cross_corr_coef=True
    )
    # Take only t > 0 values, in particular neglecting the delta peak.
    corrfct_pos = corrfct.time_slice(binsize / 2, corrfct.t_stop).flatten()

    # Calculate the timescale using trapezoidal integration
    integr = np.abs((corrfct_pos / corrfct_pos[0]).magnitude)**2
    timescale = 2 * integrate.trapz(integr, dx=binsize)
    return timescale
