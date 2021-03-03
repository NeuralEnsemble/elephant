# -*- coding: utf-8 -*-
"""
This modules provides functions to calculate correlations between spike trains.

.. autosummary::
    :toctree: _toctree/spike_train_correlation

    covariance
    correlation_coefficient
    cross_correlation_histogram
    spike_time_tiling_coefficient
    spike_train_timescale

:copyright: Copyright 2014-2020 by the Elephant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
"""
from __future__ import division, print_function, unicode_literals

import warnings

import neo
import numpy as np
import quantities as pq
import scipy.signal
from scipy import integrate

from elephant.conversion import BinnedSpikeTrain
from elephant.utils import deprecated_alias

__all__ = [
    "covariance",
    "correlation_coefficient",
    "cross_correlation_histogram",
    "spike_time_tiling_coefficient",
    "spike_train_timescale"
]

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
    binned_spiketrain_i, binned_spiketrain_j :
        elephant.conversion.BinnedSpikeTrain
        Binned spike trains to cross-correlate. The two spike trains must
        have the same `t_start` and `t_stop`.
    window : list or tuple
        List of integers - (left_edge, right_edge).
        Refer to the docs of `cross_correlation_histogram()`.
    """

    def __init__(self, binned_spiketrain_i, binned_spiketrain_j, window):
        self.binned_spiketrain_i = binned_spiketrain_i
        self.binned_spiketrain_j = binned_spiketrain_j
        self.window = window

    @staticmethod
    def get_valid_lags(binned_spiketrain_i, binned_spiketrain_j):
        """
        Computes the lags at which the cross-correlation
        of the input spiketrains can be calculated with full
        overlap.

        Parameters
        ----------
        binned_spiketrain_i, binned_spiketrain_j :
            elephant.conversion.BinnedSpikeTrain
            Binned spike trains to cross-correlate. The input spike trains can
            have any `t_start` and `t_stop`.

        Returns
        -------
        lags : np.ndarray
            Array of lags at which the cross-correlation can be computed
            at full overlap (valid mode).
        """

        bin_size = binned_spiketrain_i._bin_size

        # see cross_correlation_histogram for the examples
        if binned_spiketrain_i.n_bins < binned_spiketrain_j.n_bins:
            # ex. 1) lags range: [-2, 5] ms
            # ex. 2) lags range: [1, 2] ms
            left_edge = (binned_spiketrain_j._t_start -
                         binned_spiketrain_i._t_start) / bin_size
            right_edge = (binned_spiketrain_j._t_stop -
                          binned_spiketrain_i._t_stop) / bin_size
        else:
            # ex. 3) lags range: [-1, 3] ms
            left_edge = (binned_spiketrain_j._t_stop -
                         binned_spiketrain_i._t_stop) / bin_size
            right_edge = (binned_spiketrain_j._t_start -
                          binned_spiketrain_i._t_start) / bin_size
        right_edge = int(right_edge)
        left_edge = int(left_edge)
        lags = np.arange(left_edge, right_edge + 1, dtype=np.int32)

        return lags

    def correlate_memory(self, cch_mode):
        """
        Slow, but memory-safe mode.

        Return
        -------
        cross_corr : np.ndarray
            Cross-correlation of `self.binned_spiketrain1` and
            `self.binned_spiketrain2`.
        """
        st1_spmat = self.binned_spiketrain_i.sparse_matrix
        st2_spmat = self.binned_spiketrain_j.sparse_matrix
        left_edge, right_edge = self.window

        # extract the nonzero column indices of 1-d matrices
        st1_bin_idx_unique = st1_spmat.nonzero()[1]
        st2_bin_idx_unique = st2_spmat.nonzero()[1]

        # 'valid' mode requires bins correction due to the shift in t_starts
        # 'full' and 'pad' modes don't need this correction
        if cch_mode == "valid":
            if self.binned_spiketrain_i.n_bins > \
                    self.binned_spiketrain_j.n_bins:
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
                timediff <= right_edge)).all(), \
                'Not all the entries of cch lie in the window'
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
            Cross-correlation of `self.binned_spiketrain1` and
            `self.binned_spiketrain2`.
        """
        # Retrieve the array of the binned spike trains
        st1_arr = self.binned_spiketrain_i.to_array()[0]
        st2_arr = self.binned_spiketrain_j.to_array()[0]
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
        min_num_bins = min(self.binned_spiketrain_i.n_bins,
                           self.binned_spiketrain_j.n_bins)
        left_edge, right_edge = self.window
        valid_lags = _CrossCorrHist.get_valid_lags(self.binned_spiketrain_i,
                                                   self.binned_spiketrain_j)
        lags_to_compute = np.arange(left_edge, right_edge + 1)
        outer_subtraction = np.subtract.outer(lags_to_compute, valid_lags)
        min_distance_from_window = np.abs(outer_subtraction).min(axis=1)
        n_values_fall_in_window = min_num_bins - min_distance_from_window
        correction = float(min_num_bins) / n_values_fall_in_window
        return cross_corr * correction

    def cross_correlation_coefficient(self, cross_corr):
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
        max_num_bins = max(self.binned_spiketrain_i.n_bins,
                           self.binned_spiketrain_j.n_bins)
        n_spikes1 = self.binned_spiketrain_i.get_num_of_spikes()
        n_spikes2 = self.binned_spiketrain_j.get_num_of_spikes()
        data1 = self.binned_spiketrain_i.sparse_matrix.data
        data2 = self.binned_spiketrain_j.sparse_matrix.data
        ii = data1.dot(data1)
        jj = data2.dot(data2)
        cov_mean = n_spikes1 * n_spikes2 / max_num_bins
        std_xy = np.sqrt((ii - n_spikes1 ** 2. / max_num_bins) * (
            jj - n_spikes2 ** 2. / max_num_bins))
        cross_corr_normalized = (cross_corr - cov_mean) / std_xy
        return cross_corr_normalized

    def kernel_smoothing(self, cross_corr_array, kernel):
        """
        Performs 1-d convolution with the `kernel`.

        Parameters
        ----------
        cross_corr_array : np.ndarray
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
        return np.convolve(cross_corr_array, kernel, mode='same')


@deprecated_alias(binned_sts='binned_spiketrain')
def covariance(binned_spiketrain, binary=False, fast=True):
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
    binned_spiketrain : (N, ) elephant.conversion.BinnedSpikeTrain
        A binned spike train containing the spike trains to be evaluated.
    binary : bool, optional
        If True, the spikes of a particular spike train falling in the same bin
        are counted as 1, resulting in binary binned vectors :math:`b_i`. If
        False, the binned vectors :math:`b_i` contain the spike counts per bin.
        Default: False
    fast : bool, optional
        If `fast=True` and the sparsity of `binned_spiketrain` is `> 0.1`, use
        `np.cov()`. Otherwise, use memory efficient implementation.
        See Notes [2].
        Default: True

    Returns
    -------
    C : (N, N) np.ndarray
        The square matrix of covariances. The element :math:`C[i,j]=C[j,i]` is
        the covariance between `binned_spiketrain[i]` and
        `binned_spiketrain[j]`.

    Raises
    ------
    MemoryError
        When using `fast=True` and `binned_spiketrain` shape is large.

    Warns
    --------
    UserWarning
        If at least one row in `binned_spiketrain` is empty (has no spikes).

    See Also
    --------
    correlation_coefficient : Pearson correlation coefficient

    Notes
    -----
    1. The spike trains in the binned structure are assumed to cover the
       complete time span `[t_start, t_stop)` of `binned_spiketrain`.
    2. Using `fast=True` might lead to `MemoryError`. If it's the case,
       switch to `fast=False`.

    Examples
    --------
    Covariance matrix of two Poisson spike train processes.

    >>> import neo
    >>> import numpy as np
    >>> import quantities as pq
    >>> from elephant.spike_train_generation import homogeneous_poisson_process
    >>> from elephant.conversion import BinnedSpikeTrain
    >>> from elephant.spike_train_correlation import covariance

    >>> np.random.seed(1)
    >>> st1 = homogeneous_poisson_process(rate=10*pq.Hz, t_stop=10.0*pq.s)
    >>> st2 = homogeneous_poisson_process(rate=10*pq.Hz, t_stop=10.0*pq.s)
    >>> cov_matrix = covariance(BinnedSpikeTrain([st1, st2], bin_size=5*pq.ms))
    >>> cov_matrix
    array([[ 0.05432316, -0.00152276],
       [-0.00152276,  0.04917234]])

    """
    if binary:
        binned_spiketrain = binned_spiketrain.binarize()

    if fast and binned_spiketrain.sparsity > _SPARSITY_MEMORY_EFFICIENT_THR:
        array = binned_spiketrain.to_array()
        return np.cov(array)

    return _covariance_sparse(
        binned_spiketrain, corrcoef_norm=False)


@deprecated_alias(binned_sts='binned_spiketrain')
def correlation_coefficient(binned_spiketrain, binary=False, fast=True):
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

    Visualization of this function is covered in Viziphant:
    :func:`viziphant.spike_train_correlation.plot_corrcoef`.


    Parameters
    ----------
    binned_spiketrain : (N, ) elephant.conversion.BinnedSpikeTrain
        A binned spike train containing the spike trains to be evaluated.
    binary : bool, optional
        If True, two spikes of a particular spike train falling in the same bin
        are counted as 1, resulting in binary binned vectors :math:`b_i`. If
        False, the binned vectors :math:`b_i` contain the spike counts per bin.
        Default: False
    fast : bool, optional
        If `fast=True` and the sparsity of `binned_spiketrain` is `> 0.1`, use
        `np.corrcoef()`. Otherwise, use memory efficient implementation.
        See Notes[2]
        Default: True

    Returns
    -------
    C : (N, N) np.ndarray
        The square matrix of correlation coefficients. The element
        :math:`C[i,j]=C[j,i]` is the Pearson's correlation coefficient between
        `binned_spiketrain[i]` and `binned_spiketrain[j]`.
        If `binned_spiketrain` contains only one `neo.SpikeTrain`, C=1.0.

    Raises
    ------
    MemoryError
        When using `fast=True` and `binned_spiketrain` shape is large.

    Warns
    --------
    UserWarning
        If at least one row in `binned_spiketrain` is empty (has no spikes).

    See Also
    --------
    covariance

    Notes
    -----
    1. The spike trains in the binned structure are assumed to cover the
       complete time span `[t_start, t_stop)` of `binned_spiketrain`.
    2. Using `fast=True` might lead to `MemoryError`. If it's the case,
       switch to `fast=False`.

    Examples
    --------
    Correlation coefficient of two Poisson spike train processes.

    >>> import neo
    >>> import numpy as np
    >>> import quantities as pq
    >>> from elephant.spike_train_generation import homogeneous_poisson_process
    >>> from elephant.conversion import BinnedSpikeTrain
    >>> from elephant.spike_train_correlation import correlation_coefficient

    >>> np.random.seed(1)
    >>> st1 = homogeneous_poisson_process(rate=10*pq.Hz, t_stop=10.0*pq.s)
    >>> st2 = homogeneous_poisson_process(rate=10*pq.Hz, t_stop=10.0*pq.s)
    >>> corrcoef = correlation_coefficient(BinnedSpikeTrain([st1, st2],
    ...     bin_size=5*pq.ms))
    >>> corrcoef
    array([[ 1.        , -0.02946313],
           [-0.02946313,  1.        ]])
    """
    if binary:
        binned_spiketrain = binned_spiketrain.binarize()

    if fast and binned_spiketrain.sparsity > _SPARSITY_MEMORY_EFFICIENT_THR:
        array = binned_spiketrain.to_array()
        return np.corrcoef(array)

    return _covariance_sparse(
        binned_spiketrain, corrcoef_norm=True)


def corrcoef(*args, **kwargs):
    warnings.warn("'corrcoef' is deprecated; use 'correlation_coefficient'",
                  DeprecationWarning)
    return correlation_coefficient(*args, **kwargs)


def _covariance_sparse(binned_spiketrain, corrcoef_norm):
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
    binned_spiketrain : (N, ) elephant.conversion.BinnedSpikeTrain
        See `covariance()` or `corrcoef()`, respectively.
    corrcoef_norm : bool
        Use normalization factor for the correlation coefficient rather than
        for the covariance.

    Warns
    --------
    UserWarning
        If at least one row in `binned_spiketrain` is empty (has no spikes).

    Returns
    -------
    (N, N) np.ndarray
        Pearson correlation or covariance matrix.
    """
    spmat = binned_spiketrain.sparse_matrix
    n_bins = binned_spiketrain.n_bins

    # Check for empty spike trains
    n_spikes_per_row = spmat.sum(axis=1)
    if n_spikes_per_row.min() == 0:
        warnings.warn(
            'Detected empty spike trains (rows) in the binned_spiketrain.')

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


@deprecated_alias(binned_st1='binned_spiketrain_i',
                  binned_st2='binned_spiketrain_j',
                  cross_corr_coef='cross_correlation_coefficient')
def cross_correlation_histogram(
        binned_spiketrain_i, binned_spiketrain_j, window='full',
        border_correction=False, binary=False, kernel=None, method='speed',
        cross_correlation_coefficient=False):
    """
    Computes the cross-correlation histogram (CCH) between two binned spike
    trains `binned_spiketrain_i` and `binned_spiketrain_j`.

    Visualization of this function is covered in Viziphant:
    :func:`viziphant.spike_train_correlation.plot_cross_correlation_histogram`.

    Parameters
    ----------
    binned_spiketrain_i, binned_spiketrain_j : BinnedSpikeTrain
        Binned spike trains of lengths N and M to cross-correlate - the output
        of :class:`elephant.conversion.BinnedSpikeTrain`. The input
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
        Default: 'full'
    border_correction : bool, optional
        whether to correct for the border effect. If True, the value of the
        CCH at bin :math:`b` (for :math:`b=-H,-H+1, ...,H`, where :math:`H` is
        the CCH half-length) is multiplied by the correction factor:

        .. math::
                            (H+1)/(H+1-|b|),

        which linearly corrects for loss of bins at the edges.
        Default: False
    binary : bool, optional
        If True, spikes falling in the same bin are counted as a single spike;
        otherwise they are counted as different spikes.
        Default: False
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
        Default: None
    method : {'speed', 'memory'}, optional
        Defines the algorithm to use. "speed" uses `numpy.correlate` to
        calculate the correlation between two binned spike trains using a
        non-sparse data representation. Due to various optimizations, it is the
        fastest realization. In contrast, the option "memory" uses an own
        implementation to calculate the correlation based on sparse matrices,
        which is more memory efficient but slower than the "speed" option.
        Default: "speed"
    cross_correlation_coefficient : bool, optional
        If True, a normalization is applied to the CCH to obtain the
        cross-correlation  coefficient function ranging from -1 to 1 according
        to Equation (5.10) in :cite:`correlation-Eggermont2010_77`. See Notes.
        Default: False

    Returns
    -------
    cch_result : neo.AnalogSignal
        Containing the cross-correlation histogram between
        `binned_spiketrain_i` and `binned_spiketrain_j`.

        Offset bins correspond to correlations at delays equivalent
        to the differences between the spike times of `binned_spiketrain_i` and
        those of `binned_spiketrain_j`: an entry at positive lag corresponds to
        a spike in `binned_spiketrain_j` following a spike in
        `binned_spiketrain_i` bins to the right, and an entry at negative lag
        corresponds to a spike in `binned_spiketrain_i` following a spike in
        `binned_spiketrain_j`.

        To illustrate this definition, consider two spike trains with the same
        `t_start` and `t_stop`:
        `binned_spiketrain_i` ('reference neuron') : 0 0 0 0 1 0 0 0 0 0 0
        `binned_spiketrain_j` ('target neuron')    : 0 0 0 0 0 0 0 1 0 0 0
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
    1. The Eq. (5.10) in :cite:`correlation-Eggermont2010_77` is valid for
       binned spike trains with at most one spike per bin. For a general case,
       refer to the implementation of `_covariance_sparse()`.
    2. Alias: `cch`

    Examples
    --------
    Plot the cross-correlation histogram between two Poisson spike trains

    >>> import elephant
    >>> import quantities as pq
    >>> import numpy as np
    >>> from elephant.conversion import BinnedSpikeTrain
    >>> from elephant.spike_train_generation import homogeneous_poisson_process
    >>> from elephant.spike_train_correlation import \
    ... cross_correlation_histogram

    >>> np.random.seed(1)
    >>> binned_spiketrain_i = BinnedSpikeTrain(
    ...        homogeneous_poisson_process(
    ...            10. * pq.Hz, t_start=0 * pq.ms, t_stop=5000 * pq.ms),
    ...        bin_size=5. * pq.ms)
    >>> binned_spiketrain_j = BinnedSpikeTrain(
    ...        homogeneous_poisson_process(
    ...            10. * pq.Hz, t_start=0 * pq.ms, t_stop=5000 * pq.ms),
    ...        bin_size=5. * pq.ms)

    >>> cc_hist, lags = cross_correlation_histogram(
    ...        binned_spiketrain_i, binned_spiketrain_j, window=[-10, 10],
    ...        border_correction=False,
    ...        binary=False, kernel=None)
    >>> print(cc_hist.flatten())
    [ 5.  3.  3.  2.  4.  0.  1.  5.  3.  4.  2.  2.  2.  5.
      1.  2.  4.  2. -0.  3.  3.] dimensionless
    >>> lags
    array([-10,  -9,  -8,  -7,  -6,  -5,  -4,  -3,  -2,  -1,
         0,   1,   2,   3,   4,   5,   6,   7,   8,   9,
        10], dtype=int32)

    """

    # Check that the spike trains are binned with the same temporal
    # resolution
    if binned_spiketrain_i.shape[0] != 1 or \
            binned_spiketrain_j.shape[0] != 1:
        raise ValueError("Spike trains must be one dimensional")

    # rescale to the common units
    # this does not change the data - only its representation
    binned_spiketrain_j.rescale(binned_spiketrain_i.units)

    if not np.isclose(binned_spiketrain_i._bin_size,
                      binned_spiketrain_j._bin_size):
        raise ValueError("Bin sizes must be equal")

    bin_size = binned_spiketrain_i._bin_size
    left_edge_min = -binned_spiketrain_i.n_bins + 1
    right_edge_max = binned_spiketrain_j.n_bins - 1

    t_lags_shift = (binned_spiketrain_j._t_start -
                    binned_spiketrain_i._t_start) / bin_size
    if not np.isclose(t_lags_shift, round(t_lags_shift)):
        # For example, if bin_size=1 ms, binned_spiketrain_i.t_start=0 ms, and
        # binned_spiketrain_j.t_start=0.5 ms then there is a global shift in
        # the binning of the spike trains.
        raise ValueError(
            "Binned spiketrains time shift is not multiple of bin_size")
    t_lags_shift = int(round(t_lags_shift))

    # In the examples below we fix st2 and "move" st1.
    # Zero-lag is equal to `max(st1.t_start, st2.t_start)`.
    # Binned spiketrains (t_start and t_stop) with bin_size=1ms:
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
    if len(window) == 2 and np.issubdtype(type(window[0]), np.integer) \
            and np.issubdtype(type(window[1]), np.integer):
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
        lags = _CrossCorrHist.get_valid_lags(binned_spiketrain_i,
                                             binned_spiketrain_j)
        left_edge, right_edge = lags[(0, -1), ]
        cch_mode = window
    else:
        raise ValueError("Invalid window parameter")

    if binary:
        binned_spiketrain_i = binned_spiketrain_i.binarize()
        binned_spiketrain_j = binned_spiketrain_j.binarize()

    cch_builder = _CrossCorrHist(binned_spiketrain_i, binned_spiketrain_j,
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
    if cross_correlation_coefficient:
        cross_corr = cch_builder.cross_correlation_coefficient(cross_corr)

    normalization = 'normalized' if cross_correlation_coefficient else 'counts'
    annotations = dict(window=window, border_correction=border_correction,
                       binary=binary, kernel=kernel is not None,
                       normalization=normalization)
    annotations = dict(cch_parameters=annotations)

    # Transform the array count into an AnalogSignal
    t_start = pq.Quantity((lags[0] - 0.5) * bin_size,
                          units=binned_spiketrain_i.units, copy=False)
    cch_result = neo.AnalogSignal(
        signal=np.expand_dims(cross_corr, axis=1),
        units=pq.dimensionless,
        t_start=t_start,
        sampling_period=binned_spiketrain_i.bin_size, copy=False,
        **annotations)
    return cch_result, lags


# Alias for common abbreviation
cch = cross_correlation_histogram


@deprecated_alias(spiketrain_1='spiketrain_i', spiketrain_2='spiketrain_j')
def spike_time_tiling_coefficient(spiketrain_i, spiketrain_j, dt=0.005 * pq.s):
    """
    Calculates the Spike Time Tiling Coefficient (STTC) as described in
    :cite:`correlation-Cutts2014_14288` following their implementation in C.
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
    the original code by C. Cutts written in C and available `here
    <https://github.com/CCutts/Detecting_pairwise_correlations_in_spike_trains/
    blob/master/spike_time_tiling_coefficient.c>`_:

    Parameters
    ----------
    spiketrain_i, spiketrain_j : neo.SpikeTrain
        Spike trains to cross-correlate. They must have the same `t_start` and
        `t_stop`.
    dt : pq.Quantity.
        The synchronicity window is used for both: the quantification of the
        proportion of total recording time that lies `[-dt, +dt]` of each spike
        in each train and the proportion of spikes in `spiketrain_i` that lies
        `[-dt, +dt]` of any spike in `spiketrain_j`.
        Default : `0.005 * pq.s`

    Returns
    -------
    index : float or np.nan
        The spike time tiling coefficient (STTC). Returns np.nan if any spike
        train is empty.

    Notes
    -----
    Alias: `sttc`

    Examples
    --------
    >>> import neo
    >>> import quantities as pq
    >>> from elephant.spike_train_correlation import \
    ...    spike_time_tiling_coefficient

    >>> spiketrain1 = neo.SpikeTrain([1.3, 7.56, 15.87, 28.23, 30.9, 34.2,
    ...     38.2, 43.2], units='ms', t_stop=50)
    >>> spiketrain2 = neo.SpikeTrain([1.02, 2.71, 18.82, 28.46, 28.79, 43.6],
    ...     units='ms', t_stop=50)
    >>> spike_time_tiling_coefficient(spiketrain1, spiketrain2)
    0.4958601655933762

    """

    def run_P(spiketrain_i, spiketrain_j):
        """
        Check every spike in train 1 to see if there's a spike in train 2
        within dt
        """
        N2 = len(spiketrain_j)

        # Search spikes of spiketrain_i in spiketrain_j
        # ind will contain index of
        ind = np.searchsorted(spiketrain_j.times, spiketrain_i.times)

        # To prevent IndexErrors
        # If a spike of spiketrain_i is after the last spike of spiketrain_j,
        # the index is N2, however spiketrain_j[N2] raises an IndexError.
        # By shifting this index, the spike of spiketrain_i will be compared
        # to the last 2 spikes of spiketrain_j (negligible overhead).
        # Note: Not necessary for index 0 that will be shifted to -1,
        # because spiketrain_j[-1] is valid (additional negligible comparison)
        ind[ind == N2] = N2 - 1

        # Compare to nearest spike in spiketrain_j BEFORE spike in spiketrain_i
        close_left = np.abs(
            spiketrain_j.times[ind - 1] - spiketrain_i.times) <= dt
        # Compare to nearest spike in spiketrain_j AFTER (or simultaneous)
        # spike in spiketrain_j
        close_right = np.abs(
            spiketrain_j.times[ind] - spiketrain_i.times) <= dt

        # spiketrain_j spikes that are in [-dt, dt] range of spiketrain_i
        # spikes are counted only ONCE (as per original implementation)
        close = close_left + close_right

        # Count how many spikes in spiketrain_i have a "partner" in
        # spiketrain_j
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

    N1 = len(spiketrain_i)
    N2 = len(spiketrain_j)

    if N1 == 0 or N2 == 0:
        index = np.nan
    else:
        TA = run_T(spiketrain_i)
        TB = run_T(spiketrain_j)
        PA = run_P(spiketrain_i, spiketrain_j)
        PA = PA / N1
        PB = run_P(spiketrain_j, spiketrain_i)
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


@deprecated_alias(binned_st='binned_spiketrain', tau_max='max_tau')
def spike_train_timescale(binned_spiketrain, max_tau):
    r"""
    Calculates the auto-correlation time of a binned spike train; uses the
    definition of the auto-correlation time proposed in
    :cite:`correlation-Wieland2015_040901` (Eq. 6):

    .. math::
        \tau_\mathrm{corr} = \int_{-\tau_\mathrm{max}}^{\tau_\mathrm{max}}\
            \left[ \frac{\hat{C}(\tau)}{\hat{C}(0)} \right]^2 d\tau

    where :math:`\hat{C}(\tau) = C(\tau)-\nu\delta(\tau)` denotes
    the auto-correlation function excluding the Dirac delta at zero timelag.

    Parameters
    ----------
    binned_spiketrain : elephant.conversion.BinnedSpikeTrain
        A binned spike train containing the spike train to be evaluated.
    max_tau : pq.Quantity
        Maximal integration time :math:`\tau_{max}` of the auto-correlation
        function. It needs to be a multiple of the `bin_size` of
        `binned_spiketrain`.

    Returns
    -------
    timescale : pq.Quantity
        The auto-correlation time of the binned spiketrain with the same units
        as in the input. If `binned_spiketrain` has less than 2 spikes, a
        warning is raised and `np.nan` is returned.

    Notes
    -----
    * :math:`\tau_\mathrm{max}` is a critical parameter: numerical estimates
      of the auto-correlation functions are inherently noisy. Due to the
      square in the definition above, this noise is integrated. Thus, it is
      necessary to introduce a cutoff for the numerical integration - this
      cutoff should be neither smaller than the true auto-correlation time
      nor much bigger.
    * The bin size of `binned_spiketrain` is another critical parameter as it
      defines the discretization of the integral :math:`d\tau`. If it is too
      big, the numerical approximation of the integral is inaccurate.

    Examples
    --------
    >>> import neo
    >>> import numpy as np
    >>> import quantities as pq
    >>> from elephant.spike_train_correlation import spike_train_timescale
    >>> from elephant.conversion import BinnedSpikeTrain
    >>> spiketrain = neo.SpikeTrain([1, 5, 7, 8], units='ms', t_stop=10*pq.ms)
    >>> bst = BinnedSpikeTrain(spiketrain, bin_size=1 * pq.ms)
    >>> spike_train_timescale(bst, max_tau=5 * pq.ms)
    array(14.11111111) * ms

    """
    if binned_spiketrain.get_num_of_spikes() < 2:
        warnings.warn("Spike train contains less than 2 spikes! "
                      "np.nan will be returned.")
        return np.nan

    bin_size = binned_spiketrain._bin_size
    try:
        max_tau = max_tau.rescale(binned_spiketrain.units).item()
    except (AttributeError, ValueError):
        raise ValueError("max_tau needs units of time")

    # safe casting of max_tau/bin_size to integer
    max_tau_bins = int(round(max_tau / bin_size))
    if not np.isclose(max_tau, max_tau_bins * bin_size):
        raise ValueError("max_tau has to be a multiple of the bin_size")

    cch_window = [-max_tau_bins, max_tau_bins]
    corrfct, bin_ids = cross_correlation_histogram(
        binned_spiketrain, binned_spiketrain, window=cch_window,
        cross_correlation_coefficient=True
    )
    # Take only t > 0 values, in particular neglecting the delta peak.
    start_id = corrfct.time_index((bin_size / 2) * binned_spiketrain.units)
    corrfct = corrfct.magnitude.squeeze()[start_id:]

    # Calculate the timescale using trapezoidal integration
    integr = (corrfct / corrfct[0]) ** 2
    timescale = 2 * integrate.trapz(integr, dx=bin_size)
    return pq.Quantity(timescale, units=binned_spiketrain.units, copy=False)
