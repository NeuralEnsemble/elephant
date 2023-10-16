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

:copyright: Copyright 2014-2023 by the Elephant team, see `doc/authors.rst`.
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
    -----
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
    >>> from elephant.spike_train_generation import StationaryPoissonProcess
    >>> from elephant.conversion import BinnedSpikeTrain
    >>> from elephant.spike_train_correlation import covariance

    >>> np.random.seed(1)
    >>> st1 = StationaryPoissonProcess(
    ...                  rate=10*pq.Hz, t_stop=10.0*pq.s).generate_spiketrain()
    >>> st2 = StationaryPoissonProcess(
    ...                  rate=10*pq.Hz, t_stop=10.0*pq.s).generate_spiketrain()
    >>> cov_matrix = covariance(BinnedSpikeTrain([st1, st2], bin_size=5*pq.ms))
    >>> cov_matrix # doctest: +SKIP
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
    -----
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
    >>> from elephant.spike_train_generation import StationaryPoissonProcess
    >>> from elephant.conversion import BinnedSpikeTrain
    >>> from elephant.spike_train_correlation import correlation_coefficient

    >>> np.random.seed(1)
    >>> st1 = StationaryPoissonProcess(
    ...                  rate=10*pq.Hz, t_stop=10.0*pq.s).generate_spiketrain()
    >>> st2 = StationaryPoissonProcess(
    ...                  rate=10*pq.Hz, t_stop=10.0*pq.s).generate_spiketrain()
    >>> corrcoef = correlation_coefficient(BinnedSpikeTrain([st1, st2],
    ...     bin_size=5*pq.ms))
    >>> corrcoef # doctest: +SKIP
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


@deprecated_alias(binned_st1='signal_i',
                  binned_st2='signal_j',
                  binned_spiketrain_i='signal_i',
                  binned_spiketrain_j='signal_j',
                  cross_corr_coef='normalize',
                  cross_correlation_coefficient='normalize')
def cross_correlation_histogram(
        signal_i, signal_j, bin_size=None, window='valid',
        border_correction=False, method='speed',
        normalize=False):
    """
    Computes the cross-correlation histogram (CCH) between two signals `signal_i` and `signal_j`.
    :cite:`correlation-Eggermont2010_77`

    Visualization of this function is covered in Viziphant:
    :func:`viziphant.spike_train_correlation.plot_cross_correlation_histogram`.

    Parameters
    ----------
    signal_i, signal_j : BinnedSpikeTrain, np.array
        Signals to cross-correlate. They must have the same `t_start` and `t_stop` and exact same length.
    bin_size : pq.quantity
        Time length of each array entry, required only when the signals are given as np arrays.
    window : {'valid'} or list of int, optional
        ‘valid’: Returns output of length N + 1, since the signals must fully overlap,
              this is equivalent to a full window.
        List of integers (min_lag, max_lag):
              The entries of window are two integers representing the left and
              right extremes (expressed as number of bins) where the
              cross-correlation is computed.
    Default: 'valid'
    border_correction : bool, optional
        whether to correct for the border effect. If True, the value of the
        CCH at bin :math:`b` (for :math:`b=-H,-H+1, ...,H`, where :math:`H` is
        the CCH half-length) is multiplied by the correction factor:

        .. math::
                            (H+1)/(H+1-|b|),

        which linearly corrects for loss of bins at the edges.
        Default: False
    method : {'speed', 'memory'}, optional
        Defines the algorithm to use. "speed" uses `numpy.correlate` to
        calculate the correlation between two binned spike trains using a
        non-sparse data representation. Due to various optimizations, it is the
        fastest realization. In contrast, the option "memory" uses an own
        implementation to calculate the correlation based on sparse matrices,
        which is more memory efficient but slower than the "speed" option.
        Default: 'speed'
    normalize : bool, optional
        If True, a normalization is applied to the CCH to obtain the
        cross-correlation  coefficient function ranging from -1 to 1 according
        to Equation (5.10) in :cite:`correlation-Eggermont2010_77`. See Notes.
        Default: False

    Returns
    -------
    cch_result : neo.AnalogSignal
        Containing the cross-correlation histogram between
        `signal_i` and `signal_j`.

        Offset bins correspond to correlations at delays equivalent
        to the differences between the spike times of `signal_i` and
        those of `signal_j`: an entry at positive lag corresponds to
        a spike in `signal_j` following a spike in
        `signal_i` bins to the right, and an entry at negative lag
        corresponds to a spike in `signal_i` following a spike in
        `signal_j`.

        To illustrate this definition, consider two spike trains with the same
        `t_start` and `t_stop`:
        `signal_i` ('reference neuron') : 0 0 0 0 1 0 0 0 0 0 0
        `signal_j` ('target neuron')    : 0 0 0 0 0 0 0 1 0 0 0
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
    >>> from elephant.spike_train_generation import StationaryPoissonProcess
    >>> from elephant.spike_train_correlation import cross_correlation_histogram # noqa

    >>> np.random.seed(1)
    >>> signal_i = BinnedSpikeTrain(
    ...        StationaryPoissonProcess(
    ...            10. * pq.Hz, t_start=0 * pq.ms, t_stop=5000 * pq.ms).generate_spiketrain(),
    ...        bin_size=5. * pq.ms)
    >>> signal_j = BinnedSpikeTrain(
    ...        StationaryPoissonProcess(
    ...            10. * pq.Hz, t_start=0 * pq.ms, t_stop=5000 * pq.ms).generate_spiketrain(),
    ...        bin_size=5. * pq.ms)

    >>> cc_hist, lags = cross_correlation_histogram(
    ...        signal_i, signal_j, window=[-10, 10],
    ...        border_correction=False, kernel=None)
    >>> print(cc_hist.flatten()) # doctest: +SKIP
    [ 5.  3.  3.  2.  4.  0.  1.  5.  3.  4.  2.  2.  2.  5.
      1.  2.  4.  2. -0.  3.  3.] dimensionless

    >>> lags # doctest: +SKIP
    array([-10,  -9,  -8,  -7,  -6,  -5,  -4,  -3,  -2,  -1,
         0,   1,   2,   3,   4,   5,   6,   7,   8,   9,
        10], dtype=int32)


    """

    """ Checks of input parameters """
    if type(signal_i) is not type(signal_j):
        raise ValueError('Input signals should be of the same type')

    # Rescale units
    if type(signal_i) is not np.ndarray:
        if bin_size is not None:
            raise ValueError('Bin size should only be defined when the input signals are np.ndarray')
        signal_j.rescale(signal_i.units)
        units = signal_i.units
    else:
        units = bin_size.units

    # Checks specific to BinnedSpikeTrains
    if type(signal_i) is BinnedSpikeTrain:
        if signal_i.shape[0] != 1 or \
                signal_j.shape[0] != 1:
            raise ValueError('Signals must be one dimensional')

        # Check that the spike trains are binned with the same resolution
        if not np.isclose(signal_i._bin_size,
                          signal_j._bin_size):
            raise ValueError('Bin sizes must be equal')

        bin_size = signal_i._bin_size*units

        # Check t_start
        if not np.isclose(signal_i._t_start,
                          signal_j._t_start):
            raise ValueError('t_start must be aligned')

        # Convert to arrays
        if method == 'speed':
            signal_i = signal_i.to_array()[0]
            signal_j = signal_j.to_array()[0]
        elif method == 'memory':
            signal_i = signal_i.sparse_matrix[0]
            signal_j = signal_j.sparse_matrix[0]

    if signal_i.shape[-1] != signal_j.shape[-1]:
        raise ValueError('Input signals must have the same number of bins')

    # Extract edges
    left_edge_min = -signal_i.shape[-1] + 1
    right_edge_max = signal_i.shape[-1] - 1

    if len(window) == 2:
        if not np.issubdtype(type(window[0]), np.integer) or \
           not np.issubdtype(type(window[1]), np.integer):
            raise ValueError('Window entries must be integers')
        if window[0] >= window[1]:
            raise ValueError(
                "Window's left edge ({left}) must be lower than the right "
                "edge ({right})".format(left=window[0], right=window[1]))
        left_edge, right_edge = window
        if left_edge < left_edge_min or right_edge > right_edge_max:
            raise ValueError(
                "The window exceeds the length of the spike trains")
        lags = np.arange(window[0], window[1] + 1, dtype=np.int32)
        cch_mode = 'pad'
    elif window == 'valid':
        left_edge = left_edge_min
        right_edge = right_edge_max
        lags = np.arange(left_edge, right_edge + 1, dtype=np.int32)
        cch_mode = 'valid'
    else:
        raise ValueError("Invalid window parameter")

    """ This is where the computations happen """
    if method == 'speed':

        # Zero padding to stay between left_edge and right_edge
        if cch_mode == 'pad':
            pad_width = min(max(-left_edge, 0), max(right_edge, 0))
            signal_j = np.pad(signal_j, pad_width=pad_width, mode='constant')
            cch_mode = 'valid'

        # Cross correlate the spike trains
        cross_corr = scipy.signal.fftconvolve(signal_j, signal_i[::-1],
                                              mode=cch_mode)
        # convolution of integers is integers
        cross_corr = np.round(cross_corr)

    elif method == 'memory':
        # AITOR: This method takes several minutes in my laptop,
        # while the speed one takes 1 second for arrays with over 30K entries.
        # The memory is not an issue at all.

        # extract the nonzero column indices of 1-d matrices
        st1_bin_idx_unique = signal_i.nonzero()[1]
        st2_bin_idx_unique = signal_j.nonzero()[1]

        signal_i = signal_i.data
        signal_j = signal_j.data

        # Initialize the counts to an array of zeros,
        # and the bin IDs to integers spanning the time axis
        nr_lags = right_edge - left_edge + 1
        cross_corr = np.zeros(nr_lags)

        # Compute the CCH at lags in left_edge,...,right_edge only
        for idx, i in enumerate(st1_bin_idx_unique):
            il = np.searchsorted(st2_bin_idx_unique, left_edge + i)
            ir = np.searchsorted(st2_bin_idx_unique, right_edge + i, side='right')
            timediff = st2_bin_idx_unique[il:ir] - i
            assert ((timediff >= left_edge) & (
                timediff <= right_edge)).all(), \
                'Not all the entries of cch lie in the window'
            cross_corr[timediff - left_edge] += (
                signal_i[idx] * signal_j[il:ir])
            st2_bin_idx_unique = st2_bin_idx_unique[il:]
            signal_j = signal_j[il:]

    # AITOR: I do not know what the expected behaviour is
    # if border_correction:
    #     cross_corr = _cch_border_correction(cross_corr)

    # Normalization of the covariance values
    if normalize:
        max_num_bins = signal_i.shape[-1]
        n_spikes1 = np.sum(signal_i)
        n_spikes2 = np.sum(signal_j)
        if method == 'speed':
            ii = signal_i.dot(signal_i)
            jj = signal_j.dot(signal_j)
        elif method == 'memory':
            data1 = signal_i.data
            data2 = signal_j.data
            ii = data1.dot(data1)
            jj = data2.dot(data2)
        cov_mean = n_spikes1 * n_spikes2 / max_num_bins
        std_xy = np.sqrt((ii - n_spikes1 ** 2 / max_num_bins) *
                         (jj - n_spikes2 ** 2 / max_num_bins))
        cross_corr = (cross_corr - cov_mean) / std_xy

    # Create annotations with the parameters used to compute the CCH
    normalization = 'normalized' if normalize else 'counts'
    annotations = dict(window=window, border_correction=border_correction,
                       normalization=normalization)
    annotations = dict(cch_parameters=annotations)

    # Transform the array count into an AnalogSignal
    t_start = pq.Quantity((lags[0] - 0.5) * bin_size,
                          units=units, copy=False)
    cch_result = neo.AnalogSignal(
        signal=np.expand_dims(cross_corr, axis=1),
        units=pq.dimensionless,
        t_start=t_start,
        sampling_period=bin_size, copy=False,
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
    >>> from elephant.spike_train_correlation import spike_time_tiling_coefficient  # noqa

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

        if N == 1:  # for only a single spike in the train

            # Check difference between start of recording and single spike
            if spiketrain[0] - spiketrain.t_start < dt:
                time_A += - dt + spiketrain[0] - spiketrain.t_start

            # Check difference between single spike and end of recording
            elif spiketrain[0] + dt > spiketrain.t_stop:
                time_A += - dt - spiketrain[0] + spiketrain.t_stop

        else:  # if more than a single spike in the train

            # Calculate difference between consecutive spikes
            diff = np.diff(spiketrain)

            # Find spikes whose tiles overlap
            idx = np.where(diff < 2 * dt)[0]
            # Subtract overlapping "2*dt" tiles and add differences instead
            time_A += - 2 * dt * len(idx) + diff[idx].sum()

            # Check if spikes are within +/-dt of the start and/or end
            # if so, subtract overlap of first and/or last spike
            if (spiketrain[0] - spiketrain.t_start) < dt:
                time_A += spiketrain[0] - dt - spiketrain.t_start
            if (spiketrain.t_stop - spiketrain[N - 1]) < dt:
                time_A += - spiketrain[-1] - dt + spiketrain.t_stop

        # Calculate the proportion of total recorded time to "tiled" time
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
