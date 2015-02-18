# -*- coding: utf-8 -*-
"""
Spike train correlation

This modules provides functions to calculate correlations between spike trains.

:copyright: Copyright 2015 by the Elephant team, see AUTHORS.txt.
:license: Modified BSD, see LICENSE.txt for details.
"""
from __future__ import division
import numpy as np
import quantities as pq
import neo


def corrcoef(binned_sts, binary=False):
    '''
    Calculate the NxN matrix of pairwise Pearson's correlation coefficients
    between all combinations of N binned spike trains.

    For each pair of spike trains :math:`(i,j)`, the correlation coefficient :math:`C[i,j]`
    is given by the correlation coefficient between the vectors obtained by
    binning :math:`i` and :math:`j` at the desired bin size. Let :math:`b_i` and :math:`b_j` denote the
    binary vectors and :math:`m_i` and  :math:`m_j` their respective averages. Then

    .. math::
         C[i,j] = <b_i-m_i, b_j-m_j> /
                      \sqrt{<b_i-m_i, b_i-m_i>*<b_j-m_j,b_j-m_j>}

    where <..,.> is the scalar product of two vectors.

    For an input of n spike trains, a n x n matrix is returned.
    Each entry in the matrix is a real number ranging between -1 (perfectly
    anti-correlated spike trains) and +1 (perfectly correlated spike trains).

    If binary is True, the binned spike trains are clipped before computing the
    correlation coefficients, so that the binned vectors :math:`b_i` and :math:`b_j` are binary.

    Parameters
    ----------
    binned_sts : elephant.conversion.BinnedSpikeTrain
        A binned spike train containing the spike trains to be evaluated.
    binary : bool, optional
        If True, two spikes of a particular spike train falling in the same
        bin are counted as 1, resulting in binary binned vectors :math:`b_i`. If False,
        the binned vectors :math:`b_i` contain the spike counts per bin.
        Default: False

    Returns
    -------
    C : ndarrray
        The square matrix of correlation coefficients. The element
        :math:`C[i,j]=C[j,i]` is the Pearson's correlation coefficient between
        binned_sts[i] and binned_sts[j]. If binned_sts contains only one
        SpikeTrain, C=1.0.

    Examples
    --------
    Generate two Poisson spike trains

    >>> from elephant.spike_train_generation import homogeneous_poisson_process
    >>> st1 = homogeneous_poisson_process(rate=10.0*Hz, t_start=0.0*s, t_stop=10.0*s)
    >>> st2 = homogeneous_poisson_process(rate=10.0*Hz, t_start=0.0*s, t_stop=10.0*s)

    Calculate the correlation matrix.

    >>> from elephant.conversion import BinnedSpikeTrain
    >>> cc_matrix = corrcoef(BinnedSpikeTrain([st1, st2], binsize=5*ms))

    The correlation coefficient between the spike trains is stored in
    cc_matrix[0,1] (or cc_matrix[1,0])

    Notes
    -----
    * The spike trains in the binned structure are assumed to all cover the
      complete time span of binned_sts [t_start,t_stop).
    '''
    num_neurons = binned_sts.matrix_rows

    # Pre-allocate correlation matrix
    C = np.zeros((num_neurons, num_neurons))

    # Retrieve unclipped matrix
    spmat = binned_sts.to_sparse_array()

    # For each row, extract the nonzero column indices and the corresponding
    # data in the matrix (for performance reasons)
    bin_idx_unique = []
    bin_counts_unique = []
    if binary:
        for s in spmat:
            bin_idx_unique.append(s.nonzero()[1])
    else:
        for s in spmat:
            bin_counts_unique.append(s.data)

    # All combinations of spike trains
    for i in range(num_neurons):
        for j in range(i, num_neurons):
            # Number of spikes in i and j
            if binary:
                n_i = len(bin_idx_unique[i])
                n_j = len(bin_idx_unique[j])
            else:
                n_i = np.sum(bin_counts_unique[i])
                n_j = np.sum(bin_counts_unique[j])

            # Enumerator:
            # $$ <b_i-m_i, b_j-m_j>
            #      = <b_i, b_j> + l*m_i*m_j - <b_i, M_j> - <b_j, M_i>
            #      =:    ij     + l*m_i*m_j - n_i * m_j  - n_j * m_i
            #      =     ij     - n_i*n_j/l                         $$
            # where $n_i$ is the spike count of spike train $i$,
            # $l$ is the number of bins used (i.e., length of $b_i$ or $b_j$),
            # and $M_i$ is a vector [m_i, m_i,..., m_i].
            if binary:
                # Intersect indices to identify number of coincident spikes in
                # i and j (more efficient than directly using the dot product)
                ij = len(np.intersect1d(
                    bin_idx_unique[i], bin_idx_unique[j], assume_unique=True))
            else:
                # Calculate dot product b_i*b_j between unclipped matrices
                ij = spmat[i].dot(spmat[j].transpose()).toarray()[0][0]

            cc_enum = ij - n_i * n_j / binned_sts.num_bins

            # Denominator:
            # $$ <b_i-m_i, b_i-m_i>
            #      = <b_i, b_i> + m_i^2 - 2 <b_i, M_i>
            #      =:    ii     + m_i^2 - 2 n_i * m_i
            #      =     ii     - n_i^2 /               $$
            if binary:
                # Here, b_i*b_i is just the number of filled bins (since each
                # filled bin of a clipped spike train has value equal to 1)
                ii = len(bin_idx_unique[i])
                jj = len(bin_idx_unique[j])
            else:
                # directly calculate the dot product based on the counts of all
                # filled entries (more efficient than using the dot product of
                # the rows of the sparse matrix)
                ii = np.dot(bin_counts_unique[i], bin_counts_unique[i])
                jj = np.dot(bin_counts_unique[j], bin_counts_unique[j])

            cc_denom = np.sqrt(
                (ii - (n_i ** 2) / binned_sts.num_bins) *
                (jj - (n_j ** 2) / binned_sts.num_bins))

            # Fill entry of correlation matrix
            C[i, j] = C[j, i] = cc_enum / cc_denom
    return C


def cross_correlation_histogram(
        st1, st2, window=None, normalize=False, border_correction=False,
        binary=False, kernel=None):
    """
    Computes the cross-correlation histogram (CCH) between two binned spike
    trains st1 and st2.

    Parameters
    ----------
    st1,st2 : BinnedSpikeTrain
        Binned spike trains to cross-correlate.
    window : int or None (optional)
        histogram half-length. If specified, the cross-correlation histogram
        has a number of bins equal to 2*window+1 (up to the maximum length).
        If not specified, the full crosscorrelogram is returned
        Default: None
    normalize : bool (optional)
        whether to normalize the central value (corresponding to time lag
        0 s) to 1; the other values are rescaled accordingly.
        Default: False
    border_correction : bool (optional)
        whether to correct for the border effect. If True, the value of the
        CCH at bin b (for b=-H,-H+1, ...,H, where H is the CCH half-length)
        is multiplied by the correction factor:
                            (H+1)/(H+1-|b|),
        which linearly corrects for loss of bins at the edges.
        Default: False
    binary : bool (optional)
        whether to binary spikes from the same spike train falling in the
        same bin. If True, such spikes are considered as a single spike;
        otherwise they are considered as different spikes.
        Default: False.
    kernel : array or None (optional)
        A one dimensional array containing an optional smoothing kernel applied
        to the resulting CCH. The length N of the kernel indicates the
        smoothing window. The smoothing window cannot be larger than the
        maximum lag of the CCH. The kernel is normalized to unit area before
        being applied to the resulting CCH. Popular choices for the kernel are
          * normalized boxcar kernel: numpy.ones(N)
          * hamming: numpy.hamming(N)
          * hanning: numpy.hanning(N)
          * bartlett: numpy.bartlett(N)
        If a kernel is used and normalize is True, the kernel is applied first,
        and the result is normalize to the central bin. If None is specified,
        the CCH is not smoothed.
        Default: None

        TODO:
        * Normalize before or after smoothing -- is this good?

    Returns
    -------
    cch : AnalogSignalArray
        Containing the cross-correlation histogram between st1 and st2.

        The central bin of the histogram represents correlation at zero
        delay. Offset bins correspond to correlations at a delay equivalent
        to the difference between the spike times of st1 and those of st2: an
        entry at positive lags corresponds to a spike in st2 following a
        spike in st1 bins to the right, and an entry at negative lags
        corresponds to a spike in st1 following a spike in st2.

        To illustrate this definition, consider the two spike trains:
        st1: 0 0 0 0 1 0 0 0 0 0 0
        st2: 0 0 0 0 0 0 0 1 0 0 0
        Here, the CCH will have an entry of 1 at lag h=+3.

        Consistent with the definition of AnalogSignalArrays, the time axis
        represents the left bin borders of each histogram bin. For example,
        the time axis might be:
        np.array([-2.5 -1.5 -0.5 0.5 1.5]) * ms
    bin_ids : ndarray of int
        Contains the IDs of the individual histogram bins, where the central
        bin has ID 0, bins the left have negative IDs and bins to the right
        have positive IDs, e.g.,:
        np.array([-3, -2, -1, 0, 1, 2, 3])

    Example
    -------
        Plot the cross-correlation histogram between two Poisson spike trains

        >>> binned_st1 = elephant.conversion.BinnedSpikeTrain(
                elephant.spike_train_generation.homogeneous_poisson_process(
                    10. * pq.Hz, t_start=0 * pq.ms, t_stop=2000 * pq.s),
                binsize=1. * pq.ms)
        >>> binned_st2 = elephant.conversion.BinnedSpikeTrain(
                elephant.spike_train_generation.homogeneous_poisson_process(
                    10. * pq.Hz, t_start=0 * pq.ms, t_stop=2000 * pq.s),
                binsize=1. * pq.ms)

        >>> cc_hist = cross_correlation_histogram(
                binned_st1, binned_st2, window=20,
                normalize=True, border_correction=False,
                binary=False, kernel=None)

        >>> plt.bar(
                left=cc_hist[0].times.magnitude,
                height=cc_hist[0][:, 0].magnitude,
                width=cc_hist[0].sampling_period.magnitude)
        >>> plt.xlabel('time (' + str(cc_hist[0].times.units) + ')')
        >>> plt.ylabel('normalized cross-correlation histogram')
        >>> plt.axis('tight')
        >>> plt.show()

    Alias
    -----
    cch
    """
    if st1.binsize != st2.binsize:
        raise ValueError(
            "Input spike trains must be binned with the same bin size")

    # Retrieve unclipped matrix
    st1_spmat = st1.to_sparse_array()
    st2_spmat = st2.to_sparse_array()

    # For each row, extract the nonzero column indices and the corresponding
    # data in the matrix (for performance reasons)
    st1_bin_idx_unique = st1_spmat.nonzero()[1]
    st2_bin_idx_unique = st2_spmat.nonzero()[1]
    if binary:
        st1_bin_counts_unique = np.array(st1_spmat.data > 0, dtype=int)
        st2_bin_counts_unique = np.array(st2_spmat.data > 0, dtype=int)
    else:
        st1_bin_counts_unique = st1_spmat.data
        st2_bin_counts_unique = st2_spmat.data

    # Define the half-length of the full crosscorrelogram
    #
    # TODO: What is correct here? Why +, not max? How can we have an entry
    # beyond the maximum length of the array?
    hist_half_length = np.max([st1.num_bins, st2.num_bins]) - 1
    hist_length = 2 * hist_half_length + 1
    # hist_length = st1.num_bins + st2.num_bins - 1
    # hist_half_length = hist_length // 2
    if window is None:
        hist_bins = hist_half_length
    else:
        hist_bins = min(window, hist_half_length)

    # Initialize the counts to an array of zeroes, and the bin IDs to integers
    # spanning the time axis
    counts = np.zeros(2 * hist_bins + 1)
    bin_ids = np.arange(-hist_bins, hist_bins + 1)
    # Compute the CCH at lags in -hist_bins,...,hist_bins only
    for r, i in enumerate(st1_bin_idx_unique):
        timediff = st2_bin_idx_unique - i
        timediff_in_range = np.all(
            [timediff >= -hist_bins, timediff <= hist_bins], axis=0)
        timediff = (timediff[timediff_in_range]).reshape((-1,))
        counts[timediff + hist_bins] += st1_bin_counts_unique[r] * \
            st2_bin_counts_unique[timediff_in_range]

    # Correct the values taking into account lacking contributes at the edges
    if border_correction is True:
        correction = float(hist_half_length + 1) / np.array(
            hist_half_length + 1 - abs(np.arange(-hist_bins, hist_bins + 1)),
            float)
        counts = counts * correction

    # Define the kernel for smoothing as an ndarray
    if hasattr(kernel, '__iter__'):
        if len(kernel) > hist_length:
            raise ValueError(
                'The length of the kernel cannot be larger than the '
                'length %d of the resulting CCH.' % hist_length)
        kernel = np.array(kernel, dtype=float)
        kernel = 1. * kernel / sum(kernel)
    elif kernel is not None:
        raise ValueError('Invalid smoothing kernel.')

    # Smooth the cross-correlation histogram with the kernel
    if kernel is not None:
        counts = np.convolve(counts, kernel, mode='same')

    # Rescale the histogram so that the central bin has height 1, if requested
    if normalize:
        counts = np.array(counts, float) / float(counts[hist_bins])

    # Transform the array count into an AnalogSignalArray
    cch = neo.AnalogSignalArray(
        signal=counts.reshape(counts.size, 1),
        units=pq.dimensionless,
        t_start=(bin_ids[0] - 0.5) * st1.binsize,
        sampling_period=st1.binsize)

    # Return only the hist_bins bins and counts before and after the central
    # one
    return cch, bin_ids

# Alias for common abbreviation
cch = cross_correlation_histogram
