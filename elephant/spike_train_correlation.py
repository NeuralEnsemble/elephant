# -*- coding: utf-8 -*-
"""
Spike train correlation

This modules provides functions to calculate correlations between spike trains.

:copyright: Copyright 2015 by the Elephant team, see AUTHORS.txt.
:license: Modified BSD, see LICENSE.txt for details.
"""
from __future__ import division
import numpy as np


def covariance(binned_sts, binary=False):
    '''
    Calculate the NxN matrix of pairwise covariances between all combinations
    of N binned spike trains.

    For each pair of spike trains :math:`(i,j)`, the covariance :math:`C[i,j]`
    is obtained by binning :math:`i` and :math:`j` at the desired bin size. Let
    :math:`b_i` and :math:`b_j` denote the binary vectors and :math:`m_i` and
    :math:`m_j` their respective averages. Then

    .. math::
         C[i,j] = <b_i-m_i, b_j-m_j> / (l-1)

    where <..,.> is the scalar product of two vectors.

    For an input of n spike trains, a n x n matrix is returned containing the
    covariances for each combination of input spike trains.

    If binary is True, the binned spike trains are clipped to 0 or 1 before
    computing the covariance, so that the binned vectors :math:`b_i` and
    :math:`b_j` are binary.

    Parameters
    ----------
    binned_sts : elephant.conversion.BinnedSpikeTrain
        A binned spike train containing the spike trains to be evaluated.
    binary : bool, optional
        If True, two spikes of a particular spike train falling in the same bin
        are counted as 1, resulting in binary binned vectors :math:`b_i`. If
        False, the binned vectors :math:`b_i` contain the spike counts per bin.
        Default: False

    Returns
    -------
    C : ndarrray
        The square matrix of covariances. The element :math:`C[i,j]=C[j,i]` is
        the covariance between binned_sts[i] and binned_sts[j].

    Examples
    --------
    Generate two Poisson spike trains

    >>> from elephant.spike_train_generation import homogeneous_poisson_process
    >>> st1 = homogeneous_poisson_process(
            rate=10.0*Hz, t_start=0.0*s, t_stop=10.0*s)
    >>> st2 = homogeneous_poisson_process(
            rate=10.0*Hz, t_start=0.0*s, t_stop=10.0*s)

    Calculate the covariance matrix.

    >>> from elephant.conversion import BinnedSpikeTrain
    >>> cov_matrix = covariance(BinnedSpikeTrain([st1, st2], binsize=5*ms))

    The covariance between the spike trains is stored in cc_matrix[0,1] (or
    cov_matrix[1,0]).

    Notes
    -----
    * The spike trains in the binned structure are assumed to all cover the
      complete time span of binned_sts [t_start,t_stop).
    '''
    return __calculate_correlation_or_covariance(
        binned_sts, binary, corrcoef_norm=False)


def corrcoef(binned_sts, binary=False):
    '''
    Calculate the NxN matrix of pairwise Pearson's correlation coefficients
    between all combinations of N binned spike trains.

    For each pair of spike trains :math:`(i,j)`, the correlation coefficient
    :math:`C[i,j]` is obtained by binning :math:`i` and :math:`j` at the
    desired bin size. Let :math:`b_i` and :math:`b_j` denote the binary vectors
    and :math:`m_i` and :math:`m_j` their respective averages. Then

    .. math::
         C[i,j] = <b_i-m_i, b_j-m_j> /
                      \sqrt{<b_i-m_i, b_i-m_i>*<b_j-m_j,b_j-m_j>}

    where <..,.> is the scalar product of two vectors.

    For an input of n spike trains, a n x n matrix is returned.
    Each entry in the matrix is a real number ranging between -1 (perfectly
    anti-correlated spike trains) and +1 (perfectly correlated spike trains).

    If binary is True, the binned spike trains are clipped to 0 or 1 before
    computing the correlation coefficients, so that the binned vectors
    :math:`b_i` and :math:`b_j` are binary.

    Parameters
    ----------
    binned_sts : elephant.conversion.BinnedSpikeTrain
        A binned spike train containing the spike trains to be evaluated.
    binary : bool, optional
        If True, two spikes of a particular spike train falling in the same bin
        are counted as 1, resulting in binary binned vectors :math:`b_i`. If
        False, the binned vectors :math:`b_i` contain the spike counts per bin.
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
    >>> st1 = homogeneous_poisson_process(
            rate=10.0*Hz, t_start=0.0*s, t_stop=10.0*s)
    >>> st2 = homogeneous_poisson_process(
            rate=10.0*Hz, t_start=0.0*s, t_stop=10.0*s)

    Calculate the correlation matrix.

    >>> from elephant.conversion import BinnedSpikeTrain
    >>> cc_matrix = corrcoef(BinnedSpikeTrain([st1, st2], binsize=5*ms))

    The correlation coefficient between the spike trains is stored in
    cc_matrix[0,1] (or cc_matrix[1,0]).

    Notes
    -----
    * The spike trains in the binned structure are assumed to all cover the
      complete time span of binned_sts [t_start,t_stop).
    '''

    return __calculate_correlation_or_covariance(
        binned_sts, binary, corrcoef_norm=True)


def __calculate_correlation_or_covariance(binned_sts, binary, corrcoef_norm):
    '''
    Helper function for covariance() and corrcoef() that performs the complete
    calculation for either the covariance (corrcoef_norm=False) or correlation
    coefficient (corrcoef_norm=True). Both calculations differ only by the
    denominator.

    Parameters
    ----------
    binned_sts : elephant.conversion.BinnedSpikeTrain
        See covariance() or corrcoef(), respectively.
    binary : bool
        See covariance() or corrcoef(), respectively.
    corrcoef_norm : bool
        Use normalization factor for the correlation coefficient rather than
        for the covariance.
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

                # Number of spikes in i and j
                n_i = len(bin_idx_unique[i])
                n_j = len(bin_idx_unique[j])
            else:
                # Calculate dot product b_i*b_j between unclipped matrices
                ij = spmat[i].dot(spmat[j].transpose()).toarray()[0][0]

                # Number of spikes in i and j
                n_i = np.sum(bin_counts_unique[i])
                n_j = np.sum(bin_counts_unique[j])

            enumerator = ij - n_i * n_j / binned_sts.num_bins

            # Denominator:
            if corrcoef_norm:
                # Correlation coefficient

                # Note:
                # $$ <b_i-m_i, b_i-m_i>
                #      = <b_i, b_i> + m_i^2 - 2 <b_i, M_i>
                #      =:    ii     + m_i^2 - 2 n_i * m_i
                #      =     ii     - n_i^2 /               $$
                if binary:
                    # Here, b_i*b_i is just the number of filled bins (since
                    # each filled bin of a clipped spike train has value equal
                    # to 1)
                    ii = len(bin_idx_unique[i])
                    jj = len(bin_idx_unique[j])
                else:
                    # directly calculate the dot product based on the counts of
                    # all filled entries (more efficient than using the dot
                    # product of the rows of the sparse matrix)
                    ii = np.dot(bin_counts_unique[i], bin_counts_unique[i])
                    jj = np.dot(bin_counts_unique[j], bin_counts_unique[j])

                denominator = np.sqrt(
                    (ii - (n_i ** 2) / binned_sts.num_bins) *
                    (jj - (n_j ** 2) / binned_sts.num_bins))
            else:
                # Covariance

                # $$ l-1 $$
                denominator = (binned_sts.num_bins - 1)

            # Fill entry of correlation matrix
            C[i, j] = C[j, i] = enumerator / denominator
    return np.squeeze(C)
