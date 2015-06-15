# -*- coding: utf-8 -*-
"""
Spike train correlation

This modules provides functions to calculate correlations between spike trains.

:copyright: Copyright 2015 by the Elephant team, see AUTHORS.txt.
:license: Modified BSD, see LICENSE.txt for details.
"""
from __future__ import division
import numpy as np

def crosscorrelogram(binned_st1, binned_st2, win, chance_corrected=False):
    '''
    Calculate cross-correlogram for a pair of binned spike train. To
    caluculate auto-correlogram use the same spike train for both.

    Parameters
    ----------
    binned_st1 : elephant.conversion.BinnedSpikeTrain
        A binned spike train containing the 'post-synaptic' spikes.
    binned_st2 : elephant.conversion.BinnedSpikeTrain
        A binned spike train containing the reference ('pre-synaptic') spikes.
    win : sequence of lenght 2
        Window in which the correlogram will be correlated (minimum, maximum lag)
    chance_corrected : bool, default True
        Whether to correct for chance coincidences.

    Returns
    -------
    lags : ndarray
        Array of time lags. Useful for plotting
    xcorr : ndarray
        Array of cross-correlogram values; one per time lag.

    Examples
    --------

    Generate Poisson spike train

    >>> from quantities import Hz, ms
    >>> from elephant.spike_train_generation import homogeneous_poisson_process
    >>> st1 = homogeneous_poisson_process(rate=10.0*Hz, t_stop=10000*ms)

    Generate a second spike train by adding some jitter.

    >>> import numpy as np
    >>> st2 = st1.copy()
    >>> st2.times[:] += np.random.randn(len(st1)) * 5 * ms

    Bin spike trains
    >>> from elephant.conversion import BinnedSpikeTrain
    >>> st1b = BinnedSpikeTrain(st1, binsize = 1 * ms)
    >>> st2b = BinnedSpikeTrain(st2, binsize = 1 * ms)

    Calculate auto- and cross-correlogram

    >>> lags, acorr = crosscorrelogram(st1b, st1b, [-100*ms, 100*ms])
    >>> _, xcorr = crosscorrelogram(st1b, st2b, [-100*ms, 100*ms])

    Plot them

    >>> import matplotlib.pyplot as plt
    >>> plt.plot(lags, xcorr)
    >>> plt.plot(lags, acorr)


    Notes
    -----

    *Algorithm*

    The algorithm is implemented as a convolution between binned spike train.
    We trim the spike trains according to the selected correlogram window.
    This allows us to avoid edge effects due to undersampling of long
    inter-spike intervals, but also removes some data from calculation, which
    may be considerable amount for long windows. This method also improves
    the performance since we do not have to calculate correlogram for all
    possible lags, but only the selected ones.
    
    *Normalisation*

    By default normalisation is set such that for perfectly synchronised
    spike train (same spike train passed in binned_st1 and binned_st2) the
    maximum correlogram (at lag 0) is 1.

    If the chance_coincidences == True than the expected coincidence rate is
    subracted, such that the  expected correlogram for non-correlated spike
    train is 0.  '''

    assert binned_st1.matrix_rows == 1, "spike train must be one dimensional"
    assert binned_st2.matrix_rows == 1, "spike train must be one dimensional"
    assert binned_st1.binsize == binned_st2.binsize, "bin sizes must be equal"

    st1_arr = binned_st1.to_array()[0,:]
    st2_arr = binned_st2.to_array()[0,:]

    binsize = binned_st1.binsize
    
    def _xcorr(x, y, win, dt):

        l,r = int(win[0]/dt), int(win[1]/dt)
        n = len(x)
        # trim trains to have appropriate length of xcorr array
        if l<0:
            y = y[-l:]
        else:
            x = x[l:]
        y = y[:-r]
        mx, my = x.mean(), y.mean()
        #TODO: possibly use fftconvolve for faster calculation
        corr = np.convolve(x, y[::-1], 'valid')
        # correct for chance coincidences
        #mx = np.convolve(x, np.ones(len(y)), 'valid') / len(y)
        corr = corr / np.sum(y)

        if chance_corrected:
            corr = corr - mx


        lags = np.r_[l:r+1]
        return lags * dt, corr

    return _xcorr(st1_arr, st2_arr, win, binsize)


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
