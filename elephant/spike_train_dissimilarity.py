# -*- coding: utf-8 -*-
"""
In neuroscience one often wants to evaluate, how similar or dissimilar pairs
or even large sets of spiketrains are. For this purpose various different
spike train dissimilarity measures were introduced in the literature.
They differ, e.g., by the properties of having the mathematical properties of
a metric or by being time-scale dependent or not. Well known representatives
of spike train dissimilarity measures are the Victor-Purpura distance and the
Van Rossum distance implemented in this module, which both are metrics in the
mathematical sense and time-scale dependent.

:copyright: Copyright 2016 by the Elephant team, see AUTHORS.txt.
:license: Modified BSD, see LICENSE.txt for details.
"""

import quantities as pq
import numpy as np
import scipy as sp
import elephant.kernels as kernels
from neo.core import SpikeTrain

# Problem of conversion from Python 2 to Python 3:
# 'xrange' in Python 2 is 'range' in Python 3.
try:
    xrange
except NameError:
    xrange = range


def _create_matrix_from_indexed_function(
        shape, func, symmetric_2d=False, **func_params):
    mat = np.empty(shape)
    if symmetric_2d:
        for i in xrange(shape[0]):
            for j in xrange(i, shape[1]):
                mat[i, j] = mat[j, i] = func(i, j, **func_params)
    else:
        for idx in np.ndindex(*shape):
            mat[idx] = func(*idx, **func_params)
    return mat


def victor_purpura_dist(
        trains, q=1.0 * pq.Hz, kernel=None, sort=True, algorithm='fast'):
    """
    Calculates the Victor-Purpura's (VP) distance. It is often denoted as
    :math:`D^{\\text{spike}}[q]`.

    It is defined as the minimal cost of transforming spike train `a` into
    spike train `b` by using the following operations:

        * Inserting or deleting a spike (cost 1.0).
        * Shifting a spike from :math:`t` to :math:`t'` (cost :math:`q
          \\cdot |t - t'|`).

    A detailed description can be found in
    *Victor, J. D., & Purpura, K. P. (1996). Nature and precision of
    temporal coding in visual cortex: a metric-space analysis. Journal of
    Neurophysiology.*

    Given the average number of spikes :math:`n` in a spike train and
    :math:`N` spike trains the run-time complexity of this function is
    :math:`O(N^2 n^2)` and :math:`O(N^2 + n^2)` memory will be needed.

    Parameters
    ----------
    trains : Sequence of :class:`neo.core.SpikeTrain` objects of
        which the distance will be calculated pairwise.
    q: Quantity scalar
        Cost factor for spike shifts as inverse time scalar.
        Extreme values :math:`q=0` meaning no cost for any shift of
        spikes, or :math: `q=np.inf` meaning infinite cost for any
        spike shift and hence exclusion of spike shifts, are explicitly
        allowed. If `kernel` is not `None`, :math:`q` will be ignored.
        Default: 1.0 * pq.Hz
    kernel: :class:`.kernels.Kernel`
        Kernel to use in the calculation of the distance. If `kernel` is
        `None`, an unnormalized triangular kernel with standard deviation
        of :math:'2.0/(q * sqrt(6.0))' corresponding to a half width of
        :math:`2.0/q` will be used. Usage of the default value calculates
        the Victor-Purpura distance correctly with a triangular kernel of
        the suitable width. The choice of another kernel is enabled, but
        this leaves the framework of Victor-Purpura distances.
        Default: None
    sort: bool
        Spike trains with sorted spike times will be needed for the
        calculation. You can set `sort` to `False` if you know that your
        spike trains are already sorted to decrease calculation time.
        Default: True
    algorithm: string
        Allowed values are 'fast' or 'intuitive', each selecting an
        algorithm with which to calculate the pairwise Victor-Purpura distance.
        Typically 'fast' should be used, because while giving always the
        same result as 'intuitive', within the temporary structure of
        Python and add-on modules as numpy it is faster.
        Default: 'fast'

    Returns
    -------
        2-D array
        Matrix containing the VP distance of all pairs of spike trains.

    Example
    -------
        import elephant.spike_train_dissimilarity_measures as stdm
        q   = 1.0 / (10.0 * pq.ms)
        st_a = SpikeTrain([10, 20, 30], units='ms', t_stop= 1000.0)
        st_b = SpikeTrain([12, 24, 30], units='ms', t_stop= 1000.0)
        vp_f = stdm.victor_purpura_dist([st_a, st_b], q)[0, 1]
        vp_i = stdm.victor_purpura_dist(
                   [st_a, st_b], q, algorithm='intuitive')[0, 1]
    """
    for train in trains:
        if not (isinstance(train, (pq.quantity.Quantity, SpikeTrain)) and
                train.dimensionality.simplified ==
                pq.Quantity(1, "s").dimensionality.simplified):
            raise TypeError("Spike trains must have a time unit.")

    if not (isinstance(q, pq.quantity.Quantity) and
            q.dimensionality.simplified ==
            pq.Quantity(1, "Hz").dimensionality.simplified):
        raise TypeError("q must be a rate quantity.")

    if kernel is None:
        if q == 0.0:
            num_spikes = np.atleast_2d([st.size for st in trains])
            return np.absolute(num_spikes.T - num_spikes)
        elif q == np.inf:
            num_spikes = np.atleast_2d([st.size for st in trains])
            return num_spikes.T + num_spikes
        else:
            kernel = kernels.TriangularKernel(2.0 / (np.sqrt(6.0) * q))

    if sort:
        trains = [np.sort(st.view(type=pq.Quantity)) for st in trains]

    def compute(i, j):
        if i == j:
            return 0.0
        else:
            if algorithm == 'fast':
                return _victor_purpura_dist_for_st_pair_fast(
                    trains[i], trains[j], kernel)
            elif algorithm == 'intuitive':
                return _victor_purpura_dist_for_st_pair_intuitive(
                    trains[i], trains[j], q)
            else:
                raise NameError("algorithm must be either 'fast' "
                                "or 'intuitive'.")

    return _create_matrix_from_indexed_function(
        (len(trains), len(trains)), compute, kernel.is_symmetric())


def _victor_purpura_dist_for_st_pair_fast(train_a, train_b, kernel):
    """
    The algorithm used is based on the one given in

    J. D. Victor and K. P. Purpura, Nature and precision of temporal
    coding in visual cortex: a metric-space analysis, Journal of
    Neurophysiology, 1996.

    It constructs a matrix G[i, j] containing the minimal cost when only
    considering the first i and j spikes of the spike trains. However, one
    never needs to store more than one row and one column at the same time
    for calculating the VP distance.
    cost[0, :cost.shape[1] - i] corresponds to G[i:, i]. In the same way
    cost[1, :cost.shape[1] - i] corresponds to G[i, i:].

    Moreover, the minimum operation on the costs of the three kind of actions
    (delete, insert or move spike) can be split up in two operations. One
    operation depends only on the already calculated costs and kernel
    evaluation (insertion of spike vs moving a spike). The other minimum
    depends on that result and the cost of deleting a spike. This operation
    always depends on the last calculated element in the cost array and
    corresponds to a recursive application of
    f(accumulated_min[i]) = min(f(accumulated_min[i-1]), accumulated_min[i])
    + 1. That '+1' can be excluded from this function if the summed value for
    all recursive applications is added upfront to accumulated_min.
    Afterwards it has to be removed again except one for the currently
    processed spike to get the real costs up to the evaluation of i.

    All currently calculated costs will be considered -1 because this saves
    a number of additions as in most cases the cost would be increased by
    exactly one (the only exception is shifting, but in that calculation is
    already the addition of a constant involved, thus leaving the number of
    operations the same). The increase by one will be added after calculating
    all minima by shifting decreasing_sequence by one when removing it from
    accumulated_min.

    Parameters
    ----------
    train_a, train_b : :class:`neo.core.SpikeTrain` objects of
        which the Victor-Purpura distance will be calculated pairwise.
    kernel: :class:`.kernels.Kernel`
        Kernel to use in the calculation of the distance.

    Returns
    -------
        float
        The Victor-Purpura distance of train_a and train_b
    """

    if train_a.size <= 0 or train_b.size <= 0:
        return max(train_a.size, train_b.size)

    if train_a.size < train_b.size:
        train_a, train_b = train_b, train_a

    min_dim, max_dim = train_b.size, train_a.size + 1
    cost = np.asfortranarray(np.tile(np.arange(float(max_dim)), (2, 1)))
    decreasing_sequence = np.asfortranarray(cost[:, ::-1])
    kern = kernel((np.atleast_2d(train_a).T.view(type=pq.Quantity) -
                   train_b.view(type=pq.Quantity)))
    as_fortran = np.asfortranarray(
        ((np.sqrt(6.0) * kernel.sigma) * kern).simplified)
    k = 1 - 2 * as_fortran

    for i in xrange(min_dim):
        # determine G[i, i] == accumulated_min[:, 0]
        accumulated_min = cost[:, :-i - 1] + k[i:, i]
        accumulated_min[1, :train_b.size - i] = \
            cost[1, :train_b.size - i] + k[i, i:]
        accumulated_min = np.minimum(
            accumulated_min,  # shift
            cost[:, 1:max_dim - i])  # insert
        acc_dim = accumulated_min.shape[1]
        # delete vs min(insert, shift)
        accumulated_min[:, 0] = min(cost[1, 1], accumulated_min[0, 0])
        # determine G[i, :] and G[:, i] by propagating minima.
        accumulated_min += decreasing_sequence[:, -acc_dim - 1:-1]
        accumulated_min = np.minimum.accumulate(accumulated_min, axis=1)
        cost[:, :acc_dim] = accumulated_min - decreasing_sequence[:, -acc_dim:]
    return cost[0, -min_dim - 1]


def _victor_purpura_dist_for_st_pair_intuitive(
                                             train_a, train_b, q=1.0 * pq.Hz):
    """
    Function to calculate the Victor-Purpura distance between two spike trains
    described in *J. D. Victor and K. P. Purpura, Nature and precision of
    temporal coding in visual cortex: a metric-space analysis,
    J Neurophysiol,76(2):1310-1326, 1996*

    This function originates from the spikes-module in the signals-folder
    of the software package Neurotools. It represents the 'intuitive'
    implementation of the Victor-Purpura distance. With respect to calculation
    time at the moment this code is uncompetitive with the code implemented in
    the function _victor_purpura_dist_for_st_pair_fast. However, it is
    expected that the discrepancy in calculation time of the 2 algorithms
    decreases drastically if the temporary valid calculation speed difference
    of plain Python and Numpy routines would be removed when languages like
    cython could take over. The decision then has to be made between an
    intuitive and probably slightly slower algorithm versus a correct but
    strange optimal solution of an optimization problem under boundary
    conditions, where the boundary conditions would finally have been removed.
    Hence also this algoritm is kept here.

    Parameters
    ----------
    train_a, train_b : :class:`neo.core.SpikeTrain` objects of
        which the Victor-Purpura distance will be calculated pairwise.
    q : Quantity scalar of rate dimension
        The cost parameter.
        Default: 1.0 * pq.Hz

    Returns
    -------
        float
        The Victor-Purpura distance of train_a and train_b
    """
    nspk_a = len(train_a)
    nspk_b = len(train_b)
    scr = np.zeros((nspk_a+1, nspk_b+1))
    scr[:, 0] = xrange(0, nspk_a+1)
    scr[0, :] = xrange(0, nspk_b+1)

    if nspk_a > 0 and nspk_b > 0:
        for i in xrange(1, nspk_a+1):
            for j in xrange(1, nspk_b+1):
                scr[i, j] = min(scr[i-1, j]+1, scr[i, j-1]+1)
                scr[i, j] = min(scr[i, j], scr[i-1, j-1] + np.float64((
                               q*abs(train_a[i-1]-train_b[j-1])).simplified))
    return scr[nspk_a, nspk_b]


def van_rossum_dist(trains, tau=1.0 * pq.s, sort=True):
    """
    Calculates the van Rossum distance.

    It is defined as Euclidean distance of the spike trains convolved with a
    causal decaying exponential smoothing filter. A detailed description can
    be found in *Rossum, M. C. W. (2001). A novel spike distance. Neural
    Computation, 13(4), 751-763.* This implementation is normalized to yield
    a distance of 1.0 for the distance between an empty spike train and a
    spike train with a single spike. Divide the result by sqrt(2.0) to get
    the normalization used in the cited paper.

    Given :math:`N` spike trains with :math:`n` spikes on average the run-time
    complexity of this function is :math:`O(N^2 n)`.

    Parameters
    ----------
    trains : Sequence of :class:`neo.core.SpikeTrain` objects of
        which the van Rossum distance will be calculated pairwise.
    tau : Quantity scalar
        Decay rate of the exponential function as time scalar. Controls for
        which time scale the metric will be sensitive. This parameter will
        be ignored if `kernel` is not `None`. May also be :const:`scipy.inf`
        which will lead to only measuring differences in spike count.
        Default: 1.0 * pq.s
    sort : bool
        Spike trains with sorted spike times might be needed for the
        calculation. You can set `sort` to `False` if you know that your
        spike trains are already sorted to decrease calculation time.
        Default: True

    Returns
    -------
        2-D array
        Matrix containing the van Rossum distances for all pairs of
        spike trains.

    Example
    -------
        import elephant.spike_train_dissimilarity_measures as stdm
        tau = 10.0 * pq.ms
        st_a = SpikeTrain([10, 20, 30], units='ms', t_stop= 1000.0)
        st_b = SpikeTrain([12, 24, 30], units='ms', t_stop= 1000.0)
        vr   = stdm.van_rossum_dist([st_a, st_b], tau)[0, 1]
    """
    for train in trains:
        if not (isinstance(train, (pq.quantity.Quantity, SpikeTrain)) and
                train.dimensionality.simplified ==
                pq.Quantity(1, "s").dimensionality.simplified):
            raise TypeError("Spike trains must have a time unit.")

    if not (isinstance(tau, pq.quantity.Quantity) and
            tau.dimensionality.simplified ==
            pq.Quantity(1, "s").dimensionality.simplified):
        raise TypeError("tau must be a time quantity.")

    if tau == 0:
        spike_counts = [st.size for st in trains]
        return np.sqrt(spike_counts + np.atleast_2d(spike_counts).T)
    elif tau == np.inf:
        spike_counts = [st.size for st in trains]
        return np.absolute(spike_counts - np.atleast_2d(spike_counts).T)

    k_dist = _summed_dist_matrix(
        [st.view(type=pq.Quantity) for st in trains], tau, not sort)
    vr_dist = np.empty_like(k_dist)
    for i, j in np.ndindex(k_dist.shape):
        vr_dist[i, j] = (
            k_dist[i, i] + k_dist[j, j] - k_dist[i, j] - k_dist[j, i])
    return sp.sqrt(vr_dist)


def _summed_dist_matrix(spiketrains, tau, presorted=False):
    # The algorithm underlying this implementation is described in
    # Houghton, C., & Kreuz, T. (2012). On the efficient calculation of van
    # Rossum distances. Network: Computation in Neural Systems, 23(1-2),
    # 48-58. We would like to remark that in this paper in formula (9) the
    # left side of the equation should be divided by two.
    #
    # Given N spiketrains with n entries on average the run-time complexity is
    # O(N^2 * n). O(N^2 + N * n) memory will be needed.

    if len(spiketrains) <= 0:
        return np.zeros((0, 0))

    if not presorted:
        spiketrains = [v.copy() for v in spiketrains]
        for v in spiketrains:
            v.sort()

    sizes = np.asarray([v.size for v in spiketrains])
    values = np.empty((len(spiketrains), max(1, sizes.max())))
    values.fill(np.nan)
    for i, v in enumerate(spiketrains):
        if v.size > 0:
            values[i, :v.size] = \
                (v / tau * pq.dimensionless).simplified

    exp_diffs = np.exp(values[:, :-1] - values[:, 1:])
    markage = np.zeros(values.shape)
    for u in xrange(len(spiketrains)):
        markage[u, 0] = 0
        for i in xrange(sizes[u] - 1):
            markage[u, i + 1] = (markage[u, i] + 1.0) * exp_diffs[u, i]

    # Same spiketrain terms
    D = np.empty((len(spiketrains), len(spiketrains)))
    D[np.diag_indices_from(D)] = sizes + 2.0 * np.sum(markage, axis=1)

    # Cross spiketrain terms
    for u in xrange(D.shape[0]):
        all_ks = np.searchsorted(values[u], values, 'left') - 1
        for v in xrange(u):
            js = np.searchsorted(values[v], values[u], 'right') - 1
            ks = all_ks[v]
            slice_j = np.s_[np.searchsorted(js, 0):sizes[u]]
            slice_k = np.s_[np.searchsorted(ks, 0):sizes[v]]
            D[u, v] = np.sum(
                np.exp(values[v][js[slice_j]] - values[u][slice_j]) *
                (1.0 + markage[v][js[slice_j]]))
            D[u, v] += np.sum(
                np.exp(values[u][ks[slice_k]] - values[v][slice_k]) *
                (1.0 + markage[u][ks[slice_k]]))
            D[v, u] = D[u, v]

    return D
