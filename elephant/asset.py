"""
ASSET is a statistical method [1] for the detection of repeating sequences
of synchronous spiking events in parallel spike trains.
Given a list `sts` of spike trains, the analysis comprises the following
steps:

1) Build the intersection matrix `imat` (optional) and the associated
   probability matrix `pmat` with the desired bin size:

       >>> binsize = 5 * pq.ms
       >>> dt = 1 * pq.s
       >>> imat, xedges, yedges = intersection_matrix(sts, binsize, dt, norm=2)
       >>> pmat, xedges, yedges = probability_matrix_analytical(
               sts, binsize, dt)

2) Compute the joint probability matrix jmat, using a suitable filter:

       >>> filter_shape = (5,2)  # filter shape
       >>> nr_neigh = 5  # nr of largest neighbors
       >>> jmat = joint_probability_matrix(pmat, filter_shape, nr_neigh)

3) Create from pmat and jmat a masked version of the intersection matrix:

       >>> alpha1 = 0.99
       >>> alpha2 = 0.99999
       >>> mask = mask_matrices([pmat, jmat], [alpha1, alpha2])

4) Cluster significant elements of imat into diagonal structures ("DSs"):

       >>> epsilon = 10
       >>> minsize = 2
       >>> stretch = 5
       >>> cmat = asset.cluster_matrix_entries(mask, epsilon, minsize, stretch)

5) Extract sequences of synchronous events associated to each worm

       >>> extract_sse(sts, x_edges, y_edges, cmat)

References:

[1] Torre, Canova, Denker, Gerstein, Helias, Gruen (submitted)
"""

from __future__ import division, print_function, unicode_literals

import numpy as np
import scipy.spatial
import scipy.stats
import quantities as pq
import neo
import itertools
import elephant.conversion as conv
import elephant.spike_train_surrogates as spike_train_surrogates
from sklearn.cluster import dbscan as dbscan


# =============================================================================
# Some Utility Functions to be dealt with in some way or another
# =============================================================================


def _signals_same_tstart(signals):
    '''
    Check whether a list of signals (AnalogSignals or SpikeTrains) have same
    attribute t_start. If so return that value. Otherwise raise a ValueError.

    Parameters
    ----------
    signals : list
        a list of signals (e.g. AnalogSignals or SpikeTrains) having
        attribute `t_start`

    Returns
    -------
    t_start : Quantity
        The common attribute `t_start` of the list of signals.
        Raises a `ValueError` if the signals have a different `t_start`.
    '''

    t_start = signals[0].t_start

    if len(signals) == 1:
        return t_start
    else:
        for sig in signals[1:]:
            if sig.t_start != t_start:
                raise ValueError('signals have different t_start values')
        return t_start


def _signals_same_tstop(signals):
    '''
    Check whether a list of signals (AnalogSignals or SpikeTrains) have same
    attribute t_stop. If so return that value. Otherwise raise a ValueError.

    Parameters
    ----------
    signals : list
        a list of signals (e.g. AnalogSignals or SpikeTrains) having
        attribute t_stop

    Returns
    -------
    t_stop : Quantity
        The common attribute t_stop of the list of signals.
        If the signals have a different t_stop, a ValueError is raised.
    '''

    t_stop = signals[0].t_stop

    if len(signals) == 1:
        return t_stop
    else:
        for sig in signals[1:]:
            if sig.t_stop != t_stop:
                raise ValueError('signals have different t_stop values')
        return t_stop


def _quantities_almost_equal(x, y):
    '''
    Returns True if two quantities are almost equal, i.e. if x-y is
    "very close to 0" (not larger than machine precision for floats).

    Note: not the same as numpy.testing.assert_allclose (which does not work
    with Quantities) and numpy.testing.assert_almost_equal (which works only
    with decimals)

    Parameters
    ----------
    x : Quantity
        first Quantity to compare
    y : Quantity
        second Quantity to compare. Must have same unit type as x, but not
        necessarily the same shape. Any shapes of x and y for which x-y can
        be calculated are permitted

    Returns
    -------
    arr : ndarray of bool
        an array of bools, which is True at any position where x-y is almost
        zero
    '''
    eps = np.finfo(float).eps
    relative_diff = (x - y).magnitude
    return np.all([-eps <= relative_diff, relative_diff <= eps], axis=0)


def _transactions(spiketrains, binsize, t_start=None, t_stop=None, ids=None):
    """
    Transform parallel spike trains a into list of sublists, called
    transactions, each corresponding to a time bin and containing the list
    of spikes in spiketrains falling into that bin.

    To compute each transaction, the spike trains are binned (with adjacent
    exclusive binning) and clipped (i.e. spikes from the same train falling
    in the same bin are counted as one event). The list of spike ids within
    each bin form the corresponding transaction.

    Parameters
    ----------
    spiketrains: list of neo.SpikeTrains
        list of neo.core.SpikeTrain objects, or list of pairs
        (Train_ID, SpikeTrain), where Train_ID can be any hashable object
    binsize: quantities.Quantity
        width of each time bin; time is binned to determine synchrony
    t_start: quantity.Quantity, optional
        starting time; only spikes occurring at times t >= t_start are
        considered; the first transaction contains spikes falling into the
        time segment [t_start, t_start+binsize[.
        If None, takes the t_start value of the spike trains in spiketrains
        if the same for all of them, or returns an error.
        Default: None
    t_stop: quantities.Quantity, optional
        ending time; only spikes occurring at times t < t_stop are
        considered.
        If None, takes the t_stop value of the spike trains in spiketrains
        if the same for all of them, or returns an error.
        Default: None
    ids : list or None, optional
        list of spike train IDs. If None, IDs 0 to N-1 are used, where N
        is the number of input spike trains
        Default: None

    Returns
    -------
    trans : list of lists
        a list of transactions; each transaction corresponds to a time bin
        and represents the list of spike trains ids having a spike in that
        time bin.

    """

    # Define the spike trains and their IDs depending on the input arguments
    if all([hasattr(elem, '__iter__') and len(elem) == 2 and
            type(elem[1]) == neo.SpikeTrain for elem in spiketrains]):
        ids = [elem[0] for elem in spiketrains]
        trains = [elem[1] for elem in spiketrains]
    elif all([type(st) == neo.SpikeTrain for st in spiketrains]):
        trains = spiketrains
        if ids is None:
            ids = range(len(spiketrains))
    else:
        raise TypeError('spiketrains must be either a list of ' +
                        'SpikeTrains or a list of (id, SpikeTrain) pairs')

    # Take the minimum and maximum t_start and t_stop of all spike trains
    # TODO: the block below should be ageneral routine in elephant
    tstarts = [xx.t_start for xx in trains]
    tstops = [xx.t_stop for xx in trains]
    max_tstart = max(tstarts)
    min_tstop = min(tstops)

    # Set starting time of binning
    if t_start is None:
        start = _signals_same_tstart(trains)
    elif t_start < max_tstart:
        raise ValueError('Some SpikeTrains have a larger t_start ' +
                         'than the specified t_start value')
    else:
        start = t_start

    # Set stopping time of binning
    if t_stop is None:
        stop = _signals_same_tstop(trains)
    elif t_stop > min_tstop:
        raise ValueError(
            'Some SpikeTrains have a smaller t_stop ' +
            'than the specified t_stop value')
    else:
        stop = t_stop

    # Bin the spike trains and take for each of them the ids of filled bins
    binned = conv.BinnedSpikeTrain(
        trains, binsize=binsize, t_start=start, t_stop=stop)
    Nbins = binned.num_bins

    filled_bins = binned.spike_indices

    # Compute and return the transaction list
    return [[train_id for train_id, b in zip(ids, filled_bins)
             if bin_id in b] for bin_id in range(Nbins)]


def _analog_signal_step_interp(signal, times):
    '''
    Compute the step-wise interpolation of a signal at desired times.

    Given a signal (e.g. an AnalogSignal) s taking value s(t0) and s(t1)
    at two consecutive time points t0 and t1 (t0 < t1), the value of the
    step-wise interpolation at time t: t0 <= t < t1 is given by s(t)=s(t0).

    Parameters
    ----------
    signal : neo.AnalogSignal
        The analog signal containing the discretization of the function to
        interpolate
    times : quantities.Quantity (vector of time points)
        The time points at which the step interpolation is computed

    Returns
    -------
    quantities.Quantity object with same shape of `times`, and containing
    the values of the interpolated signal at the time points in `times`
    '''
    dt = signal.sampling_period

    # Compute the ids of the signal times to the left of each time in times
    time_ids = np.floor(
        ((times - signal.t_start) / dt).rescale(
            pq.dimensionless).magnitude).astype('i')

    return (signal.magnitude[time_ids] * signal.units).rescale(signal.units)


def _sample_quantiles(sample, p):
    r'''
    Given a sample of values extracted from a probability distribution,
    estimates the quantile(s) associated to p-value(s) p.

    Given a r.v. X with probability distribution P defined over a domain D,
    the quantile x associated to the p-value p (0 <= p <= 1) is defined by:
                q(p) = min{x \in D: P(X>=x) < p}
    Given a sample S = {x1, x2, ..., xn} of n realisations of X, an estimate
    of q(p) is given by:
            q = min{x \in S: (#{y \in S: y>=x} / #{sample}) < p}

    For instance, if p = 0.05, calculates the lowest value q in sample such
    that less than 5% other values in sample are higher than q.

    Parameters
    ----------
    sample : ndarray
        an array of sample values, which are pooled to estimate the quantile(s)
    p : float or list or floats or array, all in the range [0, 1]
        p-value(s) for which to compute the quantile(s)

    Returns
    -------
    q : float, or array of floats
        quantile(s) associated to the input p-value(s).
    '''
    # Compute the cumulative probabilities associated to the p-values
    if not isinstance(p, np.ndarray):
        p = np.array([p])
    probs = list((1 - p) * 100.)

    quantiles = np.percentile(sample.flatten(), probs)
    if hasattr(quantiles, '__len__'):
        quantiles = np.array(quantiles)

    return quantiles


def _sample_pvalue(sample, x):
    '''
    Estimates the p-value of each value in x, given a sample of values
    extracted from a probability distribution.

    Given a r.v. X with probability distribution P, the p-value of X at
    the point x is defined as:
    ..math::
                    pv(x) := P(X >= x) = 1 - cdf(x)
    The smaller pv(x), the less likely that x was extracted from the same
    probability distribution of X.
    Given a sample {x1, x2, ..., xn} of n realisations of X, an estimate of
    pv(x) is given by:
                    pv(x) ~ #{i: xi > x} / n

    Parameters
    ----------
    sample : ndarray
        a sample of realisations from a probability distribution
    x : float or list or floats or array
        p-value(s) for which to compute the quantiles

    Returns
    -------
    pv : same type and shape as x
        p-values associated to the input values x.
    '''
    # Convert x to an array
    if not isinstance(x, np.ndarray):
        x = np.array([x])

    # Convert sample to a flattened array
    if not isinstance(sample, np.ndarray):
        sample = np.array([sample])
    sample = sample.flatten()

    # Compute and return the p-values associated to the elements of x
    return np.array([(sample >= xx).sum() for xx in x]) * 1. / len(sample)


def _time_slice(signal, t_start, t_stop):
    '''
    Get the time slice of an AnalogSignal between t_start and t_stop.
    '''

    # Check which elements of the signal are between t_start and t_stop.
    # Retain those and the corresponding times
    elements_to_keep = np.all(
        [signal.times >= t_start, signal.times < t_stop], axis=0)
    times = signal.times[elements_to_keep]

    # Put the retained values and times into a new AnalogSignal
    sliced_signal = neo.AnalogSignal(
        signal[elements_to_keep].view(pq.Quantity), t_start=times[0],
        sampling_period=signal.sampling_period)

    return sliced_signal


# =============================================================================
# HERE ASSET STARTS
# =============================================================================


def intersection_matrix(
        spiketrains, binsize, dt, t_start_x=None, t_start_y=None, norm=None):
    """
    Generates the intersection matrix from a list of spike trains.

    Given a list of SpikeTrains, consider two binned versions of them
    differing for the starting time of the binning (t_start_x and t_start_y
    respectively; the two times can be identical). Then calculate the
    intersection matrix M of the two binned data, where M[i,j] is the overlap
    of bin i in the first binned data and bin j in the second binned data
    (i.e. the number of spike trains spiking both at bin i and at bin j).
    The matrix  entries can be normalized to values between 0 and 1 via
    different normalizations (see below).

    Parameters
    ----------
    spiketrains : list of neo.SpikeTrains
        list of SpikeTrains from which to compute the intersection matrix
    binsize : quantities.Quantity
        size of the time bins used to define synchronous spikes in the given
        SpikeTrains.
    dt : quantities.Quantity
        time span for which to consider the given SpikeTrains
    t_start_x : quantities.Quantity, optional
        time start of the binning for the first axis of the intersection
        matrix, respectively.
        If None (default) the attribute t_start of the SpikeTrains is used
        (if the same for all spike trains).
        Default: None
    t_start_y : quantities.Quantity, optional
        time start of the binning for the second axis of the intersection
        matrix
    norm : int, optional
        type of normalization to be applied to each entry [i,j] of the
        intersection matrix. Given the sets s_i, s_j of neuron ids in the
        bins i, j respectively, the normalisation coefficient can be:

            * norm = 0 or None: no normalisation (row counts)
            * norm = 1: len(intersection(s_i, s_j))
            * norm = 2: sqrt(len(s_1) * len(s_2))
            * norm = 3: len(union(s_i, s_j))

        Default: None

    Returns
    -------
    imat : numpy.ndarray of floats
        the intersection matrix of a list of spike trains. Has shape (n,n),
        where n is the number of bins time was discretized in.
    x_edges : numpy.ndarray
        edges of the bins used for the horizontal axis of imat. If imat is
        a matrix of shape (n, n), x_edges has length n+1
    y_edges : numpy.ndarray
        edges of the bins used for the vertical axis of imat. If imat is
        a matrix of shape (n, n), y_edges has length n+1
    """
    # Setting the start and stop time for the x and y axes:
    if t_start_x is None:
        t_start_x = _signals_same_tstart(spiketrains)
    if t_start_y is None:
        t_start_y = _signals_same_tstart(spiketrains)

    t_stop_x = dt + t_start_x
    t_stop_y = dt + t_start_y

    # Check that all SpikeTrains are defined until t_stop at least
    t_stop_max = max(t_stop_x, t_stop_y)
    for i, st in enumerate(spiketrains):
        if not (st.t_stop > t_stop_max or
                _quantities_almost_equal(st.t_stop, t_stop_max)):
            msg = 'SpikeTrain %d is shorter than the required time ' % i + \
                  'span: t_stop (%s) < %s' % (st.t_stop, t_stop_max)
            raise ValueError(msg)

    # For both x and y axis, cut all SpikeTrains between t_start and t_stop
    sts_x = [st.time_slice(t_start=t_start_x, t_stop=t_stop_x)
             for st in spiketrains]
    sts_y = [st.time_slice(t_start=t_start_y, t_stop=t_stop_y)
             for st in spiketrains]

    # Compute imat either by matrix multiplication (~20x faster) or by
    # nested for loops (more memory efficient)
    try:  # try the fast version
        # Compute the binned spike train matrices, along both time axes
        bsts_x = conv.BinnedSpikeTrain(
            sts_x, binsize=binsize,
            t_start=t_start_x, t_stop=t_stop_x).to_bool_array()
        bsts_y = conv.BinnedSpikeTrain(
            sts_y, binsize=binsize,
            t_start=t_start_y, t_stop=t_stop_y).to_bool_array()

        # Compute the number of spikes in each bin, for both time axes
        spikes_per_bin_x = bsts_x.sum(axis=0)
        spikes_per_bin_y = bsts_y.sum(axis=0)

        # Compute the intersection matrix imat
        N_bins = len(spikes_per_bin_x)
        imat = np.zeros((N_bins, N_bins), dtype=float)
        for ii in range(N_bins):
            # Compute the ii-th row of imat
            bin_ii = bsts_x[:, ii].reshape(-1, 1)
            imat[ii, :] = (bin_ii * bsts_y).sum(axis=0)
            # Normalize the row according to the specified normalization
            if norm == 0 or norm is None or bin_ii.sum() == 0:
                norm_coef = 1.
            elif norm == 1:
                norm_coef = np.minimum(
                    spikes_per_bin_x[ii], spikes_per_bin_y)
            elif norm == 2:
                norm_coef = np.sqrt(
                    spikes_per_bin_x[ii] * spikes_per_bin_y)
            elif norm == 3:
                norm_coef = ((bin_ii + bsts_y) > 0).sum(axis=0)
            imat[ii, :] = imat[ii, :] / norm_coef

        # If normalization required, for each j such that bsts_y[j] is
        # identically 0 the code above sets imat[:, j] to identically nan.
        # Substitute 0s instead. Then refill the main diagonal with 1s.
        if norm and norm >= 0.5:
            ybins_equal_0 = np.where(spikes_per_bin_y == 0)[0]
            for y_id in ybins_equal_0:
                imat[:, y_id] = 0

            np.fill_diagonal(imat, val=1.)

    except MemoryError:  # use the memory-efficient version
        # Compute the list spiking neurons per bin, along both axes
        ids_per_bin_x = _transactions(
            sts_x, binsize, t_start=t_start_x, t_stop=t_stop_x)
        ids_per_bin_y = _transactions(
            sts_y, binsize, t_start=t_start_y, t_stop=t_stop_y)

        # Generate the intersection matrix
        N_bins = len(ids_per_bin_x)
        imat = np.zeros((N_bins, N_bins))
        for ii in range(N_bins):
            for jj in range(N_bins):
                if len(ids_per_bin_x[ii]) * len(ids_per_bin_y[jj]) != 0:
                    imat[ii, jj] = len(set(ids_per_bin_x[ii]).intersection(
                        set(ids_per_bin_y[jj])))
                    # Normalise according to the desired normalisation type:
                    if norm == 1:
                        imat[ii, jj] /= float(min(len(ids_per_bin_x[ii]),
                                                  len(ids_per_bin_y[jj])))
                    elif norm == 2:
                        imat[ii, jj] /= np.sqrt(float(
                            len(ids_per_bin_x[ii]) * len(ids_per_bin_y[jj])))
                    elif norm == 3:
                        imat[ii, jj] /= float(len(set(
                            ids_per_bin_x[ii]).union(set(ids_per_bin_y[jj]))))

    # Compute the time edges corresponding to the binning employed
    t_start_x_dl = t_start_x.rescale(binsize.units).magnitude
    t_start_y_dl = t_start_y.rescale(binsize.units).magnitude
    t_stop_x_dl = t_stop_x.rescale(binsize.units).magnitude
    t_stop_y_dl = t_stop_y.rescale(binsize.units).magnitude
    xx = np.linspace(t_start_x_dl, t_stop_x_dl, N_bins + 1) * binsize.units
    yy = np.linspace(t_start_y_dl, t_stop_y_dl, N_bins + 1) * binsize.units

    # Return the intersection matrix and the edges of the bins used for the
    # x and y axes, respectively.
    return imat, xx, yy


def _reference_diagonal(x_edges, y_edges):
    '''
    Given two arrays of time bin edges :math:`x_edges = (X_1, X_2, ..., X_k)`
    and :math:`y_edges = (Y_1, Y_2, ..., Y_k)`, considers the matrix `M`
    such that :math:`M_{ij} = (X_i, Y_j)` and finds the reference diagonal of
    M, i.e. the diagonal of M whose elements are of the type `(a, a)`.
    Returns the index of such diagonal and its elements.

    For example, if :math:`x_edges = (0, 1, 2, 3) ms` and :math:`y_edges =
    (1, 2, 3, 4) ms`, then the index of the reference diagonal is -1
    (first off-diagonal below the main diagonal) and its elements are
    (-1, 0), (0, 1), (1, 2), (2, 3).

    '''
    diag_id = None
    error_msg = \
        'the time axes (%s-%s and %s-%s)' % (
            x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]) + \
        ' overlap but the overlapping bin edges are not aligned' \
        '. Bad alignment of the time axes.'

    if y_edges[0] > x_edges[0]:
        if y_edges[0] < x_edges[-1]:
            bin_ids = np.where(
                _quantities_almost_equal(x_edges, y_edges[0]))[0]
            if len(bin_ids) == 0:
                raise ValueError(error_msg)
            diag_id = - bin_ids[0]
    else:
        if y_edges[-1] > x_edges[0]:
            bin_ids = np.where(y_edges == x_edges[0])[0]
            if len(bin_ids) == 0:
                raise ValueError(error_msg)
            diag_id = bin_ids[0]

    m = len(x_edges) - 1

    if diag_id is None:
        return diag_id, np.array([])
    elif diag_id >= 0:
        elements = np.column_stack([np.arange(m - diag_id),
                                    np.arange(diag_id, m)])
    else:
        elements = np.column_stack([np.arange(diag_id, m),
                                    np.arange(m - diag_id)])

    return diag_id, elements


def mask_matrices(matrices, thresholds):
    '''
    Given a list of matrices and a list of thresholds, return a boolean matrix
    B ("mask") such that B[i,j] is True if each input matrix in the list
    strictly exceeds the corresponding threshold at that position.

    Parameters
    ----------
    matrices : list of numpy.ndarrays
        the matrices which are compared to the respective thresholds to
        build the mask. All matrices must have the same shape.
    thresholds : list of floats
        list of thresholds

    Returns
    -------
    mask : numpy.ndarray of bools
        mask matrix with same shape of the input matrices.
    '''

    # Check that input lists have same length
    L = len(matrices)
    if L != len(thresholds):
        raise ValueError('`matrices` and `thresholds` must have same length')

    # Compute mask matrix
    mask = matrices[0] > thresholds[0]
    if L > 1:
        for (mat, thresh) in zip(matrices, thresholds):
            mask = mask * (mat > thresh)

    # Replace nans, coming from False * np.inf, with 0s
    # (trick to find nans in masked: a number is nan if it's not >= - np.inf)
    mask[np.logical_xor(True, (mask >= -np.inf))] = False

    return np.array(mask, dtype=bool)


def _stretched_metric_2d(x, y, stretch, ref_angle):
    r'''
    Given a list of points on the real plane, identified by their absciss x
    and ordinate y, compute a stretched transformation of the Euclidean
    distance among each of them.
    The classical euclidean distance d between points (x1, y1) and (x2, y2),
    i.e. \sqrt((x1-x2)^2 + (y1-y2)^2), is multiplied by a factor
    .. math::

            1 + (stretch - 1.) * \abs(\sin(ref_angle - \theta)),

    where \\theta is the angle between the points and the 45deg direction
    (i.e. the line y=x).
    The stretching factor thus steadily varies between 1 (if the line
    connecting (x1, y1) and (x2, y2) has inclination ref_angle) and stretch
    (if that line has inclination 90 + ref_angle).

    Parameters
    ----------
    x : numpy.ndarray
        array of abscissa of all points among which to compute the distance
    y : numpy.ndarray (same shape as x)
        array of ordinates of all points among which to compute the distance
    stretch : float
        maximum stretching factor, applied if the line connecting the points
        has inclination 90 + ref_angle
    ref_angle : float
        reference angle, i.e. inclination along which the stretching factor
        is 1.

    Output
    ------
    D : numpy.ndarray
        square matrix of distances between all pairs of points. If x and y
        have shape (n, ) then D has shape (n, n).
    '''
    alpha = np.deg2rad(ref_angle)  # reference angle in radians

    # Create the array of points (one per row) for which to compute the
    # stretched distance
    points = np.vstack([x, y]).T

    # Compute the matrix D[i, j] of euclidean distances among points i and j
    D = scipy.spatial.distance_matrix(points, points)

    # Compute the angular coefficients of the line between each pair of points
    x_array = np.tile(x, reps=(len(x), 1))
    y_array = np.tile(y, reps=(len(y), 1))
    dX = x_array.T - x_array  # dX[i,j]: x difference between points i and j
    dY = y_array.T - y_array  # dY[i,j]: y difference between points i and j

    # Compute the matrix Theta of angles between each pair of points
    theta = np.arctan2(dY, dX)

    # Transform [-pi, pi] back to [-pi/2, pi/2]
    theta[theta < -np.pi / 2] += np.pi
    theta[theta > np.pi / 2] -= np.pi
    assert np.allclose(np.diagonal(theta), 0), \
        "Diagonal elements should be zero due to `np.arctan2(0, 0) == 0` " \
        "convention."

    # Compute the matrix of stretching factors for each pair of points
    stretch_mat = 1 + (stretch - 1.) * np.abs(np.sin(alpha - theta))

    # Return the stretched distance matrix
    return D * stretch_mat


def cluster_matrix_entries(mat, eps=10, min=2, stretch=5):
    r'''
    Given a matrix mat, replaces its positive elements with integers
    representing different cluster ids. Each cluster comprises close-by
    elements.

    In ASSET analysis, mat is a thresholded ("masked") version of an
    intersection matrix imat, whose values are those of imat only if
    considered statistically significant, and zero otherwise.

    A cluster is built by pooling elements according to their distance,
    via the DBSCAN algorithm (see sklearn.cluster.dbscan()). Elements form
    a neighbourhood if at least one of them has a distance not larger than
    eps from the others, and if they are at least min. Overlapping
    neighborhoods form a cluster.

        * Clusters are assigned integers from 1 to the total number k of
          clusters
        * Unclustered ("isolated") positive elements of mat are
          assigned value -1
        * Non-positive elements are assigned the value 0.

    The distance between the positions of two positive elements in mat is
    given by an Euclidean metric which is stretched if the two positions are
    not aligned along the 45 degree direction (the main diagonal direction),
    as more, with maximal stretching along the anti-diagonal. Specifically,
    the Euclidean distance between positions (i1, j1) and (i2, j2) is
    stretched by a factor

    .. math::
             1 + (\mathtt{stretch} - 1.) *
             \\left|\\sin((\\pi / 4) - \\theta)\\right|,

    where :math:`\\theta` is the angle between the pixels and the 45deg
    direction. The stretching factor thus varies between 1 and stretch.

    Parameters
    ----------
    mat : numpy.ndarray
        a matrix whose elements with positive values are to be clustered.
    eps : float >=0, optional
        the maximum distance for two elements in mat to be part of the same
        neighbourhood in the DBSCAN algorithm
        Default: 10
    min : int, optional
        the minimum number of elements to form a neighbourhood.
        Default: 2
    stretch : float > 1, optional
        the stretching factor of the euclidean metric for elements aligned
        along the 135 degree direction (anti-diagonal). The actual stretching
        increases from 1 to stretch as the direction of the two elements
        moves from the 45 to the 135 degree direction.
        Default: 5

    Returns
    -------
    cmat : numpy.ndarray of integers
        a matrix with the same shape of mat, each of whose elements is either
            * a positive int (cluster id) if the element is part of a cluster
            * 0 if the corresponding element in mat was non-positive
            * -1 if the element does not belong to any cluster
    '''

    # Don't do anything if mat is identically zero
    if np.all(mat == 0):
        return mat

    # List the significant pixels of mat in a 2-columns array
    xpos_sgnf, ypos_sgnf = np.where(mat > 0)

    # Compute the matrix D[i, j] of euclidean distances among pixels i and j
    D = _stretched_metric_2d(
        xpos_sgnf, ypos_sgnf, stretch=stretch, ref_angle=45)

    # Cluster positions of significant pixels via dbscan
    core_samples, config = dbscan(
        D, eps=eps, min_samples=min, metric='precomputed')

    # Construct the clustered matrix, where each element has value
    # * i = 1 to k if it belongs to a cluster i,
    # * 0 if it is not significant,
    # * -1 if it is significant but does not belong to any cluster
    cluster_mat = np.array(np.zeros(mat.shape), dtype=int)
    cluster_mat[xpos_sgnf, ypos_sgnf] = \
        config * (config == -1) + (config + 1) * (config >= 0)

    return cluster_mat


def probability_matrix_montecarlo(
        spiketrains, binsize, dt, t_start_x=None, t_start_y=None,
        surr_method='dither_spike_train', j=None, n_surr=100, verbose=False):
    '''
    Given a list of parallel spike trains, estimate the cumulative probability
     of each entry in their intersection matrix (see: intersection_matrix())
    by a Monte Carlo approach using surrogate data.
    Contrarily to the analytical version (see: probability_matrix_analytical())
    the Monte Carlo one does not incorporate the assumptions of Poissonianity
    in the null hypothesis.

    The method produces surrogate spike trains (using one of several methods
    at disposal, see below) and calculates their intersection matrix M.
    For each entry (i, j), the intersection cdf P[i, j] is then given by:

    .. centered::  P[i, j] = #(spike_train_surrogates such that M[i, j] < I[i, j]) /
                        #(spike_train_surrogates)

    If P[i, j] is large (close to 1), I[i, j] is statistically significant:
    the probability to observe an overlap equal to or larger then I[i, j]
    under the null hypothesis is 1-P[i, j], very small.

    Parameters
    ----------
    spiketrains : list of neo.SpikeTrains
        list of spike trains for which to compute the probability matrix
    binsize : quantities.Quantity
        width of the time bins used to compute the probability matrix
    dt : quantities.Quantity
        time span for which to consider the given SpikeTrains
    t_start_x, t_start_y : quantities.Quantity, optional
        time start of the binning for the first and second axes of the
        intersection matrix, respectively.
        If None (default) the attribute t_start of the SpikeTrains is used
        (if the same for all spike trains).
        Default: None
    surr_method : str, optional
        the method to use to generate surrogate spike trains. Can be one of:

            * 'dither_spike_train': see spike_train_surrogates.train_shifting() [dt needed]
            * 'spike_dithering': see spike_train_surrogates.spike_dithering() [dt needed]
            * 'spike_jittering': see spike_train_surrogates.spike_jittering() [dt needed]
            * 'spike_time_rand': see spike_train_surrogates.spike_time_rand()
            * 'isi_shuffling': see spike_train_surrogates.isi_shuffling()

        Default: 'dither_spike_train'
    j : quantities.Quantity, optional
        For methods shifting spike times randomly around their original time
        (spike dithering, train shifting) or replacing them randomly within a
        certain window (spike jittering), j represents the size of that
        shift / window. For other methods, j is ignored.
        Default: None
    n_surr : int, optional
        number of spike_train_surrogates to generate for the bootstrap
        procedure. Default: 100
    verbose : bool, optional
        Print the progress bar.

    Returns
    -------
    pmat : np.ndarray
        the cumulative probability matrix. pmat[i, j] represents the
        estimated probability of having an overlap between bins i and j
        STRICTLY LOWER than the observed overlap, under the null hypothesis
        of independence of the input spike trains.

    See also
    --------
    probability_matrix_analytical : for analytical derivation of the matrix
    '''

    # Compute the intersection matrix of the original data
    imat, x_edges, y_edges = intersection_matrix(
        spiketrains, binsize=binsize, dt=dt, t_start_x=t_start_x,
        t_start_y=t_start_y)

    # Generate surrogate spike trains as a list surrs; for each spike train
    # i, surrs[i] is a list of length n_surr, containing the
    # spike_train_surrogates of i
    surrs = [spike_train_surrogates.surrogates(
        st, n=n_surr, surr_method=surr_method, dt=j, decimals=None, edges=True)
        for st in spiketrains]

    # Compute the p-value matrix pmat; pmat[i, j] counts the fraction of
    # surrogate data whose intersection value at (i, j) whose lower than or
    # equal to that of the original data
    pmat = np.array(np.zeros(imat.shape), dtype=int)
    if verbose:
        # todo: move to tqdm
        print('pmat_bootstrap(): begin of bootstrap...')
    for i in range(n_surr):  # For each surrogate id i
        if verbose:
            print('    surr %d' % i)
        surrs_i = [st[i] for st in surrs]  # Take each i-th surrogate
        imat_surr, xx, yy = intersection_matrix(  # compute the related imat
            surrs_i, binsize=binsize, dt=dt,
            t_start_x=t_start_x, t_start_y=t_start_y)
        pmat += (imat_surr <= imat - 1)
    pmat = pmat * 1. / n_surr
    if verbose:
        print('pmat_bootstrap(): done')

    return pmat, x_edges, y_edges


def probability_matrix_analytical(
        spiketrains, binsize, dt, t_start_x=None, t_start_y=None,
        fir_rates='estimate', kernel_width=100 * pq.ms, verbose=False):
    r'''
    Given a list of spike trains, approximates the cumulative probability of
    each entry in their intersection matrix (see: intersection_matrix()).

    The approximation is analytical and works under the assumptions that the
    input spike trains are independent and Poisson. It works as follows:

        * Bin each spike train at the specified binsize: this yields a binary
          array of 1s (spike in bin) and 0s (no spike in bin) (clipping used)
        * If required, estimate the rate profile of each spike train by
          convolving the binned array with a boxcar kernel of user-defined
          length
        * For each neuron k and each pair of bins i and j, compute the
          probability p_ijk that neuron k fired in both bins i and j.
        * Approximate the probability distribution of the intersection value
          at (i, j) by a Poisson distribution with mean parameter
          l = \sum_k (p_ijk),
          justified by Le Cam's approximation of a sum of independent
          Bernouilli random variables with a Poisson distribution.

    Parameters
    ----------
    spiketrains : list of neo.SpikeTrains
        list of spike trains for whose intersection matrix to compute the
        p-values
    binsize : quantities.Quantity
        width of the time bins used to compute the probability matrix
    dt : quantities.Quantity
        time span for which to consider the given SpikeTrains
    t_start_x, t_start_y : quantities.Quantity, optional
        time start of the binning for the first and second axes of the
        intersection matrix, respectively.
        If None (default) the attribute t_start of the SpikeTrains is used
        (if the same for all spike trains).
        Default: None
    fir_rates: list of neo.AnalogSignals or 'estimate', optional
        if a list, fir_rate[i] is the firing rate of the spike train
        spiketrains[i]. If 'estimate', firing rates are estimated by simple
        boxcar kernel convolution, with specified kernel width (see below)
        Default: 'estimate'
    kernel_width : quantities.Quantity, optional
        total width of the kernel used to estimate the rate profiles when
        fir_rates='estimate'.
        Default: 100 * pq.ms
    verbose : bool, optional
        whether to print messages during the computation.
        Default: False

    Returns
    -------
    pmat : numpy.ndarray
        the cumulative probability matrix. pmat[i, j] represents the
        estimated probability of having an overlap between bins i and j
        STRICTLY LOWER THAN the observed overlap, under the null hypothesis
        of independence of the input spike trains.
    x_edges : numpy.ndarray
        edges of the bins used for the horizontal axis of pmat. If pmat is
        a matrix of shape (n, n), x_edges has length n+1
    y_edges : numpy.ndarray
        edges of the bins used for the vertical axis of pmat. If pmat is
        a matrix of shape (n, n), y_edges has length n+1
    '''

    # Bin the spike trains
    t_stop_x = None if t_start_x is None else t_start_x + dt
    t_stop_y = None if t_start_y is None else t_start_y + dt
    bsts_x = conv.BinnedSpikeTrain(
        spiketrains, binsize=binsize, t_start=t_start_x, t_stop=t_stop_x)
    bsts_y = conv.BinnedSpikeTrain(
        spiketrains, binsize=binsize, t_start=t_start_y, t_stop=t_stop_y)

    bsts_x_matrix = bsts_x.to_bool_array()
    bsts_y_matrix = bsts_y.to_bool_array()

    # Check that the duration and nr. neurons is identical between the two axes
    if bsts_x_matrix.shape != bsts_y_matrix.shape:
        raise ValueError(
            'Different spike train durations along the x and y axis!')

    # Define the firing rate profiles

    # If rates are to be estimated, create the rate profiles as Quantity
    # objects obtained by boxcar-kernel convolution
    if fir_rates == 'estimate':
        if verbose is True:
            print('compute rates by boxcar-kernel convolution...')

        # Create the boxcar kernel and convolve it with the binned spike trains
        k = int((kernel_width / binsize).rescale(pq.dimensionless))
        kernel = np.ones(k) * 1. / k
        fir_rate_x = np.vstack([np.convolve(bst, kernel, mode='same')
                                for bst in bsts_x_matrix])
        fir_rate_y = np.vstack([np.convolve(bst, kernel, mode='same')
                                for bst in bsts_y_matrix])

        # The convolution results in an array decreasing at the borders due
        # to absence of spikes beyond the borders. Replace the first and last
        # (k//2) elements with the (k//2)-th / (n-k//2)-th ones, respectively
        k2 = k // 2
        for i in range(fir_rate_x.shape[0]):
            fir_rate_x[i, :k2] = fir_rate_x[i, k2]
            fir_rate_x[i, -k2:] = fir_rate_x[i, -k2 - 1]
            fir_rate_y[i, :k2] = fir_rate_y[i, k2]
            fir_rate_y[i, -k2:] = fir_rate_y[i, -k2 - 1]

        # Multiply the firing rates by the proper unit
        fir_rate_x = fir_rate_x * (1. / binsize).rescale('Hz')
        fir_rate_y = fir_rate_y * (1. / binsize).rescale('Hz')

    # If rates provided as lists of AnalogSignals, create time slices for both
    # axes, interpolate in the time bins of interest and convert to Quantity
    elif isinstance(fir_rates, list):
        # Reshape all rates to one-dimensional array object (e.g. AnalogSignal)
        for i, rate in enumerate(fir_rates):
            if len(rate.shape) == 2:
                fir_rates[i] = rate.reshape((-1,))
            elif len(rate.shape) > 2:
                raise ValueError(
                    'elements in fir_rates have too many dimensions')

        if verbose is True:
            print('create time slices of the rates...')

        # Define the rate by time slices
        fir_rate_x = [_time_slice(signal, bsts_x.t_start, bsts_x.t_stop)
                      for signal in fir_rates]
        fir_rate_y = [_time_slice(signal, bsts_y.t_start, bsts_y.t_stop)
                      for signal in fir_rates]
        # Interpolate in the time bins and convert to Quantities
        times_x = bsts_x.bin_edges[:-1]
        times_y = bsts_y.bin_edges[:-1]
        fir_rate_x = pq.Hz * np.vstack([_analog_signal_step_interp(
            signal, times_x).rescale('Hz').magnitude for signal in fir_rates])
        fir_rate_y = pq.Hz * np.vstack([_analog_signal_step_interp(
            signal, times_y).rescale('Hz').magnitude for signal in fir_rates])

    else:
        raise ValueError('fir_rates must be a list or the string "estimate"')

    # For each neuron, compute the prob. that that neuron spikes in any bin
    if verbose is True:
        print(
            'compute the prob. that each neuron fires in each pair of bins...')

    spike_probs_x = [1. - np.exp(-(rate * binsize).rescale(
        pq.dimensionless).magnitude) for rate in fir_rate_x]
    spike_probs_y = [1. - np.exp(-(rate * binsize).rescale(
        pq.dimensionless).magnitude) for rate in fir_rate_y]

    # For each neuron k compute the matrix of probabilities p_ijk that neuron
    # k spikes in both bins i and j. (For i = j it's just spike_probs[k][i])
    spike_prob_mats = [np.outer(probx, proby) for (probx, proby) in
                       zip(spike_probs_x, spike_probs_y)]

    # Compute the matrix Mu[i, j] of parameters for the Poisson distributions
    # which describe, at each (i, j), the approximated overlap probability.
    # This matrix is just the sum of the probability matrices computed above

    if verbose is True:
        print("compute the probability matrix by Le Cam's approximation...")

    Mu = np.sum(spike_prob_mats, axis=0)

    # Compute the probability matrix obtained from imat using the Poisson pdfs
    imat, xx, yy = intersection_matrix(
        spiketrains, binsize=binsize, dt=dt, t_start_x=t_start_x,
        t_start_y=t_start_y)

    pmat = np.zeros(imat.shape)
    for i in range(imat.shape[0]):
        for j in range(imat.shape[1]):
            pmat[i, j] = scipy.stats.poisson.cdf(imat[i, j] - 1, Mu[i, j])

    # Substitute 0.5 to the elements along the main diagonal
    diag_id, elems = _reference_diagonal(xx, yy)
    if diag_id is not None:
        if verbose is True:
            print("substitute 0.5 to elements along the main diagonal...")
        for elem in elems:
            pmat[elem[0], elem[1]] = 0.5

    return pmat, xx, yy


def _jsf_uniform_orderstat_3d(u, alpha, n):
    r'''
    Considered n independent random variables X1, X2, ..., Xn all having
    uniform distribution in the interval (alpha, 1):

    .. centered::  Xi ~ Uniform(alpha, 1),

    with alpha \in [0, 1), and given a 3D matrix U = (u_ijk) where each U_ij
    is an array of length d: U_ij = [u0, u1, ..., u_{d-1}] of
    quantiles, with u1 <= u2 <= ... <= un, computes the joint survival function
    (jsf) of the d highest order statistics (U_{n-d+1}, U_{n-d+2}, ..., U_n),
    where U_i := "i-th highest X's" at each u_ij, i.e.:

    .. centered::  jsf(u_ij) = Prob(U_{n-k} >= u_ijk, k=0,1,..., d-1).


    Arguments
    ---------
    u : numpy.ndarray of shape (A, B, d)
        3D matrix of floats between 0 and 1.
        Each vertical column u_ij is an array of length d, considered a set of
        `d` largest order statistics extracted from a sample of `n` random
        variables whose cdf is F(x)=x for each x.
        The routine computes the joint cumulative probability of the `d`
        values in u_ij, for each i and j.
    alpha : float in [0, 1)
        range where the values of `u` are assumed to vary.
        alpha is 0 in the standard ASSET analysis.
    n : int
        size of the sample where the d largest order statistics u_ij are
        assumed to have been sampled from

    Returns
    -------
    S : numpy.ndarray of shape (A, B)
        matrix of joint survival probabilities. s_ij is the joint survival
        probability of the values {u_ijk, k=0,...,d-1}.
        Note: the joint probability matrix computed for the ASSET analysis
        is 1-S.
    '''
    d, A, B = u.shape

    # Define ranges [1,...,n], [2,...,n], ..., [d,...,n] for the mute variables
    # used to compute the integral as a sum over several possibilities
    lists = [range(j, n + 1) for j in range(d, 0, -1)]

    # Compute the log of the integral's coefficient
    logK = np.sum(np.log(np.arange(1, n + 1))) - n * np.log(1 - alpha)

    # Add to the 3D matrix u a bottom layer identically equal to alpha and a
    # top layer identically equal to 1. Then compute the difference dU along
    # the first dimension.
    u_extended = np.ones((d + 2, A, B))
    u_extended[0] = u_extended[0] * alpha
    for layer_idx, uu in enumerate(u):
        u_extended[layer_idx + 1] = u[layer_idx]
    dU = np.diff(u_extended, axis=0)  # shape (d+1, A, B)
    del u_extended

    # Compute the probabilities at each (a, b), a=0,...,A-1, b=0,...,B-1
    # by matrix algebra, working along the third dimension (axis 0)
    Ptot = np.zeros((A, B))  # initialize all A x B probabilities to 0
    iter_id = 0
    for i in itertools.product(*lists):
        iter_id += 1
        di = -np.diff(np.hstack([n, list(i), 0]))
        if np.all(di >= 0):
            dI = di.reshape((-1, 1, 1)) * np.ones((A, B))  # shape (d+1, A, B)

            # for each a=0,1,...,A-1 and b=0,1,...,B-1, replace dU_abk with 1
            # whenever dI_abk = 0, so that dU_abk ** dI_abk = 1 (this avoids
            # nans when both dU_abk and dI_abk are 0, and is mathematically
            # correct). dU2 still contains 0s, so that when below exp(log(U2))
            # is computed, warnings are arosen; they are no problem though.
            dU2 = dU.copy()
            dU2[dI == 0] = 1.

            # Compute for each i=0,...,A-1 and j=0,...,B-1: log(I_ij !)
            # Creates a matrix log_dIfactorial of shape (A, B)
            log_di_factorial = np.sum([np.log(np.arange(1, di_k + 1)).sum()
                                       for di_k in di if di_k >= 1])

            # Compute for each i,j the contribution to the total
            # probability given by this step, and add it to the total prob.
            logP = (dI * np.log(dU2)).sum(axis=0) - log_di_factorial
            Ptot += np.exp(logP + logK)
    return Ptot


def _pmat_neighbors(mat, filter_shape, nr_largest=None, diag=0):
    '''
    Build the 3D matrix L of largest neighbors of elements in a 2D matrix mat.

    For each entry mat[i, j], collects the nr_largest elements with largest
    values around mat[i,j], say z_i, i=1,2,...,nr_largest, and assigns them
    to L[i, j, :].
    The zone around mat[i, j] where largest neighbors are collected from is
    a rectangular area (kernel) of shape (l, w) = filter_shape centered around
    mat[i, j] and aligned along the diagonal where mat[i, j] lies into
    (if diag=0, default) or along the anti-diagonal (is diag = 1)

    Arguments
    ---------
    mat : ndarray
        a square matrix of real-valued elements

    filter_shape : tuple
        a pair (l, w) of integers representing the kernel shape

    nr_largest : int, optional
        the number of largest neighbors to collect for each entry in mat
        If None (default) the filter length l is used
        Default: 0

    diag : int, optional
        which diagonal of mat[i, j] to align the kernel to in order to
        find its largest neighbors.
        * 0: main diagonal
        * 1: anti-diagonal
        Default: 0

    Returns
    -------
    L : ndarray
        a matrix of shape (nr_largest, l, w) containing along the first
        dimension lmat[:, i, j] the largest neighbors of mat[i, j]

    '''
    l, w = filter_shape
    d = l if nr_largest is None else nr_largest

    # Check consistent arguments
    assert mat.shape[0] == mat.shape[1], 'mat must be a square matrix'
    assert diag == 0 or diag == 1, \
        'diag must be 0 (45 degree filtering) or 1 (135 degree filtering)'
    assert w < l, 'w must be lower than l'

    # Construct the kernel
    filt = np.ones((l, l), dtype=np.float32)
    filt = np.triu(filt, -w)
    filt = np.tril(filt, w)
    if diag == 1:
        filt = np.fliplr(filt)

    # Convert mat values to floats, and replaces np.infs with specified input
    # values
    mat = 1. * mat.copy()

    # Initialize the matrix of d-largest values as a matrix of zeroes
    lmat = np.zeros((d, mat.shape[0], mat.shape[1]), dtype=np.float32)

    # TODO: make this on a 3D matrix to parallelize...
    N_bin = mat.shape[0]
    bin_range = range(N_bin - l + 1)

    # Compute fmat
    try:  # try by stacking the different patches of each row of mat
        flattened_filt = filt.flatten()
        for y in bin_range:
            # creates a 2D matrix of shape (N_bin-l+1, l**2), where each row
            # is a flattened patch (length l**2) from the y-th row of mat
            row_patches = np.zeros((len(bin_range), l ** 2))
            for x in bin_range:
                row_patches[x, :] = (mat[y:y + l, x:x + l]).flatten()
            # take the l largest values in each row (patch) and assign them
            # to the corresponding row in lmat
            largest_vals = np.sort(
                row_patches * flattened_filt, axis=1)[:, -d:]
            lmat[:, y + (l // 2),
                 (l // 2): (l // 2) + N_bin - l + 1] = largest_vals.T

    except MemoryError:  # if too large, do it serially by for loops
        for y in bin_range:  # one step to the right;
            for x in bin_range:  # one step down
                patch = mat[y: y + l, x: x + l]
                mskd = np.multiply(filt, patch)
                largest_vals = np.sort(d, mskd.flatten())[-d:]
                lmat[:, y + (l // 2), x + (l // 2)] = largest_vals

    return lmat


def joint_probability_matrix(
        pmat, filter_shape, nr_largest=None, alpha=0, pvmin=1e-5):
    '''
    Map a probability matrix pmat to a joint probability matrix jmat, where
    jmat[i, j] is the joint p-value of the largest neighbors of pmat[i, j].

    The values of pmat are assumed to be uniformly distributed in the range
    [alpha, 1] (alpha=0 by default). Centered a rectangular kernel of shape
    filter_shape=(l, w) around each entry pmat[i, j], aligned along the
    diagonal where pmat[i, j] lies into, extracts the nr_largest highest values
    falling within the kernel and computes their joint p-value jmat[i, j]
    (see [1]).

    Parameters
    ----------
    pmat : ndarray
        a square matrix of cumulative probability values between alpha and 1.
        The values are assumed to be uniformly distibuted in the said range
    filter_shape : tuple
        a pair (l, w) of integers representing the kernel shape. The
    nr_largest : int, optional
        the number of largest neighbors to collect for each entry in mat
        If None (default) the filter length l is used
        Default: 0
    alpha : float in [0, 1), optional
        the left end of the range [alpha, 1]
        Default: 0
    pvmin : flaot in [0, 1), optional
        minimum p-value for individual entries in pmat. Each pmat[i, j] is
        set to min(pmat[i, j], 1-pvmin) to avoid that a single highly
        significant value in pmat (extreme case: pmat[i, j] = 1) yield
        joint significance of itself and its neighbors.
        Default: 1e-5

    Returns
    -------
    jmat : numpy.ndarray
        joint probability matrix associated to pmat


    References
    ----------
    [1] Torre et al (in prep) ...

    Example
    -------
    # Assuming to have a list sts of parallel spike trains over 1s recording,
    # the following code computes the intersection/probability/joint-prob
    # matrices imat/pmat/jmat using a bin width of 5 ms
    >>> T = 1 * pq.s
    >>> binsize = 5 * pq.ms
    >>> imat, xedges, yedges = intersection_matrix(sts, binsize=binsize, dt=T)
    >>> pmat = probability_matrix_analytical(sts, binsize, dt=T)
    >>> jmat = joint_probability_matrix(pmat, filter_shape=(fl, fw))

    '''
    # Find for each P_ij in the probability matrix its neighbors and maximize
    # them by the maximum value 1-pvmin
    pmat_neighb = _pmat_neighbors(
        pmat, filter_shape=filter_shape, nr_largest=nr_largest, diag=0)
    pmat_neighb = np.minimum(pmat_neighb, 1. - pvmin)

    # Compute the joint p-value matrix jpvmat
    l, w = filter_shape
    n = l * (1 + 2 * w) - w * (w + 1)  # number of entries covered by kernel
    jpvmat = _jsf_uniform_orderstat_3d(pmat_neighb, alpha, n)

    return 1. - jpvmat


def extract_sse(spiketrains, x_edges, y_edges, cmat, ids=None):
    '''
    Given a list of spike trains, two arrays of bin edges and a clustered
    intersection matrix obtained from those spike trains via worms analysis
    using the specified edges, extracts the sequences of synchronous events
    (SSEs) corresponding to clustered elements in the cluster matrix.

    Parameters
    ----------
    spiketrains : list of neo.SpikeTrain
        the spike trains analyzed for repeated sequences of synchronous
        events.
    x_edges : quantities.Quantity
        the first array of time bins used to compute cmat
    y_edges : quantities.Quantity
        the second array of time bins used to compute cmat. Musr have the
        same length as x_array
    cmat: numpy.ndarray
        matrix of shape (n, n), where n is the length of x_edges and
        y_edges, representing the cluster matrix in worms analysis
        (see: cluster_matrix_entries())
    ids : list or None, optional
        a list of spike train IDs. If provided, ids[i] is the identity
        of spiketrains[i]. If None, the IDs 0,1,...,n-1 are used
        Default: None

    Returns
    -------
    sse : dict
        a dictionary D of SSEs, where each SSE is a sub-dictionary Dk,
        k=1,...,K, where K is the max positive integer in cmat (the
        total number of clusters in cmat):

        .. centered:: D = {1: D1, 2: D2, ..., K: DK}

        Each sub-dictionary Dk represents the k-th diagonal structure
        (i.e. the k-th cluster) in cmat, and is of the form

        .. centered:: Dk = {(i1, j1): S1, (i2, j2): S2, ..., (iL, jL): SL}.

        The keys (i, j) represent the positions (time bin ids) of all
        elements in cmat that compose the SSE, i.e. that take value l (and
        therefore belong to the same cluster), and the values Sk are sets of
        neuron ids representing a repeated synchronous event (i.e. spiking
        at time bins i and j).
    '''

    nr_worms = cmat.max()  # number of different clusters ("worms") in cmat
    if nr_worms <= 0:
        return {}

    # Compute the transactions associated to the two binnings
    binsize_x = x_edges[1] - x_edges[0]
    t_start_x, t_stop_x = x_edges[0], x_edges[-1]
    tracts_x = _transactions(
        spiketrains, binsize=binsize_x, t_start=t_start_x, t_stop=t_stop_x,
        ids=ids)
    binsize_y = y_edges[1] - y_edges[0]
    t_start_y, t_stop_y = y_edges[0], y_edges[-1]
    tracts_y = _transactions(
        spiketrains, binsize=binsize_y, t_start=t_start_y, t_stop=t_stop_y,
        ids=ids)

    # Find the reference diagonal, whose elements correspond to same time bins
    diag_id, _ = _reference_diagonal(x_edges, y_edges)

    # Reconstruct each worm, link by link
    sse_dict = {}
    for k in range(1, nr_worms + 1):  # for each worm
        worm_k = {}  # worm k is a list of links (each link will be 1 sublist)
        pos_worm_k = np.array(np.where(cmat == k)).T  # position of all links
        # if no link lies on the reference diagonal
        if all([y - x != diag_id for (x, y) in pos_worm_k]):
            for l, (bin_x, bin_y) in enumerate(pos_worm_k):  # for each link
                link_l = set(tracts_x[bin_x]).intersection(
                    tracts_y[bin_y])  # reconstruct the link
                worm_k[(bin_x, bin_y)] = link_l  # and assign it to its pixel
            sse_dict[k] = worm_k

    return sse_dict


def sse_intersection(sse1, sse2, intersection='linkwise'):
    '''
    Given two sequences of synchronous events (SSEs) `sse1` and `sse2`, each
    consisting of a pool of positions (iK, jK) of matrix entries and
    associated synchronous events SK, finds the intersection among them.
    The intersection can be performed 'pixelwise' or 'linkwise'.

        * if 'pixelwise', it yields a new SSE which retains only events in sse1
          whose pixel position matches a pixel position in sse2. This operation
          is not symmetric: intersection(sse1, sse2) != intersection(sse2, sse1).
        * if 'linkwise', an additional step is performed where each retained
          synchronous event SK in sse1 is intersected with the corresponding
          event in sse2. This yields a symmetric operation:
          intersection(sse1, sse2) = intersection(sse2, sse1).

    Both sse1 and sse2 must be provided as dictionaries of the type

    .. centered:: {(i1, j1): S1, (i2, j2): S2, ..., (iK, jK): SK},

    where each i, j is an integer and each S is a set of neuron ids.
    (See also: extract_sse() that extracts SSEs from given spiketrains).

    Parameters
    ----------
    sse1, sse2 : each a dict
        each is a dictionary of pixel positions (i, j) as keys, and sets S of
        synchronous events as values (see above).
    intersection : str, optional
        the type of intersection to perform among the two SSEs. Either
        'pixelwise' or 'linkwise' (see above).
        Default: 'linkwise'.

    Returns
    -------
    sse : dict
        a new SSE (same structure as sse1 and sse2) which retains only the
        events of sse1 associated to keys present both in sse1 and sse2.
        If intersection = 'linkwise', such events are additionally
        intersected with the associated events in sse2 (see above).
    '''
    sse_new = sse1.copy()
    for pixel1 in sse1.keys():
        if pixel1 not in sse2.keys():
            del sse_new[pixel1]

    if intersection == 'linkwise':
        for pixel1, link1 in sse_new.items():
            sse_new[pixel1] = link1.intersection(sse2[pixel1])
            if len(sse_new[pixel1]) == 0:
                del sse_new[pixel1]
    elif intersection == 'pixelwise':
        pass
    else:
        raise ValueError(
            "intersection (=%s) can only be" % intersection +
            " 'pixelwise' or 'linkwise'")

    return sse_new


def sse_difference(sse1, sse2, difference='linkwise'):
    '''
    Given two sequences of synchronous events (SSEs) sse1 and sse2, each
    consisting of a pool of pixel positions and associated synchronous events
    (see below), computes the difference between sse1 and sse2.
    The difference can be performed 'pixelwise' or 'linkwise':

        * if 'pixelwise', it yields a new SSE which contains all (and only) the
          events in sse1 whose pixel position doesn't match any pixel in sse2.
        * if 'linkwise', for each pixel (i, j) in sse1 and corresponding
          synchronous event S1, if (i, j) is a pixel in sse2 corresponding to the
          event S2, it retains the set difference S1 - S2. If (i, j) is not a
          pixel in sse2, it retains the full set S1.

    Note that in either case the difference is a non-symmetric operation:
    intersection(sse1, sse2) != intersection(sse2, sse1).

    Both sse1 and sse2 must be provided as dictionaries of the type

    .. centered:: {(i1, j1): S1, (i2, j2): S2, ..., (iK, jK): SK},

    where each i, j is an integer and each S is a set of neuron ids.
    (See also: extract_sse() that extracts SSEs from given spiketrains).


    Parameters
    ----------
    sse1, sse2 : each a dict
        a dictionary of pixel positions (i, j) as keys, and sets S of
        synchronous events as values (see above).

    difference : str, optional
        the type of difference to perform between sse1 and sse2. Either
        'pixelwise' or 'linkwise' (see above).
        Default: 'linkwise'.

    Returns
    -------
    sse : dict
        a new SSE (same structure as sse1 and sse2) which retains the
        difference between sse1 and sse2 (see above).
    '''
    sse_new = sse1.copy()
    for pixel1 in sse1.keys():
        if pixel1 in sse2.keys():
            if difference == 'pixelwise':
                del sse_new[pixel1]
            elif difference == 'linkwise':
                sse_new[pixel1] = sse_new[pixel1].difference(sse2[pixel1])
                if len(sse_new[pixel1]) == 0:
                    del sse_new[pixel1]
            else:
                raise ValueError(
                    "difference (=%s) can only be" % difference +
                    " 'pixelwise' or 'linkwise'")

    return sse_new


def _remove_empty_events(sse):
    '''
    Given a sequence of synchronous events (SSE) sse consisting of a pool of
    pixel positions and associated synchronous events (see below), returns a
    copy of sse where all empty events have been removed.

    sse must be provided as a dictionary of type

    .. centered:: {(i1, j1): S1, (i2, j2): S2, ..., (iK, jK): SK},

    where each i, j is an integer and each S is a set of neuron ids.
    (See also: extract_sse() that extracts SSEs from given spiketrains).

    Parameters
    ----------
    sse : dict
        a dictionary of pixel positions (i, j) as keys, and sets S of
        synchronous events as values (see above).

    Returns
    -------
    sse_new : dict
        a copy of sse where all empty events have been removed.
    '''
    sse_new = sse.copy()
    for pixel, link in sse.items():
        if link == set([]):
            del sse_new[pixel]

    return sse_new


def sse_isequal(sse1, sse2):
    '''
    Given two sequences of synchronous events (SSEs) sse1 and sse2, each
    consisting of a pool of pixel positions and associated synchronous events
    (see below), determines whether sse1 is strictly contained in sse2.
    sse1 is strictly contained in sse2 if all its pixels are pixels of sse2,
    if its associated events are subsets of the corresponding events
    in sse2, and if sse2 contains events, or neuron ids in some event, which
    do not belong to sse1 (i.e. sse1 and sse2 are not identical)

    Both sse1 and sse2 must be provided as dictionaries of the type

    .. centered:: {(i1, j1): S1, (i2, j2): S2, ..., (iK, jK): SK},

    where each i, j is an integer and each S is a set of neuron ids.
    (See also: extract_sse() that extracts SSEs from given spiketrains).

    Parameters
    ----------
    sse1, sse2 : each a dict
        a dictionary of pixel positions (i, j) as keys, and sets S of
        synchronous events as values (see above).

    Returns
    -------
    is_equal : bool
        returns True if sse1 is identical to sse2

    '''
    # Remove empty links from sse11 and sse22, if any
    sse11 = _remove_empty_events(sse1)
    sse22 = _remove_empty_events(sse2)

    # Return whether sse11 == sse22
    return sse11 == sse22


def sse_isdisjoint(sse1, sse2):
    '''
    Given two sequences of synchronous events (SSEs) sse1 and sse2, each
    consisting of a pool of pixel positions and associated synchronous events
    (see below), determines whether sse1 and sse2 are disjoint.
    Two SSEs are disjoint if they don't share pixels, or if the events
    associated to common pixels are disjoint.

    Both sse1 and sse2 must be provided as dictionaries of the type

    .. centered:: {(i1, j1): S1, (i2, j2): S2, ..., (iK, jK): SK},

    where each i, j is an integer and each S is a set of neuron ids.
    (See also: extract_sse() that extracts SSEs from given spiketrains).

    Parameters
    ----------
    sse1, sse2 : each a dictionary
        a dictionary of pixel positions (i, j) as keys, and sets S of
        synchronous events as values (see above).

    Returns
    -------
    is_disjoint : bool
        returns True if sse1 is disjoint from sse2.

    '''
    # Remove empty links from sse11 and sse22, if any
    sse11 = _remove_empty_events(sse1)
    sse22 = _remove_empty_events(sse2)

    # If both SSEs are empty, return False (we consider them equal)
    if sse11 == {} and sse22 == {}:
        return False

    common_pixels = set(sse11.keys()).intersection(set(sse22.keys()))
    if common_pixels == set([]):
        return True
    elif all(sse11[p].isdisjoint(sse22[p]) for p in common_pixels):
        return True
    else:
        return False


def sse_issub(sse1, sse2):
    '''
    Given two sequences of synchronous events (SSEs) sse1 and sse2, each
    consisting of a pool of pixel positions and associated synchronous events
    (see below), determines whether sse1 is strictly contained in sse2.
    sse1 is strictly contained in sse2 if all its pixels are pixels of sse2,
    if its associated events are subsets of the corresponding events
    in sse2, and if sse2 contains non-empty events, or neuron ids in some
    event, which do not belong to sse1 (i.e. sse1 and sse2 are not identical)

    Both sse1 and sse2 must be provided as dictionaries of the type
            {(i1, j1): S1, (i2, j2): S2, ..., (iK, jK): SK},
    where each i, j is an integer and each S is a set of neuron ids.

    Parameters
    ----------
    sse1, sse2 : each a dict
        a dictionary of pixel positions (i, j) as keys, and sets S of
        synchronous events as values (see above).

    Returns
    -------
    is_sub : bool
        returns True if sse1 is a subset of sse2

    See Also
    -------
    extract_sse : extracts SSEs from given spiketrains

    '''
    # Remove empty links from sse11 and sse22, if any
    sse11 = _remove_empty_events(sse1)
    sse22 = _remove_empty_events(sse2)

    # Return False if sse11 and sse22 are disjoint
    if sse_isdisjoint(sse11, sse22):
        return False

    # Return False if any pixel in sse1 is not contained in sse2, or if any
    # link of sse1 is not a subset of the corresponding link in sse2.
    # Otherwise (if sse1 is a subset of sse2) continue
    for pixel1, link1 in sse11.items():
        if pixel1 not in sse22.keys():
            return False
        elif not link1.issubset(sse22[pixel1]):
            return False

    # Check that sse1 is a STRICT subset of sse2, i.e. that sse2 contains at
    # least one pixel or neuron id not present in sse1.
    return not sse_isequal(sse11, sse22)


def sse_issuper(sse1, sse2):
    '''
    Given two sequences of synchronous events (SSEs) sse1 and sse2, each
    consisting of a pool of pixel positions and associated synchronous events
    (see below), determines whether sse1 strictly contains sse2.
    sse1 strictly contains sse2 if it contains all pixels of sse2, if all
    associated events in sse1 contain those in sse2, and if sse1 additionally
    contains other pixels / events not contained in sse2.

    Both sse1 and sse2 must be provided as dictionaries of the type
            {(i1, j1): S1, (i2, j2): S2, ..., (iK, jK): SK},
    where each i, j is an integer and each S is a set of neuron ids.
    (See also: extract_sse() that extracts SSEs from given spiketrains).

    Note: sse_issuper(sse1, sse2) is identical to sse_issub(sse2, sse1).

    Parameters
    ----------
    sse1, sse2 : each a dict
        a dictionary of pixel positions (i, j) as keys, and sets S of
        synchronous events as values (see above).

    Returns
    -------
    is_super : bool
        returns True if sse1 strictly contains sse2.

    '''
    return sse_issub(sse2, sse1)


def sse_overlap(sse1, sse2):
    '''
    Given two sequences of synchronous events (SSEs) sse1 and sse2, each
    consisting of a pool of pixel positions and associated synchronous events
    (see below), determines whether sse1 strictly contains sse2.
    sse1 strictly contains sse2 if it contains all pixels of sse2, if all
    associated events in sse1 contain those in sse2, and if sse1 additionally
    contains other pixels / events not contained in sse2.

    Both sse1 and sse2 must be provided as dictionaries of the type
            {(i1, j1): S1, (i2, j2): S2, ..., (iK, jK): SK},
    where each i, j is an integer and each S is a set of neuron ids.
    (See also: extract_sse() that extracts SSEs from given spiketrains).

    Note: sse_issuper(sse1, sse2) is identical to sse_issub(sse2, sse1).

    Parameters
    ----------
    sse1, sse2 : each a dict
        a dictionary of pixel positions (i, j) as keys, and sets S of
        synchronous events as values (see above).

    Returns
    -------
    is_super : bool
        returns True if sse1 strictly contains sse2.

    '''
    return not (sse_issub(sse1, sse2) or sse_issuper(sse1, sse2) or
                sse_isequal(sse1, sse2) or sse_isdisjoint(sse1, sse2))
