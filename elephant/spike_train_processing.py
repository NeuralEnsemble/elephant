from __future__ import division

import neo
import elephant.conversion as conv
import quantities as pq
import numpy as np
import warnings


def get_index(lst, obj):
    for index, item in enumerate(lst):
        if item is obj:
            return index


def _check_spiketrains(spiketrains):
    if len(spiketrains) == 0:
        raise ValueError('The spiketrains should not be empty!')

    # check that all list elements are spike trains
    for spiketrain in spiketrains:
        if not isinstance(spiketrain, neo.SpikeTrain):
            raise TypeError('not all elements of spiketrains are'
                            'neo.SpikeTrain objects')


def detect_synchrofacts(spiketrains, sampling_rate, spread=2,
                        invert=False, deletion_threshold=None):
    """
    Given block with spike trains, find all spikes engaged
    in synchronous events of size *n* or higher. Two events are considered
    synchronous if they occur within spread/sampling_rate of one another.

    *Args*
    ------
    block [list]:
        a block containing neo spike trains

    segment [int or iterable or str. Default: 1]:
        indices of segments in the block. Can be an integer, an iterable object
        or a string containing 'all'. Indicates on which segments of the block
        the synchrofact removal should be performed.

    n [int. Default: 2]:
        minimum number of coincident spikes to report synchrony

    spread [int. Default: 2]:
        number of bins of size 1/sampling_rate in which to check for
        synchronous spikes.  *n* spikes within *spread* consecutive bins are
        considered synchronous.

    sampling_rate [quantity. Default: 30000/s]:
        Sampling rate of the spike trains. The spike trains are binned with
        bin_size dt = 1/sampling_rate and *n* spikes within *spread*
        consecutive bins are considered synchronous.
        Groups of *n* or more synchronous spikes are deleted/annotated.

    invert [bool. Default: True]:
        invert the mask for annotation/deletion (Default:False).
        False annotates synchrofacts with False and other spikes with True or
        deletes everything except for synchrofacts for delete = True.

    delete [bool. Default: False]:
        delete spikes engaged in synchronous activity. If set to False the
        spiketrains are array-annotated and the spike times are kept unchanged.

    unit_type [list of strings. Default 'all']:
        selects only spiketrain of certain units / channels for synchrofact
        extraction.  unit_type = 'all' considers all provided spiketrains
        Accepted unit types: 'sua', 'mua', 'idX'
                             (where X is the id number requested)
    """
    # TODO: refactor docs, correct description of spread parameter

    if deletion_threshold is not None and deletion_threshold <= 1:
        raise ValueError('A deletion_threshold <= 1 would result'
                         'in deletion of all spikes.')

    _check_spiketrains(spiketrains)

    # find times of synchrony of size >=n
    complexity_epoch = find_complexity_intervals(spiketrains,
                                                 sampling_rate,
                                                 spread=spread)
    complexity = complexity_epoch.array_annotations['complexity']
    right_edges = complexity_epoch.times + complexity_epoch.durations

    # j = index of pre-selected sts in spiketrains
    # idx = index of pre-selected sts in original
    # block.segments[seg].spiketrains
    for idx, st in enumerate(spiketrains):

        # all indices of spikes that are within the half-open intervals
        # defined by the boundaries
        # note that every second entry in boundaries is an upper boundary
        spike_to_epoch_idx = np.searchsorted(right_edges,
                                             st.times.rescale(
                                                 right_edges.units))
        complexity_per_spike = complexity[spike_to_epoch_idx]

        st.array_annotate(complexity=complexity_per_spike)

        if deletion_threshold is not None:
            mask = complexity_per_spike < deletion_threshold
            if invert:
                mask = np.invert(mask)
            old_st = st
            new_st = old_st[mask]
            spiketrains[idx] = new_st
            unit = old_st.unit
            segment = old_st.segment
            if unit is not None:
                unit.spiketrains[get_index(unit.spiketrains,
                                           old_st)] = new_st
            if segment is not None:
                segment.spiketrains[get_index(segment.spiketrains,
                                              old_st)] = new_st
            del old_st

    return complexity_epoch


def find_complexity_intervals(spiketrains, sampling_rate,
                              bin_size=None, spread=1):
    """
    Calculate the complexity (i.e. number of synchronous spikes)
    for each bin.

    For `spread = 1` this corresponds to a simple bincount.

    For `spread > 1` spikes separated by fewer than `spread - 1`
    empty bins are considered synchronous.

    Parameters
    ----------
    min_complexity : int, optional
        Minimum complexity to report
        Default: 2.
    spread : int, optional
        Number of bins in which to check for synchronous spikes.
        Spikes within `spread` consecutive bins are considered synchronous.
        Default: 2.

    Returns
    -------
    complexity_intervals : np.ndarray
        An array containing complexity values, left and right edges of all
        intervals with at least `min_complexity` spikes separated by fewer
        than `spread - 1` empty bins.
        Output shape (3, num_complexity_intervals)

    Raises
    ------
    ValueError
        When `t_stop` is smaller than `t_start`.

    Examples
    --------
    >>> import elephant.conversion as conv
    >>> import neo
    >>> import quantities as pq
    >>> st1 = neo.SpikeTrain([1, 6] * pq.ms,
    ...                      t_stop=10.0 * pq.ms)
    >>> st2 = neo.SpikeTrain([1, 7] * pq.ms,
    ...                      t_stop=10.0 * pq.ms)
    >>> bst = conv.BinnedSpikeTrain([st1, st2], num_bins=10,
    ...                             bin_size=1 * pq.ms,
    ...                             t_start=0 * pq.ms)
    >>> print(bst.complexity().magnitude.flatten())
    [0. 2. 0. 0. 0. 0. 1. 1. 0. 0.]
    >>> print(bst.complexity(spread=2).magnitude.flatten())
    [0. 2. 0. 0. 0. 0. 2. 2. 0. 0.]
    """
    _check_spiketrains(spiketrains)

    if bin_size is None:
        bin_size = 1 / sampling_rate
    elif bin_size < 1 / sampling_rate:
        raise ValueError('The bin size should be at least'
                         '1 / sampling_rate (which is the'
                         'default).')

    # TODO: documentation, example
    min_t_start = min([st.t_start for st in spiketrains])

    bst = conv.BinnedSpikeTrain(spiketrains,
                                binsize=bin_size)
    bincount = np.array(bst.to_sparse_array().sum(axis=0)).squeeze()

    if spread == 1:
        bin_indices = np.nonzero(bincount)[0]
        complexities = bincount[bin_indices]
        left_edges = bst.bin_edges[bin_indices]
        right_edges = bst.bin_edges[bin_indices + 1]
    else:
        i = 0
        complexities = []
        left_edges = []
        right_edges = []
        while i < len(bincount):
            current_bincount = bincount[i]
            if current_bincount == 0:
                i += 1
            else:
                last_window_sum = current_bincount
                last_nonzero_index = 0
                current_window = bincount[i:i+spread]
                window_sum = current_window.sum()
                while window_sum > last_window_sum:
                    last_nonzero_index = np.nonzero(current_window)[0][-1]
                    current_window = bincount[i:
                                              i + last_nonzero_index
                                              + spread]
                    last_window_sum = window_sum
                    window_sum = current_window.sum()
                complexities.append(window_sum)
                left_edges.append(
                    bst.bin_edges[i].magnitude.item())
                right_edges.append(
                    bst.bin_edges[
                        i + last_nonzero_index + 1
                    ].magnitude.item())
                i += last_nonzero_index + 1

        # we dropped units above, neither concatenate nor append works with
        # arrays of quantities
        left_edges *= bst.bin_edges.units
        right_edges *= bst.bin_edges.units

    # ensure that spikes are not on the bin edges
    bin_shift = .5 / sampling_rate
    left_edges -= bin_shift
    right_edges -= bin_shift

    # ensure that epoch does not start before the minimum t_start
    left_edges[0] = min(min_t_start, left_edges[0])

    complexity_epoch = neo.Epoch(times=left_edges,
                                 durations=right_edges - left_edges,
                                 array_annotations={'complexity':
                                                    complexities})

    return complexity_epoch

