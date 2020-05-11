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
    return None


def detect_synchrofacts(block, sampling_rate, segment='all', n=2, spread=2,
                        invert=False, delete=False, unit_type='all'):
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
        binsize dt = 1/sampling_rate and *n* spikes within *spread* consecutive
        bins are considered synchronous.
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

    if isinstance(segment, str):
        if 'all' in segment.lower():
            segment = range(len(block.segments))
        else:
            raise ValueError('Input parameter segment not understood.')

    elif isinstance(segment, int):
        segment = [segment]

    # make sure all quantities have units s
    binsize = (1 / sampling_rate).rescale(pq.s)

    for seg in segment:
        # data check
        if len(block.segments[seg].spiketrains) == 0:
            warnings.warn(
                'Segment {0} does not contain any spiketrains!'.format(seg))
            continue

        selected_sts, index = [], []

        # considering all spiketrains for unit_type == 'all'
        if isinstance(unit_type, str):
            if 'all' in unit_type.lower():
                selected_sts = block.segments[seg].spiketrains
                index = range(len(block.segments[seg].spiketrains))

        else:
            # extracting spiketrains which should be used for synchrofact
            # extraction based on given unit type
            # possible improvement by using masks for different conditions
            # and adding them up
            for i, st in enumerate(block.segments[seg].spiketrains):
                take_it = False
                for utype in unit_type:
                    if (utype[:2] == 'id' and
                        st.annotations['unit_id'] == int(
                            utype[2:])):
                        take_it = True
                    elif ((utype == 'sua' or utype == 'mua')
                          and utype in st.annotations
                          and st.annotations[utype]):
                        take_it = True
                if take_it:
                    selected_sts.append(st)
                    index.append(i)

        # if no spiketrains were selected
        if len(selected_sts) == 0:
            warnings.warn(
                'No matching spike trains for given unit selection'
                'criteria %s found' % unit_type)
            # we can skip to the next segment immediately since there are no
            # spiketrains to perform synchrofact detection on
            continue
        else:
            # find times of synchrony of size >=n
            bst = conv.BinnedSpikeTrain(selected_sts,
                                        binsize=binsize)
            # TODO: adapt everything below, find_complexity_intervals should
            #       return a neo.Epoch instead
            # TODO: we can probably clean up all implicit units once we use
            #       neo.Epoch for intervals
            # TODO: use conversion._detect_rounding_errors to ensure that
            #       there are no rounding errors
            complexity_intervals = find_complexity_intervals(bst,
                                                             min_complexity=n,
                                                             spread=spread)
            # get a sorted flattened array of the interval edges
            boundaries = complexity_intervals[1:].flatten(order='F')

        # j = index of pre-selected sts in selected_sts
        # idx = index of pre-selected sts in original
        # block.segments[seg].spiketrains
        for j, idx in enumerate(index):

            # all indices of spikes that are within the half-open intervals
            # defined by the boundaries
            # note that every second entry in boundaries is an upper boundary
            mask = np.array(
                np.searchsorted(boundaries,
                                selected_sts[j].times.rescale(pq.s).magnitude,
                                side='right') % 2,
                dtype=np.bool)
            if invert:
                mask = np.invert(mask)

            if delete:
                old_st = selected_sts[j]
                new_st = old_st[np.logical_not(mask)]
                block.segments[seg].spiketrains[idx] = new_st
                unit = old_st.unit
                if unit is not None:
                    unit.spiketrains[get_index(unit.spiketrains,
                                               old_st)] = new_st
                del old_st
            else:
                block.segments[seg].spiketrains[idx].array_annotate(
                    synchrofacts=mask)


def find_complexity_intervals(bst, min_complexity=2, spread=1):
    """
    Calculate the complexity (i.e. number of synchronous spikes)
    for each bin.

    For `spread = 1` this corresponds to a simple bincount.

    For `spread > 1` jittered synchrony is included, then spikes within
    `spread` consecutive bins are considered to be synchronous.
    Every bin of such a jittered synchronous event is assigned the
    complexity of the whole event, see example below.

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
    ...                             binsize=1 * pq.ms,
    ...                             t_start=0 * pq.ms)
    >>> print(bst.complexity().magnitude.flatten())
    [0. 2. 0. 0. 0. 0. 1. 1. 0. 0.]
    >>> print(bst.complexity(spread=2).magnitude.flatten())
    [0. 2. 0. 0. 0. 0. 2. 2. 0. 0.]
    """
    # TODO: documentation, example
    bincount = np.array(bst.to_sparse_array().sum(axis=0)).squeeze()

    if spread == 1:
        bin_indices = np.where(bincount >= min_complexity)[0]
        complexities = bincount[bin_indices]
        left_edges = bst.bin_edges[bin_indices].rescale(pq.s).magnitude
        right_edges = bst.bin_edges[bin_indices + 1].rescale(pq.s).magnitude
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
                if window_sum >= min_complexity:
                    complexities.append(window_sum)
                    left_edges.append(
                        bst.bin_edges[i].rescale(pq.s).magnitude.item())
                    right_edges.append(
                        bst.bin_edges[
                            i + last_nonzero_index + 1
                        ].rescale(pq.s).magnitude.item())
                i += last_nonzero_index + 1

    # TODO: return a neo.Epoch instead
    complexity_intervals = np.vstack((complexities, left_edges, right_edges))

    return complexity_intervals

