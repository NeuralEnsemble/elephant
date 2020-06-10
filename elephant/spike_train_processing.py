# -*- coding: utf-8 -*-
"""
Module for spike train processing

:copyright: Copyright 2014-2020 by the Elephant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
"""

from __future__ import division

import neo
import elephant.conversion as conv
import numpy as np
import quantities as pq
from .utils import _check_consistency_of_spiketrainlist
from elephant.statistics import time_histogram
import warnings


class complexity:
    """
    docstring TODO

    COPIED FROM PREVIOUS GET EPOCHS AS IS:
    Calculate the complexity (i.e. number of synchronous spikes found)
    at `sampling_rate` precision in a list of spiketrains.

    Complexity is calculated by counting the number of spikes (i.e. non-empty
    bins) that occur separated by `spread - 1` or less empty bins, within and
    across spike trains in the `spiketrains` list.

    Parameters
    ----------
    spiketrains : list of neo.SpikeTrains
        a list of neo.SpikeTrains objects. These spike trains should have been
        recorded simultaneously.
    sampling_rate : pq.Quantity
        Sampling rate of the spike trains.
    spread : int, optional
        Number of bins in which to check for synchronous spikes.
        Spikes that occur separated by `spread - 1` or less empty bins are
        considered synchronous.
          * `spread = 0` corresponds to a bincount accross spike trains.
          * `spread = 1` corresponds to counting consecutive spikes.
          * `spread = 2` corresponds to counting consecutive spikes and
            spikes separated by exactly 1 empty bin.
          * `spread = n` corresponds to counting spikes separated by exactly
            or less than `n - 1` empty bins.
        Default: 0

    Returns
    -------
    complexity_intervals : neo.Epoch
        An epoch object containing complexity values, left edges and durations
        of all intervals with at least one spike.

    Raises
    ------
    ValueError
        When `t_stop` is smaller than `t_start`.

    Examples
    --------
    Here the behavior of
    `elephant.spike_train_processing.precise_complexity_intervals` is shown, by
    applying the function to some sample spiketrains.

    >>> import neo
    >>> import quantities as pq

    >>> sampling_rate = 1/pq.ms
    >>> st1 = neo.SpikeTrain([1, 4, 6] * pq.ms, t_stop=10.0 * pq.ms)
    >>> st2 = neo.SpikeTrain([1, 5, 8] * pq.ms, t_stop=10.0 * pq.ms)

    >>> # spread = 0, a simple bincount
    >>> ep1 = precise_complexity_intervals([st1, st2], sampling_rate)
    >>> print(ep1.array_annotations['complexity'].flatten())
    [2 1 1 1 1]
    >>> print(ep1.times)
    [0.  3.5 4.5 5.5 7.5] ms
    >>> print(ep1.durations)
    [1.5 1.  1.  1.  1. ] ms

    >>> # spread = 1, consecutive spikes
    >>> ep2 = precise_complexity_intervals([st1, st2], sampling_rate, spread=1)
    >>> print(ep2.array_annotations['complexity'].flatten())
    [2 3 1]
    >>> print(ep2.times)
    [0.  3.5 7.5] ms
    >>> print(ep2.durations)
    [1.5 3.  1. ] ms

    >>> # spread = 2, consecutive spikes and separated by 1 empty bin
    >>> ep3 = precise_complexity_intervals([st1, st2], sampling_rate, spread=2)
    >>> print(ep3.array_annotations['complexity'].flatten())
    [2 4]
    >>> print(ep3.times)
    [0.  3.5] ms
    >>> print(ep3.durations)
    [1.5 5. ] ms
    """

    def __init__(self, spiketrains,
                 sampling_rate=None,
                 bin_size=None,
                 binary=True,
                 spread=0):

        if isinstance(spiketrains, list):
            _check_consistency_of_spiketrainlist(spiketrains)
        else:
            raise TypeError('spiketrains should be a list of neo.SpikeTrain')
        self.input_spiketrains = spiketrains
        self.sampling_rate = sampling_rate
        self.bin_size = bin_size
        self.binary = binary
        self.spread = spread

        if bin_size is None and sampling_rate is None:
            raise ValueError('No bin_size or sampling_rate was speficied!')
        elif bin_size is None and sampling_rate is not None:
            self.bin_size = 1 / self.sampling_rate

        if spread < 0:
            raise ValueError('Spread must be >=0')
        elif spread == 0:
            self.time_histogram, self.histogram = self._histogram_no_spread()
        else:
            print('Complexity calculated at sampling rate precision')
            # self.epoch = self.precise_complexity_intervals()
            self.epoch = self.get_epoch()
            self.histogram = self._histogram_with_spread()

        return self

    @property
    def pdf(self):
        """
        Normalization of the Complexity Histogram (probabilty distribution)
        """
        norm_hist = self.histogram / self.histogram.sum()
        # Convert the Complexity pdf to an neo.AnalogSignal
        pdf = neo.AnalogSignal(
            np.array(norm_hist).reshape(len(norm_hist), 1) *
            pq.dimensionless, t_start=0 * pq.dimensionless,
            sampling_period=1 * pq.dimensionless)
        return pdf

    # @property
    # def epoch(self):
    #     if self.spread == 0:
    #         warnings.warn('No epoch for cases with spread = 0')
    #         return None
    #     else:
    #         return self._epoch

    def _histogram_no_spread(self):
        """
        Complexity Distribution of a list of `neo.SpikeTrain` objects.

        Probability density computed from the complexity histogram which is the
        histogram of the entries of the population histogram of clipped
        (binary) spike trains computed with a bin width of `binsize`.
        It provides for each complexity (== number of active neurons per bin)
        the number of occurrences. The normalization of that histogram to 1 is
        the probability density.

        Implementation is based on [1]_.

        Returns
        -------
        complexity_distribution : neo.AnalogSignal
            A `neo.AnalogSignal` object containing the histogram values.
            `neo.AnalogSignal[j]` is the histogram computed between
            `t_start + j * binsize` and `t_start + (j + 1) * binsize`.

        See also
        --------
        elephant.conversion.BinnedSpikeTrain

        References
        ----------
        .. [1] S. Gruen, M. Abeles, & M. Diesmann, "Impact of higher-order
               correlations on coincidence distributions of massively parallel
               data," In "Dynamic Brain - from Neural Spikes to Behaviors",
               pp. 96-114, Springer Berlin Heidelberg, 2008.

        """
        # Computing the population histogram with parameter binary=True to
        # clip the spike trains before summing
        pophist = time_histogram(self.input_spiketrains,
                                 self.bin_size,
                                 binary=self.binary)

        # Computing the histogram of the entries of pophist
        complexity_hist = np.histogram(
            pophist.magnitude,
            bins=range(0, len(self.input_spiketrains) + 2))[0]

        return pophist, complexity_hist

    def _histogram_with_spread(self):
        """
        Calculate the complexity histogram;
        the number of occurrences of events of different complexities.

        Returns
        -------
        complexity_histogram : np.ndarray
            A histogram of complexities. `complexity_histogram[i]` corresponds
            to the number of events of complexity `i` for `i > 0`.
        """
        complexity_histogram = np.bincount(
            self.epoch.array_annotations['complexity'])
        return complexity_histogram

    def get_epoch(self):
        bst = conv.BinnedSpikeTrain(self.input_spiketrains,
                                    binsize=self.bin_size)

        if self.binary:
            binarized = bst.to_sparse_bool_array()
            bincount = np.array(binarized.sum(axis=0)).squeeze()
        else:
            bincount = np.array(bst.to_sparse_array().sum(axis=0)).squeeze()

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
                current_window = bincount[i:i + self.spread + 1]
                window_sum = current_window.sum()
                while window_sum > last_window_sum:
                    last_nonzero_index = np.nonzero(current_window)[0][-1]
                    current_window = bincount[i:
                                              i + last_nonzero_index
                                              + self.spread + 1]
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

        # we dropped units above, neither concatenate nor append works
        # with arrays of quantities
        left_edges *= bst.bin_edges.units
        right_edges *= bst.bin_edges.units

        if self.sampling_rate:
            # ensure that spikes are not on the bin edges
            bin_shift = .5 / self.sampling_rate
            left_edges -= bin_shift
            right_edges -= bin_shift
        else:
            warnings.warn('No sampling rate specified. '
                          'Note that using the complexity epoch to get '
                          'precise spike times can lead to rounding errors.')

        # ensure that an epoch does not start before the minimum t_start
        min_t_start = min([st.t_start for st in self.input_spiketrains])
        left_edges[0] = min(min_t_start, left_edges[0])

        complexity_epoch = neo.Epoch(times=left_edges,
                                     durations=right_edges - left_edges,
                                     array_annotations={'complexity':
                                                        complexities})

        return complexity_epoch


class synchrotool(complexity):
    complexity.__doc_

    def __init__(self, spiketrains,
                 sampling_rate=None,
                 spread=1):
        self.spiketrains = spiketrains
        self.sampling_rate = sampling_rate
        self.spread = spread

        # self.super(...).__init__()

        # find times of synchrony of size >=n
        # complexity_epoch =

        # ...
        return self

    def annotate_synchrofacts(self):
        return None

    def delete_synchrofacts(self):
        return None

    def extract_synchrofacts(self):
        return None

    # def delete_synchrofacts(self, in_place=False):
    #
    #     if not in_place:
    #         # return clean_spiketrains
    #
    # @property
    # def synchrofacts(self):
    #     self.synchrofacts = self.detect_synchrofacts(deletion_threshold=1,
    #                                                  invert_delete=True)

    def detect_synchrofacts(self,
                            deletion_threshold=None,
                            invert_delete=False):
        """
        Given a list of neo.Spiketrain objects, calculate the number of synchronous
        spikes found and optionally delete or extract them from the given list
        *in-place*.

        The spike trains are binned at sampling precission
        (i.e. bin_size = 1 / `sampling_rate`)

        Two spikes are considered synchronous if they occur separated by strictly
        fewer than `spread - 1` empty bins from one another. See
        `elephant.statistics.precise_complexity_intervals` for a detailed
        description of how synchronous events are counted.

        Synchronous events are considered within the same spike train and across
        different spike trains in the `spiketrains` list. Such that, synchronous
        events can be found both in multi-unit and single-unit spike trains.

        The spike trains in the `spiketrains` list are annotated with the
        complexity value of each spike in their :attr:`array_annotations`.


        Parameters
        ----------
        spiketrains : list of neo.SpikeTrains
            a list of neo.SpikeTrains objects. These spike trains should have been
            recorded simultaneously.
        sampling_rate : pq.Quantity
            Sampling rate of the spike trains. The spike trains are binned with
            bin_size = 1 / `sampling_rate`.
        spread : int
            Number of bins in which to check for synchronous spikes.
            Spikes that occur separated by `spread - 1` or less empty bins are
            considered synchronous.
            Default: 1
        deletion_threshold : int, optional
            Threshold value for the deletion of spikes engaged in synchronous
            activity.
              * `deletion_threshold = None` leads to no spikes being deleted, spike
                trains are array-annotated and the spike times are kept unchanged.
              * `deletion_threshold >= 2` leads to all spikes with a larger or
                equal complexity value to be deleted *in-place*.
              * `deletion_threshold` cannot be set to 1 (this would delete all
                spikes and there are definitely more efficient ways of doing this)
              * `deletion_threshold <= 0` leads to a ValueError.
            Default: None
        invert_delete : bool
            Inversion of the mask for deletion of synchronous events.
              * `invert_delete = False` leads to the deletion of all spikes with
                complexity >= `deletion_threshold`,
                i.e. deletes synchronous spikes.
              * `invert_delete = True` leads to the deletion of all spikes with
                complexity < `deletion_threshold`, i.e. returns synchronous spikes.
            Default: False

        Returns
        -------
        complexity_epoch : neo.Epoch
            An epoch object containing complexity values, left edges and durations
            of all intervals with at least one spike.
            Calculated with
            `elephant.spike_train_processing.precise_complexity_intervals`.
              * ``complexity_epoch.array_annotations['complexity']`` contains the
                complexity values per spike.
              * ``complexity_epoch.times`` contains the left edges.
              * ``complexity_epoch.durations`` contains the durations.

        See also
        --------
        elephant.spike_train_processing.precise_complexity_intervals

        """
        if deletion_threshold is not None and deletion_threshold <= 1:
            raise ValueError('A deletion_threshold <= 1 would result '
                             'in deletion of all spikes.')

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
                if invert_delete:
                    mask = np.invert(mask)
                old_st = st
                new_st = old_st[mask]
                spiketrains[idx] = new_st
                unit = old_st.unit
                segment = old_st.segment
                if unit is not None:
                    unit.spiketrains[self._get_index(unit.spiketrains,
                                                old_st)] = new_st
                if segment is not None:
                    segment.spiketrains[self._get_index(segment.spiketrains,
                                                   old_st)] = new_st
                del old_st

        return complexity_epoch

    def _get_index(lst, obj):
        for index, item in enumerate(lst):
            if item is obj:
                return index
