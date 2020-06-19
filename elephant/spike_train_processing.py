# -*- coding: utf-8 -*-
"""
Module for spike train processing

:copyright: Copyright 2014-2020 by the Elephant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
"""

from __future__ import division

import numpy as np


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
