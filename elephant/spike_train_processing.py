# -*- coding: utf-8 -*-
"""
Module for spike train processing.


.. autosummary::
    :toctree: toctree/spike_train_processing/

    Synchrotool


:copyright: Copyright 2014-2020 by the Elephant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
"""

from __future__ import division, print_function, unicode_literals

from copy import deepcopy

import numpy as np

from elephant.statistics import Complexity

__all__ = [
    "Synchrotool"
]


class Synchrotool(Complexity):
    """
    Tool class to find, remove and/or annotate the presence of synchronous
    spiking events across multiple spike trains.

    The complexity is used to characterize synchronous events within the same
    spike train and across different spike trains in the `spiketrains` list.
    Such that, synchronous events can be found both in multi-unit and
    single-unit spike trains.

    This class inherits from :func:`elephant.statistics.Complexity`, see its
    documentation for more details and input parameters description.

    See also
    --------
    elephant.statistics.Complexity

    """

    def __init__(self, spiketrains,
                 sampling_rate,
                 bin_size=None,
                 binary=True,
                 spread=0,
                 tolerance=1e-8):

        self.annotated = False

        super(Synchrotool, self).__init__(spiketrains=spiketrains,
                                          bin_size=bin_size,
                                          sampling_rate=sampling_rate,
                                          binary=binary,
                                          spread=spread,
                                          tolerance=tolerance)

    def delete_synchrofacts(self, threshold, in_place=False, mode='delete'):
        """
        Delete or extract synchronous spiking events.

        Parameters
        ----------
        threshold : int
            Threshold value for the deletion of spikes engaged in synchronous
            activity.
              * `deletion_threshold >= 2` leads to all spikes with a larger or
                equal complexity value to be deleted/extracted.
              * `deletion_threshold <= 1` leads to a ValueError, since this
              would delete/extract all spikes and there are definitely more
              efficient ways of doing so.
        in_place : bool
            Determines whether the modification are made in place
            on ``self.input_spiketrains``.
            Default: False
        mode : bool
            Inversion of the mask for deletion of synchronous events.
              * ``'delete'`` leads to the deletion of all spikes with
                complexity >= `threshold`,
                i.e. deletes synchronous spikes.
              * ``'extract'`` leads to the deletion of all spikes with
                complexity < `threshold`, i.e. extracts synchronous spikes.
            Default: 'delete'

        Returns
        -------
        list of neo.SpikeTrain
            List of spiketrains where the spikes with
            ``complexity >= threshold`` have been deleted/extracted.
              * If ``in_place`` is True, the returned list is the same as
                ``self.input_spiketrains``.
              * If ``in_place`` is False, the returned list is a deepcopy of
                ``self.input_spiketrains``.

        """

        if not self.annotated:
            self.annotate_synchrofacts()

        if mode not in ['delete', 'extract']:
            raise ValueError(str(mode) + ' is not a valid mode. '
                             "valid modes are ['delete', 'extract']")

        if threshold <= 1:
            raise ValueError('A deletion threshold <= 1 would result '
                             'in the deletion of all spikes.')

        if in_place:
            spiketrain_list = self.input_spiketrains
        else:
            spiketrain_list = deepcopy(self.input_spiketrains)

        for idx, st in enumerate(spiketrain_list):
            mask = st.array_annotations['complexity'] < threshold
            if mode == 'extract':
                mask = np.invert(mask)
            new_st = st[mask]
            spiketrain_list[idx] = new_st
            if in_place:
                unit = st.unit
                segment = st.segment
                if unit is not None:
                    new_index = self._get_spiketrain_index(
                        unit.spiketrains, st)
                    unit.spiketrains[new_index] = new_st
                if segment is not None:
                    new_index = self._get_spiketrain_index(
                        segment.spiketrains, st)
                    segment.spiketrains[new_index] = new_st

        return spiketrain_list

    def annotate_synchrofacts(self):
        """
        Annotate the complexity of each spike in the
        ``self.epoch.array_annotations`` *in-place*.
        """
        epoch_complexities = self.epoch.array_annotations['complexity']
        right_edges = (
            self.epoch.times.magnitude.flatten()
            + self.epoch.durations.rescale(
                self.epoch.times.units).magnitude.flatten()
        )

        for idx, st in enumerate(self.input_spiketrains):

            # all indices of spikes that are within the half-open intervals
            # defined by the boundaries
            # note that every second entry in boundaries is an upper boundary
            spike_to_epoch_idx = np.searchsorted(
                right_edges,
                st.times.rescale(self.epoch.times.units).magnitude.flatten())
            complexity_per_spike = epoch_complexities[spike_to_epoch_idx]

            st.array_annotate(complexity=complexity_per_spike)

        self.annotated = True

    def _get_spiketrain_index(self, spiketrain_list, spiketrain):
        for index, item in enumerate(spiketrain_list):
            if item is spiketrain:
                return index
        raise ValueError("Spiketrain is not found in the list")
