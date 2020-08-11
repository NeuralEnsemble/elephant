# -*- coding: utf-8 -*-
"""
Module for spike train processing.


.. autosummary::
    :toctree: toctree/spike_train_processing/

    synchrotool


:copyright: Copyright 2014-2020 by the Elephant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
"""

from __future__ import division, print_function, unicode_literals

from copy import deepcopy

import numpy as np

from elephant.statistics import complexity


class synchrotool(complexity):
    """
    Tool class to find, remove and/or annotate the presence of synchronous
    spiking events across multiple spike trains.

    The complexity is used to characterize synchronous events within the same
    spike train and across different spike trains in the `spiketrains` list.
    Such that, synchronous events can be found both in multi-unit and
    single-unit spike trains.

    This class inherits from ``elephant.statistics.complexity``, see its
    documentation for more details.

    *The rest of this documentation is copied from
    ``elephant.statistics.complexity`` !!!
    TODO: Figure out a better way to merge the docstrings.*

    Parameters
    ----------
    spiketrains : list of neo.SpikeTrain
        Spike trains with a common time axis (same `t_start` and `t_stop`)
    sampling_rate : pq.Quantity, optional
        Sampling rate of the spike trains with units of 1/time.
        Default: None
    bin_size : pq.Quantity, optional
        FIXME: no bin_size detected
        Width of the histogram's time bins with units of time.
        The user must specify the `bin_size` or the `sampling_rate`.
          * If no `bin_size` is specified and the `sampling_rate` is available
            1/`sampling_rate` is used.
          * If both are given then `bin_size` is used.
        Default: None
    binary : bool, optional
          * If `True` then the time histograms will be binary.
          * If `False` the total number of synchronous spikes is counted in the
            time histogram.
        Default: True
    spread : int, optional
        Number of bins in which to check for synchronous spikes.
        Spikes that occur separated by `spread - 1` or less empty bins are
        considered synchronous.
          * ``spread = 0`` corresponds to a bincount accross spike trains.
          * ``spread = 1`` corresponds to counting consecutive spikes.
          * ``spread = 2`` corresponds to counting consecutive spikes and
            spikes separated by exactly 1 empty bin.
          * ``spread = n`` corresponds to counting spikes separated by exactly
            or less than `n - 1` empty bins.
        Default: 0
    tolerance : float, optional
        Tolerance for rounding errors in the binning process and in the input
        data.
        Default: 1e-8

    Attributes
    ----------
    epoch : neo.Epoch
        An epoch object containing complexity values, left edges and durations
        of all intervals with at least one spike.
          * ``epoch.array_annotations['complexity']`` contains the
            complexity values per spike.
          * ``epoch.times`` contains the left edges.
          * ``epoch.durations`` contains the durations.
    time_histogram : neo.Analogsignal
        A `neo.AnalogSignal` object containing the histogram values.
        `neo.AnalogSignal[j]` is the histogram computed between
        `t_start + j * binsize` and `t_start + (j + 1) * binsize`.
          * If ``binary = True`` : Number of neurons that spiked in each bin,
            regardless of the number of spikes.
          * If ``binary = False`` : Number of neurons and spikes per neurons
            in each bin.
    complexity_histogram : np.ndarray
        The number of occurrences of events of different complexities.
        `complexity_hist[i]` corresponds to the number of events of
        complexity `i` for `i > 0`.

    Raises
    ------
    ValueError
        When `t_stop` is smaller than `t_start`.

        When both `sampling_rate` and `bin_size` are not specified.

        When `spread` is not a positive integer.

        When `spiketrains` is an empty list.

        When `t_start` is not the same for all spiketrains

        When `t_stop` is not the same for all spiketrains

    TypeError
        When `spiketrains` is not a list.

        When the elements in `spiketrains` are not instances of neo.SpikeTrain

    Notes
    -----
    Note that with most common parameter combinations spike times can end up
    on bin edges. This makes the binning susceptible to rounding errors which
    is accounted for by moving spikes which are within tolerance of the next
    bin edge into the following bin. This can be adjusted using the tolerance
    parameter and turned off by setting `tolerance=None`.

    See also
    --------
    elephant.statistics.complexity

    """

    def __init__(self, spiketrains,
                 sampling_rate,
                 binary=True,
                 spread=0,
                 tolerance=1e-8):

        self.annotated = False

        super(synchrotool, self).__init__(spiketrains=spiketrains,
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
        Annotate the complexity of each spike in the array_annotations
        *in-place*.
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
