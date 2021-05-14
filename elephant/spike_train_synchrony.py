# -*- coding: utf-8 -*-
"""
Functions to measure the synchrony of several spike trains.


Synchrony Measures
------------------

.. autosummary::
    :toctree: _toctree/spike_train_synchrony/

    spike_contrast
    Synchrotool


:copyright: Copyright 2014-2020 by the Elephant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
"""
from __future__ import division, print_function, unicode_literals

import warnings
from collections import namedtuple
from copy import deepcopy

import neo
import numpy as np
import quantities as pq

from elephant.statistics import Complexity
from elephant.utils import is_time_quantity, check_same_units

SpikeContrastTrace = namedtuple("SpikeContrastTrace", (
    "contrast", "active_spiketrains", "synchrony", "bin_size"))


__all__ = [
    "SpikeContrastTrace",
    "spike_contrast",
    "Synchrotool"
]


def _get_theta_and_n_per_bin(spiketrains, t_start, t_stop, bin_size):
    """
    Calculates theta (amount of spikes per bin) and the amount of active spike
    trains per bin of one spike train.
    """
    bin_step = bin_size / 2
    edges = np.arange(t_start, t_stop + bin_step, bin_step)
    # Calculate histogram for every spike train
    histogram = np.vstack([
        _binning_half_overlap(st, edges=edges)
        for st in spiketrains
    ])
    # Amount of spikes per bin
    theta = histogram.sum(axis=0)
    # Amount of active spike trains per bin
    n_active_per_bin = np.count_nonzero(histogram, axis=0)

    return theta, n_active_per_bin


def _binning_half_overlap(spiketrain, edges):
    """
    Referring to [1] overlapping the bins creates a better result.
    """
    histogram, bin_edges = np.histogram(spiketrain, bins=edges)
    histogram = histogram[:-1] + histogram[1:]
    return histogram


def spike_contrast(spiketrains, t_start=None, t_stop=None,
                   min_bin=10 * pq.ms, bin_shrink_factor=0.9,
                   return_trace=False):
    """
    Calculates the synchrony of spike trains, according to
    :cite:`synchrony-Ciba18_136`. The spike trains can have different lengths.

    Original implementation by: Philipp Steigerwald [s160857@th-ab.de]

    Visualization is covered in
    :func:`viziphant.spike_train_synchrony.plot_spike_contrast`.

    Parameters
    ----------
    spiketrains : list of neo.SpikeTrain
        A list of input spike trains to calculate the synchrony from.
    t_start : pq.Quantity, optional
        The beginning of the spike train. If None, it's taken as the minimum
        value of `t_start`s of the input spike trains.
        Default: None
    t_stop : pq.Quantity, optional
        The end of the spike train. If None, it's taken as the maximum value
        of `t_stop` of the input spike trains.
        Default: None
    min_bin : pq.Quantity, optional
        Sets the minimum value for the `bin_min` that is calculated by the
        algorithm and defines the smallest bin size to compute the histogram
        of the input `spiketrains`.
        Default: 0.01 ms
    bin_shrink_factor : float, optional
        A multiplier to shrink the bin size on each iteration. The value must
        be in range `(0, 1)`.
        Default: 0.9
    return_trace : bool, optional
        If set to True, returns a history of spike-contrast synchrony, computed
        for a range of different bin sizes, alongside with the maximum value of
        the synchrony.
        Default: False

    Returns
    -------
    synchrony : float
        Returns the synchrony of the input spike trains.
    spike_contrast_trace : namedtuple
        If `return_trace` is set to True, a `SpikeContrastTrace` namedtuple is
        returned with the following attributes:

          `.contrast` - the average sum of differences of the number of spikes
          in subsuequent bins;

          `.active_spiketrains` - the average number of spikes per bin,
          weighted by the number of spike trains containing at least one spike
          inside the bin;

          `.synchrony` - the product of `contrast` and `active_spiketrains`;

          `.bin_size` - the X axis, a list of bin sizes that correspond to
          these traces.

    Raises
    ------
    ValueError
        If `bin_shrink_factor` is not in (0, 1) range.

        If the input spike trains constist of a single spiketrain.

        If all input spike trains contain no more than 1 spike.
    TypeError
        If the input spike trains is not a list of `neo.SpikeTrain` objects.

        If `t_start`, `t_stop`, or `min_bin` are not time quantities.

    Examples
    --------
    >>> import quantities as pq
    >>> from elephant.spike_train_generation import homogeneous_poisson_process
    >>> from elephant.spike_train_synchrony import spike_contrast
    >>> spiketrain_1 = homogeneous_poisson_process(rate=20*pq.Hz,
    ...     t_stop=1000*pq.ms)
    >>> spiketrain_2 = homogeneous_poisson_process(rate=20*pq.Hz,
    ...     t_stop=1000*pq.ms)
    >>> spike_contrast([spiketrain_1, spiketrain_2])
    0.4192546583850932

    """
    if not 0. < bin_shrink_factor < 1.:
        raise ValueError(f"'bin_shrink_factor' ({bin_shrink_factor}) must be "
                         "in range (0, 1).")
    if not len(spiketrains) > 1:
        raise ValueError("Spike contrast measure requires more than 1 input "
                         "spiketrain.")
    check_same_units(spiketrains, object_type=neo.SpikeTrain)
    if not is_time_quantity(t_start, t_stop, allow_none=True):
        raise TypeError("'t_start' and 't_stop' must be time quantities.")
    if not is_time_quantity(min_bin):
        raise TypeError("'min_bin' must be a time quantity.")

    if t_start is None:
        t_start = min(st.t_start for st in spiketrains)
    if t_stop is None:
        t_stop = max(st.t_stop for st in spiketrains)

    # convert everything to spiketrain units
    units = spiketrains[0].units
    spiketrains = [st.magnitude for st in spiketrains]
    t_start = t_start.rescale(units).item()
    t_stop = t_stop.rescale(units).item()
    min_bin = min_bin.rescale(units).item()

    spiketrains = [times[(times >= t_start) & (times <= t_stop)]
                   for times in spiketrains]

    n_spiketrains = len(spiketrains)
    n_spikes_total = sum(map(len, spiketrains))

    duration = t_stop - t_start
    bin_max = duration / 2

    try:
        isi_min = min(np.diff(st).min() for st in spiketrains if len(st) > 1)
    except TypeError:
        raise ValueError("All input spiketrains contain no more than 1 spike.")
    bin_min = max(isi_min / 2, min_bin)

    contrast_list = []
    active_spiketrains = []
    synchrony_curve = []

    # Set new time boundaries
    t_start = t_start - isi_min
    t_stop = t_stop + isi_min

    bin_sizes = []
    bin_size = bin_max
    while bin_size >= bin_min:
        bin_sizes.append(bin_size)
        # Calculate Theta and n
        theta_k, n_k = _get_theta_and_n_per_bin(spiketrains,
                                                t_start=t_start,
                                                t_stop=t_stop,
                                                bin_size=bin_size)

        # calculate synchrony_curve = contrast * active_st
        active_st = (np.sum(n_k * theta_k) / np.sum(theta_k) - 1) / (
                    n_spiketrains - 1)
        contrast = np.sum(np.abs(np.diff(theta_k))) / (2 * n_spikes_total)
        # Contrast: sum(|derivation|) / (2*#Spikes)
        synchrony = contrast * active_st

        contrast_list.append(contrast)
        active_spiketrains.append(active_st)
        synchrony_curve.append(synchrony)

        # New bin size
        bin_size *= bin_shrink_factor

    # Sync value is maximum of the cost function C
    synchrony = max(synchrony_curve)

    if return_trace:
        spike_contrast_trace = SpikeContrastTrace(
            contrast=contrast_list,
            active_spiketrains=active_spiketrains,
            synchrony=synchrony_curve,
            bin_size=bin_sizes * units,
        )
        return synchrony, spike_contrast_trace

    return synchrony


class Synchrotool(Complexity):
    """
    Tool class to find, remove and/or annotate the presence of synchronous
    spiking events across multiple spike trains.

    The complexity is used to characterize synchronous events within the same
    spike train and across different spike trains in the `spiketrains` list.
    This way synchronous events can be found both in multi-unit and
    single-unit spike trains.

    This class inherits from :class:`elephant.statistics.Complexity`, see its
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
        in_place : bool, optional
            Determines whether the modification are made in place
            on ``self.input_spiketrains``.
            Default: False
        mode : {'delete', 'extract'}, optional
            Inversion of the mask for deletion of synchronous events.
              * ``'delete'`` leads to the deletion of all spikes with
                complexity >= `threshold`,
                i.e. deletes synchronous spikes.
              * ``'extract'`` leads to the deletion of all spikes with
                complexity < `threshold`, i.e. extracts synchronous spikes.
            Default: 'delete'

        Raises
        ------
        ValueError
            If `mode` is not one in {'delete', 'extract'}.

            If `threshold <= 1`.

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
            raise ValueError(f"Invalid mode '{mode}'. Valid modes are: "
                             f"'delete', 'extract'")

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
            if in_place and st.segment is not None:
                segment = st.segment

                try:
                    # replace link to spiketrain in segment
                    new_index = self._get_spiketrain_index(
                        segment.spiketrains, st)
                    segment.spiketrains[new_index] = new_st
                except ValueError:
                    # st is not in this segment even though it points to it
                    warnings.warn(f"The SpikeTrain at index {idx} of the "
                                  "input list spiketrains has a "
                                  "unidirectional uplink to a segment in "
                                  "whose segment.spiketrains list it does not "
                                  "appear. Only the spiketrains in the input "
                                  "list will be replaced. You can suppress "
                                  "this warning by setting "
                                  "spiketrain.segment=None for the input "
                                  "spiketrains.")

                block = segment.block
                if block is not None:
                    # replace link to spiketrain in groups
                    for group in block.groups:
                        try:
                            idx = self._get_spiketrain_index(
                                group.spiketrains,
                                st)
                        except ValueError:
                            # st is not in this group, move to next group
                            continue

                        # st found in group, replace with new_st
                        group.spiketrains[idx] = new_st
            spiketrain_list[idx] = new_st

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
