# -*- coding: utf-8 -*-
"""
Functions to measure the synchrony of several spike trains.


Synchrony Measures
------------------

.. autosummary::
    :toctree: toctree/spike_train_synchrony/

    spike_contrast


:copyright: Copyright 2015-2020 by the Elephant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
"""
from __future__ import division, print_function, unicode_literals

from collections import namedtuple

import neo
import numpy as np
import quantities as pq

from elephant.utils import is_time_quantity

SpikeContrastTrace = namedtuple("SpikeContrastTrace", (
    "contrast", "active_spiketrains", "synchrony"))


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

          `.synchrony` - the product of `contrast` and `active_spiketrains`.

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
        raise ValueError("'bin_shrink_factor' ({}) must be in range (0, 1)."
                         .format(bin_shrink_factor))
    if not len(spiketrains) > 1:
        raise ValueError("Spike contrast measure requires more than 1 input "
                         "spiketrain.")
    if not all(isinstance(st, neo.SpikeTrain) for st in spiketrains):
        raise TypeError("Input spike trains must be a list of neo.SpikeTrain.")
    if not is_time_quantity(t_start, allow_none=True) \
            or not is_time_quantity(t_stop, allow_none=True):
        raise TypeError("'t_start' and 't_stop' must be time quantities.")
    if not is_time_quantity(min_bin):
        raise TypeError("'min_bin' must be a time quantity.")

    if t_start is None:
        t_start = min(st.t_start for st in spiketrains)
    if t_stop is None:
        t_stop = max(st.t_stop for st in spiketrains)
    spiketrains = [st.time_slice(t_start=t_start, t_stop=t_stop)
                   for st in spiketrains]

    # convert everything to seconds
    spiketrains = [st.simplified.magnitude for st in spiketrains]
    t_start = t_start.simplified.item()
    t_stop = t_stop.simplified.item()
    min_bin = min_bin.simplified.item()

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

    bin_size = bin_max
    while bin_size >= bin_min:
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
            synchrony=synchrony_curve
        )
        return synchrony, spike_contrast_trace

    return synchrony
