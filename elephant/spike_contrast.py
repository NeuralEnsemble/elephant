# -*- coding: utf-8 -*-
"""
Functions to measure the synchrony of several spike trains (based on [1]).


* get_theta_and_n_per_bin
    This function calculates the amount of spikes per bin and the amount of active spike trains per bin.
    Note: Bin overlap of half bin size.
* binning_half_overlap
    Current spike train is binned (calculating histogram) with an overlapping bin (overlap: half the bin size).
* spike_contrast
    Calculates the synchrony of the spike trains according to [1].

References
----------
[1] Manuel Ciba (2018). Spike-contrast: A novel time scale independent
    and multivariate measure of spike train synchrony. Journal of Neuroscience Methods. 2018; 293: 136-143.


Original implementation by: Philipp Steigerwald [s160857@th-ab.de]
:copyright: Copyright 2015-2016 by the Elephant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
"""

import numpy as np


def _get_theta_and_n_per_bin(spiketrains, t_start, t_stop, bin_size):
    """
    Calculates theta (amount of spikes per bin) and n (amount of active spike
    trains per bin) of one spike train.
    """
    # Calculate histogram for every spike train
    histogram = np.vstack([
        _binning_half_overlap(st[st.nonzero()], t_start=t_start, t_stop=t_stop,
                              bin_size=bin_size)
        for st in spiketrains
    ])
    # Amount of spikes per bin
    theta = histogram.sum(axis=0)
    # Amount of active spike trains per bin
    n = np.count_nonzero(histogram, axis=0)

    return theta, n


def _binning_half_overlap(spiketrain, t_start, t_stop, bin_size):
    """
    Referring to [1] overlapping the bins creates a better result.
    """
    bin_step = bin_size / 2
    edges = np.arange(t_start, t_stop + bin_step, bin_step)
    histogram, bin_edges = np.histogram(spiketrain, edges)
    histogram = histogram[:-1] + histogram[1:]
    return histogram


def spike_contrast(spiketrains, t_start, t_stop, min_bin=0.01):
    """
    Calculates the synchrony of several spike trains. The spikes trains do not have to have the same length, the
    algorithm takes care of that.

    Parameters
    ----------
    spiketrains : list of neo.SpikeTrain or list of np.ndarray
        Contains all the spike trains.
    t_start : float
        The beginning of the spike train.
    t_stop : float
        The end of the spike train.
    min_bin : float
        The variable bin_min gets calculated by the algorithm, but if the calculated
        value is smaller than max_bin_min it takes max_bin_min to not get smaller than it.
        Default: 0.01

    Returns
    -------
    synchrony : float
        Returns the synchrony of the input spike trains.

    Examples
    --------
    >>> import quantities as pq
    >>> import elephant.spike_train_generation
    >>> import elephant.spike_contrast
    >>> spike_train_1 = homogeneous_poisson_process(
    ...     20*pq.Hz, t_start=5000*pq.ms, t_stop=10000*pq.ms, as_array=True)
    >>> spikes_train_2 = homogeneous_poisson_process(50*pq.Hz, t_start=0*pq.ms,
    ...     t_stop=1000*pq.ms, refractory_period = 3*pq.ms)
    >>> spike_trains = np.array([spike_train_1, spike_train_2])
    >>> print(spike_contrast(spiketrains_padded, 0, 10000))

    """
    n_spiketrains = len(spiketrains)
    n_spikes_total = sum(map(len, spiketrains))

    # Detect the longest array
    biggest_array_count = max(map(len, spiketrains))
    # pad the spiketrains with NaN
    spiketrains_padded = np.vstack([
        np.pad(st, pad_width=(0, biggest_array_count - len(st)),
               constant_values=np.nan)
        for st in spiketrains
    ])
    # Get the transposed matrix for the algorithm
    duration = t_stop - t_start

    # parameter
    bin_shrink_factor = 0.9  # bin size decreases by factor 0.9 for each iteration
    bin_max = duration / 2
    isi_min = min(min(np.diff(st)) for st in spiketrains)
    bin_min = max(isi_min / 2, min_bin)

    # initialization
    num_iterations = np.ceil(
        np.log(bin_min / bin_max) / np.log(bin_shrink_factor))
    num_iterations = int(num_iterations)
    active_st = np.zeros(num_iterations)
    contrast = np.zeros(num_iterations)
    synchrony_curve = np.zeros(num_iterations)

    bin_size = bin_max
    for iter_id in range(num_iterations):
        # Set the new boundaries for the time
        time_start = -isi_min
        time_end = duration + isi_min
        # Calculate Theta and n
        theta_k, n_k = _get_theta_and_n_per_bin(spiketrains_padded,
                                                t_start=time_start,
                                                t_stop=time_end,
                                                bin_size=bin_size)

        # calculate synchrony_curve = contrast * active_st
        active_st[iter_id] = ((np.sum(n_k * theta_k)) / (np.sum(theta_k)) - 1) / (
                    n_spiketrains - 1)
        contrast[iter_id] = (np.sum(np.abs(np.diff(theta_k))) / (n_spikes_total * 2))
        # Contrast: sum(|derivation|) / (2*#Spikes)
        synchrony_curve[iter_id] = contrast[iter_id] * active_st[iter_id]  # synchrony_curve
        # New bin size
        bin_size *= bin_shrink_factor
    # Sync value is maximum of cost function C
    synchrony = np.max(synchrony_curve)
    return synchrony
