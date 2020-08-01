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
    Calculates theta (amount of spikes per bin) and the amount of active spike
    trains per bin of one spike train.
    """
    # Calculate histogram for every spike train
    histogram = np.vstack([
        _binning_half_overlap(st, t_start=t_start, t_stop=t_stop,
                              bin_size=bin_size)
        for st in spiketrains
    ])
    # Amount of spikes per bin
    theta = histogram.sum(axis=0)
    # Amount of active spike trains per bin
    n_active_per_bin = np.count_nonzero(histogram, axis=0)

    return theta, n_active_per_bin


def _binning_half_overlap(spiketrain, t_start, t_stop, bin_size):
    """
    Referring to [1] overlapping the bins creates a better result.
    """
    bin_step = bin_size / 2
    edges = np.arange(t_start, t_stop + bin_step, bin_step)
    histogram, bin_edges = np.histogram(spiketrain, edges)
    histogram = histogram[:-1] + histogram[1:]
    return histogram


def spike_contrast(spiketrains, t_start, t_stop, min_bin=0.01,
                   bin_shrink_factor=0.9):
    """
    Calculates the synchrony of spike trains. The spike trains can have
    different lengths.

    Parameters
    ----------
    spiketrains : list of neo.SpikeTrain or list of np.ndarray
        Contains all the spike trains.
    t_start : float
        The beginning of the spike train.
    t_stop : float
        The end of the spike train.
    min_bin : float, optional
        Sets the minimum value for the `bin_min` that is calculated by the
        algorithm and defines the smallest bin size to compute the histogram
        of the input `spiketrains`.
        Default: 0.01
    bin_shrink_factor : float, optional
        A multiplier to shrink the bin size on each iteration. The value must
        be in range `(0, 1)`.
        Default: 0.9

    Returns
    -------
    synchrony : float
        Returns the synchrony of the input spike trains.

    Examples
    --------
    >>> import quantities as pq
    >>> from elephant.spike_train_generation import homogeneous_poisson_process
    >>> from elephant.spike_contrast import spike_contrast
    >>> spiketrain_1 = homogeneous_poisson_process(rate=20*pq.Hz,
    ...     t_start=5000*pq.ms, t_stop=10000*pq.ms, as_array=True)
    >>> spiketrain_2 = homogeneous_poisson_process(50*pq.Hz, t_start=0*pq.ms,
    ...     t_stop=1000*pq.ms, refractory_period=3*pq.ms, as_array=True)
    >>> spike_contrast([spiketrain_1, spiketrain_2], t_start=0, t_stop=10000)
    0.0

    """
    if not 0. < bin_shrink_factor < 1.:
        raise ValueError("'bin_shrink_factor' ({}) must be in range (0, 1)."
                         .format(bin_shrink_factor))

    n_spiketrains = len(spiketrains)
    n_spikes_total = sum(map(len, spiketrains))

    duration = t_stop - t_start
    bin_max = duration / 2
    isi_min = min(np.min(np.diff(st)) for st in spiketrains)
    bin_min = max(isi_min / 2, min_bin)

    synchrony_curve = []
    bin_size = bin_max
    while bin_size >= bin_min:
        # Set the new boundaries for the time
        time_start = -isi_min
        time_end = duration + isi_min
        # Calculate Theta and n
        theta_k, n_k = _get_theta_and_n_per_bin(spiketrains,
                                                t_start=time_start,
                                                t_stop=time_end,
                                                bin_size=bin_size)

        # calculate synchrony_curve = contrast * active_st
        active_st = (np.sum(n_k * theta_k) / np.sum(theta_k) - 1) / (
                    n_spiketrains - 1)
        contrast = np.sum(np.abs(np.diff(theta_k))) / (2 * n_spikes_total)
        # Contrast: sum(|derivation|) / (2*#Spikes)
        synchrony_curve_i = contrast * active_st
        synchrony_curve.append(synchrony_curve_i)
        # New bin size
        bin_size *= bin_shrink_factor

    # Sync value is maximum of cost function C
    synchrony = max(synchrony_curve)
    return synchrony
