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


def get_theta_and_n_per_bin(spike_trains, t_start, t_stop, bin_size):
    """
        Calculates theta (amount of spikes per bin) and n (amount of active spike trains per bin) of one spike train.

        Parameters
        ----------
        spike_trains : np.ndarray
            A spike train.
        t_start : int
            The beginning of the spike train.
        t_stop : int
            The end of the spike train.
        bin_size : int
            The time precision used to discretize the spiketrains (binning).

        Returns
        -------
        theta : int
            Contains the amount of spikes per bin
        n : int
            Contains the amount of active spike trains per bin.

    """
    # Smoothing
    bin_step = bin_size / 2
    # Creating vector with t_start as start and t_end as end with step size bin_step
    edges = np.arange(t_start, t_stop+bin_step, bin_step)
    # Amount of spike trains
    tmp = spike_trains.shape
    amount_of_spikes = tmp[1]
    histogram = np.zeros((len(edges) - 2, amount_of_spikes))
    # Calculate histogram for every spike train
    for i in range(0, amount_of_spikes):  # for all spike trains
        spike_train_i = spike_trains[:, i]
        # only use non-nan values
        spike_train_i = spike_train_i[~np.isnan(spike_train_i)]
        histogram[:, i] = binning_half_overlap(spike_train_i, t_start, t_stop, bin_size)
    # Calculate parameter over all spike trains
    # Amount of spikes per bin
    theta = np.sum(histogram, 1)
    # Create matrix with a 1 for every non 0 element
    mask = histogram != 0
    # Amount of active spike trains
    n = np.sum(mask, 1)

    return theta, n


def binning_half_overlap(spike_train_i, t_start, t_stop, bin_size):
    """
        Referring to [1] overlapping the bins creates a better result.

        Parameters
        ----------
        spike_train_i : np.array
            Contains all spikes of one spiketrain.
        t_start : int
            The beginning of the spike train.
        t_stop : int
            The end of the spike train.
        bin_size : int
            The time precision used to discretize the spike trains (binning).

        Returns
        -------
        histogram : np.ndarray
            Contains the histogram of one spike train.
    """
    bin_step = bin_size / 2
    edges = np.arange(t_start, t_stop+bin_step, bin_step)
    histogram, bin_edges = np.histogram(spike_train_i, edges)
    histogram[0:len(histogram) - 1] = histogram[0:len(histogram) - 1] + histogram[1:len(histogram)]
    return histogram[0:len(histogram) - 1]


def spike_contrast(spike_trains_elephant, t_start, t_stop, max_bin_min=0.01):
    """
        Calculates the synchrony of several spike trains. The spikes trains do not have to have the same length, the
        algorithm takes care of that.

        Parameters
        ----------
        spike_trains_elephant : neo.SpikeTrain or np.ndarray
            Contains all the spike trains.
        t_start : float
            The beginning of the spike train.
        t_stop : float
            The end of the spike train.
        max_bin_min : float
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
    >>> print(spike_contrast(spike_trains, 0, 10000))

    """
    # Pad all spike trains with zeros, so every array has the same length
    zero_array = np.zeros((1, spike_trains_elephant.shape[0]))
    for i in range(0, spike_trains_elephant.shape[0]):
        length_spike_train = spike_trains_elephant[i].shape
        zero_array[0][i] = length_spike_train[0]
    # Detect the longest array
    biggest_array_count = int(np.max(zero_array))
    # Great new array where every spike train has the same length
    spike_trains_zeros = np.zeros((spike_trains_elephant.shape[0], biggest_array_count))
    for i in range(0, spike_trains_elephant.shape[0]):
        spike_trains_zeros[i] = np.pad(spike_trains_elephant[i], (0, biggest_array_count -
                                                                  len(spike_trains_elephant[i])))
    # Get the Dimension of the spike train.
    tmp = spike_trains_zeros.shape
    # Get the transposed matrix for the algorithm
    spike_trains = np.zeros((tmp[1], tmp[0]))
    for i in range(tmp[0]):
        for y in range(tmp[1]):
            spike_trains[y][i] = spike_trains_zeros[i][y]
    time = t_stop - t_start
    # Set zeros to NaN (zero-padding)
    spike_trains = np.where(spike_trains == 0, np.nan, spike_trains)
    mask = np.isnan(spike_trains)
    # Make a masked array
    spike_trains_ma = np.ma.MaskedArray(spike_trains, mask)

    tmp = spike_trains.shape
    amount_of_spikes = int(tmp[1])

    # parameter
    bin_shrink_factor = 0.9  # bin size decreases by factor 0.9 for each iteration
    bin_max = time / 2
    isi = np.diff(spike_trains_ma, axis=0)
    isi_min = np.min(isi)
    bin_min = np.max([isi_min / 2, max_bin_min])

    # initialization
    num_iterations = np.ceil(np.log(bin_min / bin_max) / np.log(bin_shrink_factor))
    num_iterations = int(num_iterations)
    active_st = np.zeros((num_iterations, 1))
    contrast = np.zeros((num_iterations, 1))
    synchrony_curve = np.zeros((num_iterations, 1))

    num_all_spikes = spike_trains_ma.count()
    bin_size = bin_max
    # for 0, 1, 2, ... num_iterations
    for i in range(0, num_iterations):
        # Set the new boundaries for the time
        time_start = -isi_min
        time_end = time + isi_min
        # Calculate Theta and n
        theta_k, n_k = get_theta_and_n_per_bin(spike_trains, time_start, time_end, bin_size)

        # calculate synchrony_curve = contrast * active_st
        active_st[i] = ((np.sum(n_k * theta_k)) / (np.sum(theta_k)) - 1) / (amount_of_spikes - 1)
        contrast[i] = (np.sum(np.abs(np.diff(theta_k))) / (num_all_spikes * 2))
        # Contrast: sum(|derivation|) / (2*#Spikes)
        synchrony_curve[i] = contrast[i] * active_st[i]  # synchrony_curve
        # New bin size
        bin_size *= bin_shrink_factor
    # Sync value is maximum of cost function C
    synchrony = np.max(synchrony_curve)
    return synchrony
