# -*- coding: utf-8 -*-
"""
Methods for performing phase analysis.

:copyright: Copyright 2014-2018 by the Elephant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
"""

from __future__ import division, print_function, unicode_literals

import numpy as np
import quantities as pq

__all__ = [
    "spike_triggered_phase"
]


def spike_triggered_phase(hilbert_transform, spiketrains, interpolate):
    """
    Calculate the set of spike-triggered phases of a `neo.AnalogSignal`.

    Parameters
    ----------
    hilbert_transform : neo.AnalogSignal or list of neo.AnalogSignal
        `neo.AnalogSignal` of the complex analytic signal (e.g., returned by
        the `elephant.signal_processing.hilbert` function).
        If `hilbert_transform` is only one signal, all spike trains are
        compared to this signal. Otherwise, length of `hilbert_transform` must
        match the length of `spiketrains`.
    spiketrains : neo.SpikeTrain or list of neo.SpikeTrain
        Spike trains on which to trigger `hilbert_transform` extraction.
    interpolate : bool
        If True, the phases and amplitudes of `hilbert_transform` for spikes
        falling between two samples of signal is interpolated.
        If False, the closest sample of `hilbert_transform` is used.

    Returns
    -------
    phases : list of np.ndarray
        Spike-triggered phases. Entries in the list correspond to the
        `neo.SpikeTrain`s in `spiketrains`. Each entry contains an array with
        the spike-triggered angles (in rad) of the signal.
    amp : list of pq.Quantity
        Corresponding spike-triggered amplitudes.
    times : list of pq.Quantity
        A list of times corresponding to the signal. They correspond to the
        times of the `neo.SpikeTrain` referred by the list item.

    Raises
    ------
    ValueError
        If the number of spike trains and number of phase signals don't match,
        and neither of the two are a single signal.

    Examples
    --------
    Create a 20 Hz oscillatory signal sampled at 1 kHz and a random Poisson
    spike train, then calculate spike-triggered phases and amplitudes of the
    oscillation:

    >>> import neo
    >>> import elephant
    >>> import quantities as pq
    >>> import numpy as np
    ...
    >>> f_osc = 20. * pq.Hz
    >>> f_sampling = 1 * pq.ms
    >>> tlen = 100 * pq.s
    ...
    >>> time_axis = np.arange(
    ...     0, tlen.magnitude,
    ...     f_sampling.rescale(pq.s).magnitude) * pq.s
    >>> analogsignal = neo.AnalogSignal(
    ...     np.sin(2 * np.pi * (f_osc * time_axis).simplified.magnitude),
    ...     units=pq.mV, t_start=0*pq.ms, sampling_period=f_sampling)
    >>> spiketrain = (elephant.spike_train_generation.
    ...     homogeneous_poisson_process(
    ...     50 * pq.Hz, t_start=0.0*pq.ms, t_stop=tlen.rescale(pq.ms)))
    ...
    >>> phases, amps, times = elephant.phase_analysis.spike_triggered_phase(
    ...     elephant.signal_processing.hilbert(analogsignal),
    ...     spiketrain,
    ...     interpolate=True)

    """

    # Convert inputs to lists
    if not isinstance(spiketrains, list):
        spiketrains = [spiketrains]

    if not isinstance(hilbert_transform, list):
        hilbert_transform = [hilbert_transform]

    # Number of signals
    num_spiketrains = len(spiketrains)
    num_phase = len(hilbert_transform)

    if num_spiketrains != 1 and num_phase != 1 and \
            num_spiketrains != num_phase:
        raise ValueError(
            "Number of spike trains and number of phase signals"
            "must match, or either of the two must be a single signal.")

    # For each trial, select the first input
    start = [elem.t_start for elem in hilbert_transform]
    stop = [elem.t_stop for elem in hilbert_transform]

    result_phases = []
    result_amps = []
    result_times = []

    # Step through each signal
    for spiketrain_i, spiketrain in enumerate(spiketrains):
        # Check which hilbert_transform AnalogSignal to look at - if there is
        # only one then all spike trains relate to this one, otherwise the two
        # lists of spike trains and phases are matched up
        if num_phase > 1:
            phase_i = spiketrain_i
        else:
            phase_i = 0

        # Take only spikes which lie directly within the signal segment -
        # ignore spikes sitting on the last sample
        sttimeind = np.where(np.logical_and(
            spiketrain >= start[phase_i], spiketrain < stop[phase_i]))[0]

        # Find index into signal for each spike
        ind_at_spike = np.round(
            (spiketrain[sttimeind] - hilbert_transform[phase_i].t_start) /
            hilbert_transform[phase_i].sampling_period). \
            simplified.magnitude.astype(int)

        # Extract times for speed reasons
        times = hilbert_transform[phase_i].times

        # Append new list to the results for this spiketrain
        result_phases.append([])
        result_amps.append([])
        result_times.append([])

        # Step through all spikes
        for spike_i, ind_at_spike_j in enumerate(ind_at_spike):
            # Difference vector between actual spike time and sample point,
            # positive if spike time is later than sample point
            dv = spiketrain[sttimeind[spike_i]] - times[ind_at_spike_j]

            # Make sure ind_at_spike is to the left of the spike time
            if dv < 0 and ind_at_spike_j > 0:
                ind_at_spike_j = ind_at_spike_j - 1

            if interpolate:
                # Get relative spike occurrence between the two closest signal
                # sample points
                # if z->0 spike is more to the left sample
                # if z->1 more to the right sample
                z = (spiketrain[sttimeind[spike_i]] - times[ind_at_spike_j]) /\
                    hilbert_transform[phase_i].sampling_period

                # Save hilbert_transform (interpolate on circle)
                p1 = np.angle(hilbert_transform[phase_i][ind_at_spike_j])
                p2 = np.angle(hilbert_transform[phase_i][ind_at_spike_j + 1])
                result_phases[spiketrain_i].append(
                    np.angle(
                        (1 - z) * np.exp(np.complex(0, p1)) +
                        z * np.exp(np.complex(0, p2))))

                # Save amplitude
                result_amps[spiketrain_i].append(
                    (1 - z) * np.abs(
                        hilbert_transform[phase_i][ind_at_spike_j]) +
                    z * np.abs(hilbert_transform[phase_i][ind_at_spike_j + 1]))
            else:
                p1 = np.angle(hilbert_transform[phase_i][ind_at_spike_j])
                result_phases[spiketrain_i].append(p1)

                # Save amplitude
                result_amps[spiketrain_i].append(
                    np.abs(hilbert_transform[phase_i][ind_at_spike_j]))

            # Save time
            result_times[spiketrain_i].append(spiketrain[sttimeind[spike_i]])

    # Convert outputs to arrays
    for i, entry in enumerate(result_phases):
        result_phases[i] = np.array(entry).flatten()
    for i, entry in enumerate(result_amps):
        result_amps[i] = pq.Quantity(entry, units=entry[0].units).flatten()
    for i, entry in enumerate(result_times):
        result_times[i] = pq.Quantity(entry, units=entry[0].units).flatten()

    return result_phases, result_amps, result_times


def phase_locking_value(phases_x, phases_y):
    """
    TODO: add the description

    Parameters:
    -----------
        - phases_x: array of arrays
            time-series of signal x with n trials
        - phases_y: array of arrays
            time-series of signal y with n trials

    Returns:
    --------
        - plv: phase-locking value (float)


    References:
    -----------
    "Measuring Phase Synchrony in Brain Signals" - Jean-Philippe Lachaux,
    Eugenio Rodriguez, Jacques Martinerie, and Francisco J. Varela*

    Mathematics:
    ------------
    PLV_t = 1/N * abs(sum_n=1_to_N(exp{i * theta(t, n)} ) )
    where theta(t, n) is the phase difference phi_x(t, n) - phi_y(t, n)
    """

    if len(phases_x) == len(phases_y):
        num_trial = len(phases_x)
        num_time_points = len(phases_x[0])
    else:
        raise ValueError("trial number of signal x and y must be equal")

    # trial by trial and time-resolved
    # version 0.2: signal x and y have multiple trials
    # with discrete values/phases

    # list of trial averaged plv at time t
    list_plv_t = []
    for time_i in range(num_time_points):
        # list of phase differences at time i and
        # for each trial j from signal x and y
        list_adiff_i = []
        for trial_j in range(num_trial):
            adiff_i_j = angular_difference(phases_x[trial_j][time_i],
                                           phases_y[trial_j][time_i])
            list_adiff_i.append(adiff_i_j)
        plv_theta_i, plv_r_i = mean_vector(list_adiff_i)
        list_plv_t.append(plv_r_i)
    return list_plv_t


def mean_vector(phases):
    """
    This function calculates the mean direction & the mean vector length
    of the phases-set.

    Parameters
    ----------
    - phases: array-like object
        phases of circular data

    Returns
    -------
    - theta_bar: mean direction of the phases
    - r: length of the mean vector
    """
    # applying trigonometric functions to calculate the mean direction
    # n: number of phases
    n = len(phases)
    # x_i and y_i: cartesian coordinates of phase_i
    x_i = np.cos(phases)
    y_i = np.sin(phases)
    # x_bar and y_bar: mean cartesian coordinates of the phases
    x_bar = np.sum(x_i) / n
    y_bar = np.sum(y_i) / n
    # r: length of the mean vector
    r = np.sqrt(x_bar**2 + y_bar**2)
    # theta_bar: mean direction of the phases in radians
    if x_bar > 0:
        theta_bar = np.arctan(y_bar / x_bar)
    elif x_bar < 0:
        theta_bar = np.pi + np.arctan(y_bar / x_bar)
    elif x_bar == 0:
        if y_bar > 0:
            theta_bar = np.pi/2
        elif y_bar < 0:
            theta_bar = 3/2 * np.pi
        else:
            print("undetermined")
    theta_bar %= (2*np.pi)
    return theta_bar, r


def angular_difference(alpha, beta):
    """
    This function calculates the difference between a pair of angles.

    Parameters
    ----------
    - alpha: array-like object
        angles in radians
    - beta: array-like object
        angles in radians

    Returns
    -------
    - adiff: float
        angle difference between alpha and beta TODO:in range of [-pi, pi]

    """
    adiff = (alpha - beta + np.pi) % (2*np.pi) - np.pi
    return adiff
