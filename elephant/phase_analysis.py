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
    Calculates the phase locking value (PLV).

    This function expects phases of two signals (with multiple trials).
    For each trial-pair it calculates the phase difference at each given
    time-point. Than it calculates the mean vectors of those phase differences
    across all trials for each given time-point.
    The PLV at time t is the length of the corresponding mean vector.

    Parameters:
    -----------
    phases_x, phases_y: array-like object
        time-series of signal x and signal y with n trials each

    Returns:
    --------
    plv: array-like object
        phase-locking value (float)
        range: [0, 1]

    Raises:
    -------
    ValueError:
        if shapes of phases x and y are different

    Notes
    -----
    This implementation is based on the formula taken from [1] (pp. 195).

    PLV_t = 1/N * abs(sum_n=1_to_N(exp{i * theta(t, n)} ) )

    where theta(t, n) is the phase difference phi_x(t, n) - phi_y(t, n).

    References:
    -----------
    [1] Jean-Philippe Lachaux, Eugenio Rodriguez, Jacques Martinerie,
    and Francisco J. Varela, "Measuring Phase Synchrony in Brain Signals"
    Human Brain Mapping, vol 8, pp. 194-208, 1999.
    """
    if np.shape(phases_x) != np.shape(phases_y):
        raise ValueError("trial number and trial length of signal x and y "
                         "must be equal")

    # trial by trial and time-resolved
    # version 0.2: signal x and y have multiple trials
    # with discrete values/phases

    phase_diff = phase_difference(phases_x, phases_y)
    theta, r = mean_vector(phase_diff, axis=0)
    return r


# draft for phase_locking_value() with list of neo.AnalogSignal as input
def phase_locking_value_analog_signal(phase_data):
    """
    Calculates the phase locking value (PLV).

    This function expects joined phase_data of two signals (with multiple
    trials). For each trial-pair it calculates the phase difference at each
    given time-point. Than it calculates the mean vectors of those phase
    differences across all trials for each given time-point. The PLV at time
    t is the length of the corresponding mean vector.

    Parameters:
    -----------
    phases_data: list of neo.AnalogSignals objects with multiple trials
        time-series of two signals with n trials each
        # version_0:
        axis: 0 -> signal x/y, 1 -> trial, 2 -> phases
        # version_1:
        axis: 0 -> trial, 1 -> signal x/y, 2 -> phases

    Returns:
    --------
    plv: array-like object
        phase-locking value (float)
        range: [0, 1]

    Notes
    -----
    This implementation is based on the formula taken from [1] (pp. 195).

    PLV_t = 1/N * abs(sum_n=1_to_N(exp{i * theta(t, n)} ) )

    where theta(t, n) is the phase difference phi_x(t, n) - phi_y(t, n).

    References:
    -----------
    [1] Jean-Philippe Lachaux, Eugenio Rodriguez, Jacques Martinerie,
    and Francisco J. Varela, "Measuring Phase Synchrony in Brain Signals"
    Human Brain Mapping, vol 8, pp. 194-208, 1999.
    """
    # version_0: phase_data has shape(signal x & y, trial, phases)
    # if np.shape(phase_data[0]) != np.shape(phase_data[1]):
    #     raise ValueError("trial number and trial length of signal x and y "
    #                      "must be equal")
    # phase_diff = phase_difference(phase_data[0], phase_data[1])

    # version_1: phase_data has shape(trial, signal x & y, phases)
    try:
        if (np.shape(np.asarray([signal[0] for signal in phase_data])) !=
                np.shape(np.asarray([signal[1] for signal in phase_data]))):
            raise ValueError("trial number and trial length of signal x and y "
                             "must be equal")
    except IndexError as ie:
        raise ie
    phase_diff = phase_difference(
        np.asarray([signal[0] for signal in phase_data]),
        np.asarray([signal[1] for signal in phase_data]))

    theta, r = mean_vector(phase_diff, axis=0)
    return r


def mean_vector(phases, axis=0):
    """
    Calculates the mean vector of phases.

    This function expects phases (in radians) and uses their representation as
    complex numbers to calculate the direction 'theta' and the length 'r'
    of the mean vector.

    Parameters
    ----------
    phases: array-like object
        phases in radians
    axis: {0, 1, None}
        axis along which the mean_vector will be calculated
        - None: across flattened array
        - 0: across columns of array (default)
        - 1: across rows of array

    Returns
    -------
    z_mean_theta: array-like object
        angle of the mean vector
        range: (-pi, pi]
    z_mean_r: array-like object
        length of the mean vector
        range: [0, 1]
    """
    # use complex number representation
    # z_phases = np.cos(phases) + 1j * np.sin(phases)
    z_phases = np.exp(1j * np.asarray(phases))
    z_mean = np.mean(z_phases, axis=axis)
    z_mean_theta = np.angle(z_mean)
    z_mean_r = np.abs(z_mean)
    return z_mean_theta, z_mean_r


def phase_difference(alpha, beta):
    """
    Calculates the difference between a pair of phases. The output is in range
    of -pi to pi.

    Parameters
    ----------
    alpha: array-like object
        phases in radians
    beta: array-like object
        phases in radians

    Returns
    -------
    phase_diff: float
        phase difference between alpha and beta
        range: [-pi, pi]

    Notes
    -----
    The usage of arctan2 assures that the range of the phase difference
    is [-pi, pi] and is located in the correct quadrant.
    """
    delta = alpha - beta
    phase_diff = np.arctan2(np.sin(delta), np.cos(delta))
    return phase_diff
