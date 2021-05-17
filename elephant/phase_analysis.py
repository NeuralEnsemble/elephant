# -*- coding: utf-8 -*-
"""
Methods for performing phase analysis.

.. autosummary::
    :toctree: _toctree/phase_analysis

    spike_triggered_phase
    phase_locking_value
    mean_phase_vector
    phase_difference

:copyright: Copyright 2014-2020 by the Elephant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
"""

from __future__ import division, print_function, unicode_literals

import numpy as np
import quantities as pq
import neo

__all__ = [
    "spike_triggered_phase",
    "phase_locking_value",
    "mean_phase_vector",
    "phase_difference"
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
    >>> phases
    [array([-0.57890515,  1.03105904, -0.82241075, ...,  0.90023903,
             2.23702263,  2.93744259])]
    >>> amps
    [array([0.86117412, 1.08918248, 0.98256318, ..., 1.05760518, 1.08407016,
        1.01927305]) * dimensionless]
    >>> times
    [array([6.41327152e+00, 2.02715221e+01, 1.05827312e+02, ...,
        9.99692942e+04, 9.99808429e+04, 9.99870120e+04]) * ms]

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

        # Extract times for speed reasons
        times = hilbert_transform[phase_i].times

        # Find index into signal for each spike
        ind_at_spike = (
            (spiketrain[sttimeind] - hilbert_transform[phase_i].t_start) /
            hilbert_transform[phase_i].sampling_period). \
            simplified.magnitude.astype(int)

        # Append new list to the results for this spiketrain
        result_phases.append([])
        result_amps.append([])
        result_times.append([])

        # Step through all spikes
        for spike_i, ind_at_spike_j in enumerate(ind_at_spike):

            if interpolate and ind_at_spike_j+1 < len(times):
                # Get relative spike occurrence between the two closest signal
                # sample points
                # if z->0 spike is more to the left sample
                # if z->1 more to the right sample
                z = (spiketrain[sttimeind[spike_i]] - times[ind_at_spike_j]) /\
                    hilbert_transform[phase_i].sampling_period

                # Save hilbert_transform (interpolate on circle)
                p1 = np.angle(hilbert_transform[phase_i][ind_at_spike_j])
                p2 = np.angle(hilbert_transform[phase_i][ind_at_spike_j + 1])
                interpolation = (1 - z) * np.exp(np.complex(0, p1)) \
                                    + z * np.exp(np.complex(0, p2))
                p12 = np.angle([interpolation])
                result_phases[spiketrain_i].append(p12)

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


def phase_locking_value(phases_i, phases_j):
    r"""
    Calculates the phase locking value (PLV).

    This function expects the phases of two signals (each containing multiple
    trials). For each trial pair, it calculates the phase difference at each
    time point. Then it calculates the mean vectors of those phase differences
    across all trials. The PLV at time `t` is the length of the corresponding
    mean vector.

    Parameters
    ----------
    phases_i, phases_j : (t, n) np.ndarray
        Time-series of the first and second signals, with `t` time points and
        `n` trials.

    Returns
    -------
    plv : (t,) np.ndarray
        Vector of floats with the phase-locking value at each time point.
        Range: :math:`[0, 1]`

    Raises
    ------
    ValueError
        If the shapes of `phases_i` and `phases_j` are different.

    Notes
    -----
    This implementation is based on the formula taken from [1] (pp. 195):

    .. math::
        PLV_t = \frac{1}{N} \left |
        \sum_{n=1}^N \exp(i \cdot \theta(t, n)) \right | \\

    where :math:`\theta(t, n) = \phi_x(t, n) - \phi_y(t, n)`
    is the phase difference at time `t` for trial `n`.

    References
    ----------
    [1] Jean-Philippe Lachaux, Eugenio Rodriguez, Jacques Martinerie,
    and Francisco J. Varela, "Measuring Phase Synchrony in Brain Signals"
    Human Brain Mapping, vol 8, pp. 194-208, 1999.
    """
    if np.shape(phases_i) != np.shape(phases_j):
        raise ValueError("trial number and trial length of signal x and y "
                         "must be equal")

    # trial by trial and time-resolved
    # version 0.2: signal x and y have multiple trials
    # with discrete values/phases

    phase_diff = phase_difference(phases_i, phases_j)
    theta, r = mean_phase_vector(phase_diff, axis=0)
    return r


def mean_phase_vector(phases, axis=0):
    r"""
    Calculates the mean vector of phases.

    This function expects phases (in radians) and uses their representation as
    complex numbers to calculate the direction :math:`\theta` and the length
    `r` of the mean vector.

    Parameters
    ----------
    phases : np.ndarray
        Phases in radians.
    axis : int, optional
        Axis along which the mean vector will be calculated.
        If None, it will be computed across the flattened array.
        Default: 0

    Returns
    -------
    z_mean_theta : np.ndarray
        Angle of the mean vector.
        Range: :math:`(-\pi, \pi]`
    z_mean_r : np.ndarray
        Length of the mean vector.
        Range: :math:`[0, 1]`
    """
    # use complex number representation
    # z_phases = np.cos(phases) + 1j * np.sin(phases)
    z_phases = np.exp(1j * np.asarray(phases))
    z_mean = np.mean(z_phases, axis=axis)
    z_mean_theta = np.angle(z_mean)
    z_mean_r = np.abs(z_mean)
    return z_mean_theta, z_mean_r


def phase_difference(alpha, beta):
    r"""
    Calculates the difference between a pair of phases.

    The output is in range from :math:`-\pi` to :math:`\pi`.

    Parameters
    ----------
    alpha : np.ndarray
        Phases in radians.
    beta : np.ndarray
        Phases in radians.

    Returns
    -------
    phase_diff : np.ndarray
        Difference between phases `alpha` and `beta`.
        Range: :math:`[-\pi, \pi]`

    Notes
    -----
    The usage of `np.arctan2` ensures that the range of the phase difference
    is :math:`[-\pi, \pi]` and is located in the correct quadrant.
    """
    delta = alpha - beta
    phase_diff = np.arctan2(np.sin(delta), np.cos(delta))
    return phase_diff
