# -*- coding: utf-8 -*-
"""
Methods for performing phase analysis.

:copyright: Copyright 2014-2018 by the Elephant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
"""

from __future__ import division, print_function, unicode_literals

import numpy as np
import quantities as pq
from neo import AnalogSignal, SpikeTrain
from scipy.interpolate import interp1d


def spike_triggered_phase(hilbert_transform, spiketrains, interpolate, as_array=True):
    """
    Calculate the spike-triggered phases of hilbert-transforms (or analogsignals)
    at spiketimes in spiketrains. The function receives n_a analog signals and n_s spiketrains.

    Parameters
    ----------
    hilbert_transform : neo.AnalogSignal or list of neo.Analogsignal
        # TODO: also allow non-complex signals
        `neo.AnalogSignal` of the complex analytic signal (e.g., returned by
        the `elephant.signal_processing.hilbert` function). Analogsignals should
        have shape n_samples x 1
        # TODO: allow 2D asigs?
    spiketrains : neo.SpikeTrain or list of neo.SpikeTrain
        Spike trains on which to trigger `hilbert_transform` extraction.
    interpolate : bool
        If True, the phases and amplitudes of `hilbert_transform` for spikes
        falling between two samples of signal is interpolated.
        If False, the closest sample of `hilbert_transform` is used.
    as_array : bool
        If True, the phases, amplitudes and times are returned per spiketrain
        as a single array
        If False, the phases, amplitdues and times are returned a set of lists per
        entry in hilbert_transform

    Returns
    -------
    phases : list of np.ndarray or list of lists of nd.array
        Spike-triggered phases. If as_array is True:
        List of length n_s (nr. spiketrains) with an np.array n_spikes (number of spikes
        during the hilbert_transform intervals)
        If as_array is False, list of length n_s (nr. spiketrains) containing n_a
        (nr. of analog signals) lists with n_spike entrys, matching the number of spikes
        during the analogsignal interval
    amp : list of nd.array or list of lists of nd.array
        Similar to phases. Corresponding spike-triggered amplitudes.
    times : list of nd.array or list of lists of nd.array
        Similar to phases. Corresponding spike-times

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
    if not isinstance(hilbert_transform, list):
        hilbert_transform = [hilbert_transform]

    if not isinstance(spiketrains, list):
        spiketrains = [spiketrains]

    # checks on hilbert transform
    for entry in hilbert_transform:
        if not isinstance(entry, AnalogSignal):
            raise ValueError("Signal must be an AnalogSignnal, "
                             "not %s." % type(entry))
        if np.any(np.isreal(entry)):
            raise ValueError("Signal must be a complex signal"
                             "(use elphant.singal_processing.hilbert")
        if entry.shape[0] == 1 or entry.shape[1] > 1:
            raise ValueError("Signal must be of shape N samples x 1")

    # checks on spiketrains
    for entry in spiketrains:
        if not isinstance(entry, SpikeTrain):
            "spiketrains must be a SpikeTrain, not %s." % type(entry)

    # Number of signals
    num_spiketrains = len(spiketrains)
    num_trials = len(hilbert_transform)

    # For each trial, select the first input
    t_starts = [elem.t_start for elem in hilbert_transform]
    t_stops = [elem.t_stop for elem in hilbert_transform]
    phases = [np.angle(elem.magnitude.flatten()) for elem in hilbert_transform]
    amps = [np.abs(elem.magnitude.flatten()) for elem in hilbert_transform]
    times = [elem.times for elem in hilbert_transform]

    # output is a list length nun_spiketrains
    result_phases = [[] for _ in range(num_spiketrains)]
    result_amps = [[] for _ in range(num_spiketrains)]
    result_times = [[] for _ in range(num_spiketrains)]

    # Step through each signal
    for spiketrain_i in range(num_spiketrains):
        full_sp = spiketrains[spiketrain_i]
        for trial_i in range(num_trials):
            spike_times = full_sp.time_slice(t_starts[trial_i], t_stops[trial_i]).times  # leave out last spike
            time = times[trial_i]
            phase = phases[trial_i]
            amp = amps[trial_i]

            # cut righthand border
            if spike_times[-1] == time[-1]:
                spike_times = spike_times[:-1]

            if interpolate:
                # use the scipy.interp1d fits to get phases at spiketimes
                phase_fit = interp1d(time, phase, kind='linear')
                amp_fit = interp1d(time, amp, kind='linear')
                result_phases[spiketrain_i].append(phase_fit(spike_times))
                result_amps[spiketrain_i].append(amp_fit(spike_times))
                result_times[spiketrain_i].append(spike_times)
            else:

                # for each spike, get index in time closest to spiketime
                spike_phase_idx = [np.argmin(np.abs(time - sptime)) for sptime in spike_times]
                result_phases[spiketrain_i].append(phase[spike_phase_idx])
                result_amps[spiketrain_i].append(amp[spike_phase_idx])
                result_times[spiketrain_i].append(spike_times)

    # Convert outputs to arrays
    if as_array:
        for sp_i in range(num_spiketrains):
            result_phases[sp_i] = np.hstack(result_phases[sp_i]).flatten()
            result_amps[sp_i] = np.hstack(result_amps[sp_i]).flatten()
            result_times[sp_i] = np.hstack(result_times[sp_i]).flatten()

    return result_phases, result_amps, result_times
