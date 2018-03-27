# -*- coding: utf-8 -*-
'''
Functions to calculate spike-triggered average and spike-field coherence of
analog signals.

:copyright: Copyright 2015-2016 by the Elephant team, see AUTHORS.txt.
:license: Modified BSD, see LICENSE.txt for details.
'''

from __future__ import division
import numpy as np
import scipy.signal
import quantities as pq
from neo.core import AnalogSignal, SpikeTrain
import warnings
from .conversion import BinnedSpikeTrain


def spike_triggered_average(signal, spiketrains, window):
    """
    Calculates the spike-triggered averages of analog signals in a time window
    relative to the spike times of a corresponding spiketrain for multiple
    signals each. The function receives n analog signals and either one or
    n spiketrains. In case it is one spiketrain this one is muliplied n-fold
    and used for each of the n analog signals.

    Parameters
    ----------
    signal : neo AnalogSignal object
        'signal' contains n analog signals.
    spiketrains : one SpikeTrain or one numpy ndarray or a list of n of either of these.
        'spiketrains' contains the times of the spikes in the spiketrains.
    window : tuple of 2 Quantity objects with dimensions of time.
        'window' is the start time and the stop time, relative to a spike, of
        the time interval for signal averaging.
        If the window size is not a multiple of the sampling interval of the 
        signal the window will be extended to the next multiple. 

    Returns
    -------
    result_sta : neo AnalogSignal object
        'result_sta' contains the spike-triggered averages of each of the
        analog signals with respect to the spikes in the corresponding
        spiketrains. The length of 'result_sta' is calculated as the number
        of bins from the given start and stop time of the averaging interval
        and the sampling rate of the analog signal. If for an analog signal
        no spike was either given or all given spikes had to be ignored
        because of a too large averaging interval, the corresponding returned
        analog signal has all entries as nan. The number of used spikes and
        unused spikes for each analog signal are returned as annotations to 
        the returned AnalogSignal object.

    Examples
    --------

    >>> signal = neo.AnalogSignal(np.array([signal1, signal2]).T, units='mV',
    ...                                sampling_rate=10/ms)
    >>> stavg = spike_triggered_average(signal, [spiketrain1, spiketrain2],
    ...                                 (-5 * ms, 10 * ms))

    """

    # checking compatibility of data and data types
    # window_starttime: time to specify the start time of the averaging
    # interval relative to a spike
    # window_stoptime: time to specify the stop time of the averaging
    # interval relative to a spike
    window_starttime, window_stoptime = window
    if not (isinstance(window_starttime, pq.quantity.Quantity) and
            window_starttime.dimensionality.simplified ==
            pq.Quantity(1, "s").dimensionality):
        raise TypeError("The start time of the window (window[0]) "
                        "must be a time quantity.")
    if not (isinstance(window_stoptime, pq.quantity.Quantity) and
            window_stoptime.dimensionality.simplified ==
            pq.Quantity(1, "s").dimensionality):
        raise TypeError("The stop time of the window (window[1]) "
                        "must be a time quantity.")
    if window_stoptime <= window_starttime:
        raise ValueError("The start time of the window (window[0]) must be "
                         "earlier than the stop time of the window (window[1]).")

    # checks on signal
    if not isinstance(signal, AnalogSignal):
        raise TypeError(
            "Signal must be an AnalogSignal, not %s." % type(signal))
    if len(signal.shape) > 1:
        # num_signals: number of analog signals
        num_signals = signal.shape[1]
    else:
        raise ValueError("Empty analog signal, hence no averaging possible.")
    if window_stoptime - window_starttime > signal.t_stop - signal.t_start:
        raise ValueError("The chosen time window is larger than the "
                         "time duration of the signal.")

    # spiketrains type check
    if isinstance(spiketrains, (np.ndarray, SpikeTrain)):
        spiketrains = [spiketrains]
    elif isinstance(spiketrains, list):
        for st in spiketrains:
            if not isinstance(st, (np.ndarray, SpikeTrain)):
                raise TypeError(
                    "spiketrains must be a SpikeTrain, a numpy ndarray, or a "
                    "list of one of those, not %s." % type(spiketrains))
    else:
        raise TypeError(
            "spiketrains must be a SpikeTrain, a numpy ndarray, or a list of "
            "one of those, not %s." % type(spiketrains))

    # multiplying spiketrain in case only a single spiketrain is given
    if len(spiketrains) == 1 and num_signals != 1:
        template = spiketrains[0]
        spiketrains = []
        for i in range(num_signals):
            spiketrains.append(template)

    # checking for matching numbers of signals and spiketrains
    if num_signals != len(spiketrains):
        raise ValueError(
            "The number of signals and spiketrains has to be the same.")

    # checking the times of signal and spiketrains
    for i in range(num_signals):
        if spiketrains[i].t_start < signal.t_start:
            raise ValueError(
                "The spiketrain indexed by %i starts earlier than "
                "the analog signal." % i)
        if spiketrains[i].t_stop > signal.t_stop:
            raise ValueError(
                "The spiketrain indexed by %i stops later than "
                "the analog signal." % i)

    # *** Main algorithm: ***

    # window_bins: number of bins of the chosen averaging interval
    window_bins = int(np.ceil(((window_stoptime - window_starttime) *
        signal.sampling_rate).simplified))
    # result_sta: array containing finally the spike-triggered averaged signal
    result_sta = AnalogSignal(np.zeros((window_bins, num_signals)),
        sampling_rate=signal.sampling_rate, units=signal.units)
    # setting of correct times of the spike-triggered average
    # relative to the spike
    result_sta.t_start = window_starttime
    used_spikes = np.zeros(num_signals, dtype=int)
    unused_spikes = np.zeros(num_signals, dtype=int)
    total_used_spikes = 0

    for i in range(num_signals):
        # summing over all respective signal intervals around spiketimes
        for spiketime in spiketrains[i]:
            # checks for sufficient signal data around spiketime
            if (spiketime + window_starttime >= signal.t_start and
                    spiketime + window_stoptime <= signal.t_stop):
                # calculating the startbin in the analog signal of the
                # averaging window for spike
                startbin = int(np.floor(((spiketime + window_starttime -
                    signal.t_start) * signal.sampling_rate).simplified))
                # adds the signal in selected interval relative to the spike
                result_sta[:, i] += signal[
                    startbin: startbin + window_bins, i]
                # counting of the used spikes
                used_spikes[i] += 1
            else:
                # counting of the unused spikes
                unused_spikes[i] += 1

        # normalization
        result_sta[:, i] = result_sta[:, i] / used_spikes[i]

        total_used_spikes += used_spikes[i]

    if total_used_spikes == 0:
        warnings.warn(
            "No spike at all was either found or used for averaging")
    result_sta.annotate(used_spikes=used_spikes, unused_spikes=unused_spikes)

    return result_sta


def spike_field_coherence(signal, spiketrain, **kwargs):
    """
    Calculates the spike-field coherence between a analog signal(s) and a
    (binned) spike train.

    The current implementation makes use of scipy.signal.coherence(). Additional
    kwargs will will be directly forwarded to scipy.signal.coherence(),
    except for the axis parameter and the sampling frequency, which will be
    extracted from the input signals.

    The spike_field_coherence function receives an analog signal array and
    either a binned spike train or a spike train containing the original spike
    times. In case of original spike times the spike train is binned according
    to the sampling rate of the analog signal array.

    The AnalogSignal object can contain one or multiple signal traces. In case
    of multiple signal traces, the spike field coherence is calculated
    individually for each signal trace and the spike train.

    Parameters
    ----------
    signal : neo AnalogSignal object
        'signal' contains n analog signals.
    spiketrain : SpikeTrain or BinnedSpikeTrain
        Single spike train to perform the analysis on. The binsize of the
        binned spike train must match the sampling_rate of signal.

    KWArgs
    ------
    All KWArgs are passed to scipy.signal.coherence().

    Returns
    -------
    coherence : complex Quantity array
        contains the coherence values calculated for each analog signal trace
        in combination with the spike train. The first dimension corresponds to
        the frequency, the second to the number of the signal trace.
    frequencies : Quantity array
        contains the frequency values corresponding to the first dimension of
        the 'coherence' array

    Example
    -------

    Plot the SFC between a regular spike train at 20 Hz, and two sinusoidal
    time series at 20 Hz and 23 Hz, respectively.

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from quantities import s, ms, mV, Hz, kHz
    >>> import neo, elephant

    >>> t = pq.Quantity(range(10000),units='ms')
    >>> f1, f2 = 20. * Hz, 23. * Hz
    >>> signal = neo.AnalogSignal(np.array([
            np.sin(f1 * 2. * np.pi * t.rescale(s)),
            np.sin(f2 * 2. * np.pi * t.rescale(s))]).T,
            units=pq.mV, sampling_rate=1. * kHz)
    >>> spiketrain = neo.SpikeTrain(
        range(t[0], t[-1], 50), units='ms',
        t_start=t[0], t_stop=t[-1])
    >>> sfc, freqs = elephant.sta.spike_field_coherence(
        signal, spiketrain, window='boxcar')

    >>> plt.plot(freqs, sfc[:,0])
    >>> plt.plot(freqs, sfc[:,1])
    >>> plt.xlabel('Frequency [Hz]')
    >>> plt.ylabel('SFC')
    >>> plt.xlim((0, 60))
    >>> plt.show()
    """

    if not hasattr(scipy.signal, 'coherence'):
        raise AttributeError('scipy.signal.coherence is not available. The sfc '
                             'function uses scipy.signal.coherence for '
                             'the coherence calculation. This function is '
                             'available for scipy version 0.16 or newer. '
                             'Please update you scipy version.')

    # spiketrains type check
    if not isinstance(spiketrain, (SpikeTrain, BinnedSpikeTrain)):
        raise TypeError(
            "spiketrain must be of type SpikeTrain or BinnedSpikeTrain, "
            "not %s." % type(spiketrain))

    # checks on analogsignal
    if not isinstance(signal, AnalogSignal):
        raise TypeError(
            "Signal must be an AnalogSignal, not %s." % type(signal))
    if len(signal.shape) > 1:
        # num_signals: number of individual traces in the analog signal
        num_signals = signal.shape[1]
    elif len(signal.shape) == 1:
        num_signals = 1
    else:
        raise ValueError("Empty analog signal.")
    len_signals = signal.shape[0]

    # bin spiketrain if necessary
    if isinstance(spiketrain, SpikeTrain):
        spiketrain = BinnedSpikeTrain(
            spiketrain, binsize=signal.sampling_period)

    # check the start and stop times of signal and spike trains
    if spiketrain.t_start < signal.t_start:
        raise ValueError(
            "The spiketrain starts earlier than the analog signal.")
    if spiketrain.t_stop > signal.t_stop:
        raise ValueError(
            "The spiketrain stops later than the analog signal.")

    # check equal time resolution for both signals
    if spiketrain.binsize != signal.sampling_period:
        raise ValueError(
            "The spiketrain and signal must have a "
            "common sampling frequency / binsize")

    # calculate how many bins to add on the left of the binned spike train
    delta_t = spiketrain.t_start - signal.t_start
    if delta_t % spiketrain.binsize == 0:
        left_edge = int((delta_t / spiketrain.binsize).magnitude)
    else:
        raise ValueError("Incompatible binning of spike train and LFP")
    right_edge = int(left_edge + spiketrain.num_bins)

    # duplicate spike trains
    spiketrain_array = np.zeros((1, len_signals))
    spiketrain_array[0, left_edge:right_edge] = spiketrain.to_array()
    spiketrains_array = np.repeat(spiketrain_array, repeats=num_signals, axis=0).transpose()

    # calculate coherence
    frequencies, sfc = scipy.signal.coherence(
        spiketrains_array, signal.magnitude,
        fs=signal.sampling_rate.rescale('Hz').magnitude,
        axis=0, **kwargs)

    return (pq.Quantity(sfc, units=pq.dimensionless),
            pq.Quantity(frequencies, units=pq.Hz))
