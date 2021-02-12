"""
This module generates Poisson spike trains or analog data that are used in the
example scripts.
"""
import numpy as np
import quantities as pq
import neo

from elephant.spike_train_generation import homogeneous_poisson_process


def get_spike_trains(firing_rate, n_spiketrains, t_stop):
    """
    Generates a list of Poisson spike trains.

    Parameters
    ----------
    firing_rate : pq.Quantity
        Firing rate of the spike trains.
    n_spiketrains : int
        Number of spike trains to generate.
    t_stop : pq.Quantity
        Stop time of the generated spike trains.

    Returns
    -------
    list
        List of `neo.SpikeTrain` objects.
    """
    return [homogeneous_poisson_process(firing_rate, t_stop=t_stop)
            for _ in range(n_spiketrains)]


def get_analog_signal(frequency, n_channels, t_stop, amplitude, sampling_rate):
    """
    Generates a sine wave with a particular frequency, duration and amplitude.
    The signal is repeated to simulate multichannel data.

    Parameters
    ----------
    frequency : pq.Quantity
        A quantity describing the sine wave frequency.
    n_channels : int
        Number of channels to generate.
    t_stop : pq.Quantity
        A time quantity to define the duration of the signal.
    amplitude : pq.Quantity
        A voltage quantity defining the amplitude of the oscillation.
    sampling_rate : pq.Quantity
        A frequency quantity describing the sampling frequency of the
        generated signal.

    Returns
    -------
    neo.AnalogSignal
        It will have the shape (T, N) where `T` is the total number of samples
        considering the `sampling_rate` and `t_stop` and `N` is `n_channels`.
    """
    end_time = t_stop.rescale(pq.s).magnitude
    period = 1/sampling_rate.rescale(pq.Hz).magnitude
    freq = frequency.rescale(pq.Hz).magnitude

    samples = np.arange(0, end_time, period)
    base_signal = np.sin(2*np.pi*freq*samples) * amplitude.magnitude

    signal = np.tile(base_signal, (n_channels, 1))

    array_annotations = {
        'channel_names': np.array([f"chan{ch+1}" for ch in range(n_channels)])
    }
    return neo.AnalogSignal(signal.T, units=amplitude.units,
                            sampling_rate=sampling_rate,
                            array_annotations=array_annotations)
