"""
This module generates Poisson spike trains that are used in the example
scripts.
"""

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
