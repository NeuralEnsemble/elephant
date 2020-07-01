from __future__ import division, print_function, unicode_literals

import numpy as np

from neo import SpikeTrain


def is_binary(array):
    """
    Parameters
    ----------
    array: np.ndarray or list

    Returns
    -------
    bool
        Whether the input array is binary or nor.

    """
    array = np.asarray(array)
    return ((array == 0) | (array == 1)).all()


def _check_consistency_of_spiketrainlist(spiketrains,
                                         same_t_start=False,
                                         same_t_stop=False,
                                         same_units=False):
    """
    Private function to check the consistency of a list of neo.SpikeTrain

    Raises
    ------
    TypeError
        When `spiketrains` is not a list.
    ValueError
        When `spiketrains` is an empty list.
    TypeError
        When the elements in `spiketrains` are not instances of neo.SpikeTrain
    ValueError
        When `t_start` is not the same for all spiketrains,
        if same_t_start=True
    ValueError
        When `t_stop` is not the same for all spiketrains,
        if same_t_stop=True
    ValueError
        When `units` are not the same for all spiketrains,
        if same_units=True
    """
    if not isinstance(spiketrains, list):
        raise TypeError('spiketrains should be a list of neo.SpikeTrain')
    if len(spiketrains) == 0:
        raise ValueError('The spiketrains list is empty!')
    for st in spiketrains:
        if not isinstance(st, SpikeTrain):
            raise TypeError(
                'elements in spiketrains list must be instances of '
                ':class:`SpikeTrain` of Neo!'
                'Found: %s, value %s' % (type(st), str(st)))
        if same_t_start and not st.t_start == spiketrains[0].t_start:
            raise ValueError(
                "the spike trains must have the same t_start!")
        if same_t_stop and not st.t_stop == spiketrains[0].t_stop:
            raise ValueError(
                "the spike trains must have the same t_stop!")
        if same_units and not st.units == spiketrains[0].units:
            raise ValueError('The spike trains must have the same units!')
