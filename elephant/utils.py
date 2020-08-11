from __future__ import division, print_function, unicode_literals

import warnings
from functools import wraps

import numpy as np
import quantities as pq

from neo import SpikeTrain


def is_binary(array):
    """
    Parameters
    ----------
    array: np.ndarray or list

    Returns
    -------
    bool
        Whether the input array is binary or not.

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


def deprecated_alias(**aliases):
    """
    A deprecation decorator constructor.

    Parameters
    ----------
    aliases: str
        The key-value pairs of mapping old --> new argument names of a
        function.

    Returns
    -------
    callable
        A decorator for the specific mapping of deprecated argument names.

    Examples
    --------
    In the example below, `my_function(binsize)` signature is marked as
    deprecated (but still usable) and changed to `my_function(bin_size)`.

    >>> @deprecated_alias(binsize='bin_size')
    ... def my_function(bin_size):
    ...     pass

    """
    def deco(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            _rename_kwargs(func.__name__, kwargs, aliases)
            return func(*args, **kwargs)

        return wrapper

    return deco


def _rename_kwargs(func_name, kwargs, aliases):
    for old, new in aliases.items():
        if old in kwargs:
            if new in kwargs:
                raise TypeError("{} received both '{}' and '{}'".format(
                    func_name, old, new))
            warnings.warn("'{}' is deprecated; use '{}'".format(old, new),
                          DeprecationWarning)
            kwargs[new] = kwargs.pop(old)


def is_time_quantity(x, allow_none=False):
    """
    Parameters
    ----------
    x : array-like
        A scalar or array-like to check for being a Quantity with time units.
    allow_none : bool
        Allow `x` to be None or not.

    Returns
    -------
    bool
        Whether the input is a time Quantity (True) or not (False).
        If the input is None and `allow_none` is set to True, returns True.

    """
    if x is None and allow_none:
        return True
    if not isinstance(x, pq.Quantity):
        return False
    return x.dimensionality.simplified == pq.Quantity(1, "s").dimensionality
