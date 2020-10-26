from __future__ import division, print_function, unicode_literals

import warnings
from functools import wraps

import numpy as np
import quantities as pq
import neo


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


def get_common_start_stop_times(neo_objects):
    """
    Extracts the `t_start`and the `t_stop` from the input neo objects.

    If a single neo object is given, its `t_start` and `t_stop` is returned.
    Otherwise, the aligned times are returned: the maximal `t_start` and
    minimal `t_stop` across `neo_objects`.

    Parameters
    ----------
    neo_objects : neo.SpikeTrain or neo.AnalogSignal or list
        A neo object or a list of neo objects that have `t_start` and `t_stop`
        attributes.

    Returns
    -------
    t_start, t_stop : pq.Quantity
        Shared start and stop times.

    Raises
    ------
    AttributeError
        If the input neo objects do not have `t_start` and `t_stop` attributes.
    ValueError
        If there is no shared interval ``[t_start, t_stop]`` across the input
        neo objects.
    """
    if isinstance(neo_objects, neo.SpikeTrain):
        return neo_objects.t_start, neo_objects.t_stop
    try:
        t_start = max(elem.t_start for elem in neo_objects)
        t_stop = min(elem.t_stop for elem in neo_objects)
    except AttributeError:
        raise AttributeError("Input neo objects must have 't_start' and "
                             "'t_stop' attributes")
    if t_stop < t_start:
        raise ValueError("t_stop ({t_stop}) is smaller than t_start "
                         "({t_start})".format(t_stop=t_stop, t_start=t_start))
    return t_start, t_stop


def check_consistency_of_spiketrains(spiketrains, t_start=None,
                                     t_stop=None):
    for st in spiketrains:
        if not isinstance(st, neo.SpikeTrain):
            raise TypeError("The spike trains must be instances of "
                            "neo.SpikeTrain. Found: '{}'".
                            format(type(st)))

        if t_start is None and not st.t_start == spiketrains[0].t_start:
            raise ValueError("The spike trains must have the same t_start.")
        if t_stop is None and not st.t_stop == spiketrains[0].t_stop:
            raise ValueError("The spike trains must have the same t_stop.")
        if not st.units == spiketrains[0].units:
            raise ValueError("The spike trains must have the same units.")


def rescale_magnitude(neo_object, units):
    factor = neo_object.units.rescale(units).item()
    return neo_object.magnitude * factor
