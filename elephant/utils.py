from __future__ import division, print_function, unicode_literals

import warnings
from functools import wraps

import neo
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
    if hasattr(neo_objects, 't_start') and hasattr(neo_objects, 't_stop'):
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


def check_neo_consistency(neo_objects, object_type, t_start=None,
                          t_stop=None, tolerance=1e-6):
    """
    Checks that all input neo objects share the same units, t_start, and
    t_stop.

    Parameters
    ----------
    neo_objects : list of neo.SpikeTrain or neo.AnalogSignal
        A list of neo spike trains or analog signals.
    object_type : type
        The common type.
    t_start, t_stop : pq.Quantity or None, optional
        If None, check for exact match of t_start/t_stop across the input.
    tolerance : float, optional
        The absolute affordable tolerance for the discrepancies between
        t_start/stop magnitude values across trials.
        Default : 1e-6

    Raises
    ------
    TypeError
        If input objects are not instances of the specified `object_type`.
    ValueError
        If input object units, t_start, or t_stop do not match across trials.
    """
    if not isinstance(neo_objects, (list, tuple)):
        neo_objects = [neo_objects]
    try:
        units = neo_objects[0].units
        start = neo_objects[0].t_start.item()
        stop = neo_objects[0].t_stop.item()
    except AttributeError:
        raise TypeError("The input must be a list of {}. Got {}".format(
                object_type.__name__, type(neo_objects[0]).__name__))
    if tolerance is None:
        tolerance = 0
    for neo_obj in neo_objects:
        if not isinstance(neo_obj, object_type):
            raise TypeError("The input must be a list of {}. Got {}".format(
                object_type.__name__, type(neo_obj).__name__))
        if neo_obj.units != units:
            raise ValueError("The input must have the same units.")
        if t_start is None and abs(neo_obj.t_start.item() - start) > tolerance:
            raise ValueError("The input must have the same t_start.")
        if t_stop is None and abs(neo_obj.t_stop.item() - stop) > tolerance:
            raise ValueError("The input must have the same t_stop.")


def check_same_units(quantities, object_type=pq.Quantity):
    """
    Check that all input quantities are of the same type and share common
    units. Raise an error if the check is unsuccessful.

    Parameters
    ----------
    quantities : list of pq.Quantity or pq.Quantity
        A list of quantities, neo objects or a single neo object.
    object_type : type, optional
        The common type.
        Default: pq.Quantity

    Raises
    ------
    TypeError
        If input objects are not instances of the specified `object_type`.
    ValueError
        If input objects do not share common units.
    """
    if not isinstance(quantities, (list, tuple)):
        quantities = [quantities]
    for quantity in quantities:
        if not isinstance(quantity, object_type):
            raise TypeError("The input must be a list of {}. Got {}".format(
                object_type.__name__, type(quantity).__name__))
        if quantity.units != quantities[0].units:
            raise ValueError("The input quantities must have the same units, "
                             "which is achieved with object.rescale('ms') "
                             "operation.")
