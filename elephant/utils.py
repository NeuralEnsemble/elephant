from __future__ import division, print_function, unicode_literals

import warnings
from functools import wraps

import numpy as np
import quantities as pq


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


def check_same_units(quantities, object_type=pq.Quantity):
    """
    Check that all input quantities are of the same type and share common
    units. Raise an error if the check is unsuccessful.

    Parameters
    ----------
    quantities : list of pq.Quantity or pq.Quantity
        A list of quantities, neo objects or a single neo object.
    object_type : type
        The common type.

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
