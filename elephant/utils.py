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


def deprecate_binsize(func):
    @wraps(func)
    def deprecated_func(*args, **kwargs):
        if 'binsize' in kwargs:
            warnings.warn("'binsize' is deprecated and renamed to 'bin_size'; "
                          "'binsize' will be removed in v0.9.0 release. "
                          "Please use 'bin_size'.", DeprecationWarning)
            bin_size = kwargs.pop('binsize')
            kwargs['bin_size'] = bin_size
        return func(*args, **kwargs)

    return deprecated_func


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
