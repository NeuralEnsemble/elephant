from __future__ import division, print_function, unicode_literals

import warnings
from functools import wraps

import numpy as np


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
