from __future__ import division, print_function, unicode_literals

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
