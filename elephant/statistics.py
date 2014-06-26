# -*- coding: utf-8 -*-
"""
docstring goes here.

:copyright: Copyright 2014 by the Elephant team, see AUTHORS.txt.
:license: Modified BSD, see LICENSE.txt for details.
"""

from __future__ import division, print_function

import numpy as np
import quantities as pq
import scipy.stats


def isi(spiketrain, axis=-1):
    """
    Return an array containing the inter-spike intervals of the SpikeTrain.

    Accepts a Neo SpikeTrain, a Quantity array, or a plain NumPy array.
    If either a SpikeTrain or Quantity array is provided, the return value will
    be a quantities array, otherwise a plain NumPy array. The units of
    the quantities array will be the same as spiketrain.

    Parameters
    ----------

    spiketrain : Neo SpikeTrain or Quantity array or NumPy ndarray
                 The spike times.
    axis : int, optional
           The axis along which the difference is taken.
           Default is the last axis.

    Returns
    -------

    NumPy array or quantities array.

    """
    if axis is None:
        axis = -1
    intervals = np.diff(spiketrain, axis=axis)
    if hasattr(spiketrain, 'waveforms'):
        intervals = pq.Quantity(intervals.magnitude, units=spiketrain.units)
    return intervals


# we make `cv` an alias for scipy.stats.variation for the convenience
# of former NeuroTools users
cv = scipy.stats.variation
