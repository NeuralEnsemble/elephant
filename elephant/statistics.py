"""
docstring goes here

:copyright: Copyright 2014 by the Elephant team, see AUTHORS.txt.
:license: CeCILL, see LICENSE.txt for details.
"""

from __future__ import division, print_function
import numpy as np
import scipy.stats
import quantities as pq


def isi(spiketrain, units=None):
    """
    Return an array containing the inter-spike intervals of the
    SpikeTrain.

    Accepts either a Neo SpikeTrain or a plain NumPy array. If either a
    SpikeTrain is provided or `units` is specified, the return value will
    be a quantities array, otherwise a plain NumPy array. The units of
    the quantities array will be the same as `units`, if given, or the
    units of the SpikeTrain otherwise.

    Parameters
    ----------

    spiketrain : Neo SpikeTrain or NumPy ndarray containing spike times
    units : str, optional

    Returns
    -------

    NumPy array or quantities array.

    """
    intervals = np.diff(spiketrain)
    if hasattr(spiketrain, "units"):
        intervals = pq.Quantity(np.array(intervals), units=spiketrain.units)
        if units is not None:
            intervals = intervals.rescale(units)
    elif units is not None:
        intervals = pq.Quantity(intervals, units)
    return intervals


# we make `cv` an alias for scipy.stats.variation for the convenience of former NeuroTools users
cv = scipy.stats.variation
