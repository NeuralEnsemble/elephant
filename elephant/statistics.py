"""
docstring goes here

:copyright: Copyright 2014 by the Elephant team, see AUTHORS.txt.
:license: CeCILL, see LICENSE.txt for details.
"""

import numpy as np
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
        if units is not None:
            intervals.units = units
        return pq.Quantity(np.array(intervals), intervals.units)
    elif units is not None:
        return pq.Quantity(intervals, units)
    else:
        return intervals


def cv_isi(spiketrain):
    """
    Return the coefficient of variation of the inter-spike intervals.

    Accepts either a Neo SpikeTrain or a plain NumPy array, containing
    at least two spike times. If there are fewer than two spikes,
    raises ValueError.

    Parameters
    ----------

    spiketrain : Neo SpikeTrain or NumPy ndarray containing spike times

    Returns
    -------

    float

    Raises
    ------

    ValueError
        If the spiketrain contains fewer than two spikes.
    """
    if spiketrain.size > 1:
        intervals = isi(spiketrain)
        return np.std(intervals)/np.mean(intervals)
    else:
        # alternative is to return NaN
        raise ValueError("Spike train must contain at least two spikes.")
