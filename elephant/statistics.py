"""
docstring goes here

:copyright: Copyright 2014 by the ElePhAnT team, see AUTHORS.txt.
:license: CeCILL, see LICENSE.txt for details.
"""

import numpy as np
import quantities as pq


def isi(spiketrain):
    """
    Return a quantities array containing the inter-spike intervals of the
    SpikeTrain.
    """
    return pq.Quantity(np.array(np.diff(spiketrain)), spiketrain.units)


def cv_isi(spiketrain):
    """
    Return the coefficient of variation of the inter-spike intervals.
    """
    if spiketrain.size > 1:
        intervals = isi(spiketrain)
        return np.std(intervals)/np.mean(intervals)
    else:
        # should perhaps emit a warning?
        return np.nan
