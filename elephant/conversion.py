# -*- coding: utf-8 -*-
"""
docstring goes here.

:copyright: Copyright 2014 by the Elephant team, see AUTHORS.txt.
:license: CeCILL, see LICENSE.txt for details.
"""

from __future__ import division, print_function

import numpy as np
import quantities as pq


def binarize(spiketrain, sampling_rate=None, t_start=None, t_stop=None,
             return_times=None):
    """
    Return an array indicating if spikes occured at individual time points.

    The array contains boolean values identifying whether one or more spikes
    happened in the corresponding time bin.  Time bins start at `t_start`
    and end at `t_stop`, spaced in `1/sampling_rate` intervals.

    Accepts either a Neo SpikeTrain, a Quantity array, or a plain NumPy array.
    Returns a boolean array with each element being the presence or absence of
    a spike in that time bin.  The number of spikes in a time bin is not
    considered.

    Optionally also returns an array of time points corresponding to the
    elements of the boolean array.  The units of this array will be the same as
    the units of the SpikeTrain, if any.

    Parameters
    ----------

    spiketrain : Neo SpikeTrain or Quantity array or NumPy array
                 The spike times.  Does not have to be sorted.
    sampling_rate : float or Quantity scalar, optional
                    The sampling rate to use for the time points.
                    If not specified, retrieved from the `sampling_rate`
                    attribute of `spiketrain`.
    t_start : float or Quantity scalar, optional
              The start time to use for the time points.
              If not specified, retrieved from the `t_start`
              attribute of `spiketrain`.  If that is not present, default to
              `0`.  Any value from `spiketrain` below this value is
              ignored.
    t_stop : float or Quantity scalar, optional
             The start time to use for the time points.
             If not specified, retrieved from the `t_stop`
             attribute of `spiketrain`.  If that is not present, default to
             the maximum value of `sspiketrain`.  Any value from
             `spiketrain` above this value is ignored.
    return_times : bool
                   If True, also return the corresponding time points.

    Returns
    -------

    values : NumPy array of bools
             A `True``value at a particular index indicates the presence of
             one or more spikes at the corresponding time point.
    times : NumPy array or Quantity array, optional
            The time points.  This will have the same units as `spiketrain`.
            If `spiketrain` has no units, this will be an NumPy array.

    Notes
    -----
    Spike times are placed in the bin of the closest time point, going to the
    higher bin if exactly between two bins.

    So in the case where the bins are `5.5` and `6.5`, with the spike time
    being `6.0`, the spike will be placed in the `6.5` bin.

    The upper edge of the last bin, equal to `t_stop`, is inclusive.  That is,
    a spike time exactly equal to `t_stop` will be included.

    If `spiketrain` is a Quantity or Neo SpikeTrain and
    `t_start`, `t_stop` or `sampling_rate` is not, then the arguments that
    are not quantities will be assumed to have the same units as`spiketrain`.

    Raises
    ------

    TypeError
        If `spiketrain` is a NumPy array and `t_start`, `t_stop`, or
        `sampling_rate` is a Quantity..

    ValueError
        `t_start` and `t_stop` can be inferred from `spiketrain` if
        not explicitly defined and not an attribute of `spiketrain`.
        `sampling_rate` cannot, so an exception is raised if it is not
        explicitly defined and not present as an attribute of `spiketrain`.
    """
    # get the values from spiketrain if they are not specified.
    if sampling_rate is None:
        sampling_rate = getattr(spiketrain, 'sampling_rate', None)
        if sampling_rate is None:
            raise ValueError('sampling_rate must either be explicitly defined '
                             'or must be an attribute of spiketrain')
    if t_start is None:
        t_start = getattr(spiketrain, 't_start', 0)
    if t_stop is None:
        t_stop = getattr(spiketrain, 't_stop', np.max(spiketrain))

    # we don't actually want the sampling rate, we want the sampling period
    sampling_period = 1./sampling_rate

    # figure out what units, if any, we are dealing with
    if hasattr(spiketrain, 'units'):
        units = spiketrain.units
        spiketrain = spiketrain.magnitude
    else:
        units = None

    # convert everything to the same units, then get the magnitude
    if hasattr(sampling_period, 'units'):
        if units is None:
            raise TypeError('sampling_period cannot be a Quantity if '
                            'spiketrain is not a quantity')
        sampling_period = sampling_period.rescale(units).magnitude
    if hasattr(t_start, 'units'):
        if units is None:
            raise TypeError('t_start cannot be a Quantity if '
                            'spiketrain is not a quantity')
        t_start = t_start.rescale(units).magnitude
    if hasattr(t_stop, 'units'):
        if units is None:
            raise TypeError('t_stop cannot be a Quantity if '
                            'spiketrain is not a quantity')
        t_stop = t_stop.rescale(units).magnitude

    # figure out the bin edges
    edges = np.arange(t_start-sampling_period/2, t_stop+sampling_period*3/2,
                      sampling_period)
    # we don't want to count any spikes before t_start or after t_stop
    if edges[-2] > t_stop:
        edges = edges[:-1]
    if edges[1] < t_start:
        edges = edges[1:]
    edges[0] = t_start
    edges[-1] = t_stop

    # this is where we actually get the binarized spike train
    res = np.histogram(spiketrain, edges)[0].astype('bool')

    # figure out what to output
    if not return_times:
        return res
    elif units is None:
        return res, np.arange(t_start, t_stop+sampling_period, sampling_period)
    else:
        return res, pq.Quantity(np.arange(t_start, t_stop+sampling_period,
                                          sampling_period), units=units)
