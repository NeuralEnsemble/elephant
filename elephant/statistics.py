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

from elephant.neoinfo import NeoInfo


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


def mean_firing_rate(spiketrain, t_start=None, t_stop=None, axis=None):
    """
    Return the firing rate of the SpikeTrain.

    Accepts a Neo SpikeTrain, a Quantity array, or a plain NumPy array.
    If either a SpikeTrain or Quantity array is provided, the return value will
    be a quantities array, otherwise a plain NumPy array. The units of
    the quantities array will be the inverse of the spiketrain.

    The interval over which the firing rate is calculated can be optionally
    controlled with `t_start` and `t_stop`

    Parameters
    ----------

    spiketrain : Neo SpikeTrain or Quantity array or NumPy ndarray
                 The spike times.
    t_start : float or Quantity scalar, optional
              The start time to use for the inveral.
              If not specified, retrieved from the``t_start`
              attribute of `spiketrain`.  If that is not present, default to
              `0`.  Any value from `spiketrain` below this value is ignored.
    t_stop : float or Quantity scalar, optional
             The stop time to use for the time points.
             If not specified, retrieved from the `t_stop`
             attribute of `spiketrain`.  If that is not present, default to
             the maximum value of `spiketrain`.  Any value from
             `spiketrain` above this value is ignored.
    axis : int, optional
           The axis over which to do the calculation.
           Default is `None`, do the calculation over the flattened array.

    Returns
    -------

    float, quantities scalar, NumPy array or quantities array.

    Notes
    -----

    If `spiketrain` is a Quantity or Neo SpikeTrain and `t_start` or `t_stop`
    are not, `t_start` and `t_stop` are assumed to have the same units as
    `spiketrain`.

    Raises
    ------

    TypeError
        If `spiketrain` is a NumPy array and `t_start` or `t_stop`
        is a quantity scalar.

    """
    if t_start is None:
        t_start = getattr(spiketrain, 't_start', 0)

    found_t_start = False
    if t_stop is None:
        if hasattr(spiketrain, 't_stop'):
            t_stop = spiketrain.t_stop
        else:
            t_stop = np.max(spiketrain, axis=axis)
            found_t_start = True

    # figure out what units, if any, we are dealing with
    if hasattr(spiketrain, 'units'):
        units = spiketrain.units
    else:
        units = None

    # convert everything to the same units
    if hasattr(t_start, 'units'):
        if units is None:
            raise TypeError('t_start cannot be a Quantity if '
                            'spiketrain is not a quantity')
        t_start = t_start.rescale(units)
    elif units is not None:
        t_start = pq.Quantity(t_start, units=units)
    if hasattr(t_stop, 'units'):
        if units is None:
            raise TypeError('t_stop cannot be a Quantity if '
                            'spiketrain is not a quantity')
        t_stop = t_stop.rescale(units)
    elif units is not None:
        t_stop = pq.Quantity(t_stop, units=units)

    if not axis or not found_t_start:
        return np.sum((spiketrain >= t_start) & (spiketrain <= t_stop),
                      axis=axis) / (t_stop - t_start)
    else:
        # this is needed to handle broadcasting between spiketrain and t_stop
        t_stop_test = np.expand_dims(t_stop, axis)
        return np.sum((spiketrain >= t_start) & (spiketrain <= t_stop_test),
                      axis=axis) / (t_stop - t_start)


def fano(data):
    """
    Evaluate the empirical Fano Factor (FF) of the spike counts of a list of
    spike trains.

    The Fano factor is calculated as the variance of the spike count across
    trials of a fixed duration, divided by the mean of the spike count.

    Parameters
    ----------
    data : Neo object
        The input must consist of N>1 trials of data. Each trial must contain
        exactly one spike train.

    Returns
    -------
    ff : float
        The Fano factor of the spike trains.

    Raises
    ------
    ValueError
        Raised if:
            * no trials containing exactly one spike train are available in
              the input or
            * all spike trains are empty
            * or the valid trials are not of equal length

    Example
    -------
    This example creates 300 spike trains containing up to 20 spikes each, and
    calculates the Fano factor.

    >>> import numpy as np
    >>> import numpy.random
    >>> import neo.core
    >>> numpy.random.seed(100)
    >>> blk = neo.core.Block()
    >>> for i in range(300):
    >>>     seg = neo.core.Segment(name='segment %d' % i, index=i)
    >>>     st = neo.core.SpikeTrain(
    >>>         numpy.random.rand(numpy.random.randint(19) + 1) * pq.s,
    >>>         t_start=0 * pq.s,
    >>>         t_stop=10.0 * pq.s)
    >>>     seg.spiketrains.append(st)
    >>>     blk.segments.append(seg)
    >>> print fano(blk)
    """
    ni = NeoInfo(data)

    # test for exactly one spike train per trial
    ni.set_trial_conditions(
        trial_has_exact_st=(True, 1))

    # test if we have valid trials
    if not ni.has_trials():
        raise ValueError('Unable to find trials.')

    # test if all valid trials are of equal length
    if not ni.is_trials_equal_len(ni.valid_trial_ids):
        raise ValueError("Not all valid trials are of equal length")

    # get all SpikeTrains from valid trials
    # output format: [[trialid,[spiketrain]],[trialid,[spiketrain]],...]
    trial_list = ni.get_spiketrains_from_valid_trials()

    # create an array of all spike counts
    counts = np.array([len(sp[1][0].times) for sp in trial_list])

    var_counts = np.var(counts)
    mean_counts = np.mean(counts)
    if mean_counts == 0.:
        raise ValueError(
            "Unable to compute Fano factor: all spike trains are empty.")
    return var_counts / mean_counts


# we make `cv` an alias for scipy.stats.variation for the convenience
# of former NeuroTools users
cv = scipy.stats.variation
