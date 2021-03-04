# -*- coding: utf-8 -*-
"""
Statistical measures of spike trains (e.g., Fano factor) and functions to
estimate firing rates.

Rate estimation
***************

.. autosummary::
    :toctree: _toctree/statistics/

    mean_firing_rate
    instantaneous_rate
    time_histogram
    optimal_kernel_bandwidth


Spike interval statistics
*************************

.. autosummary::
    :toctree: _toctree/statistics/

    isi
    cv
    cv2
    lv
    lvr


Statistics across spike trains
******************************

.. autosummary::
    :toctree: _toctree/statistics/

    fanofactor
    complexity_pdf
    Complexity


Tutorial
********

:doc:`View tutorial <../tutorials/statistics>`

Run tutorial interactively:

.. image:: https://mybinder.org/badge.svg
   :target: https://mybinder.org/v2/gh/NeuralEnsemble/elephant/master
            ?filepath=doc/tutorials/statistics.ipynb


References
----------

.. bibliography:: ../bib/elephant.bib
   :labelprefix: st
   :keyprefix: statistics-
   :style: unsrt


:copyright: Copyright 2014-2020 by the Elephant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
"""

from __future__ import division, print_function

import math
import warnings

import neo
import numpy as np
import quantities as pq
import scipy.stats

import elephant.conversion as conv
import elephant.kernels as kernels
from elephant.conversion import BinnedSpikeTrain
from elephant.utils import deprecated_alias, check_neo_consistency, \
    is_time_quantity, round_binning_errors

# do not import unicode_literals
# (quantities rescale does not work with unicodes)

__all__ = [
    "isi",
    "mean_firing_rate",
    "fanofactor",
    "cv",
    "cv2",
    "lv",
    "lvr",
    "instantaneous_rate",
    "time_histogram",
    "complexity_pdf",
    "Complexity",
    "fftkernel",
    "optimal_kernel_bandwidth"
]

cv = scipy.stats.variation


def isi(spiketrain, axis=-1):
    """
    Return an array containing the inter-spike intervals of the spike train.

    Accepts a `neo.SpikeTrain`, a `pq.Quantity` array, a `np.ndarray`, or a
    list of time spikes. If either a `neo.SpikeTrain` or `pq.Quantity` is
    provided, the return value will be `pq.Quantity`, otherwise `np.ndarray`.
    The units of `pq.Quantity` will be the same as `spiketrain`.

    Visualization of this function is covered in Viziphant:
    :func:`viziphant.statistics.plot_isi_histogram`.

    Parameters
    ----------
    spiketrain : neo.SpikeTrain or pq.Quantity or array-like
        The spike times.
    axis : int, optional
        The axis along which the difference is taken.
        Default: the last axis

    Returns
    -------
    intervals : np.ndarray or pq.Quantity
        The inter-spike intervals of the `spiketrain`.

    Warns
    -----
    UserWarning
        When the input array is not sorted, negative intervals are returned
        with a warning.

    Examples
    --------
    >>> from elephant import statistics
    >>> statistics.isi([0.3, 4.5, 6.7, 9.3])
    array([4.2, 2.2, 2.6])

    """
    if isinstance(spiketrain, neo.SpikeTrain):
        intervals = np.diff(spiketrain.magnitude, axis=axis)
        # np.diff makes a copy
        intervals = pq.Quantity(intervals, units=spiketrain.units, copy=False)
    else:
        intervals = np.diff(spiketrain, axis=axis)
    if (intervals < 0).any():
        warnings.warn("ISI evaluated to negative values. "
                      "Please sort the input array.")

    return intervals


def mean_firing_rate(spiketrain, t_start=None, t_stop=None, axis=None):
    """
    Return the firing rate of the spike train.

    The firing rate is calculated as the number of spikes in the spike train
    in the range `[t_start, t_stop]` divided by the time interval
    `t_stop - t_start`. See the description below for cases when `t_start` or
    `t_stop` is None.

    Accepts a `neo.SpikeTrain`, a `pq.Quantity` array, or a plain
    `np.ndarray`. If either a `neo.SpikeTrain` or `pq.Quantity` array is
    provided, the return value will be a `pq.Quantity` array, otherwise a
    plain `np.ndarray`. The units of the `pq.Quantity` array will be the
    inverse of the `spiketrain`.

    Parameters
    ----------
    spiketrain : neo.SpikeTrain or pq.Quantity or np.ndarray
        The spike times.
    t_start : float or pq.Quantity, optional
        The start time to use for the interval.
        If None, retrieved from the `t_start` attribute of `spiketrain`. If
        that is not present, default to 0. All spiketrain's spike times below
        this value are ignored.
        Default: None
    t_stop : float or pq.Quantity, optional
        The stop time to use for the time points.
        If not specified, retrieved from the `t_stop` attribute of
        `spiketrain`. If that is not present, default to the maximum value of
        `spiketrain`. All spiketrain's spike times above this value are
        ignored.
        Default: None
    axis : int, optional
        The axis over which to do the calculation; has no effect when the
        input is a neo.SpikeTrain, because a neo.SpikeTrain is always a 1-d
        vector. If None, do the calculation over the flattened array.
        Default: None

    Returns
    -------
    float or pq.Quantity or np.ndarray
        The firing rate of the `spiketrain`

    Raises
    ------
    TypeError
        If the input spiketrain is a `np.ndarray` but `t_start` or `t_stop` is
        `pq.Quantity`.

        If the input spiketrain is a `neo.SpikeTrain` or `pq.Quantity` but
        `t_start` or `t_stop` is not `pq.Quantity`.
    ValueError
        If the input spiketrain is empty.

    Examples
    --------
    >>> from elephant import statistics
    >>> statistics.mean_firing_rate([0.3, 4.5, 6.7, 9.3])
    0.4301075268817204

    """
    if isinstance(spiketrain, neo.SpikeTrain) and t_start is None \
            and t_stop is None and axis is None:
        # a faster approach for a typical use case
        n_spikes = len(spiketrain)
        time_interval = spiketrain.t_stop - spiketrain.t_start
        time_interval = time_interval.rescale(spiketrain.units)
        rate = n_spikes / time_interval
        return rate

    if isinstance(spiketrain, pq.Quantity):
        # Quantity or neo.SpikeTrain
        if not is_time_quantity(t_start, allow_none=True):
            raise TypeError("'t_start' must be a Quantity or None")
        if not is_time_quantity(t_stop, allow_none=True):
            raise TypeError("'t_stop' must be a Quantity or None")

        units = spiketrain.units
        if t_start is None:
            t_start = getattr(spiketrain, 't_start', 0 * units)
        t_start = t_start.rescale(units).magnitude
        if t_stop is None:
            t_stop = getattr(spiketrain, 't_stop',
                             np.max(spiketrain, axis=axis))
        t_stop = t_stop.rescale(units).magnitude

        # calculate as a numpy array
        rates = mean_firing_rate(spiketrain.magnitude, t_start=t_start,
                                 t_stop=t_stop, axis=axis)

        rates = pq.Quantity(rates, units=1. / units)
    elif isinstance(spiketrain, (np.ndarray, list, tuple)):
        if isinstance(t_start, pq.Quantity) or isinstance(t_stop, pq.Quantity):
            raise TypeError("'t_start' and 't_stop' cannot be quantities if "
                            "'spiketrain' is not a Quantity.")
        spiketrain = np.asarray(spiketrain)
        if len(spiketrain) == 0:
            raise ValueError("Empty input spiketrain.")
        if t_start is None:
            t_start = 0
        if t_stop is None:
            t_stop = np.max(spiketrain, axis=axis)
        time_interval = t_stop - t_start
        if axis and isinstance(t_stop, np.ndarray):
            t_stop = np.expand_dims(t_stop, axis)
        rates = np.sum((spiketrain >= t_start) & (spiketrain <= t_stop),
                       axis=axis) / time_interval
    else:
        raise TypeError("Invalid input spiketrain type: '{}'. Allowed: "
                        "neo.SpikeTrain, Quantity, ndarray".
                        format(type(spiketrain)))
    return rates


def fanofactor(spiketrains, warn_tolerance=0.1 * pq.ms):
    r"""
    Evaluates the empirical Fano factor F of the spike counts of
    a list of `neo.SpikeTrain` objects.

    Given the vector v containing the observed spike counts (one per
    spike train) in the time window [t0, t1], F is defined as:

    .. math::
        F := \frac{var(v)}{mean(v)}

    The Fano factor is typically computed for spike trains representing the
    activity of the same neuron over different trials. The higher F, the
    larger the cross-trial non-stationarity. In theory for a time-stationary
    Poisson process, F=1.

    Parameters
    ----------
    spiketrains : list
        List of `neo.SpikeTrain` or `pq.Quantity` or `np.ndarray` or list of
        spike times for which to compute the Fano factor of spike counts.
    warn_tolerance : pq.Quantity
        In case of a list of input neo.SpikeTrains, if their durations vary by
        more than `warn_tolerence` in their absolute values, throw a warning
        (see Notes).
        Default: 0.1 ms

    Returns
    -------
    fano : float
        The Fano factor of the spike counts of the input spike trains.
        Returns np.NaN if an empty list is specified, or if all spike trains
        are empty.

    Raises
    ------
    TypeError
        If the input spiketrains are neo.SpikeTrain objects, but
        `warn_tolerance` is not a quantity.

    Notes
    -----
    The check for the equal duration of the input spike trains is performed
    only if the input is of type`neo.SpikeTrain`: if you pass a numpy array,
    please make sure that they all have the same duration manually.

    Examples
    --------
    >>> import neo
    >>> from elephant import statistics
    >>> spiketrains = [
    ...     neo.SpikeTrain([0.3, 4.5, 6.7, 9.3], t_stop=10, units='s'),
    ...     neo.SpikeTrain([1.4, 3.3, 8.2], t_stop=10, units='s')
    ... ]
    >>> statistics.fanofactor(spiketrains)
    0.07142857142857142

    """
    # Build array of spike counts (one per spike train)
    spike_counts = np.array([len(st) for st in spiketrains])

    # Compute FF
    if all(count == 0 for count in spike_counts):
        # empty list of spiketrains reaches this branch, and NaN is returned
        return np.nan

    if all(isinstance(st, neo.SpikeTrain) for st in spiketrains):
        if not is_time_quantity(warn_tolerance):
            raise TypeError("'warn_tolerance' must be a time quantity.")
        durations = [(st.t_stop - st.t_start).simplified.item()
                     for st in spiketrains]
        durations_min = min(durations)
        durations_max = max(durations)
        if durations_max - durations_min > warn_tolerance.simplified.item():
            warnings.warn("Fano factor calculated for spike trains of "
                          "different duration (minimum: {_min}s, maximum "
                          "{_max}s).".format(_min=durations_min,
                                             _max=durations_max))

    fano = spike_counts.var() / spike_counts.mean()
    return fano


def __variation_check(v, with_nan):
    # ensure the input ia a vector
    if v.ndim != 1:
        raise ValueError("The input must be a vector, not a {}-dim matrix.".
                         format(v.ndim))

    # ensure we have enough entries
    if v.size < 2:
        if with_nan:
            warnings.warn("The input size is too small. Please provide"
                          "an input with more than 1 entry. Returning `NaN`"
                          "since the argument `with_nan` is `True`")
            return np.NaN
        raise ValueError("Input size is too small. Please provide "
                         "an input with more than 1 entry. Set 'with_nan' "
                         "to True to replace the error by a warning.")

    return None


@deprecated_alias(v='time_intervals')
def cv2(time_intervals, with_nan=False):
    r"""
    Calculate the measure of Cv2 for a sequence of time intervals between
    events :cite:`statistics-Holt1996_1806`.

    Given a vector :math:`I` containing a sequence of intervals, the Cv2 is
    defined as:

    .. math::
        Cv2 := \frac{1}{N} \sum_{i=1}^{N-1}
                           \frac{2|I_{i+1}-I_i|}
                          {|I_{i+1}+I_i|}

    The Cv2 is typically computed as a substitute for the classical
    coefficient of variation (Cv) for sequences of events which include some
    (relatively slow) rate fluctuation.  As with the Cv, Cv2=1 for a sequence
    of intervals generated by a Poisson process.

    Parameters
    ----------
    time_intervals : pq.Quantity or np.ndarray or list
        Vector of consecutive time intervals.
    with_nan : bool, optional
        If True, `cv2` of a spike train with less than two spikes results in a
        np.NaN value and a warning is raised.
        If False, `ValueError` exception is raised with a spike train with
        less than two spikes.
        Default: True

    Returns
    -------
    float
        The Cv2 of the inter-spike interval of the input sequence.

    Raises
    ------
    ValueError
        If an empty list is specified, or if the sequence has less than two
        entries and `with_nan` is False.

        If a matrix is passed to the function. Only vector inputs are
        supported.

    Warns
    -----
    UserWarning
        If `with_nan` is True and `cv2` is calculated for a sequence with less
        than two entries, generating a np.NaN.

    Examples
    --------
    >>> from elephant import statistics
    >>> statistics.cv2([0.3, 4.5, 6.7, 9.3])
    0.8226190476190478

    """
    # convert to array, cast to float
    time_intervals = np.asarray(time_intervals)
    np_nan = __variation_check(time_intervals, with_nan)
    if np_nan is not None:
        return np_nan

    # calculate Cv2 and return result
    cv_i = np.diff(time_intervals) / (time_intervals[:-1] + time_intervals[1:])
    return 2. * np.mean(np.abs(cv_i))


@deprecated_alias(v='time_intervals')
def lv(time_intervals, with_nan=False):
    r"""
    Calculate the measure of local variation Lv for a sequence of time
    intervals between events :cite:`statistics-Shinomoto2003_2823`.

    Given a vector :math:`I` containing a sequence of intervals, the Lv is
    defined as:

    .. math::
        Lv := \frac{1}{N} \sum_{i=1}^{N-1}
                          \frac{3(I_i-I_{i+1})^2}
                          {(I_i+I_{i+1})^2}

    The Lv is typically computed as a substitute for the classical coefficient
    of variation for sequences of events which include some (relatively slow)
    rate fluctuation.  As with the Cv, Lv=1 for a sequence of intervals
    generated by a Poisson process.

    Parameters
    ----------
    time_intervals : pq.Quantity or np.ndarray or list
        Vector of consecutive time intervals.
    with_nan : bool, optional
        If True, the Lv of a spike train with less than two spikes results in a
        `np.NaN` value and a warning is raised.
        If False, a `ValueError` exception is raised with a spike train with
        less than two spikes.
        Default: True

    Returns
    -------
    float
        The Lv of the inter-spike interval of the input sequence.

    Raises
    ------
    ValueError
        If an empty list is specified, or if the sequence has less than two
        entries and `with_nan` is False.

        If a matrix is passed to the function. Only vector inputs are
        supported.

    Warns
    -----
    UserWarning
        If `with_nan` is True and the Lv is calculated for a spike train
        with less than two spikes, generating a np.NaN.

    Examples
    --------
    >>> from elephant import statistics
    >>> statistics.lv([0.3, 4.5, 6.7, 9.3])
    0.8306154336734695

    """
    # convert to array, cast to float
    time_intervals = np.asarray(time_intervals)
    np_nan = __variation_check(time_intervals, with_nan)
    if np_nan is not None:
        return np_nan

    cv_i = np.diff(time_intervals) / (time_intervals[:-1] + time_intervals[1:])
    return 3. * np.mean(np.power(cv_i, 2))


def lvr(time_intervals, R=5*pq.ms, with_nan=False):
    r"""
    Calculate the measure of revised local variation LvR for a sequence of time
    intervals between events :cite:`statistics-Shinomoto2009_e1000433`.

    Given a vector :math:`I` containing a sequence of intervals, the LvR is
    defined as:

    .. math::
        LvR := \frac{3}{N-1} \sum_{i=1}^{N-1}
                            \left(1-\frac{4 I_i I_{i+1}}
                            {(I_i+I_{i+1})^2}\right)
                            \left(1+\frac{4 R}{I_i+I_{i+1}}\right)

    The LvR is a revised version of the Lv, with enhanced invariance to firing
    rate fluctuations by introducing a refractoriness constant R. The LvR with
    `R=5ms` was shown to outperform other ISI variability measures in spike
    trains with firing rate fluctuations and sensory stimuli
    :cite:`statistics-Shinomoto2009_e1000433`.

    Parameters
    ----------
    time_intervals : pq.Quantity or np.ndarray or list
        Vector of consecutive time intervals. Must have time units, if not unit
        is passed `ms` are assumed.
    R : pq.Quantity or int or float
        Refractoriness constant (R >= 0). If no quantity is passed `ms` are
        assumed.
        Default: 5 ms
    with_nan : bool, optional
        If True, LvR of a spike train with less than two spikes results in a
        np.NaN value and a warning is raised.
        If False, a `ValueError` exception is raised with a spike train with
        less than two spikes.
        Default: True

    Returns
    -------
    float
        The LvR of the inter-spike interval of the input sequence.

    Raises
    ------
    ValueError
        If an empty list is specified, or if the sequence has less than two
        entries and `with_nan` is False.

        If a matrix is passed to the function. Only vector inputs are
        supported.

    Warns
    -----
    UserWarning
        If `with_nan` is True and the `lvr` is calculated for a spike train
        with less than two spikes, generating a np.NaN.
        If R is passed without any units attached milliseconds are assumed.

    Examples
    --------
    >>> from elephant import statistics
    >>> statistics.lvr([0.3, 4.5, 6.7, 9.3], R=0.005)
    0.833907445980624
    """
    if isinstance(R, pq.Quantity):
        R = R.rescale('ms').magnitude
    else:
        warnings.warn('No units specified for R, assuming milliseconds (ms)')

    if R < 0:
        raise ValueError('R must be >= 0')

    # check units of intervals if available
    if isinstance(time_intervals, pq.Quantity):
        time_intervals = time_intervals.rescale('ms').magnitude
    else:
        warnings.warn('No units specified for time_intervals,'
                      ' assuming milliseconds (ms)')

    # convert to array, cast to float
    time_intervals = np.asarray(time_intervals)
    np_nan = __variation_check(time_intervals, with_nan)
    if np_nan is not None:
        return np_nan

    N = len(time_intervals)
    t = time_intervals[:-1] + time_intervals[1:]
    frac1 = 4 * time_intervals[:-1] * time_intervals[1:] / t**2
    frac2 = 4 * R / t
    lvr = (3 / (N-1)) * np.sum((1-frac1) * (1+frac2))
    return lvr


@deprecated_alias(spiketrain='spiketrains')
def instantaneous_rate(spiketrains, sampling_period, kernel='auto',
                       cutoff=5.0, t_start=None, t_stop=None, trim=False,
                       center_kernel=True):
    """
    Estimates instantaneous firing rate by kernel convolution.

    Visualization of this function is covered in Viziphant:
    :func:`viziphant.statistics.plot_instantaneous_rates_colormesh`.


    Parameters
    ----------
    spiketrains : neo.SpikeTrain or list of neo.SpikeTrain
        Neo object(s) that contains spike times, the unit of the time stamps,
        and `t_start` and `t_stop` of the spike train.
    sampling_period : pq.Quantity
        Time stamp resolution of the spike times. The same resolution will
        be assumed for the kernel.
    kernel : 'auto' or Kernel, optional
        The string 'auto' or callable object of class `kernels.Kernel`.
        The kernel is used for convolution with the spike train and its
        standard deviation determines the time resolution of the instantaneous
        rate estimation. Currently implemented kernel forms are rectangular,
        triangular, epanechnikovlike, gaussian, laplacian, exponential, and
        alpha function.
        If 'auto', the optimized kernel width for the rate estimation is
        calculated according to :cite:`statistics-Shimazaki2010_171` and with
        this width a gaussian kernel is constructed. Automatized calculation
        of the kernel width is not available for other than gaussian kernel
        shapes.
        Default: 'auto'
    cutoff : float, optional
        This factor determines the cutoff of the probability distribution of
        the kernel, i.e., the considered width of the kernel in terms of
        multiples of the standard deviation sigma.
        Default: 5.0
    t_start : pq.Quantity, optional
        Start time of the interval used to compute the firing rate.
        If None, `t_start` is assumed equal to `t_start` attribute of
        `spiketrain`.
        Default: None
    t_stop : pq.Quantity, optional
        End time of the interval used to compute the firing rate (included).
        If None, `t_stop` is assumed equal to `t_stop` attribute of
        `spiketrain`.
        Default: None
    trim : bool, optional
        Accounts for the asymmetry of a kernel.
        If False, the output of the Fast Fourier Transformation being a longer
        vector than the input vector by the size of the kernel is reduced back
        to the original size of the considered time interval of the
        `spiketrain` using the median of the kernel. False (no trimming) is
        equivalent to 'same' convolution mode for symmetrical kernels.
        If True, only the region of the convolved signal is returned, where
        there is complete overlap between kernel and spike train. This is
        achieved by reducing the length of the output of the Fast Fourier
        Transformation by a total of two times the size of the kernel, and
        `t_start` and `t_stop` are adjusted. True (trimming) is equivalent to
        'valid' convolution mode for symmetrical kernels.
        Default: False
    center_kernel : bool, optional
        If set to True, the kernel will be translated such that its median is
        centered on the spike, thus putting equal weight before and after the
        spike. If False, no adjustment is performed such that the spike sits at
        the origin of the kernel.
        Default: True

    Returns
    -------
    rate : neo.AnalogSignal
        2D matrix that contains the rate estimation in unit hertz (Hz) of shape
        ``(time, len(spiketrains))`` or ``(time, 1)`` in case of a single
        input spiketrain. `rate.times` contains the time axis of the rate
        estimate: the unit of this property is the same as the resolution that
        is given via the argument `sampling_period` to the function.

    Raises
    ------
    TypeError
        If `spiketrain` is not an instance of `neo.SpikeTrain`.

        If `sampling_period` is not a `pq.Quantity`.

        If `sampling_period` is not larger than zero.

        If `kernel` is neither instance of `kernels.Kernel` nor string 'auto'.

        If `cutoff` is neither `float` nor `int`.

        If `t_start` and `t_stop` are neither None nor a `pq.Quantity`.

        If `trim` is not `bool`.
    ValueError
        If `sampling_period` is smaller than zero.

        If `kernel` is 'auto' and the function was unable to calculate optimal
        kernel width for instantaneous rate from input data.

    Warns
    -----
    UserWarning
        If `cutoff` is less than `min_cutoff` attribute of `kernel`, the width
        of the kernel is adjusted to a minimally allowed width.

    Notes
    -----
    The resulting instantaneous firing rate values smaller than ``0``, which
    can happen due to machine precision errors, are clipped to zero.

    Examples
    --------
    Example 1. Automatic kernel estimation.

    >>> import neo
    >>> import quantities as pq
    >>> from elephant import statistics
    >>> spiketrain = neo.SpikeTrain([0.3, 4.5, 6.7, 9.3], t_stop=10, units='s')
    >>> rate = statistics.instantaneous_rate(spiketrain,
    ...                                      sampling_period=10 * pq.ms,
    ...                                      kernel='auto')
    >>> rate
    AnalogSignal with 1 channels of length 1000; units Hz; datatype float64
    annotations: {'t_stop': array(10.) * s,
      'kernel': {'type': 'GaussianKernel',
       'sigma': '7.273225922958104 s',
       'invert': False}}
    sampling rate: 0.1 1/ms
    time: 0.0 s to 10.0 s

    Example 2. Manually set kernel.

    >>> from elephant import kernels
    >>> spiketrain = neo.SpikeTrain([0], t_stop=1, units='s')
    >>> kernel = kernels.GaussianKernel(sigma=300 * pq.ms)
    >>> rate = statistics.instantaneous_rate(spiketrain,
    ...        sampling_period=200 * pq.ms, kernel=kernel, t_start=-1 * pq.s)
    >>> rate
    AnalogSignal with 1 channels of length 10; units Hz; datatype float64
    annotations: {'t_stop': array(1.) * s,
      'kernel': {'type': 'GaussianKernel',
       'sigma': '300.0 ms',
       'invert': False}}
    sampling rate: 0.005 1/ms
    time: -1.0 s to 1.0 s
    >>> rate.magnitude
    array([[0.01007419],
       [0.05842767],
       [0.22928759],
       [0.60883028],
       [1.0938699 ],
       [1.3298076 ],
       [1.0938699 ],
       [0.60883028],
       [0.22928759],
       [0.05842767]])

    """
    def optimal_kernel(st):
        width_sigma = None
        if len(st) > 0:
            width_sigma = optimal_kernel_bandwidth(
                st.magnitude, times=None, bootstrap=False)['optw']
        if width_sigma is None:
            raise ValueError("Unable to calculate optimal kernel width for "
                             "instantaneous rate from input data.")
        return kernels.GaussianKernel(width_sigma * st.units)

    if isinstance(spiketrains, neo.SpikeTrain):
        if kernel == 'auto':
            kernel = optimal_kernel(spiketrains)
        spiketrains = [spiketrains]
    elif not isinstance(spiketrains, (list, tuple)):
        raise TypeError(
            "'spiketrains' must be a list of neo.SpikeTrain's or a single "
            "neo.SpikeTrain. Found: '{}'".format(type(spiketrains)))

    if not is_time_quantity(sampling_period):
        raise TypeError(
            "The 'sampling_period' must be a time Quantity. \n"
            "Found: {}".format(type(sampling_period)))

    if sampling_period.magnitude < 0:
        raise ValueError("The 'sampling_period' ({}) must be non-negative.".
                         format(sampling_period))

    if not (isinstance(kernel, kernels.Kernel) or kernel == 'auto'):
        raise TypeError(
            "'kernel' must be either instance of class elephant.kernels.Kernel"
            " or the string 'auto'. Found: %s, value %s" % (type(kernel),
                                                            str(kernel)))

    if not isinstance(cutoff, (float, int)):
        raise TypeError("'cutoff' must be float or integer")

    if not is_time_quantity(t_start, allow_none=True):
        raise TypeError("'t_start' must be a time Quantity")

    if not is_time_quantity(t_stop, allow_none=True):
        raise TypeError("'t_stop' must be a time Quantity")

    if not isinstance(trim, bool):
        raise TypeError("'trim' must be bool")

    check_neo_consistency(spiketrains,
                          object_type=neo.SpikeTrain,
                          t_start=t_start, t_stop=t_stop)
    if kernel == 'auto':
        if len(spiketrains) == 1:
            kernel = optimal_kernel(spiketrains[0])
        else:
            raise ValueError("Cannot estimate a kernel for a list of spike "
                             "trains. Please provide a kernel explicitly "
                             "rather than 'auto'.")

    if t_start is None:
        t_start = spiketrains[0].t_start
    if t_stop is None:
        t_stop = spiketrains[0].t_stop

    units = pq.CompoundUnit(
        "{}*s".format(sampling_period.rescale('s').item()))
    t_start = t_start.rescale(spiketrains[0].units)
    t_stop = t_stop.rescale(spiketrains[0].units)

    n_bins = int(((t_stop - t_start) / sampling_period).simplified) + 1
    time_vectors = np.zeros((len(spiketrains), n_bins), dtype=np.float64)
    hist_range_end = t_stop + sampling_period.rescale(spiketrains[0].units)
    hist_range = (t_start.item(), hist_range_end.item())
    for i, st in enumerate(spiketrains):
        time_vectors[i], _ = np.histogram(st.magnitude, bins=n_bins,
                                          range=hist_range)

    if cutoff < kernel.min_cutoff:
        cutoff = kernel.min_cutoff
        warnings.warn("The width of the kernel was adjusted to a minimally "
                      "allowed width.")

    # An odd number of points correctly resolves the median index and the
    # fact that the peak of an instantaneous rate should be centered at t=0
    # for symmetric kernels applied on a single spike at t=0.
    # See issue https://github.com/NeuralEnsemble/elephant/issues/360
    n_half = math.ceil(cutoff * (
            kernel.sigma / sampling_period).simplified.item())
    cutoff_sigma = cutoff * kernel.sigma.rescale(units).magnitude
    if center_kernel:
        # t_arr must be centered at the kernel median.
        # Not centering on the kernel median leads to underestimating the
        # instantaneous rate in cases when sampling_period >> kernel.sigma.
        median = kernel.icdf(0.5).rescale(units).item()
    else:
        median = 0
    t_arr = np.linspace(-cutoff_sigma + median, stop=cutoff_sigma + median,
                        num=2 * n_half + 1, endpoint=True) * units

    if center_kernel:
        # keep the full convolve range and do the trimming afterwards;
        # trimming is performed according to the kernel median index
        fft_mode = 'full'
    elif trim:
        # no median index trimming is involved
        fft_mode = 'valid'
    else:
        # no median index trimming is involved
        fft_mode = 'same'

    time_vectors = time_vectors.T  # make it (time, units)
    kernel_arr = np.expand_dims(kernel(t_arr).rescale(pq.Hz).magnitude, axis=1)
    rate = scipy.signal.fftconvolve(time_vectors,
                                    kernel_arr,
                                    mode=fft_mode)
    # the convolution of non-negative vectors is non-negative
    rate = np.clip(rate, a_min=0, a_max=None, out=rate)

    if center_kernel:  # account for the kernel asymmetry
        median_id = kernel.median_index(t_arr)
        # the size of kernel() output matches the input size, len(t_arr)
        kernel_array_size = len(t_arr)
        if not trim:
            rate = rate[median_id: -kernel_array_size + median_id]
        else:
            rate = rate[2 * median_id: -2 * (kernel_array_size - median_id)]
            t_start = t_start + median_id * units
            t_stop = t_stop - (kernel_array_size - median_id) * units
    else:
        # FIXME: don't shrink the output array
        # (to be consistent with center_kernel=True)
        # n points have n-1 intervals;
        # instantaneous rate is a list of intervals;
        # hence, the last element is excluded
        rate = rate[:-1]

    kernel_annotation = dict(type=type(kernel).__name__,
                             sigma=str(kernel.sigma),
                             invert=kernel.invert)

    rate = neo.AnalogSignal(signal=rate,
                            sampling_period=sampling_period,
                            units=pq.Hz, t_start=t_start, t_stop=t_stop,
                            kernel=kernel_annotation)

    return rate


@deprecated_alias(binsize='bin_size')
def time_histogram(spiketrains, bin_size, t_start=None, t_stop=None,
                   output='counts', binary=False):
    """
    Time Histogram of a list of `neo.SpikeTrain` objects.

    Visualization of this function is covered in Viziphant:
    :func:`viziphant.statistics.plot_time_histogram`.

    Parameters
    ----------
    spiketrains : list of neo.SpikeTrain
        `neo.SpikeTrain`s with a common time axis (same `t_start` and `t_stop`)
    bin_size : pq.Quantity
        Width of the histogram's time bins.
    t_start : pq.Quantity, optional
        Start time of the histogram. Only events in `spiketrains` falling
        between `t_start` and `t_stop` (both included) are considered in the
        histogram.
        If None, the maximum `t_start` of all `neo.SpikeTrain`s is used as
        `t_start`.
        Default: None
    t_stop : pq.Quantity, optional
        Stop time of the histogram. Only events in `spiketrains` falling
        between `t_start` and `t_stop` (both included) are considered in the
        histogram.
        If None, the minimum `t_stop` of all `neo.SpikeTrain`s is used as
        `t_stop`.
        Default: None
    output : {'counts', 'mean', 'rate'}, optional
        Normalization of the histogram. Can be one of:
        * 'counts': spike counts at each bin (as integer numbers)
        * 'mean': mean spike counts per spike train
        * 'rate': mean spike rate per spike train. Like 'mean', but the
          counts are additionally normalized by the bin width.
        Default: 'counts'
    binary : bool, optional
        If True, indicates whether all `neo.SpikeTrain` objects should first
        be binned to a binary representation (using the
        `conversion.BinnedSpikeTrain` class) and the calculation of the
        histogram is based on this representation.
        Note that the output is not binary, but a histogram of the converted,
        binary representation.
        Default: False

    Returns
    -------
    neo.AnalogSignal
        A `neo.AnalogSignal` object containing the histogram values.
        `neo.AnalogSignal[j]` is the histogram computed between
        `t_start + j * bin_size` and `t_start + (j + 1) * bin_size`.

    Raises
    ------
    ValueError
        If `output` is not 'counts', 'mean' or 'rate'.

    Warns
    -----
    UserWarning
        If `t_start` is None and the objects in `spiketrains` have different
        `t_start` values.
        If `t_stop` is None and the objects in `spiketrains` have different
        `t_stop` values.

    See also
    --------
    elephant.conversion.BinnedSpikeTrain

    Examples
    --------
    >>> import neo
    >>> import quantities as pq
    >>> from elephant import statistics
    >>> spiketrains = [
    ...     neo.SpikeTrain([0.3, 4.5, 6.7, 9.3], t_stop=10, units='s'),
    ...     neo.SpikeTrain([0.7, 4.3, 8.2], t_stop=10, units='s')
    ... ]
    >>> hist = statistics.time_histogram(spiketrains, bin_size=1 * pq.s)
    >>> hist
    AnalogSignal with 1 channels of length 10; units dimensionless; datatype
    int64
    annotations: {'normalization': 'counts'}
    sampling rate: 1.0 1/s
    time: 0.0 s to 10.0 s
    >>> hist.magnitude.flatten()
    array([2, 0, 0, 0, 2, 0, 1, 0, 1, 1])

    """
    # Bin the spike trains and sum across columns
    bs = BinnedSpikeTrain(spiketrains, t_start=t_start, t_stop=t_stop,
                          bin_size=bin_size)

    if binary:
        bs = bs.binarize(copy=False)
    bin_hist = bs.get_num_of_spikes(axis=0)
    # Flatten array
    bin_hist = np.ravel(bin_hist)
    # Renormalise the histogram
    if output == 'counts':
        # Raw
        bin_hist = pq.Quantity(bin_hist, units=pq.dimensionless, copy=False)
    elif output == 'mean':
        # Divide by number of input spike trains
        bin_hist = pq.Quantity(bin_hist / len(spiketrains),
                               units=pq.dimensionless, copy=False)
    elif output == 'rate':
        # Divide by number of input spike trains and bin width
        bin_hist = bin_hist / (len(spiketrains) * bin_size)
    else:
        raise ValueError(f'Parameter output ({output}) is not valid.')

    return neo.AnalogSignal(signal=np.expand_dims(bin_hist, axis=1),
                            sampling_period=bin_size, units=bin_hist.units,
                            t_start=bs.t_start, normalization=output,
                            copy=False)


@deprecated_alias(binsize='bin_size')
def complexity_pdf(spiketrains, bin_size):
    """
    Complexity Distribution of a list of `neo.SpikeTrain` objects
    :cite:`statistics-Gruen2007_96`.

    Deprecated in favor of :meth:`Complexity.pdf`.

    Probability density computed from the complexity histogram which is the
    histogram of the entries of the population histogram of clipped (binary)
    spike trains computed with a bin width of `bin_size`.
    It provides for each complexity (== number of active neurons per bin) the
    number of occurrences. The normalization of that histogram to 1 is the
    probability density.

    Parameters
    ----------
    spiketrains : list of neo.SpikeTrain
        Spike trains with a common time axis (same `t_start` and `t_stop`)
    bin_size : pq.Quantity
        Width of the histogram's time bins.

    Returns
    -------
    complexity_distribution : neo.AnalogSignal
        A `neo.AnalogSignal` object containing the histogram values.
        `neo.AnalogSignal[j]` is the histogram computed between
        `t_start + j * bin_size` and `t_start + (j + 1) * bin_size`.

    See also
    --------
    elephant.conversion.BinnedSpikeTrain
    """
    warnings.warn("'complexity_pdf' is deprecated in favor of the Complexity "
                  "class which has a 'pdf' method", DeprecationWarning)

    complexity = Complexity(spiketrains, bin_size=bin_size)

    return complexity.pdf()


class Complexity(object):
    """
    Class for complexity distribution (i.e. number of synchronous spikes found)
    :cite:`statistics-Gruen2007_96` of a list of `neo.SpikeTrain` objects.

    Complexity is calculated by counting the number of spikes (i.e. non-empty
    bins) that occur separated by `spread - 1` or less empty bins, within and
    across spike trains in the `spiketrains` list.

    Implementation (without spread) is based on the cited above paper.

    Parameters
    ----------
    spiketrains : list of neo.SpikeTrain
        Spike trains with a common time axis (same `t_start` and `t_stop`)
    sampling_rate : pq.Quantity or None, optional
        Sampling rate of the spike trains with units of 1/time.
        Used to shift the epoch edges in order to avoid rounding errors.
        If None using the epoch to slice spike trains may introduce
        rounding errors.
        Default: None
    bin_size : pq.Quantity or None, optional
        Width of the histogram's time bins with units of time.
        The user must specify the `bin_size` or the `sampling_rate`.
          * If None and the `sampling_rate` is available
          1/`sampling_rate` is used.
          * If both are given then `bin_size` is used.
        Default: None
    binary : bool, optional
          * If True then the time histograms will be binary.
          * If False the total number of synchronous spikes is counted in the
            time histogram.
        Default: True
    spread : int, optional
        Number of bins in which to check for synchronous spikes.
        Spikes that occur separated by `spread - 1` or less empty bins are
        considered synchronous.
          * ``spread = 0`` corresponds to a bincount accross spike trains.
          * ``spread = 1`` corresponds to counting consecutive spikes.
          * ``spread = 2`` corresponds to counting consecutive spikes and
            spikes separated by exactly 1 empty bin.
          * ``spread = n`` corresponds to counting spikes separated by exactly
            or less than `n - 1` empty bins.
        Default: 0
    tolerance : float or None, optional
        Tolerance for rounding errors in the binning process and in the input
        data.
        If None possible binning errors are not accounted for.
        Default: 1e-8

    Attributes
    ----------
    epoch : neo.Epoch
        An epoch object containing complexity values, left edges and durations
        of all intervals with at least one spike.
          * ``epoch.array_annotations['complexity']`` contains the
            complexity values per spike.
          * ``epoch.times`` contains the left edges.
          * ``epoch.durations`` contains the durations.
    time_histogram : neo.Analogsignal
        A `neo.AnalogSignal` object containing the histogram values.
        `neo.AnalogSignal[j]` is the histogram computed between
        `t_start + j * binsize` and `t_start + (j + 1) * binsize`.
          * If ``binary = True`` : Number of neurons that spiked in each bin,
            regardless of the number of spikes.
          * If ``binary = False`` : Number of neurons and spikes per neurons
            in each bin.
    complexity_histogram : np.ndarray
        The number of occurrences of events of different complexities.
        `complexity_hist[i]` corresponds to the number of events of
        complexity `i` for `i > 0`.

    Raises
    ------
    ValueError
        When `t_stop` is smaller than `t_start`.

        When both `sampling_rate` and `bin_size` are not specified.

        When `spread` is not a positive integer.

        When `spiketrains` is an empty list.

        When `t_start` is not the same for all spiketrains

        When `t_stop` is not the same for all spiketrains

    TypeError
        When `spiketrains` is not a list.

        When the elements in `spiketrains` are not instances of neo.SpikeTrain

    Warns
    -----
    UserWarning
        If no sampling rate is supplied which may lead to rounding errors
        when using the epoch to slice spike trains.

    Notes
    -----
    * Note that with most common parameter combinations spike times can end up
      on bin edges. This makes the binning susceptible to rounding errors which
      is accounted for by moving spikes which are within tolerance of the next
      bin edge into the following bin. This can be adjusted using the tolerance
      parameter and turned off by setting `tolerance=None`.

    See also
    --------
    elephant.conversion.BinnedSpikeTrain
    elephant.spike_train_synchrony.Synchrotool

    Examples
    --------
    >>> import neo
    >>> import quantities as pq
    >>> from elephant.statistics import Complexity

    >>> sampling_rate = 1/pq.ms
    >>> st1 = neo.SpikeTrain([1, 4, 6] * pq.ms, t_stop=10.0 * pq.ms)
    >>> st2 = neo.SpikeTrain([1, 5, 8] * pq.ms, t_stop=10.0 * pq.ms)
    >>> sts = [st1, st2]

    >>> # spread = 0, a simple bincount
    >>> cpx = Complexity(sts, sampling_rate=sampling_rate)
    Complexity calculated at sampling rate precision
    >>> print(cpx.complexity_histogram)
    [5 4 1]
    >>> print(cpx.time_histogram.flatten())
    [0 2 0 0 1 1 1 0 1 0] dimensionless
    >>> print(cpx.time_histogram.times)
    [0. 1. 2. 3. 4. 5. 6. 7. 8. 9.] ms

    >>> # spread = 1, consecutive spikes
    >>> cpx = Complexity(sts, sampling_rate=sampling_rate, spread=1)
    Complexity calculated at sampling rate precision
    >>> print(cpx.complexity_histogram)
    [5 4 1]
    >>> print(cpx.time_histogram.flatten())
    [0 2 0 0 3 3 3 0 1 0] dimensionless

    >>> # spread = 2, consecutive spikes and separated by 1 empty bin
    >>> cpx = Complexity(sts, sampling_rate=sampling_rate, spread=2)
    Complexity calculated at sampling rate precision
    >>> print(cpx.complexity_histogram)
    [4 0 1 0 1]
    >>> print(cpx.time_histogram.flatten())
    [0 2 0 0 4 4 4 4 4 0] dimensionless
    >>> pdf = cpx.pdf()
    >>> pdf
    AnalogSignal with 1 channels of length 3; units dimensionless;
    datatype float64
    sampling rate: 1.0 dimensionless
    time: 0.0 dimensionless to 3.0 dimensionless
    >>> pdf.magnitude
    array([[0.5],
           [0.4],
           [0.1]])
    """

    def __init__(self, spiketrains,
                 sampling_rate=None,
                 bin_size=None,
                 binary=True,
                 spread=0,
                 tolerance=1e-8):

        check_neo_consistency(spiketrains, object_type=neo.SpikeTrain)

        if bin_size is None and sampling_rate is None:
            raise ValueError('No bin_size or sampling_rate was specified!')

        if spread < 0:
            raise ValueError('Spread must be >=0')

        self.input_spiketrains = spiketrains
        self.t_start = spiketrains[0].t_start
        self.t_stop = spiketrains[0].t_stop
        self.sampling_rate = sampling_rate
        self.bin_size = bin_size
        self.binary = binary
        self.spread = spread
        self.tolerance = tolerance

        if bin_size is None and sampling_rate is not None:
            self.bin_size = 1 / self.sampling_rate

        if spread == 0:
            self.time_histogram, self.complexity_histogram = \
                self._histogram_no_spread()
            self.epoch = self._epoch_no_spread()
        else:
            self.epoch = self._epoch_with_spread()
            self.time_histogram, self.complexity_histogram = \
                self._histogram_with_spread()

    def pdf(self):
        """
        Probability density computed from the complexity histogram.

        Returns
        -------
        pdf : neo.AnalogSignal
            A `neo.AnalogSignal` object containing the pdf values.
            `neo.AnalogSignal[j]` is the histogram computed between
            `t_start + j * binsize` and `t_start + (j + 1) * binsize`.
        """
        norm_hist = self.complexity_histogram / self.complexity_histogram.sum()
        # Convert the Complexity pdf to an neo.AnalogSignal
        pdf = neo.AnalogSignal(
            np.expand_dims(norm_hist, axis=1),
            units=pq.dimensionless,
            t_start=0 * pq.dimensionless,
            sampling_period=1 * pq.dimensionless)
        return pdf

    def _histogram_no_spread(self):
        """
        Calculate the complexity histogram and time histogram for `spread` = 0
        """
        # Computing the population histogram with parameter binary=True to
        # clip the spike trains before summing
        time_hist = time_histogram(self.input_spiketrains,
                                   self.bin_size,
                                   binary=self.binary)

        # Computing the histogram of the entries of pophist
        complexity_hist = np.histogram(
            time_hist.magnitude,
            bins=range(0, len(self.input_spiketrains) + 2))[0]

        return time_hist, complexity_hist

    def _histogram_with_spread(self):
        """
        Calculate the complexity histogram and time histogram for `spread` > 0
        """
        complexity_hist = np.bincount(
            self.epoch.array_annotations['complexity'])
        num_bins = (self.t_stop - self.t_start).rescale(
            self.bin_size.units).item() / self.bin_size.item()
        num_bins = round_binning_errors(num_bins, tolerance=self.tolerance)
        time_hist = np.zeros(num_bins, dtype=int)

        start_bins = (self.epoch.times - self.t_start).rescale(
            self.bin_size.units).magnitude / self.bin_size.item()
        stop_bins = (self.epoch.times + self.epoch.durations - self.t_start
                     ).rescale(self.bin_size.units
                               ).magnitude / self.bin_size.item()

        if self.sampling_rate is not None:
            shift = (.5 / self.sampling_rate / self.bin_size).simplified.item()
            # account for the first bin not being shifted in the epoch creation
            # if the shift would move it past t_start
            if self.epoch.times[0] == self.t_start:
                start_bins[1:] += shift
            else:
                start_bins += shift
            stop_bins += shift

        start_bins = round_binning_errors(start_bins, tolerance=self.tolerance)
        stop_bins = round_binning_errors(stop_bins, tolerance=self.tolerance)

        for idx, (start, stop) in enumerate(zip(start_bins, stop_bins)):
            time_hist[start:stop] = \
                    self.epoch.array_annotations['complexity'][idx]

        time_hist = neo.AnalogSignal(
            signal=np.expand_dims(time_hist, axis=1),
            sampling_period=self.bin_size, units=pq.dimensionless,
            t_start=self.t_start)

        empty_bins = (self.t_stop - self.t_start - self.epoch.durations.sum())
        empty_bins = empty_bins.rescale(self.bin_size.units
                                        ).magnitude / self.bin_size.item()
        empty_bins = round_binning_errors(empty_bins, tolerance=self.tolerance)
        complexity_hist[0] = empty_bins

        return time_hist, complexity_hist

    def _epoch_no_spread(self):
        """
        Get an epoch object of the complexity distribution with `spread` = 0
        """
        left_edges = self.time_histogram.times
        durations = self.bin_size * np.ones(self.time_histogram.shape)

        if self.sampling_rate:
            # ensure that spikes are not on the bin edges
            bin_shift = .5 / self.sampling_rate
            left_edges -= bin_shift

            # Ensure that an epoch does not start before the minimum t_start.
            # Note: all spike trains share the same t_start and t_stop.
            if left_edges[0] < self.t_start:
                left_edges[0] = self.t_start
                durations[0] -= bin_shift
        else:
            warnings.warn('No sampling rate specified. '
                          'Note that using the complexity epoch to get '
                          'precise spike times can lead to rounding errors.')

        epoch = neo.Epoch(left_edges,
                          durations=durations,
                          array_annotations={
                              'complexity':
                              self.time_histogram.magnitude.flatten()})
        return epoch

    def _epoch_with_spread(self):
        """
        Get an epoch object of the complexity distribution with `spread` > 0
        """
        bst = conv.BinnedSpikeTrain(self.input_spiketrains,
                                    binsize=self.bin_size,
                                    tolerance=self.tolerance)

        if self.binary:
            bst = bst.binarize(copy=False)
        bincount = bst.get_num_of_spikes(axis=0)

        nonzero_indices = np.nonzero(bincount)[0]
        left_diff = np.diff(nonzero_indices,
                            prepend=-self.spread - 1)
        right_diff = np.diff(nonzero_indices,
                             append=len(bincount) + self.spread + 1)

        # standalone bins (no merging required)
        single_bin_indices = np.logical_and(left_diff > self.spread,
                                            right_diff > self.spread)
        single_bins = nonzero_indices[single_bin_indices]

        # bins separated by fewer than spread bins form clusters
        # that have to be merged
        cluster_start_indices = np.logical_and(left_diff > self.spread,
                                               right_diff <= self.spread)
        cluster_starts = nonzero_indices[cluster_start_indices]
        cluster_stop_indices = np.logical_and(left_diff <= self.spread,
                                              right_diff > self.spread)
        cluster_stops = nonzero_indices[cluster_stop_indices] + 1

        single_bin_complexities = bincount[single_bins]
        cluster_complexities = [bincount[start:stop].sum()
                                for start, stop in zip(cluster_starts,
                                                       cluster_stops)]

        # merge standalone bins and clusters and sort them
        combined_starts = np.concatenate((single_bins, cluster_starts))
        combined_stops = np.concatenate((single_bins + 1, cluster_stops))
        combined_complexities = np.concatenate((single_bin_complexities,
                                                cluster_complexities))
        sorting = np.argsort(combined_starts, kind='mergesort')
        left_edges = bst.bin_edges[combined_starts[sorting]]
        right_edges = bst.bin_edges[combined_stops[sorting]]
        complexities = combined_complexities[sorting].astype(int)

        if self.sampling_rate:
            # ensure that spikes are not on the bin edges
            bin_shift = .5 / self.sampling_rate
            left_edges -= bin_shift
            right_edges -= bin_shift
        else:
            warnings.warn('No sampling rate specified. '
                          'Note that using the complexity epoch to get '
                          'precise spike times can lead to rounding errors.')

        # Ensure that an epoch does not start before the minimum t_start.
        # Note: all spike trains share the same t_start and t_stop.
        left_edges[0] = max(self.t_start, left_edges[0])

        complexity_epoch = neo.Epoch(times=left_edges,
                                     durations=right_edges - left_edges,
                                     array_annotations={'complexity':
                                                        complexities})

        return complexity_epoch


def nextpow2(x):
    """
    Return the smallest integral power of 2 that is equal or larger than `x`.
    """
    log2_n = math.ceil(math.log2(x))
    n = 2 ** log2_n
    return n


def fftkernel(x, w):
    """
    Applies the Gauss kernel smoother to an input signal using FFT algorithm.

    Parameters
    ----------
    x : np.ndarray
        Vector with sample signal.
    w : float
        Kernel bandwidth (the standard deviation) in unit of the sampling
        resolution of `x`.

    Returns
    -------
    y : np.ndarray
        The smoothed signal.

    Notes
    -----
    1. MAY 5/23, 2012 Author Hideaki Shimazaki
       RIKEN Brain Science Insitute
       http://2000.jukuin.keio.ac.jp/shimazaki
    2. Ported to Python: Subhasis Ray, NCBS. Tue Jun 10 10:42:38 IST 2014

    """
    L = len(x)
    Lmax = L + 3 * w
    n = nextpow2(Lmax)
    X = np.fft.fft(x, n)
    f = np.arange(0, n, 1.0) / n
    f = np.concatenate((-f[:int(n / 2)], f[int(n / 2):0:-1]))
    K = np.exp(-0.5 * (w * 2 * np.pi * f) ** 2)
    y = np.fft.ifft(X * K, n)
    y = y[:L].copy()
    return y


def logexp(x):
    if x < 1e2:
        y = np.log(1 + np.exp(x))
    else:
        y = x
    return y


def ilogexp(x):
    if x < 1e2:
        y = np.log(np.exp(x) - 1)
    else:
        y = x
    return y


def cost_function(x, N, w, dt):
    """
    Computes the cost function for `sskernel`.

    Cn(w) = sum_{i,j} int k(x - x_i) k(x - x_j) dx - 2 sum_{i~=j} k(x_i - x_j)

    """
    yh = np.abs(fftkernel(x, w / dt))  # density
    # formula for density
    C = np.sum(yh ** 2) * dt - 2 * np.sum(yh * x) * \
        dt + 2 / np.sqrt(2 * np.pi) / w / N
    C = C * N * N
    # formula for rate
    # C = dt*sum( yh.^2 - 2*yh.*y_hist + 2/sqrt(2*pi)/w*y_hist )
    return C, yh


@deprecated_alias(tin='times', w='bandwidth')
def optimal_kernel_bandwidth(spiketimes, times=None, bandwidth=None,
                             bootstrap=False):
    """
    Calculates optimal fixed kernel bandwidth
    :cite:`statistics-Shimazaki2010_171`, given as the standard deviation
    sigma.

    Original matlab code (sskernel.m)
    http://2000.jukuin.keio.ac.jp/shimazaki/res/kernel.html has been ported to
    Python by Subhasis Ray, NCBS.

    Parameters
    ----------
    spiketimes : np.ndarray
        Sequence of spike times (sorted to be ascending).
    times : np.ndarray or None, optional
        Time points at which the kernel bandwidth is to be estimated.
        If None, `spiketimes` is used.
        Default: None
    bandwidth : np.ndarray or None, optional
        Vector of kernel bandwidths (standard deviation sigma).
        If specified, optimal bandwidth is selected from this.
        If None, `bandwidth` is obtained through a golden-section search on a
        log-exp scale.
        Default: None
    bootstrap : bool, optional
        If True, calculates the 95% confidence interval using Bootstrap.
        Default: False

    Returns
    -------
    dict
        'y' : np.ndarray
            Estimated density.
        't' : np.ndarray
            Points at which estimation was computed.
        'optw' : float
            Optimal kernel bandwidth given as standard deviation sigma
        'w' : np.ndarray
            Kernel bandwidths examined (standard deviation sigma).
        'C' : np.ndarray
            Cost functions of `bandwidth`.
        'confb95' : tuple of np.ndarray
            Bootstrap 95% confidence interval: (lower level, upper level).
            If `bootstrap` is False, `confb95` is None.
        'yb' : np.ndarray
            Bootstrap samples.
            If `bootstrap` is False, `yb` is None.

        If no optimal kernel could be found, all entries of the dictionary are
        set to None.

    """

    if times is None:
        time = np.max(spiketimes) - np.min(spiketimes)
        isi = np.diff(spiketimes)
        isi = isi[isi > 0].copy()
        dt = np.min(isi)
        times = np.linspace(np.min(spiketimes),
                            np.max(spiketimes),
                            min(int(time / dt + 0.5),
                                1000))  # The 1000 seems somewhat arbitrary
        t = times
    else:
        time = np.max(times) - np.min(times)
        spiketimes = spiketimes[(spiketimes >= np.min(times)) &
                                (spiketimes <= np.max(times))].copy()
        isi = np.diff(spiketimes)
        isi = isi[isi > 0].copy()
        dt = np.min(isi)
        if dt > np.min(np.diff(times)):
            t = np.linspace(np.min(times), np.max(times),
                            min(int(time / dt + 0.5), 1000))
        else:
            t = times
    dt = np.min(np.diff(times))
    yhist, bins = np.histogram(spiketimes, np.r_[t - dt / 2, t[-1] + dt / 2])
    N = np.sum(yhist)
    yhist = yhist / (N * dt)  # density
    optw = None
    y = None
    if bandwidth is not None:
        C = np.zeros(len(bandwidth))
        Cmin = np.inf
        for k, w_ in enumerate(bandwidth):
            C[k], yh = cost_function(yhist, N, w_, dt)
            if C[k] < Cmin:
                Cmin = C[k]
                optw = w_
                y = yh
    else:
        # Golden section search on a log-exp scale
        wmin = 2 * dt
        wmax = max(spiketimes) - min(spiketimes)
        imax = 20  # max iterations
        bandwidth = np.zeros(imax)
        C = np.zeros(imax)
        tolerance = 1e-5
        phi = 0.5 * (np.sqrt(5) + 1)  # The Golden ratio
        a = ilogexp(wmin)
        b = ilogexp(wmax)
        c1 = (phi - 1) * a + (2 - phi) * b
        c2 = (2 - phi) * a + (phi - 1) * b
        f1, y1 = cost_function(yhist, N, logexp(c1), dt)
        f2, y2 = cost_function(yhist, N, logexp(c2), dt)
        k = 0
        while (np.abs(b - a) > (tolerance * (np.abs(c1) + np.abs(c2)))) \
                and (k < imax):
            if f1 < f2:
                b = c2
                c2 = c1
                c1 = (phi - 1) * a + (2 - phi) * b
                f2 = f1
                f1, y1 = cost_function(yhist, N, logexp(c1), dt)
                bandwidth[k] = logexp(c1)
                C[k] = f1
                optw = logexp(c1)
                y = y1 / (np.sum(y1 * dt))
            else:
                a = c1
                c1 = c2
                c2 = (2 - phi) * a + (phi - 1) * b
                f1 = f2
                f2, y2 = cost_function(yhist, N, logexp(c2), dt)
                bandwidth[k] = logexp(c2)
                C[k] = f2
                optw = logexp(c2)
                y = y2 / np.sum(y2 * dt)
            k = k + 1
    # Bootstrap confidence intervals
    confb95 = None
    yb = None
    # If bootstrap is requested, and an optimal kernel was found
    if bootstrap and optw:
        nbs = 1000
        yb = np.zeros((nbs, len(times)))
        for ii in range(nbs):
            idx = np.floor(np.random.rand(N) * N).astype(int)
            xb = spiketimes[idx]
            y_histb, bins = np.histogram(
                xb, np.r_[t - dt / 2, t[-1] + dt / 2]) / dt / N
            yb_buf = fftkernel(y_histb, optw / dt).real
            yb_buf = yb_buf / np.sum(yb_buf * dt)
            yb[ii, :] = np.interp(times, t, yb_buf)
        ybsort = np.sort(yb, axis=0)
        y95b = ybsort[np.floor(0.05 * nbs).astype(int), :]
        y95u = ybsort[np.floor(0.95 * nbs).astype(int), :]
        confb95 = (y95b, y95u)
    # Only perform interpolation if y could be calculated
    if y is not None:
        y = np.interp(times, t, y)
    return {'y': y,
            't': times,
            'optw': optw,
            'w': bandwidth,
            'C': C,
            'confb95': confb95,
            'yb': yb}


def sskernel(*args, **kwargs):
    warnings.warn("'sskernel' function is deprecated; "
                  "use 'optimal_kernel_bandwidth'", DeprecationWarning)
    return optimal_kernel_bandwidth(*args, **kwargs)
