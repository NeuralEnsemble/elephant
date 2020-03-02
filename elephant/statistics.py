# -*- coding: utf-8 -*-
"""
Statistical measures of spike trains (e.g., Fano factor) and functions to
estimate firing rates.

Tutorial
--------

:doc:`View tutorial <../tutorials/statistics>`

Run tutorial interactively:

.. image:: https://mybinder.org/badge.svg
   :target: https://mybinder.org/v2/gh/NeuralEnsemble/elephant/master
            ?filepath=doc/tutorials/statistics.ipynb


.. current_module elephant.statistics

Functions overview
------------------

Rate estimation
~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: statistics/

    mean_firing_rate
    instantaneous_rate
    time_histogram
    sskernel


Spike interval statistics
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: statistics/

    isi
    cv
    lv
    cv2


Statistics across spike trains
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: statistics/

    fanofactor
    complexity_pdf

:copyright: Copyright 2014-2020 by the Elephant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
"""

from __future__ import division, print_function
# do not import unicode_literals
# (quantities rescale does not work with unicodes)

import numpy as np
import quantities as pq
import scipy.stats
import scipy.signal
import neo
from neo.core import SpikeTrain
import elephant.conversion as conv
import elephant.kernels as kernels
import warnings

cv = scipy.stats.variation


def isi(spiketrain, axis=-1):
    """
    Return an array containing the inter-spike intervals of the spike train.

    Accepts a `neo.SpikeTrain`, a `pq.Quantity` array, or a plain
    `np.ndarray`. If either a `neo.SpikeTrain` or `pq.Quantity` is provided,
    the return value will be `pq.Quantity`, otherwise `np.ndarray`. The units
    of `pq.Quantity` will be the same as `spiketrain`.

    Parameters
    ----------
    spiketrain : neo.SpikeTrain or pq.Quantity or np.ndarray
        The spike times.
    axis : int, optional
        The axis along which the difference is taken.
        Default: the last axis.

    Returns
    -------
    intervals : np.ndarray or pq.Quantity
        The inter-spike intervals of the `spiketrain`.

    """
    if axis is None:
        axis = -1
    if isinstance(spiketrain, neo.SpikeTrain):
        intervals = np.diff(
            np.sort(spiketrain.times.view(pq.Quantity)), axis=axis)
    else:
        intervals = np.diff(np.sort(spiketrain), axis=axis)
    return intervals


def mean_firing_rate(spiketrain, t_start=None, t_stop=None, axis=None):
    """
    Return the firing rate of the spike train.

    Accepts a `neo.SpikeTrain`, a `pq.Quantity` array, or a plain
    `np.ndarray`. If either a `neo.SpikeTrain` or `pq.Quantity` array is
    provided, the return value will be a `pq.Quantity` array, otherwise a
    plain `np.ndarray`. The units of the `pq.Quantity` array will be the
    inverse of the `spiketrain`.

    The interval over which the firing rate is calculated can be optionally
    controlled with `t_start` and `t_stop`.

    Parameters
    ----------
    spiketrain : neo.SpikeTrain or pq.Quantity or np.ndarray
        The spike times.
    t_start : float or pq.Quantity, optional
        The start time to use for the interval.
        If None, retrieved from the `t_start` attribute of `spiketrain`. If
        that is not present, default to 0. Any value from `spiketrain` below
        this value is ignored.
        Default: None.
    t_stop : float or pq.Quantity, optional
        The stop time to use for the time points.
        If not specified, retrieved from the `t_stop` attribute of
        `spiketrain`. If that is not present, default to the maximum value of
        `spiketrain`. Any value from `spiketrain` above this value is ignored.
        Default: None.
    axis : int, optional
        The axis over which to do the calculation.
        If None, do the calculation over the flattened array.
        Default: None.

    Returns
    -------
    float or pq.Quantity or np.ndarray
        The firing rate of the `spiketrain`

    Raises
    ------
    TypeError
        If `spiketrain` is a `np.ndarray` and `t_start` or `t_stop` is
        `pq.Quantity`.

    Notes
    -----
    If `spiketrain` is a `pq.Quantity` or `neo.SpikeTrain`, and `t_start` or
    `t_stop` are not `pq.Quantity`, `t_start` and `t_stop` are assumed to have
    the same units as `spiketrain`.

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


def fanofactor(spiketrains):
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

    Returns
    -------
    fano : float
        The Fano factor of the spike counts of the input spike trains.
        Returns np.NaN if an empty list is specified, or if all spike trains
        are empty.

    """
    # Build array of spike counts (one per spike train)
    spike_counts = np.array([len(t) for t in spiketrains])

    # Compute FF
    if all([count == 0 for count in spike_counts]):
        fano = np.nan
    else:
        fano = spike_counts.var() / spike_counts.mean()
    return fano


def lv(v, with_nan=False):
    r"""
    Calculate the measure of local variation LV for a sequence of time
    intervals between events.

    Given a vector v containing a sequence of intervals, the LV is defined as:

    .. math::
        LV := \frac{1}{N} \sum_{i=1}^{N-1}
                          \frac{3(isi_i-isi_{i+1})^2}
                          {(isi_i+isi_{i+1})^2}

    The LV is typically computed as a substitute for the classical coefficient
    of variation for sequences of events which include some (relatively slow)
    rate fluctuation.  As with the CV, LV=1 for a sequence of intervals
    generated by a Poisson process.

    Parameters
    ----------
    v : pq.Quantity or np.ndarray or list
        Vector of consecutive time intervals.
    with_nan : bool, optional
        If True, `lv` of a spike train with less than two spikes results in a
        np.NaN value and a warning is raised.
        If False, a `ValueError` exception is raised with a spike train with
        less than two spikes.
        Default: True.

    Returns
    -------
    float
        The LV of the inter-spike interval of the input sequence.

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
        If `with_nan` is True and the `lv` is calculated for a spike train
        with less than two spikes, generating a np.NaN.

    References
    ----------
    .. [1] S. Shinomoto, K. Shima, & J. Tanji, "Differences in spiking
           patterns among cortical neurons," Neural Computation, vol. 15,
           pp. 2823–2842, 2003.

    """
    # convert to array, cast to float
    v = np.asarray(v)
    # ensure the input is a vector
    if len(v.shape) > 1:
        raise ValueError("Input shape is larger than 1. Please provide "
                         "a vector as an input.")

    # ensure we have enough entries
    if v.size < 2:
        if with_nan:
            warnings.warn("Input size is too small. Please provide "
                          "an input with more than 1 entry. lv returns 'NaN'"
                          "since the argument `with_nan` is True")
            return np.NaN

        else:
            raise ValueError("Input size is too small. Please provide "
                             "an input with more than 1 entry. lv returned any"
                             "value since the argument `with_nan` is False")

    # calculate LV and return result
    # raise error if input is multi-dimensional
    return 3. * np.mean(np.power(np.diff(v) / (v[:-1] + v[1:]), 2))


def cv2(v, with_nan=False):
    r"""
    Calculate the measure of CV2 for a sequence of time intervals between
    events.

    Given a vector v containing a sequence of intervals, the CV2 is defined
    as:

    .. math::
        CV2 := \frac{1}{N} \sum{i=1}^{N-1}
                           \frac{2|isi_{i+1}-isi_i|}
                          {|isi_{i+1}+isi_i|}

    The CV2 is typically computed as a substitute for the classical
    coefficient of variation (CV) for sequences of events which include some
    (relatively slow) rate fluctuation.  As with the CV, CV2=1 for a sequence
    of intervals generated by a Poisson process.

    Parameters
    ----------
    v : pq.Quantity or np.ndarray or list
        Vector of consecutive time intervals.
    with_nan : bool, optional
        If True, `cv2` of a spike train with less than two spikes results in a
        np.NaN value and a warning is raised.
        If False, `ValueError` exception is raised with a spike train with
        less than two spikes.
        Default: True.

    Returns
    -------
    float
        The CV2 of the inter-spike interval of the input sequence.

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

    References
    ----------
    .. [1] G. R. Holt, W. R. Softky, C. Koch, & R. J. Douglas, "Comparison of
           discharge variability in vitro and in vivo in cat visual cortex
           neurons," Journal of Neurophysiology, vol. 75, no. 5, pp. 1806-1814,
           1996.

    """
    # convert to array, cast to float
    v = np.asarray(v)

    # ensure the input ia a vector
    if len(v.shape) > 1:
        raise ValueError("Input shape is larger than 1. Please provide "
                         "a vector as an input.")

    # ensure we have enough entries
    if v.size < 2:
        if with_nan:
            warnings.warn("Input size is too small. Please provide"
                          "an input with more than 1 entry. cv2 returns `NaN`"
                          "since the argument `with_nan` is `True`")
            return np.NaN
        else:
            raise ValueError("Input size is too small. Please provide "
                             "an input with more than 1 entry. cv2 returns any"
                             "value since the argument `with_nan` is `False`")

    # calculate CV2 and return result
    return 2. * np.mean(np.absolute(np.diff(v)) / (v[:-1] + v[1:]))


def instantaneous_rate(spiketrain, sampling_period, kernel='auto',
                       cutoff=5.0, t_start=None, t_stop=None, trim=False):
    """
    Estimates instantaneous firing rate by kernel convolution.

    Parameters
    -----------
    spiketrain : neo.SpikeTrain or list of neo.SpikeTrain
        Neo object(s) that contains spike times, the unit of the time stamps,
        and `t_start` and `t_stop` of the spike train.
    sampling_period : pq.Quantity
        Time stamp resolution of the spike times. The same resolution will
        be assumed for the kernel.
    kernel : str or `kernels.Kernel`, optional
        The string 'auto' or callable object of class `kernels.Kernel`.
        The kernel is used for convolution with the spike train and its
        standard deviation determines the time resolution of the instantaneous
        rate estimation. Currently implemented kernel forms are rectangular,
        triangular, epanechnikovlike, gaussian, laplacian, exponential, and
        alpha function.
        If 'auto', the optimized kernel width for the rate estimation is
        calculated according to [1]_ and with this width a gaussian kernel is
        constructed. Automatized calculation of the kernel width is not
        available for other than gaussian kernel shapes.
        Default: 'auto'.
    cutoff : float, optional
        This factor determines the cutoff of the probability distribution of
        the kernel, i.e., the considered width of the kernel in terms of
        multiples of the standard deviation sigma.
        Default: 5.0.
    t_start : pq.Quantity, optional
        Start time of the interval used to compute the firing rate.
        If None, `t_start` is assumed equal to `t_start` attribute of
        `spiketrain`.
        Default: None.
    t_stop : pq.Quantity, optional
        End time of the interval used to compute the firing rate (included).
        If None, `t_stop` is assumed equal to `t_stop` attribute of
        `spiketrain`.
        Default: None.
    trim : bool, optional
        If False, the output of the Fast Fourier Transformation being a longer
        vector than the input vector by the size of the kernel is reduced back
        to the original size of the considered time interval of the
        `spiketrain` using the median of the kernel.
        If True, only the region of the convolved signal is returned, where
        there is complete overlap between kernel and spike train. This is
        achieved by reducing the length of the output of the Fast Fourier
        Transformation by a total of two times the size of the kernel, and
        `t_start` and `t_stop` are adjusted.
        Default: False.

    Returns
    -------
    rate : neo.AnalogSignal
        Contains the rate estimation in unit hertz (Hz). In case a list of
        spike trains was given, this is the combined rate of all spike trains
        (not the average rate). `rate.times` contains the time axis of the rate
        estimate. The unit of this property is the same as the resolution that
        is given via the argument `sampling_period` to the function.

    Raises
    ------
    TypeError:
        If `spiketrain` is not an instance of `neo.SpikeTrain`.

        If `sampling_period` is not a `pq.Quantity`.

        If `sampling_period` is not larger than zero.

        If `kernel` is neither instance of `kernels.Kernel` nor string 'auto'.

        If `cutoff` is neither `float` nor `int`.

        If `t_start` and `t_stop` are neither None nor a `pq.Quantity`.

        If `trim` is not `bool`.

    ValueError:
        If `sampling_period` is smaller than zero.

        If `kernel` is 'auto' and the function was unable to calculate optimal
        kernel width for instantaneous rate from input data.

    Warns
    -----
    UserWarning
        If `cutoff` is less than `min_cutoff` attribute of `kernel`, the width
        of the kernel is adjusted to a minimally allowed width.

        If the instantaneous firing rate approximation contains negative values
        with respect to a tolerance (less than -1e-8), possibly due to machine
        precision errors.

    References
    ----------
    .. [1] H. Shimazaki, & S. Shinomoto, "Kernel bandwidth optimization in
           spike rate estimation," J Comput Neurosci, vol. 29, pp. 171–182,
           2010.

    Examples
    --------
    >>> import quantities as pq
    >>> from elephant import kernels
    >>> kernel = kernels.AlphaKernel(sigma=0.05*pq.s, invert=True)
    >>> rate = instantaneous_rate(spiketrain, sampling_period=2*pq.ms,
    ...     kernel=kernel)

    """
    # Merge spike trains if list of spike trains given:
    if isinstance(spiketrain, list):
        _check_consistency_of_spiketrainlist(
            spiketrain, t_start=t_start, t_stop=t_stop)
        if t_start is None:
            t_start = spiketrain[0].t_start
        if t_stop is None:
            t_stop = spiketrain[0].t_stop
        spikes = np.concatenate([st.magnitude for st in spiketrain])
        merged_spiketrain = SpikeTrain(np.sort(spikes),
                                       units=spiketrain[0].units,
                                       t_start=t_start, t_stop=t_stop)
        return instantaneous_rate(merged_spiketrain,
                                  sampling_period=sampling_period,
                                  kernel=kernel, cutoff=cutoff,
                                  t_start=t_start,
                                  t_stop=t_stop, trim=trim)

    # Checks of input variables:
    if not isinstance(spiketrain, SpikeTrain):
        raise TypeError(
            "spiketrain must be instance of :class:`SpikeTrain` of Neo!\n"
            "    Found: %s, value %s" % (type(spiketrain), str(spiketrain)))

    if not (isinstance(sampling_period, pq.Quantity) and
            sampling_period.dimensionality.simplified ==
            pq.Quantity(1, "s").dimensionality):
        raise TypeError(
            "The sampling period must be a time quantity!\n"
            "    Found: %s, value %s" % (
                type(sampling_period), str(sampling_period)))

    if sampling_period.magnitude < 0:
        raise ValueError("The sampling period must be larger than zero.")

    if kernel == 'auto':
        kernel_width_sigma = sskernel(
            spiketrain.magnitude, tin=None, bootstrap=False)['optw']
        if kernel_width_sigma is None:
            raise ValueError(
                "Unable to calculate optimal kernel width for "
                "instantaneous rate from input data.")
        kernel = kernels.GaussianKernel(kernel_width_sigma * spiketrain.units)
    elif not isinstance(kernel, kernels.Kernel):
        raise TypeError(
            "kernel must be either instance of :class:`Kernel` "
            "or the string 'auto'!\n"
            "    Found: %s, value %s" % (type(kernel), str(kernel)))

    if not (isinstance(cutoff, float) or isinstance(cutoff, int)):
        raise TypeError("cutoff must be float or integer!")

    if not (t_start is None or (isinstance(t_start, pq.Quantity) and
                                t_start.dimensionality.simplified ==
                                pq.Quantity(1, "s").dimensionality)):
        raise TypeError("t_start must be a time quantity!")

    if not (t_stop is None or (isinstance(t_stop, pq.Quantity) and
                               t_stop.dimensionality.simplified ==
                               pq.Quantity(1, "s").dimensionality)):
        raise TypeError("t_stop must be a time quantity!")

    if not (isinstance(trim, bool)):
        raise TypeError("trim must be bool!")

    # main function:
    units = pq.CompoundUnit(
        "{}*s".format(sampling_period.rescale('s').item()))
    spiketrain = spiketrain.rescale(units)
    if t_start is None:
        t_start = spiketrain.t_start
    else:
        t_start = t_start.rescale(spiketrain.units)

    if t_stop is None:
        t_stop = spiketrain.t_stop
    else:
        t_stop = t_stop.rescale(spiketrain.units)

    time_vector = np.zeros(int((t_stop - t_start)) + 1)

    spikes_slice = spiketrain.time_slice(t_start, t_stop) \
        if len(spiketrain) else np.array([])

    for spike in spikes_slice:
        index = int((spike - t_start))
        time_vector[index] += 1

    if cutoff < kernel.min_cutoff:
        cutoff = kernel.min_cutoff
        warnings.warn("The width of the kernel was adjusted to a minimally "
                      "allowed width.")

    t_arr = np.arange(-cutoff * kernel.sigma.rescale(units).magnitude,
                      cutoff * kernel.sigma.rescale(units).magnitude +
                      sampling_period.rescale(units).magnitude,
                      sampling_period.rescale(units).magnitude) * units

    r = scipy.signal.fftconvolve(time_vector,
                                 kernel(t_arr).rescale(pq.Hz).magnitude,
                                 'full')
    if np.any(r < -1e-8):  # abs tolerance in np.isclose
        warnings.warn("Instantaneous firing rate approximation contains "
                      "negative values, possibly caused due to machine "
                      "precision errors.")

    if not trim:
        r = r[kernel.median_index(t_arr):-(kernel(t_arr).size -
                                           kernel.median_index(t_arr))]
    elif trim:
        r = r[2 * kernel.median_index(t_arr):-2 * (kernel(t_arr).size -
                                                   kernel.median_index(t_arr))]
        t_start += kernel.median_index(t_arr) * spiketrain.units
        t_stop -= (kernel(t_arr).size -
                   kernel.median_index(t_arr)) * spiketrain.units

    rate = neo.AnalogSignal(signal=r.reshape(r.size, 1),
                            sampling_period=sampling_period,
                            units=pq.Hz, t_start=t_start, t_stop=t_stop)

    return rate


def time_histogram(spiketrains, binsize, t_start=None, t_stop=None,
                   output='counts', binary=False):
    """
    Time Histogram of a list of `neo.SpikeTrain` objects.

    Parameters
    ----------
    spiketrains : list of neo.SpikeTrain
        `neo.SpikeTrain`s with a common time axis (same `t_start` and `t_stop`)
    binsize : pq.Quantity
        Width of the histogram's time bins.
    t_start : pq.Quantity, optional
        Start time of the histogram. Only events in `spiketrains` falling
        between `t_start` and `t_stop` (both included) are considered in the
        histogram.
        If None, the maximum `t_start` of all `neo.SpikeTrain`s is used as
        `t_start`.
        Default: None.
    t_stop : pq.Quantity, optional
        Stop time of the histogram. Only events in `spiketrains` falling
        between `t_start` and `t_stop` (both included) are considered in the
        histogram.
        If None, the minimum `t_stop` of all `neo.SpikeTrain`s is used as
        `t_stop`.
        Default: None.
    output : {'counts', 'mean', 'rate'}, optional
        Normalization of the histogram. Can be one of:
        * 'counts': spike counts at each bin (as integer numbers)
        * 'mean': mean spike counts per spike train
        * 'rate': mean spike rate per spike train. Like 'mean', but the
          counts are additionally normalized by the bin width.
        Default: 'counts'.
    binary : bool, optional
        If True, indicates whether all `neo.SpikeTrain` objects should first
        be binned to a binary representation (using the
        `conversion.BinnedSpikeTrain` class) and the calculation of the
        histogram is based on this representation.
        Note that the output is not binary, but a histogram of the converted,
        binary representation.
        Default: False.

    Returns
    -------
    neo.AnalogSignal
        A `neo.AnalogSignal` object containing the histogram values.
        `neo.AnalogSignal[j]` is the histogram computed between
        `t_start + j * binsize` and `t_start + (j + 1) * binsize`.

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

    """
    min_tstop = 0
    if t_start is None:
        # Find the internal range for t_start, where all spike trains are
        # defined; cut all spike trains taking that time range only
        max_tstart, min_tstop = conv._get_start_stop_from_input(spiketrains)
        t_start = max_tstart
        if not all([max_tstart == t.t_start for t in spiketrains]):
            warnings.warn(
                "Spiketrains have different t_start values -- "
                "using maximum t_start as t_start.")

    if t_stop is None:
        # Find the internal range for t_stop
        if min_tstop:
            t_stop = min_tstop
            if not all([min_tstop == t.t_stop for t in spiketrains]):
                warnings.warn(
                    "Spiketrains have different t_stop values -- "
                    "using minimum t_stop as t_stop.")
        else:
            min_tstop = conv._get_start_stop_from_input(spiketrains)[1]
            t_stop = min_tstop
            if not all([min_tstop == t.t_stop for t in spiketrains]):
                warnings.warn(
                    "Spiketrains have different t_stop values -- "
                    "using minimum t_stop as t_stop.")

    sts_cut = [st.time_slice(t_start=t_start, t_stop=t_stop) for st in
               spiketrains]

    # Bin the spike trains and sum across columns
    bs = conv.BinnedSpikeTrain(sts_cut, t_start=t_start, t_stop=t_stop,
                               binsize=binsize)

    if binary:
        bin_hist = bs.to_sparse_bool_array().sum(axis=0)
    else:
        bin_hist = bs.to_sparse_array().sum(axis=0)
    # Flatten array
    bin_hist = np.ravel(bin_hist)
    # Renormalise the histogram
    if output == 'counts':
        # Raw
        bin_hist = bin_hist * pq.dimensionless
    elif output == 'mean':
        # Divide by number of input spike trains
        bin_hist = bin_hist * 1. / len(spiketrains) * pq.dimensionless
    elif output == 'rate':
        # Divide by number of input spike trains and bin width
        bin_hist = bin_hist * 1. / len(spiketrains) / binsize
    else:
        raise ValueError('Parameter output is not valid.')

    return neo.AnalogSignal(signal=bin_hist.reshape(bin_hist.size, 1),
                            sampling_period=binsize, units=bin_hist.units,
                            t_start=t_start)


def complexity_pdf(spiketrains, binsize):
    """
    Complexity Distribution of a list of `neo.SpikeTrain` objects.

    Probability density computed from the complexity histogram which is the
    histogram of the entries of the population histogram of clipped (binary)
    spike trains computed with a bin width of `binsize`.
    It provides for each complexity (== number of active neurons per bin) the
    number of occurrences. The normalization of that histogram to 1 is the
    probability density.

    Implementation is based on [1]_.

    Parameters
    ----------
    spiketrains : list of neo.SpikeTrain
        Spike trains with a common time axis (same `t_start` and `t_stop`)
    binsize : pq.Quantity
        Width of the histogram's time bins.

    Returns
    -------
    complexity_distribution : neo.AnalogSignal
        A `neo.AnalogSignal` object containing the histogram values.
        `neo.AnalogSignal[j]` is the histogram computed between
        `t_start + j * binsize` and `t_start + (j + 1) * binsize`.

    See also
    --------
    elephant.conversion.BinnedSpikeTrain

    References
    ----------
    .. [1] S. Gruen, M. Abeles, & M. Diesmann, "Impact of higher-order
           correlations on coincidence distributions of massively parallel
           data," In "Dynamic Brain - from Neural Spikes to Behaviors",
           pp. 96-114, Springer Berlin Heidelberg, 2008.

    """
    # Computing the population histogram with parameter binary=True to clip the
    # spike trains before summing
    pophist = time_histogram(spiketrains, binsize, binary=True)

    # Computing the histogram of the entries of pophist (=Complexity histogram)
    complexity_hist = np.histogram(
        pophist.magnitude, bins=range(0, len(spiketrains) + 2))[0]

    # Normalization of the Complexity Histogram to 1 (probabilty distribution)
    complexity_hist = complexity_hist / complexity_hist.sum()
    # Convert the Complexity pdf to an neo.AnalogSignal
    complexity_distribution = neo.AnalogSignal(
        np.array(complexity_hist).reshape(len(complexity_hist), 1) *
        pq.dimensionless, t_start=0 * pq.dimensionless,
        sampling_period=1 * pq.dimensionless)

    return complexity_distribution


"""
Kernel Bandwidth Optimization.

Python implementation by Subhasis Ray.

Original matlab code (sskernel.m) here:
http://2000.jukuin.keio.ac.jp/shimazaki/res/kernel.html

This was translated into Python by Subhasis Ray, NCBS. Tue Jun 10
23:01:43 IST 2014

"""


def nextpow2(x):
    """
    Return the smallest integral power of 2 that is equal or larger than `x`.
    """
    n = 2
    while n < x:
        n = 2 * n
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


def sskernel(spiketimes, tin=None, w=None, bootstrap=False):
    """
    Calculates optimal fixed kernel bandwidth, given as the standard deviation
    sigma.

    Parameters
    ----------
    spiketimes : np.ndarray
        Sequence of spike times (sorted to be ascending).
    tin : np.ndarray, optional
        Time points at which the kernel bandwidth is to be estimated.
        If None, `spiketimes` is used.
        Default: None.
    w : np.ndarray, optional
        Vector of kernel bandwidths (standard deviation sigma).
        If specified, optimal bandwidth is selected from this.
        If None, `w` is obtained through a golden-section search on a log-exp
        scale.
        Default: None.
    bootstrap : bool, optional
        If True, calculates the 95% confidence interval using Bootstrap.
        Default: False.

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
            Cost functions of `w`.
        'confb95' : tuple of np.ndarray
            Bootstrap 95% confidence interval: (lower level, upper level).
            If `bootstrap` is False, `confb95` is None.
        'yb' : np.ndarray
            Bootstrap samples.
            If `bootstrap` is False, `yb` is None.

        If no optimal kernel could be found, all entries of the dictionary are
        set to None.

    References
    ----------
    .. [1] H. Shimazaki, & S. Shinomoto, "Kernel bandwidth optimization in
           spike rate estimation," Journal of Computational Neuroscience,
           vol. 29, no. 1-2, pp. 171-82, 2010. doi:10.1007/s10827-009-0180-4.

    """

    if tin is None:
        time = np.max(spiketimes) - np.min(spiketimes)
        isi = np.diff(spiketimes)
        isi = isi[isi > 0].copy()
        dt = np.min(isi)
        tin = np.linspace(np.min(spiketimes),
                          np.max(spiketimes),
                          min(int(time / dt + 0.5),
                              1000))  # The 1000 seems somewhat arbitrary
        t = tin
    else:
        time = np.max(tin) - np.min(tin)
        spiketimes = spiketimes[(spiketimes >= np.min(tin)) &
                                (spiketimes <= np.max(tin))].copy()
        isi = np.diff(spiketimes)
        isi = isi[isi > 0].copy()
        dt = np.min(isi)
        if dt > np.min(np.diff(tin)):
            t = np.linspace(np.min(tin), np.max(tin),
                            min(int(time / dt + 0.5), 1000))
        else:
            t = tin
    dt = np.min(np.diff(tin))
    yhist, bins = np.histogram(spiketimes, np.r_[t - dt / 2, t[-1] + dt / 2])
    N = np.sum(yhist)
    yhist = yhist / (N * dt)  # density
    optw = None
    y = None
    if w is not None:
        C = np.zeros(len(w))
        Cmin = np.inf
        for k, w_ in enumerate(w):
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
        w = np.zeros(imax)
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
                w[k] = logexp(c1)
                C[k] = f1
                optw = logexp(c1)
                y = y1 / (np.sum(y1 * dt))
            else:
                a = c1
                c1 = c2
                c2 = (2 - phi) * a + (phi - 1) * b
                f1 = f2
                f2, y2 = cost_function(yhist, N, logexp(c2), dt)
                w[k] = logexp(c2)
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
        yb = np.zeros((nbs, len(tin)))
        for ii in range(nbs):
            idx = np.floor(np.random.rand(N) * N).astype(int)
            xb = spiketimes[idx]
            y_histb, bins = np.histogram(
                xb, np.r_[t - dt / 2, t[-1] + dt / 2]) / dt / N
            yb_buf = fftkernel(y_histb, optw / dt).real
            yb_buf = yb_buf / np.sum(yb_buf * dt)
            yb[ii, :] = np.interp(tin, t, yb_buf)
        ybsort = np.sort(yb, axis=0)
        y95b = ybsort[np.floor(0.05 * nbs).astype(int), :]
        y95u = ybsort[np.floor(0.95 * nbs).astype(int), :]
        confb95 = (y95b, y95u)
    # Only perform interpolation if y could be calculated
    if y is not None:
        y = np.interp(tin, t, y)
    return {'y': y,
            't': tin,
            'optw': optw,
            'w': w,
            'C': C,
            'confb95': confb95,
            'yb': yb}


def _check_consistency_of_spiketrainlist(spiketrainlist, t_start=None,
                                         t_stop=None):
    for spiketrain in spiketrainlist:
        if not isinstance(spiketrain, SpikeTrain):
            raise TypeError(
                "spike train must be instance of :class:`SpikeTrain` of Neo!\n"
                "    Found: %s, value %s" % (
                    type(spiketrain), str(spiketrain)))
        if t_start is None and not spiketrain.t_start == spiketrainlist[
                0].t_start:
            raise ValueError(
                "the spike trains must have the same t_start!")
        if t_stop is None and not spiketrain.t_stop == spiketrainlist[
                0].t_stop:
            raise ValueError(
                "the spike trains must have the same t_stop!")
        if not spiketrain.units == spiketrainlist[0].units:
            raise ValueError(
                "the spike trains must have the same units!")
