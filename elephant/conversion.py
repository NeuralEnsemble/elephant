# -*- coding: utf-8 -*-
"""
This module allows to convert standard data representations
(e.g., a spike train stored as Neo SpikeTrain object)
into other representations useful to perform calculations on the data.
An example is the representation of a spike train as a sequence of 0-1 values
(binned spike train).

:copyright: Copyright 2014-2016 by the Elephant team, see `doc/authors.rst`.
:license: BSD, see LICENSE.txt for details.
"""

from __future__ import division, print_function, unicode_literals

import warnings
from copy import deepcopy

import neo
import numpy as np
import quantities as pq
import scipy.sparse as sps

from elephant.utils import is_binary, deprecated_alias


def binarize(spiketrain, sampling_rate=None, t_start=None, t_stop=None,
             return_times=False):
    """
    Return an array indicating if spikes occurred at individual time points.

    The array contains boolean values identifying whether at least one spike
    occurred in the corresponding time bin. Time bins start at `t_start`
    and end at `t_stop`, spaced in `1/sampling_rate` intervals.

    Accepts either a `neo.SpikeTrain`, a `pq.Quantity` array, or a plain
    `np.ndarray`.
    Returns a boolean array with each element indicating the presence or
    absence of a spike in that time bin.

    Optionally also returns an array of time points corresponding to the
    elements of the boolean array.  The units of this array will be the same as
    the units of the neo.SpikeTrain, if any.

    Parameters
    ----------
    spiketrain : neo.SpikeTrain or pq.Quantity or np.ndarray
        The spike times.  Does not have to be sorted.
    sampling_rate : float or pq.Quantity, optional
        The sampling rate to use for the time points.
        If not specified, retrieved from the `sampling_rate` attribute of
        `spiketrain`.
        Default: None.
    t_start : float or pq.Quantity, optional
        The start time to use for the time points.
        If not specified, retrieved from the `t_start` attribute of
        `spiketrain`. If this is not present, defaults to `0`.  Any element of
        `spiketrain` lower than `t_start` is ignored.
        Default: None.
    t_stop : float or pq.Quantity, optional
        The stop time to use for the time points.
        If not specified, retrieved from the `t_stop` attribute of
        `spiketrain`. If this is not present, defaults to the maximum value of
        `spiketrain`. Any element of `spiketrain` higher than `t_stop` is
        ignored.
        Default: None.
    return_times : bool, optional
        If True, also return the corresponding time points.
        Default: False.

    Returns
    -------
    values : np.ndarray of bool
        A True value at a particular index indicates the presence of one or
        more spikes at the corresponding time point.
    times : np.ndarray or pq.Quantity, optional
        The time points.  This will have the same units as `spiketrain`.
        If `spiketrain` has no units, this will be an `np.ndarray` array.

    Raises
    ------
    TypeError
        If `spiketrain` is an `np.ndarray` and `t_start`, `t_stop`, or
        `sampling_rate` is a `pq.Quantity`.
    ValueError
        If `sampling_rate` is not explicitly defined and not present as an
        attribute of `spiketrain`.

    Notes
    -----
    Spike times are placed in the bin of the closest time point, going to the
    higher bin if exactly between two bins.

    So in the case where the bins are `5.5` and `6.5`, with the spike time
    being `6.0`, the spike will be placed in the `6.5` bin.

    The upper edge of the last bin, equal to `t_stop`, is inclusive.  That is,
    a spike time exactly equal to `t_stop` will be included.

    If `spiketrain` is a `pq.Quantity` or `neo.SpikeTrain` and `t_start`,
    `t_stop` or `sampling_rate` is not, then the arguments that are not
    `pq.Quantity` will be assumed to have the same units as `spiketrain`.

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
    sampling_period = 1. / sampling_rate

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
    edges = np.arange(t_start - sampling_period / 2,
                      t_stop + sampling_period * 3 / 2,
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
        return res, np.arange(t_start, t_stop + sampling_period,
                              sampling_period)
    else:
        return res, pq.Quantity(np.arange(t_start, t_stop + sampling_period,
                                          sampling_period), units=units)


###########################################################################
#
# Methods to calculate parameters, t_start, t_stop, bin size,
# number of bins
#
###########################################################################


def _detect_rounding_errors(values, tolerance):
    """
    Finds rounding errors in values that will be cast to int afterwards.
    Returns True for values that are within tolerance of the next integer.
    Works for both scalars and numpy arrays.
    """
    if tolerance is None:
        return np.zeros_like(values, dtype=bool)
    return 1 - (values % 1) <= tolerance


def _calc_tstart(n_bins, bin_size, t_stop):
    """
    Calculates the start point from given parameters.

    Calculates the start point `t_start` from the three parameters
    `n_bins`, `bin_size`, `t_stop`.

    Parameters
    ----------
    n_bins : int
        Number of bins
    bin_size : pq.Quantity
        Size of Bins
    t_stop : pq.Quantity
        Stop time

    Returns
    -------
    t_start : pq.Quantity
        Starting point calculated from given parameters.
    """
    if n_bins is not None and bin_size is not None and t_stop is not None:
        return t_stop.rescale(bin_size.units) - n_bins * bin_size


def _calc_tstop(n_bins, bin_size, t_start):
    """
    Calculates the stop point from given parameters.

    Calculates the stop point `t_stop` from the three parameters
    `n_bins`, `bin_size`, `t_start`.

    Parameters
    ----------
    n_bins : int
        Number of bins
    bin_size : pq.Quantity
        Size of bins
    t_start : pq.Quantity
        Start time

    Returns
    -------
    t_stop : pq.Quantity
        Stopping point calculated from given parameters.
    """
    if n_bins is not None and bin_size is not None and t_start is not None:
        return t_start.rescale(bin_size.units) + n_bins * bin_size


def _calc_number_of_bins(bin_size, t_start, t_stop, tolerance):
    """
    Calculates the number of bins from given parameters.

    Calculates the number of bins `n_bins` from the three parameters
    `bin_size`, `t_start`, `t_stop`.

    Parameters
    ----------
    bin_size : pq.Quantity
        Size of Bins
    t_start : pq.Quantity
        Start time
    t_stop : pq.Quantity
        Stop time
    tolerance : float
        tolerance for detection of rounding errors before casting
        the resulting num. of bins to integer

    Returns
    -------
    n_bins : int
       Number of bins calculated from given parameters.

    Raises
    ------
    ValueError
        When `t_stop` is smaller than `t_start`".

    """
    if bin_size is not None and t_start is not None and t_stop is not None:
        if t_stop < t_start:
            raise ValueError("t_stop (%s) is smaller than t_start (%s)"
                             % (t_stop, t_start))
        n_bins = ((t_stop - t_start).rescale(
                        bin_size.units) / bin_size.magnitude).item()
        if _detect_rounding_errors(n_bins, tolerance):
            warnings.warn('Correcting a rounding error in the calculation '
                          'of n_bins by increasing n_bins by 1. '
                          'You can set tolerance=None to disable this '
                          'behaviour.')
            n_bins += 1
        return int(n_bins)


def _calc_bin_size(n_bins, t_start, t_stop):
    """
    Calculates the stop point from given parameters.

    Calculates the size of bins `bin_size` from the three parameters
    `n_bins`, `t_start` and `t_stop`.

    Parameters
    ----------
    n_bins : int
        Number of bins
    t_start : pq.Quantity
        Start time
    t_stop : pq.Quantity
        Stop time

    Returns
    -------
    bin_size : pq.Quantity
        Size of bins calculated from given parameters.

    Raises
    ------
    ValueError
        When `t_stop` is smaller than `t_start`.
    """

    if n_bins is not None and t_start is not None and t_stop is not None:
        if t_stop < t_start:
            raise ValueError("t_stop (%s) is smaller than t_start (%s)"
                             % (t_stop, t_start))
        return (t_stop - t_start) / n_bins


def _get_start_stop_from_input(spiketrains):
    """
    Extracts the `t_start`and the `t_stop` from 'spiketrains'.

    If a single `neo.SpikeTrain` is given, the `t_start `and
    `t_stop` of this spike train is returned.
    Otherwise, the aligned times are returned: the maximal `t_start` and
    minimal `t_stop` across `spiketrains`.

    Parameters
    ----------
    spiketrains : neo.SpikeTrain or list or np.ndarray of neo.SpikeTrain
        `neo.SpikeTrain`s to extract `t_start` and `t_stop` from.

    Returns
    -------
    start : pq.Quantity
        Start point extracted from input `spiketrains`
    stop : pq.Quantity
        Stop point extracted from input `spiketrains`

    Raises
    ------
    AttributeError
        If spiketrains (or any element of it) do not have `t_start` or `t_stop`
        attribute.
    """
    if isinstance(spiketrains, neo.SpikeTrain):
        return spiketrains.t_start, spiketrains.t_stop
    else:
        try:
            start = max([elem.t_start for elem in spiketrains])
            stop = min([elem.t_stop for elem in spiketrains])
        except AttributeError as ae:
            raise AttributeError(ae, 'Please provide t_start or t_stop')
    return start, stop


class BinnedSpikeTrain(object):
    """
    Class which calculates a binned spike train and provides methods to
    transform the binned spike train to a boolean matrix or a matrix with
    counted time points.

    A binned spike train represents the occurrence of spikes in a certain time
    frame.
    I.e., a time series like [0.5, 0.7, 1.2, 3.1, 4.3, 5.5, 6.7] is
    represented as [0, 0, 1, 3, 4, 5, 6]. The outcome is dependent on given
    parameter such as size of bins, number of bins, start and stop points.

    A boolean matrix represents the binned spike train in a binary (True/False)
    manner. Its rows represent the number of spike trains and the columns
    represent the binned index position of a spike in a spike train.
    The calculated matrix entry containing `True` indicates a spike.

    A matrix with counted time points is calculated the same way, but its
    entries contain the number of spikes that occurred in the given bin of the
    given spike train.

    Note that with most common parameter combinations spike times can end up
    on bin edges. This makes the binning susceptible to rounding errors which
    is accounted for by moving spikes which are within tolerance of the next
    bin edge into the following bin. This can be adjusted using the tolerance
    parameter and turned off by setting `tolerance=None`.

    Parameters
    ----------
    spiketrains : neo.SpikeTrain or list of neo.SpikeTrain or np.ndarray
        Spike train(s) to be binned.
    bin_size : pq.Quantity, optional
        Width of a time bin.
        Default: None
    n_bins : int, optional
        Number of bins of the binned spike train.
        Default: None
    t_start : pq.Quantity, optional
        Time of the left edge of the first bin (left extreme; included).
        Default: None
    t_stop : pq.Quantity, optional
        Time of the right edge of the last bin (right extreme; excluded).
        Default: None
    tolerance : float, optional
        Tolerance for rounding errors in the binning process and in the input
        data
        Default: 1e-8

    Raises
    ------
    AttributeError
        If less than 3 optional parameters are `None`.
    TypeError
        If `spiketrains` is an np.ndarray with dimensionality different than
        NxM or
        if type of `n_bins` is not an `int` or `n_bins` < 0.
    ValueError
        When number of bins calculated from `t_start`, `t_stop` and `bin_size`
        differs from provided `n_bins` or
        if `t_stop` of any spike train is smaller than any `t_start` or
        if any spike train does not cover the full [`t_start`, t_stop`] range.

    Warns
    -----
    UserWarning
        If some spikes fall outside of [`t_start`, `t_stop`] range

    See also
    --------
    _convert_to_binned
    spike_indices
    to_bool_array
    to_array

    Notes
    -----
    There are four minimal configurations of the optional parameters which have
    to be provided, otherwise a `ValueError` will be raised:
    * `t_start`, `n_bins`, `bin_size`
    * `t_start`, `n_bins`, `t_stop`
    * `t_start`, `bin_size`, `t_stop`
    * `t_stop`, `n_bins`, `bin_size`

    If `spiketrains` is a `neo.SpikeTrain` or a list thereof, it is enough to
    explicitly provide only one parameter: `n_bins` or `bin_size`. The
    `t_start` and `t_stop` will be calculated from given `spiketrains` (max
    `t_start` and min `t_stop` of `neo.SpikeTrain`s).
    Missing parameter will be calculated automatically.
    All parameters will be checked for consistency. A corresponding error will
    be raised, if one of the four parameters does not match the consistency
    requirements.

    """

    @deprecated_alias(binsize='bin_size', num_bins='n_bins')
    def __init__(self, spiketrains, bin_size=None, n_bins=None, t_start=None,
                 t_stop=None, tolerance=1e-8):
        """
        Defines a BinnedSpikeTrain class

        """
        self.is_spiketrain = _check_neo_spiketrain(spiketrains)
        if not self.is_spiketrain:
            self.is_binned = _check_binned_array(spiketrains)
        else:
            self.is_binned = False
        # Converting spiketrains to a list, if spiketrains is one
        # SpikeTrain object
        if isinstance(spiketrains,
                      neo.SpikeTrain) and self.is_spiketrain:
            spiketrains = [spiketrains]

        # Link to input
        self.input_spiketrains = spiketrains
        # Set given parameters
        self.t_start = t_start
        self.t_stop = t_stop
        self.n_bins = n_bins
        self.bin_size = bin_size
        self.tolerance = tolerance
        # Empty matrix for storage, time points matrix
        self._mat_u = None
        # Variables to store the sparse matrix
        self._sparse_mat_u = None
        # Check all parameter, set also missing values
        if self.is_binned:
            self.n_bins = np.shape(spiketrains)[1]
        self._calc_start_stop(spiketrains)
        self._check_init_params(
            self.bin_size, self.n_bins, self.t_start, self.t_stop)
        self._check_consistency(spiketrains, self.bin_size, self.n_bins,
                                self.t_start, self.t_stop)
        # Now create sparse matrix
        self._convert_to_binned(spiketrains)

        if self.is_spiketrain:
            n_spikes = sum(map(len, spiketrains))
            n_spikes_binned = self.get_num_of_spikes()
            if n_spikes != n_spikes_binned:
                warnings.warn("Binning discarded {n} last spike(s) in the "
                              "input spiketrain.".format(
                                  n=n_spikes - n_spikes_binned))

    @property
    def matrix_rows(self):
        return self._sparse_mat_u.shape[0]

    @property
    def matrix_columns(self):
        return self._sparse_mat_u.shape[1]

    @property
    def binsize(self):
        warnings.warn("'.binsize' is deprecated; use '.bin_size'",
                      DeprecationWarning)
        return self.bin_size

    @property
    def lst_input(self):
        warnings.warn("'.lst_input' is deprecated; use '.input_spiketrains'",
                      DeprecationWarning)
        return self.input_spiketrains

    @property
    def num_bins(self):
        warnings.warn("'.num_bins' is deprecated; use '.n_bins'")
        return self.n_bins

    # =========================================================================
    # There are four cases the given parameters must fulfill, or a `ValueError`
    # will be raised:
    # t_start, n_bins, bin_size
    # t_start, n_bins, t_stop
    # t_start, bin_size, t_stop
    # t_stop, n_bins, bin_size
    # =========================================================================

    def _check_init_params(self, bin_size, n_bins, t_start, t_stop):
        """
        Checks given parameters.
        Calculates also missing parameter.

        Parameters
        ----------
        bin_size : pq.Quantity
            Size of bins
        n_bins : int
            Number of bins
        t_start: pq.Quantity
            Start time for the binned spike train
        t_stop: pq.Quantity
            Stop time for the binned spike train

        Raises
        ------
        TypeError
            If type of `n_bins` is not an `int`.
        ValueError
            When `t_stop` is smaller than `t_start`.

        """
        # Check if n_bins is an integer (special case)
        if n_bins is not None:
            if not np.issubdtype(type(n_bins), np.integer):
                raise TypeError("'n_bins' is not an integer!")
        # Check if all parameters can be calculated, otherwise raise ValueError
        if t_start is None:
            self.t_start = _calc_tstart(n_bins, bin_size, t_stop)
        elif t_stop is None:
            self.t_stop = _calc_tstop(n_bins, bin_size, t_start)
        elif n_bins is None:
            self.n_bins = _calc_number_of_bins(bin_size, t_start, t_stop,
                                               self.tolerance)
        elif bin_size is None:
            self.bin_size = _calc_bin_size(n_bins, t_start, t_stop)

    def _calc_start_stop(self, spiketrains):
        """
        Calculates `t_start`, `t_stop` from given spike trains.

        The start and stop points are calculated from given spike trains only
        if they are not calculable from given parameters or the number of
        parameters is less than three.

        Parameters
        ----------
        spiketrains : neo.SpikeTrain or list or np.ndarray of neo.SpikeTrain

        """
        if self._count_params() is False:
            start, stop = _get_start_stop_from_input(spiketrains)
            if self.t_start is None:
                self.t_start = start
            if self.t_stop is None:
                self.t_stop = stop

    def _count_params(self):
        """
        Checks the number of explicitly provided parameters and returns `True`
        if the count is greater or equal `3`.

        The calculation of the binned matrix is only possible if there are at
        least three parameters (fourth parameter will be calculated out of
        them).
        This method checks if the necessary parameters are not `None` and
        returns `True` if the count is greater or equal to `3`.

        Returns
        -------
        bool
            True, if the count of not None parameters is greater or equal to
            `3`, False otherwise.

        """
        return sum(x is not None for x in
                   [self.t_start, self.t_stop, self.bin_size,
                    self.n_bins]) >= 3

    def _check_consistency(self, spiketrains, bin_size, n_bins, t_start,
                           t_stop):
        """
        Checks the given parameters for consistency

        Raises
        ------
        AttributeError
            If there is an insufficient number of parameters.
        TypeError
            If `n_bins` is not an `int` or is <0.
        ValueError
            If an inconsistency regarding the parameters appears, e.g.
            `t_start` > `t_stop`.

        """
        if self._count_params() is False:
            raise AttributeError("Too few parameters given. Please provide "
                                 "at least one of the parameter which are "
                                 "None.\n"
                                 "t_start: %s, t_stop: %s, bin_size: %s, "
                                 "n_bins: %s" % (
                                     self.t_start,
                                     self.t_stop,
                                     self.bin_size,
                                     self.n_bins))
        if self.is_spiketrain:
            t_starts = [elem.t_start for elem in spiketrains]
            t_stops = [elem.t_stop for elem in spiketrains]
            max_tstart = max(t_starts)
            min_tstop = min(t_stops)
            if max_tstart >= min_tstop:
                raise ValueError("Starting time of each spike train must be "
                                 "smaller than each stopping time")
            if t_start < max_tstart or t_start > min_tstop:
                raise ValueError(
                    'some spike trains are not defined in the time given '
                    'by t_start')
            if not (t_start < t_stop <= min_tstop):
                raise ValueError(
                    'too many / too large time bins. Some spike trains are '
                    'not defined in the ending time')

        # account for rounding errors in the reference num_bins
        n_bins_test = ((
            (t_stop - t_start).rescale(
                bin_size.units) / bin_size).magnitude)
        if _detect_rounding_errors(n_bins_test, tolerance=self.tolerance):
            n_bins_test += 1
        n_bins_test = int(n_bins_test)
        if n_bins != n_bins_test:
            raise ValueError(
                "Inconsistent arguments t_start (%s), " % t_start +
                "t_stop (%s), bin_size (%s) " % (t_stop, bin_size) +
                "and n_bins (%d)" % n_bins)
        if n_bins - int(n_bins) != 0 or n_bins < 0:
            raise TypeError(
                "Number of bins ({}) is not an integer or < 0".format(n_bins))

    @property
    def bin_edges(self):
        """
        Returns all time edges as a quantity array with :attr:`n_bins` bins.

        The borders of all time steps between :attr:`t_start` and
        :attr:`t_stop` with a step :attr:`bin_size`. It is crucial for many
        analyses that all bins have the same size, so if
        :attr:`t_stop` - :attr:`t_start` is not divisible by :attr:`bin_size`,
        there will be some leftover time at the end
        (see https://github.com/NeuralEnsemble/elephant/issues/255).
        The length of the returned array should match :attr:`n_bins`.

        Returns
        -------
        bin_edges : pq.Quantity
            All edges in interval [:attr:`t_start`, :attr:`t_stop`] with
            :attr:`n_bins` bins are returned as a quantity array.
        """
        t_start = self.t_start.rescale(self.bin_size.units).magnitude
        bin_edges = np.linspace(t_start, t_start + self.n_bins *
                                self.bin_size.magnitude,
                                num=self.n_bins + 1, endpoint=True)
        return pq.Quantity(bin_edges, units=self.bin_size.units)

    @property
    def bin_centers(self):
        """
        Returns each center time point of all bins between :attr:`t_start` and
        :attr:`t_stop` points.

        The center of each bin of all time steps between start and stop.

        Returns
        -------
        bin_edges : pq.Quantity
            All center edges in interval (:attr:`start`, :attr:`stop`).

        """
        return self.bin_edges[:-1] + self.bin_size / 2

    def to_sparse_array(self):
        """
        Getter for sparse matrix with time points.

        Returns
        -------
        scipy.sparse.csr_matrix
            Sparse matrix, version with spike counts.

        See also
        --------
        scipy.sparse.csr_matrix
        to_array

        """
        return self._sparse_mat_u

    def to_sparse_bool_array(self):
        """
        Getter for boolean version of the sparse matrix, calculated from
        sparse matrix with counted time points.

        Returns
        -------
        scipy.sparse.csr_matrix
            Sparse matrix, binary, boolean version.

        See also
        --------
        scipy.sparse.csr_matrix
        to_bool_array

        """
        # Return sparse Matrix as a copy
        tmp_mat = self._sparse_mat_u.copy()
        tmp_mat[tmp_mat.nonzero()] = 1
        return tmp_mat.astype(bool)

    def get_num_of_spikes(self, axis=None):
        """
        Compute the number of binned spikes.

        Parameters
        ----------
        axis : int, optional
            If `None`, compute the total num. of spikes.
            Otherwise, compute num. of spikes along axis.
            If axis is `1`, compute num. of spikes per spike train (row).
            Default is `None`.

        Returns
        -------
        n_spikes_per_row : int or np.ndarray
            The number of binned spikes.

        """
        if axis is None:
            return self._sparse_mat_u.sum(axis=axis)
        n_spikes_per_row = self._sparse_mat_u.sum(axis=axis)
        n_spikes_per_row = np.asarray(n_spikes_per_row)[:, 0]
        return n_spikes_per_row

    @property
    def spike_indices(self):
        """
        A list of lists for each spike train (i.e., rows of the binned matrix),
        that in turn contains for each spike the index into the binned matrix
        where this spike enters.

        In contrast to `to_sparse_array().nonzero()`, this function will report
        two spikes falling in the same bin as two entries.

        Examples
        --------
        >>> import elephant.conversion as conv
        >>> import neo as n
        >>> import quantities as pq
        >>> st = n.SpikeTrain([0.5, 0.7, 1.2, 3.1, 4.3, 5.5, 6.7] * pq.s,
        ...                   t_stop=10.0 * pq.s)
        >>> x = conv.BinnedSpikeTrain(st, n_bins=10, bin_size=1 * pq.s,
        ...                           t_start=0 * pq.s)
        >>> print(x.spike_indices)
        [[0, 0, 1, 3, 4, 5, 6]]
        >>> print(x.to_sparse_array().nonzero()[1])
        [0 1 3 4 5 6]
        >>> print(x.to_array())
        [[2, 1, 0, 1, 1, 1, 1, 0, 0, 0]]

        """
        spike_idx = []
        for row in self._sparse_mat_u:
            # Extract each non-zeros column index and how often it exists,
            # i.e., how many spikes fall in this column
            n_cols = np.repeat(row.indices, row.data)
            spike_idx.append(n_cols)
        return spike_idx

    @property
    def is_binary(self):
        """
        Checks and returns `True` if given input is a binary input.
        Beware, that the function does not know if the input is binary
        because e.g `to_bool_array()` was used before or if the input is just
        sparse (i.e. only one spike per bin at maximum).

        Returns
        -------
        bool
            True for binary input, False otherwise.
        """

        return is_binary(self.input_spiketrains)

    def to_bool_array(self):
        """
        Returns a matrix, in which the rows correspond to the spike trains and
        the columns correspond to the bins in the `BinnedSpikeTrain`.
        `True` indicates a spike in given bin of given spike train and
        `False` indicates lack of spikes.

        Returns
        -------
        numpy.ndarray
            Returns a dense matrix representation of the sparse matrix,
            with `True` indicating a spike and `False` indicating a no-spike.
            The columns represent the index position of the bins and rows
            represent the number of spike trains.

        See also
        --------
        scipy.sparse.csr_matrix
        scipy.sparse.csr_matrix.toarray

        Examples
        --------
        >>> import elephant.conversion as conv
        >>> import neo as n
        >>> import quantities as pq
        >>> a = n.SpikeTrain([0.5, 0.7, 1.2, 3.1, 4.3, 5.5, 6.7] * pq.s,
        ...                  t_stop=10.0 * pq.s)
        >>> x = conv.BinnedSpikeTrain(a, n_bins=10, bin_size=1 * pq.s,
        ...                           t_start=0 * pq.s)
        >>> print(x.to_bool_array())
        [[ True  True False  True  True  True  True False False False]]

        """
        return self.to_array().astype(bool)

    def to_array(self, store_array=False):
        """
        Returns a dense matrix, calculated from the sparse matrix, with counted
        time points of spikes. The rows correspond to spike trains and the
        columns correspond to bins in a `BinnedSpikeTrain`.
        Entries contain the count of spikes that occurred in the given bin of
        the given spike train.
        If the boolean :attr:`store_array` is set to `True`, the matrix
        will be stored in memory.

        Returns
        -------
        matrix : np.ndarray
            Matrix with spike counts. Columns represent the index positions of
            the binned spikes and rows represent the spike trains.

        Examples
        --------
        >>> import elephant.conversion as conv
        >>> import neo as n
        >>> import quantities as pq
        >>> a = n.SpikeTrain([0.5, 0.7, 1.2, 3.1, 4.3, 5.5, 6.7] * pq.s,
        ...                  t_stop=10.0 * pq.s)
        >>> x = conv.BinnedSpikeTrain(a, n_bins=10, bin_size=1 * pq.s,
        ...                           t_start=0 * pq.s)
        >>> print(x.to_array())
        [[2 1 0 1 1 1 1 0 0 0]]

        See also
        --------
        scipy.sparse.csr_matrix
        scipy.sparse.csr_matrix.toarray

        """
        if self._mat_u is not None:
            return self._mat_u
        if store_array:
            self._store_array()
            return self._mat_u
        # Matrix on demand
        else:
            return self._sparse_mat_u.toarray()

    def _store_array(self):
        """
        Stores the matrix with counted time points in memory.

        """
        if self._mat_u is None:
            self._mat_u = self._sparse_mat_u.toarray()

    def remove_stored_array(self):
        """
        Unlinks the matrix with counted time points from memory.
        """
        self._mat_u = None

    def binarize(self, copy=True):
        """
        Clip the internal array (no. of spikes in a bin) to `0` (no spikes) or
        `1` (at least one spike) values only.

        Parameters
        ----------
        copy : bool
            Perform the clipping in-place (False) or on a copy (True).
            Default: True.

        Returns
        -------
        bst : BinnedSpikeTrain
            `BinnedSpikeTrain` with both sparse and dense (if present) array
            representation clipped to `0` (no spike) or `1` (at least one
            spike) entries.

        """
        if copy:
            bst = deepcopy(self)
        else:
            bst = self
        bst._sparse_mat_u.data.clip(max=1, out=bst._sparse_mat_u.data)
        if bst._mat_u is not None:
            bst._mat_u.clip(max=1, out=bst._mat_u)
        return bst

    @property
    def sparsity(self):
        """
        Returns
        -------
        float
            Matrix sparsity defined as no. of nonzero elements divided by
            the matrix size
        """
        num_nonzero = self._sparse_mat_u.data.shape[0]
        return num_nonzero / np.prod(self._sparse_mat_u.shape)

    def _convert_to_binned(self, spiketrains):
        """
        Converts `neo.SpikeTrain` objects to a sparse matrix
        (`scipy.sparse.csr_matrix`), which contains the binned spike times, and
        stores it in :attr:`_sparse_mat_u`.

        Parameters
        ----------
        spiketrains : neo.SpikeTrain or list of neo.SpikeTrain
            Spike trains to bin.

        """
        if not self.is_spiketrain:
            self._sparse_mat_u = sps.csr_matrix(spiketrains, dtype=int)
            return

        row_ids, column_ids = [], []
        # data
        counts = []

        for idx, st in enumerate(spiketrains):
            times = (st.times - self.t_start).rescale(self.bin_size.units)
            scale = np.array((times / self.bin_size).magnitude)

            # shift spikes that are very close
            # to the right edge into the next bin
            rounding_error_indices = _detect_rounding_errors(scale,
                                                             self.tolerance)
            num_rounding_corrections = rounding_error_indices.sum()
            if num_rounding_corrections > 0:
                warnings.warn('Correcting {} rounding errors by shifting '
                              'the affected spikes into the following bin. '
                              'You can set tolerance=None to disable this '
                              'behaviour.'.format(num_rounding_corrections))
            scale[rounding_error_indices] += .5

            scale = scale.astype(int)

            la = np.logical_and(times >= 0 * self.bin_size.units,
                                times <= (self.t_stop
                                          - self.t_start).rescale(
                                              self.bin_size.units))
            filled_tmp = scale[la]
            filled_tmp = filled_tmp[filled_tmp < self.n_bins]
            f, c = np.unique(filled_tmp, return_counts=True)
            column_ids.extend(f)
            counts.extend(c)
            row_ids.extend([idx] * len(f))
        csr_matrix = sps.csr_matrix((counts, (row_ids, column_ids)),
                                    shape=(len(spiketrains),
                                           self.n_bins),
                                    dtype=int)
        self._sparse_mat_u = csr_matrix


def _check_neo_spiketrain(matrix):
    """
    Checks if given input contains neo.SpikeTrain objects

    Parameters
    ----------
    matrix
        Object to test for `neo.SpikeTrain`s

    Returns
    -------
    bool
        True if `matrix` is a neo.SpikeTrain or a list or tuple thereof,
        otherwise False.

    """
    # Check for single spike train
    if isinstance(matrix, neo.SpikeTrain):
        return True
    # Check for list or tuple
    if isinstance(matrix, (list, tuple)):
        return all(map(_check_neo_spiketrain, matrix))
    return False


def _check_binned_array(matrix):
    """
    Checks if given input is a binned array

    Parameters
    ----------
    matrix
        Object to test

    Returns
    -------
    bool
        True if `matrix` is an 2D array-like object,
        otherwise False.

    Raises
    ------
    TypeError
        If `matrix` is not 2-dimensional.

    """
    matrix = np.asarray(matrix)
    # Check for proper dimension MxN
    if matrix.ndim == 2:
        return True
    elif matrix.dtype == np.dtype('O'):
        raise TypeError('Please check the dimensions of the input, '
                        'it should be an MxN array, '
                        'the input has the shape: {}'.format(matrix.shape))
    else:
        # Otherwise not supported
        raise TypeError('Input not supported. Please check again')
