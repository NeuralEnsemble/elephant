# -*- coding: utf-8 -*-
"""
This module allows to convert standard data representations
(e.g., a spike train stored as Neo SpikeTrain object)
into other representations useful to perform calculations on the data.
An example is the representation of a spike train as a sequence of 0-1 values
(binned spike train).

:copyright: Copyright 2014-2015 by the Elephant team, see AUTHORS.txt.
:license: CeCILL, see LICENSE.txt for details.
"""

from __future__ import division, print_function

import neo
import scipy
import scipy.sparse as sps
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
             A ``True`` value at a particular index indicates the presence of
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
    are not quantities will be assumed to have the same units as `spiketrain`.

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

###########################################################################
#
# Methods to calculate parameters, t_start, t_stop, bin size,
# number of bins
#
###########################################################################
def _calc_tstart(num_bins, binsize, t_stop):
    """
    Calculates the start point from given parameter.

    Calculates the start point :attr:`t_start` from the three parameter
    :attr:`t_stop`, :attr:`num_bins` and :attr`binsize`.

    Parameters
    ----------
    num_bins: int
        Number of bins
    binsize: quantities.Quantity
        Size of Bins
    t_stop: quantities.Quantity
        Stop time

    Returns
    -------
    t_start : quantities.Quantity
        Starting point calculated from given parameter.
    """
    if num_bins is not None and binsize is not None and t_stop is not None:
        return t_stop.rescale(binsize.units) - num_bins * binsize


def _calc_tstop(num_bins, binsize, t_start):
    """
    Calculates the stop point from given parameter.

    Calculates the stop point :attr:`t_stop` from the three parameter
    :attr:`t_start`, :attr:`num_bins` and :attr`binsize`.

    Parameters
    ----------
    num_bins: int
        Number of bins
    binsize: quantities.Quantity
        Size of bins
    t_start: quantities.Quantity
        Start time

    Returns
    -------
    t_stop : quantities.Quantity
        Stoping point calculated from given parameter.
    """
    if num_bins is not None and binsize is not None and t_start is not None:
        return t_start.rescale(binsize.units) + num_bins * binsize


def _calc_num_bins(binsize, t_start, t_stop):
    """
    Calculates the number of bins from given parameter.

    Calculates the number of bins :attr:`num_bins` from the three parameter
    :attr:`t_start`, :attr:`t_stop` and :attr`binsize`.

    Parameters
    ----------
    binsize: quantities.Quantity
        Size of Bins
    t_start : quantities.Quantity
        Start time
    t_stop: quantities.Quantity
        Stop time

    Returns
    -------
    num_bins : int
       Number of bins  calculated from given parameter.

    Raises
    ------
    ValueError :
        Raised when :attr:`t_stop` is smaller than :attr:`t_start`".

    """
    if binsize is not None and t_start is not None and t_stop is not None:
        if t_stop < t_start:
            raise ValueError("t_stop (%s) is smaller than t_start (%s)"
                             % (t_stop, t_start))
        return int(((t_stop - t_start).rescale(
            binsize.units) / binsize).magnitude)


def _calc_binsize(num_bins, t_start, t_stop):
    """
    Calculates the stop point from given parameter.

    Calculates the size of bins :attr:`binsize` from the three parameter
    :attr:`num_bins`, :attr:`t_start` and :attr`t_stop`.

    Parameters
    ----------
    num_bins: int
        Number of bins
    t_start: quantities.Quantity
        Start time
    t_stop
       Stop time

    Returns
    -------
    binsize : quantities.Quantity
        Size of bins calculated from given parameter.

    Raises
    ------
    ValueError :
        Raised when :attr:`t_stop` is smaller than :attr:`t_start`".
    """

    if num_bins is not None and t_start is not None and t_stop is not None:
        if t_stop < t_start:
            raise ValueError("t_stop (%s) is smaller than t_start (%s)"
                             % (t_stop, t_start))
        return (t_stop - t_start) / num_bins


def _get_start_stop_from_input(spiketrains):
    """
    Returns the start :attr:`t_start`and stop :attr:`t_stop` point
    from given input.

    If one neo.SpikeTrain objects is given the start :attr:`t_stop `and stop
    :attr:`t_stop` of the spike train is returned.
    Otherwise the aligned times are returned, which are the maximal start point
    and minimal stop point.

    Parameters
    ----------
    spiketrains: neo.SpikeTrain object, list or array of neo.core.SpikeTrain
                 objects
        List of neo.core SpikeTrain objects to extract `t_start` and
        `t_stop` from.

    Returns
    -------
    start : quantities.Quantity
        Start point extracted from input :attr:`spiketrains`
    stop : quantities.Quantity
        Stop point extracted from input :attr:`spiketrains`
    """
    if isinstance(spiketrains, neo.SpikeTrain):
        return spiketrains.t_start, spiketrains.t_stop
    else:
        start = max([elem.t_start for elem in spiketrains])
        stop = min([elem.t_stop for elem in spiketrains])
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
    manner.
    Its rows represent the number of spike trains
    and the columns represent the binned index position of a spike in a
    spike train.
    The calculated matrix columns contains **Trues**, which indicate
    a spike.

    A matrix with counted time points is calculated the same way, but its
    columns contain the number of spikes that occurred in the spike train(s).
    It counts the occurrence of the timing of a spike in its respective
    spike train.

    Parameters
    ----------
    spiketrains : List of `neo.SpikeTrain` or a `neo.SpikeTrain` object
        Spiketrain(s)to be binned.
    binsize : quantities.Quantity
        Width of each time bin.
        Default is `None`
    num_bins : int
        Number of bins of the binned spike train.
        Default is `None`
    t_start : quantities.Quantity
        Time of the first bin (left extreme; included).
        Default is `None`
    t_stop : quantities.Quantity
        Stopping time of the last bin (right extreme; excluded).
        Default is `None`

    See also
    --------
    _convert_to_binned
    spike_indices
    to_bool_array
    to_array

    Notes
    -----
    There are four cases the given parameters must fulfill.
    Each parameter must be a combination of following order or it will raise
    a value error:
    * t_start, num_bins, binsize
    * t_start, num_bins, t_stop
    * t_start, bin_size, t_stop
    * t_stop, num_bins, binsize

    It is possible to give the SpikeTrain objects and one parameter
    (:attr:`num_bins` or :attr:`binsize`). The start and stop time will be
    calculated from given SpikeTrain objects (max start and min stop point).
    Missing parameter will also be calculated automatically.
    All parameters will be checked for consistency. A corresponding error will
    be risen, if one of the four parameters does not match the consistency
    requirements.

    """

    def __init__(self, spiketrains, binsize=None, num_bins=None, t_start=None,
                 t_stop=None):
        """
        Defines a binned spike train class

        """
        # Converting spiketrains to a list, if spiketrains is one
        # SpikeTrain object
        if isinstance(spiketrains, neo.SpikeTrain):
            spiketrains = [spiketrains]

        # Check that spiketrains is a list of neo Spike trains.
        if not all([type(elem) == neo.core.SpikeTrain for elem in spiketrains]):
            raise TypeError(
                "All elements of the input list must be neo.core.SpikeTrain "
                "objects ")
        # Link to input
        self.lst_input = spiketrains
        # Set given parameter
        self.t_start = t_start
        self.t_stop = t_stop
        self.num_bins = num_bins
        self.binsize = binsize
        self.matrix_columns = num_bins
        self.matrix_rows = len(spiketrains)
        # Empty matrix for storage, time points matrix
        self._mat_u = None
        # Check all parameter, set also missing values
        self._calc_start_stop(spiketrains)
        self._check_init_params(binsize, num_bins, self.t_start, self.t_stop)
        self._check_consistency(spiketrains, self.binsize, self.num_bins,
                                self.t_start, self.t_stop)
        # Variables to store the sparse matrix
        self._sparse_mat_u = None
        # Now create sparse matrix
        self._convert_to_binned(spiketrains)

    # =========================================================================
    # There are four cases the given parameters must fulfill
    # Each parameter must be a combination of following order or it will raise
    # a value error:
    # t_start, num_bins, binsize
    # t_start, num_bins, t_stop
    # t_start, bin_size, t_stop
    # t_stop, num_bins, binsize
    # ==========================================================================

    def _check_init_params(self, binsize, num_bins, t_start, t_stop):
        """
        Checks given parameter.
        Calculates also missing parameter.

        Parameters
        ----------
        binsize : quantity.Quantity
            Size of Bins
        num_bins : int
            Number of Bins
        t_start: quantity.Quantity
            Start time of the spike
        t_stop: quantity.Quantity
            Stop time of the spike

        Raises
        ------
        ValueError :
            If all parameters are `None`, a ValueError is raised.
        TypeError:
            If type of :attr:`num_bins` is not an Integer.

        """
        # Check if num_bins is an integer (special case)
        if num_bins is not None:
            if not isinstance(num_bins, int):
                raise TypeError("num_bins is not an integer!")
        # Check if all parameters can be calculated, otherwise raise ValueError
        if t_start is None:
            self.t_start = _calc_tstart(num_bins, binsize, t_stop)
        elif t_stop is None:
            self.t_stop = _calc_tstop(num_bins, binsize, t_start)
        elif num_bins is None:
            self.num_bins = _calc_num_bins(binsize, t_start, t_stop)
            if self.matrix_columns is None:
                self.matrix_columns = self.num_bins
        elif binsize is None:
            self.binsize = _calc_binsize(num_bins, t_start, t_stop)

    def _calc_start_stop(self, spiketrains):
        """
        Calculates start, stop from given spike trains.

        The start and stop points are calculated from given spike trains, only
        if they are not calculable from given parameter or the number of
        parameters is less than three.

        """
        if self._count_params() is False:
            start, stop = _get_start_stop_from_input(spiketrains)
            if self.t_start is None:
                self.t_start = start
            if self.t_stop is None:
                self.t_stop = stop

    def _count_params(self):
        """
        Counts up if one parameter is not `None` and returns **True** if the
        count is greater or equal `3`.

        The calculation of the binned matrix is only possible if there are at
        least three parameter (fourth parameter will be calculated out of
        them).
        This method checks if the necessary parameter are not `None` and
        returns **True** if the count is greater or equal to `3`.

        Returns
        -------
        bool :
            True, if the count is greater or equal to `3`.
            False, otherwise.

        """
        return sum(x is not None for x in
                   [self.t_start, self.t_stop, self.binsize,
                    self.num_bins]) >= 3

    def _check_consistency(self, spiketrains, binsize, num_bins, t_start,
                           t_stop):
        """
        Checks the given parameters for consistency

        Raises
        ------
        ValueError :
            A ValueError is raised if an inconsistency regarding the parameter
            appears.
        AttributeError :
            An AttributeError is raised if there is an insufficient number of
            parameters.

        """
        if self._count_params() is False:
            raise AttributeError("Too less parameter given. Please provide "
                                 "at least one of the parameter which are "
                                 "None.\n"
                                 "t_start: %s, t_stop: %s, binsize: %s, "
                                 "numb_bins: %s" % (
                                     self.t_start,
                                     self.t_stop,
                                     self.binsize,
                                     self.num_bins))
        t_starts = [elem.t_start for elem in spiketrains]
        t_stops = [elem.t_stop for elem in spiketrains]
        max_tstart = max(t_starts)
        min_tstop = min(t_stops)
        if max_tstart >= min_tstop:
            raise ValueError(
                "Starting time of each spike train must be smaller than each "
                "stopping time")
        elif t_start < max_tstart or t_start > min_tstop:
            raise ValueError(
                'some spike trains are not defined in the time given '
                'by t_start')
        elif num_bins != int((
                    (t_stop - t_start).rescale(binsize.units) / binsize).magnitude):
            raise ValueError(
                "Inconsistent arguments t_start (%s), " % t_start +
                "t_stop (%s), binsize (%d) " % (t_stop, binsize) +
                "and num_bins (%d)" % num_bins)
        elif not (t_start < t_stop <= min_tstop):
            raise ValueError(
                'too many / too large time bins. Some spike trains are '
                'not defined in the ending time')
        elif num_bins - int(num_bins) != 0 or num_bins < 0:
            raise TypeError(
                "Number of bins (num_bins) is not an integer or < 0: " + str(
                    num_bins))

    @property
    def bin_edges(self):
        """
        Returns all time edges with :attr:`num_bins` bins as a quantity array.

        The borders of all time steps between start and stop [start, stop]
        with:attr:`num_bins` bins are regarded as edges.
        The border of the last bin is included.

        Returns
        -------
        bin_edges : quantities.Quantity array
            All edges in interval [start, stop] with :attr:`num_bins` bins
            are returned as a quantity array.

        """
        return np.linspace(self.t_start, self.t_stop,
                           self.num_bins + 1, endpoint=True)

    @property
    def bin_centers(self):
        """
        Returns each center time point of all bins between start and stop
        points.

        The center of each bin of all time steps between start and stop
        (start, stop).

        Returns
        -------
        bin_edges : quantities.Quantity array
            All center edges in interval (start, stop) are returned as
            a quantity array.

        """
        return self.bin_edges[:-1] + self.binsize / 2

    def to_sparse_array(self):
        """
        Getter for sparse matrix with time points.

        Returns
        -------
        matrix: scipy.sparse.csr_matrix
            Sparse matrix, counted version.

        See also
        --------
        scipy.sparse.csr_matrix
        to_array

        """
        return self._sparse_mat_u

    def to_sparse_bool_array(self):
        """
        Getter for **boolean** version of the sparse matrix, calculated from
        sparse matrix with counted time points.

        Returns
        -------
        matrix: scipy.sparse.csr_matrix
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
        >>> st = n.SpikeTrain([0.5, 0.7, 1.2, 3.1, 4.3, 5.5, 6.7] * pq.s, t_stop=10.0 * pq.s)
        >>> x = conv.BinnedSpikeTrain(st, num_bins=10, binsize=1 * pq.s, t_start=0 * pq.s)
        >>> print(x.spike_indices)
        [[0, 0, 1, 3, 4, 5, 6]]
        >>> print(x.to_sparse_array().nonzero()[1])
        [0 1 3 4 5 6]

        """
        spike_idx = []
        for row in self._sparse_mat_u:
            l = []
            # Extract each non-zeros column index and how often it exists,
            # i.e., how many spikes fall in this column
            for col, count in zip(row.nonzero()[1], row.data):
                # Append the column index for each spike
                l.extend([col] * count)
            spike_idx.append(l)
        return spike_idx

    def to_bool_array(self):
        """
        Returns a dense matrix (`scipy.sparse.csr_matrix`), which rows
        represent the number of spike trains and the columns represent the
        binned index position of a spike in a spike train.
        The matrix columns contain **True**, which indicate a spike and
        **False** for non spike.

        Returns
        -------
        bool matrix : numpy.ndarray
            Returns a dense matrix representation of the sparse matrix,
            with **True** indicating a spike and **False*** for
            non spike.
            The **Trues** in the columns represent the index
            position of the spike in the spike train and rows represent the
            number of spike trains.

        Examples
        --------
        >>> import elephant.conversion as conv
        >>> import neo as n
        >>> import quantities as pq
        >>> a = n.SpikeTrain([0.5, 0.7, 1.2, 3.1, 4.3, 5.5, 6.7] * pq.s, t_stop=10.0 * pq.s)
        >>> x = conv.BinnedSpikeTrain(a, num_bins=10, binsize=1 * pq.s, t_start=0 * pq.s)
        >>> print(x.to_bool_array())
        [[ True  True False  True  True  True  True False False False]]

        See also
        --------
        scipy.sparse.csr_matrix
        scipy.sparse.csr_matrix.toarray
        """
        return abs(scipy.sign(self.to_array())).astype(bool)

    def to_array(self, store_array=False):
        """
        Returns a dense matrix, calculated from the sparse matrix with counted
        time points, which rows represents the number of spike trains and the
        columns represents the binned index position of a spike in a
        spike train.
        The  matrix columns contain the number of spikes that
        occurred in the spike train(s).
        If the **boolean** :attr:`store_array` is set to **True** the matrix
        will be stored in memory.

        Returns
        -------
        matrix : numpy.ndarray
            Matrix with spike times. Columns represent the index position of
            the binned spike and rows represent the number of spike trains.

        Examples
        --------
        >>> import elephant.conversion as conv
        >>> import neo as n
        >>> a = n.SpikeTrain([0.5, 0.7, 1.2, 3.1, 4.3, 5.5, 6.7] * pq.s, t_stop=10.0 * pq.s)
        >>> x = conv.BinnedSpikeTrain(a, num_bins=10, binsize=1 * pq.s, t_start=0 * pq.s)
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
            self._mat_u = self.to_sparse_array().toarray()

    def remove_stored_array(self):
        """
        Removes the matrix with counted time points from memory.

        """
        if self._mat_u is not None:
            del self._mat_u
            self._mat_u = None

    def _convert_to_binned(self, spiketrains):
        """
        Converts neo.core.SpikeTrain objects to a sparse matrix
        (`scipy.sparse.csr_matrix`), which contains the binned times.

        Parameters
        ----------
        spiketrains : neo.SpikeTrain object or list of SpikeTrain objects
           The binned time array :attr:`spike_indices` is calculated from a
           SpikeTrain object or from a list of SpikeTrain objects.

        """
        lil_mat = sps.lil_matrix((self.matrix_rows, self.matrix_columns),
                                 dtype=int)
        for idx, elem in enumerate(spiketrains):
            ev = elem.view(pq.Quantity)
            scale = np.array(((ev - self.t_start).rescale(
                self.binsize.units) / self.binsize).magnitude, dtype=int)
            l = np.logical_and(ev >= self.t_start.rescale(self.binsize.units),
                               ev <= self.t_stop.rescale(self.binsize.units))
            filled = scale[l]
            filled = filled[filled < self.num_bins]
            for inner_elem in filled:
                lil_mat[idx, inner_elem] += 1
        self._sparse_mat_u = lil_mat.tocsr()