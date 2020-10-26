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

from elephant.utils import is_binary, deprecated_alias, rescale_magnitude, \
    check_consistency_of_spiketrains, get_common_start_stop_times

__all__ = [
    "binarize",
    "BinnedSpikeTrain"
]


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
    # same as '1 - (values % 1) <= tolerance' but faster
    return 1 - tolerance <= values % 1


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
        # Converting spiketrains to a list, if spiketrains is one
        # SpikeTrain object
        if isinstance(spiketrains, neo.SpikeTrain):
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
        self._resolve_input_parameters(spiketrains)
        self._round_number_of_bins()
        # Now create sparse matrix
        n_discarded = self._convert_to_binned(spiketrains)

        if n_discarded > 0:
            warnings.warn("Binning discarded {} last spike(s) in the "
                          "input spiketrain".format(n_discarded))

    @property
    def shape(self):
        return self._sparse_mat_u.shape

    @property
    def binsize(self):
        warnings.warn("'.binsize' is deprecated; use '.bin_size'",
                      DeprecationWarning)
        return self.bin_size

    @property
    def num_bins(self):
        warnings.warn("'.num_bins' is deprecated; use '.n_bins'")
        return self.n_bins

    def __resolve_binned(self, spiketrains):
        spiketrains = np.asarray(spiketrains)
        if spiketrains.ndim != 2 or spiketrains.dtype == np.dtype('O'):
            raise ValueError("If the input is not a spiketrain(s), it "
                             "must be an MxN numpy array, each cell of "
                             "which represents the number of (binned) "
                             "spikes that fall in an interval - not "
                             "raw spike times.")
        if self.n_bins is not None:
            raise ValueError("When the input is a binned matrix, 'n_bins' "
                             "must be set to None - it's extracted from the "
                             "input shape.")
        self.n_bins = spiketrains.shape[1]
        if self.bin_size is None:
            if self.t_start is None or self.t_stop is None:
                raise ValueError("To determine the bin size, both 't_start' "
                                 "and 't_stop' must be set")
            self.bin_size = (self.t_stop - self.t_start) / self.n_bins
        if self.t_start is None and self.t_stop is None:
            raise ValueError("Either 't_start' or 't_stop' must be set")
        if self.t_start is None:
            self.t_start = self.t_stop - self.bin_size * self.n_bins
        if self.t_stop is None:
            self.t_stop = self.t_start + self.bin_size * self.n_bins

    def _resolve_input_parameters(self, spiketrains):
        """
        Calculates `t_start`, `t_stop` from given spike trains.

        The start and stop points are calculated from given spike trains only
        if they are not calculable from given parameters or the number of
        parameters is less than three.

        Parameters
        ----------
        spiketrains : neo.SpikeTrain or list or np.ndarray of neo.SpikeTrain

        """
        if not _check_neo_spiketrain(spiketrains):
            self.__resolve_binned(spiketrains)
            return

        def check_bin_size_n_bins():
            if self.bin_size is None or self.n_bins is None:
                raise ValueError("When only 't_start' or 't_stop' is given, "
                                 "both 'bin_size' and 'n_bins' must be set")

        if self.bin_size is None and self.n_bins is None:
            raise ValueError("Either 'bin_size' or 'n_bins' must be given")

        # The input is a list a spiketrains
        if self.t_start is None and self.t_stop is None:
            check_consistency_of_spiketrains(spiketrains)
            self.t_start = spiketrains[0].t_start
            self.t_stop = spiketrains[0].t_stop
        elif self.t_start is None and self.t_stop is not None:
            check_bin_size_n_bins()
            self.t_start = self.t_stop - self.bin_size * self.n_bins
        elif self.t_start is not None and self.t_stop is None:
            check_bin_size_n_bins()
            self.t_stop = self.t_start + self.bin_size * self.n_bins
        start_shared, stop_shared = get_common_start_stop_times(spiketrains)
        if self.t_start < start_shared or self.t_stop > stop_shared:
            raise ValueError("'t_start' ({t_start}) or 't_stop' ({t_stop}) is "
                             "outside of the shared [{start_shared}, "
                             "{stop_shared}] interval".format(
                t_start=self.t_start, t_stop=self.t_stop,
                start_shared=start_shared, stop_shared=stop_shared))

        if self.n_bins is None:
            # bin_size is provided
            n_bins = ((self.t_stop - self.t_start) / self.bin_size
                      ).simplified.item()
            if _detect_rounding_errors(n_bins, self.tolerance):
                warnings.warn('Correcting a rounding error in the calculation '
                              'of n_bins by increasing n_bins by 1. '
                              'You can set tolerance=None to disable this '
                              'behaviour.')
                n_bins += 1
            self.n_bins = int(n_bins)
        elif self.bin_size is None:
            # n_bins is provided
            self.bin_size = (self.t_stop - self.t_start) / self.n_bins

        if self.t_start >= self.t_stop:
            raise ValueError("t_start must be smaller than t_stop")

    def _round_number_of_bins(self):
        # account for rounding errors in the reference num_bins
        n_bins_rounded = ((self.t_stop - self.t_start) / self.bin_size
                          ).simplified.item()
        if _detect_rounding_errors(n_bins_rounded, tolerance=self.tolerance):
            n_bins_rounded += 1
        n_bins_rounded = int(n_bins_rounded)
        if self.n_bins != n_bins_rounded:
            raise ValueError("Inconsistent arguments: t_start ({t_start}), "
                             "t_stop ({t_stop}), bin_size ({bin_size}), and "
                             "n_bins ({n_bins})".format(
                t_start=self.t_start, t_stop=self.t_stop,
                bin_size=self.bin_size, n_bins=self.n_bins))
        if not isinstance(self.n_bins, int) or self.n_bins <= 0:
            raise TypeError("The number of bins ({}) must be a positive "
                            "integer".format(self.n_bins))

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
        t_start = self.t_start.rescale(self.bin_size.units).item()
        bin_edges = np.linspace(t_start, t_start + self.n_bins *
                                self.bin_size.item(),
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
        tmp_mat.data = tmp_mat.data.astype(bool)
        return tmp_mat

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
        else:  # Matrix on demand
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
        n_discarded = 0

        if not _check_neo_spiketrain(spiketrains):
            # a binned numpy array
            self._sparse_mat_u = sps.csr_matrix(spiketrains, dtype=np.int32)
            return n_discarded

        row_ids, column_ids = [], []
        # data
        counts = []

        for idx, st in enumerate(spiketrains):
            times = rescale_magnitude(st.times - self.t_start,
                                      units=self.bin_size.units)
            duration = (self.t_stop - self.t_start).rescale(
                self.bin_size.units).magnitude
            time_slice = np.logical_and(times >= 0, times <= duration)
            bins = times[time_slice] / self.bin_size.item()

            # shift spikes that are very close
            # to the right edge into the next bin
            rounding_error_indices = _detect_rounding_errors(bins,
                                                             self.tolerance)
            num_rounding_corrections = rounding_error_indices.sum()
            if num_rounding_corrections > 0:
                warnings.warn('Correcting {} rounding errors by shifting '
                              'the affected spikes into the following bin. '
                              'You can set tolerance=None to disable this '
                              'behaviour.'.format(num_rounding_corrections))
            bins[rounding_error_indices] += .5

            bins = bins.astype(np.int32)
            valid_bins = bins[bins < self.n_bins]
            n_discarded += len(bins) - len(valid_bins)
            f, c = np.unique(valid_bins, return_counts=True)
            column_ids.extend(f)
            counts.extend(c)
            row_ids.extend([idx] * len(f))
        csr_matrix = sps.csr_matrix((counts, (row_ids, column_ids)),
                                    shape=(len(spiketrains), self.n_bins),
                                    dtype=np.int32)
        self._sparse_mat_u = csr_matrix

        return n_discarded


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
