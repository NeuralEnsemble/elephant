# -*- coding: utf-8 -*-
"""
This module allows to convert standard data representations
(e.g., a spike train stored as Neo SpikeTrain object)
into other representations useful to perform calculations on the data.
An example is the representation of a spike train as a sequence of 0-1 values
(binned spike train).


.. autosummary::
    :toctree: _toctree/conversion

    BinnedSpikeTrain
    BinnedSpikeTrainView
    binarize

Examples
********
>>> import neo
>>> import quantities as pq
>>> from elephant.conversion import BinnedSpikeTrain
>>> spiketrains = [
...   neo.SpikeTrain([0.5, 0.7, 1.2, 3.1, 4.3, 5.5, 6.7], t_stop=9, units='s'),
...   neo.SpikeTrain([0.1, 0.7, 1.2, 2.2, 4.3, 5.5, 8.0], t_stop=9, units='s')
... ]
>>> bst = BinnedSpikeTrain(spiketrains, bin_size=1 * pq.s)
>>> bst
BinnedSpikeTrain(t_start=0.0 s, t_stop=9.0 s, bin_size=1.0 s; shape=(2, 9))
>>> bst.to_array()
array([[2, 1, 0, 1, 1, 1, 1, 0, 0],
       [2, 1, 1, 0, 1, 1, 0, 0, 1]], dtype=int32)

Binarizing the binned matrix.

>>> bst.to_bool_array()
array([[ True,  True, False,  True,  True,  True,  True, False, False],
       [ True,  True,  True, False,  True,  True, False, False,  True]])

>>> bst_binary = bst.binarize()
>>> bst_binary
BinnedSpikeTrainView(t_start=0.0 s, t_stop=9.0 s, bin_size=1.0 s; shape=(2, 9))
>>> bst_binary.to_array()
array([[1, 1, 0, 1, 1, 1, 1, 0, 0],
       [1, 1, 1, 0, 1, 1, 0, 0, 1]], dtype=int32)

Slicing.

>>> bst.time_slice(t_stop=3.5 * pq.s)
BinnedSpikeTrainView(t_start=0.0 s, t_stop=3.0 s, bin_size=1.0 s; shape=(2, 3))
>>> bst[0, 1:-3]
BinnedSpikeTrainView(t_start=1.0 s, t_stop=6.0 s, bin_size=1.0 s; shape=(1, 5))

Generate a realisation of spike trains from the binned version.

>>> bst.to_spike_trains(spikes='center')
[<SpikeTrain(array([0.33333333, 0.66666667, 1.5       , 3.5       , 4.5       ,
       5.5       , 6.5       ]) * s, [0.0 s, 9.0 s])>,
<SpikeTrain(array([0.33333333, 0.66666667, 1.5       , 2.5       , 4.5       ,
       5.5       , 8.5       ]) * s, [0.0 s, 9.0 s])>]

Check the correctness of a spike trains realosation

>>> BinnedSpikeTrain(bst.to_spike_trains(), bin_size=bst.bin_size) == bst
True

Rescale the units of a binned spike train without changing the data.

>>> bst.rescale('ms')
>>> bst
BinnedSpikeTrain(t_start=0.0 ms, t_stop=9000.0 ms, bin_size=1000.0 ms;
shape=(2, 9))

:copyright: Copyright 2014-2020 by the Elephant team, see `doc/authors.rst`.
:license: BSD, see LICENSE.txt for details.
"""

from __future__ import division, print_function, unicode_literals

import math
import warnings

import neo
import numpy as np
import quantities as pq
import scipy.sparse as sps

from elephant.utils import is_binary, deprecated_alias, is_time_quantity, \
    check_neo_consistency, round_binning_errors

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
        Default: None
    t_start : float or pq.Quantity, optional
        The start time to use for the time points.
        If not specified, retrieved from the `t_start` attribute of
        `spiketrain`. If this is not present, defaults to `0`.  Any element of
        `spiketrain` lower than `t_start` is ignored.
        Default: None
    t_stop : float or pq.Quantity, optional
        The stop time to use for the time points.
        If not specified, retrieved from the `t_stop` attribute of
        `spiketrain`. If this is not present, defaults to the maximum value of
        `spiketrain`. Any element of `spiketrain` higher than `t_stop` is
        ignored.
        Default: None
    return_times : bool, optional
        If True, also return the corresponding time points.
        Default: False

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
    if units is None:
        return res, np.arange(t_start, t_stop + sampling_period,
                              sampling_period)
    return res, pq.Quantity(np.arange(t_start, t_stop + sampling_period,
                                      sampling_period), units=units)


###########################################################################
#
# Methods to calculate parameters, t_start, t_stop, bin size,
# number of bins
#
###########################################################################


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
    sparse_format : {'csr', 'csc'}, optional
        The sparse matrix format. By default, CSR format is used to perform
        slicing and computations efficiently.
        Default: 'csr'

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

    Notes
    -----
    There are four minimal configurations of the optional parameters which have
    to be provided, otherwise a `ValueError` will be raised:
      * t_start, n_bins, bin_size
      * t_start, n_bins, t_stop
      * t_start, bin_size, t_stop
      * t_stop, n_bins, bin_size

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
                 t_stop=None, tolerance=1e-8, sparse_format="csr"):
        if sparse_format not in ("csr", "csc"):
            raise ValueError(f"Invalid 'sparse_format': {sparse_format}. "
                             "Available: 'csr' and 'csc'")

        # Converting spiketrains to a list, if spiketrains is one
        # SpikeTrain object
        if isinstance(spiketrains, neo.SpikeTrain):
            spiketrains = [spiketrains]

        # The input params will be rescaled later to unit-less floats
        self.tolerance = tolerance
        self._t_start = t_start
        self._t_stop = t_stop
        self.n_bins = n_bins
        self._bin_size = bin_size
        self.units = None  # will be set later
        # Check all parameter, set also missing values
        self._resolve_input_parameters(spiketrains)
        # Now create the sparse matrix
        self.sparse_matrix = self._create_sparse_matrix(
            spiketrains, sparse_format=sparse_format)

    @property
    def shape(self):
        """
        The shape of the sparse matrix.
        """
        return self.sparse_matrix.shape

    @property
    def bin_size(self):
        """
        Bin size quantity.
        """
        return pq.Quantity(self._bin_size, units=self.units, copy=False)

    @property
    def t_start(self):
        """
        t_start quantity; spike times below this value have been ignored.
        """
        return pq.Quantity(self._t_start, units=self.units, copy=False)

    @property
    def t_stop(self):
        """
        t_stop quantity; spike times above this value have been ignored.
        """
        return pq.Quantity(self._t_stop, units=self.units, copy=False)

    @property
    def binsize(self):
        """
        Deprecated in favor of :attr:`bin_size`.
        """
        warnings.warn("'.binsize' is deprecated; use '.bin_size'",
                      DeprecationWarning)
        return self._bin_size

    @property
    def num_bins(self):
        """
        Deprecated in favor of :attr:`n_bins`.
        """
        warnings.warn("'.num_bins' is deprecated; use '.n_bins'",
                      DeprecationWarning)
        return self.n_bins

    def __repr__(self):
        return f"{type(self).__name__}(t_start={self.t_start}, " \
               f"t_stop={self.t_stop}, bin_size={self.bin_size}; " \
               f"shape={self.shape}, " \
               f"format={self.sparse_matrix.__class__.__name__})"

    def rescale(self, units):
        """
        Inplace rescaling to the new quantity units.

        Parameters
        ----------
        units : pq.Quantity or str
            New quantity units.

        Raises
        ------
        TypeError
            If the input units are not quantities.

        """
        if isinstance(units, str):
            units = pq.Quantity(1, units=units)
        if units == self.units:
            # do nothing
            return
        if not isinstance(units, pq.Quantity):
            raise TypeError("The input units must be quantities or string")
        scale = self.units.rescale(units).item()
        self._t_stop *= scale
        self._t_start *= scale
        self._bin_size *= scale
        self.units = units

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
        if self._bin_size is None:
            if self._t_start is None or self._t_stop is None:
                raise ValueError("To determine the bin size, both 't_start' "
                                 "and 't_stop' must be set")
            self._bin_size = (self._t_stop - self._t_start) / self.n_bins
        if self._t_start is None and self._t_stop is None:
            raise ValueError("Either 't_start' or 't_stop' must be set")
        if self._t_start is None:
            self._t_start = self._t_stop - self._bin_size * self.n_bins
        if self._t_stop is None:
            self._t_stop = self._t_start + self._bin_size * self.n_bins

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
        def get_n_bins():
            n_bins = (self._t_stop - self._t_start) / self._bin_size
            if isinstance(n_bins, pq.Quantity):
                n_bins = n_bins.simplified.item()
            n_bins = round_binning_errors(n_bins, tolerance=self.tolerance)
            return n_bins

        def check_n_bins_consistency():
            if self.n_bins != get_n_bins():
                raise ValueError(
                    "Inconsistent arguments: t_start ({t_start}), "
                    "t_stop ({t_stop}), bin_size ({bin_size}), and "
                    "n_bins ({n_bins})".format(
                        t_start=self.t_start, t_stop=self.t_stop,
                        bin_size=self.bin_size, n_bins=self.n_bins))

        def check_consistency():
            if self.t_start >= self.t_stop:
                raise ValueError("t_start must be smaller than t_stop")
            if not isinstance(self.n_bins, int) or self.n_bins <= 0:
                raise TypeError("The number of bins ({}) must be a positive "
                                "integer".format(self.n_bins))

        if not _check_neo_spiketrain(spiketrains):
            # a binned numpy matrix
            self.__resolve_binned(spiketrains)
            self.units = self._bin_size.units
            check_n_bins_consistency()
            check_consistency()
            self._t_start = self._t_start.rescale(self.units).item()
            self._t_stop = self._t_stop.rescale(self.units).item()
            self._bin_size = self._bin_size.rescale(self.units).item()
            return

        if self._bin_size is None and self.n_bins is None:
            raise ValueError("Either 'bin_size' or 'n_bins' must be given")

        try:
            check_neo_consistency(spiketrains,
                                  object_type=neo.SpikeTrain,
                                  t_start=self._t_start,
                                  t_stop=self._t_stop,
                                  tolerance=self.tolerance)
        except ValueError as er:
            # different t_start/t_stop
            raise ValueError(er, "If you want to bin over the shared "
                                 "[t_start, t_stop] interval, provide "
                                 "shared t_start and t_stop explicitly, "
                                 "which can be obtained like so: "
                                 "t_start, t_stop = elephant.utils."
                                 "get_common_start_stop_times(spiketrains)"
                             )

        if self._t_start is None:
            self._t_start = spiketrains[0].t_start
        if self._t_stop is None:
            self._t_stop = spiketrains[0].t_stop
        # At this point, all spiketrains share the same units.
        self.units = spiketrains[0].units

        # t_start and t_stop are checked to be time quantities in the
        # check_neo_consistency call.
        self._t_start = self._t_start.rescale(self.units).item()
        self._t_stop = self._t_stop.rescale(self.units).item()

        start_shared = max(st.t_start.item() for st in spiketrains)
        stop_shared = min(st.t_stop.item() for st in spiketrains)

        tolerance = self.tolerance
        if tolerance is None:
            tolerance = 0
        if self._t_start < start_shared - tolerance \
                or self._t_stop > stop_shared + tolerance:
            raise ValueError("'t_start' ({t_start}) or 't_stop' ({t_stop}) is "
                             "outside of the shared [{start_shared}, "
                             "{stop_shared}] interval".format(
                                 t_start=self.t_start, t_stop=self.t_stop,
                                 start_shared=start_shared,
                                 stop_shared=stop_shared))

        if self.n_bins is None:
            # bin_size is provided
            self._bin_size = self._bin_size.rescale(self.units).item()
            self.n_bins = get_n_bins()
        elif self._bin_size is None:
            # n_bins is provided
            self._bin_size = (self._t_stop - self._t_start) / self.n_bins
        else:
            # both n_bins are bin_size are given
            self._bin_size = self._bin_size.rescale(self.units).item()
            check_n_bins_consistency()

        check_consistency()

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
        bin_edges = np.linspace(self._t_start, self._t_start + self.n_bins *
                                self._bin_size,
                                num=self.n_bins + 1, endpoint=True)
        return pq.Quantity(bin_edges, units=self.units, copy=False)

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
        start = self._t_start + self._bin_size / 2
        stop = start + (self.n_bins - 1) * self._bin_size
        bin_centers = np.linspace(start=start,
                                  stop=stop,
                                  num=self.n_bins, endpoint=True)
        bin_centers = pq.Quantity(bin_centers, units=self.units, copy=False)
        return bin_centers

    def to_sparse_array(self):
        """
        Getter for sparse matrix with time points. Deprecated in favor of
        :attr:`sparse_matrix`.

        Returns
        -------
        scipy.sparse.csr_matrix or scipy.sparse.csc_matrix
            Sparse matrix, version with spike counts.

        See also
        --------
        scipy.sparse.csr_matrix
        to_array

        """
        warnings.warn("'.to_sparse_array()' function is deprecated; "
                      "use '.sparse_matrix' attribute directly",
                      DeprecationWarning)
        return self.sparse_matrix

    def to_sparse_bool_array(self):
        """
        Getter for boolean version of the sparse matrix, calculated from
        sparse matrix with counted time points.

        Returns
        -------
        scipy.sparse.csr_matrix or scipy.sparse.csc_matrix
            Sparse matrix, binary, boolean version.

        See also
        --------
        scipy.sparse.csr_matrix
        to_bool_array

        """
        # Return sparse Matrix as a copy
        spmat_copy = self.sparse_matrix.copy()
        spmat_copy.data = spmat_copy.data.astype(bool)
        return spmat_copy

    def __eq__(self, other):
        if not isinstance(other, BinnedSpikeTrain):
            return False
        if self.n_bins != other.n_bins:
            return
        dt_start = other.t_start.rescale(self.units).item() - self._t_start
        dt_stop = other.t_stop.rescale(self.units).item() - self._t_stop
        dbin_size = other.bin_size.rescale(self.units).item() - self._bin_size
        tol = 0 if self.tolerance is None else self.tolerance
        if any(abs(diff) > tol for diff in [dt_start, dt_stop, dbin_size]):
            return False
        sp1 = self.sparse_matrix
        sp2 = other.sparse_matrix
        if sp1.__class__ is not sp2.__class__ or sp1.shape != sp2.shape \
                or sp1.data.shape != sp2.data.shape:
            return False
        return (sp1.data == sp2.data).all() and \
            (sp1.indptr == sp2.indptr).all() and \
            (sp1.indices == sp2.indices).all()

    def copy(self):
        """
        Copies the binned sparse matrix and returns a view. Any changes to
        the copied object won't affect the original object.

        Returns
        -------
        BinnedSpikeTrainView
            A copied view of itself.
        """
        return BinnedSpikeTrainView(t_start=self._t_start,
                                    t_stop=self._t_stop,
                                    bin_size=self._bin_size,
                                    units=self.units,
                                    sparse_matrix=self.sparse_matrix.copy(),
                                    tolerance=self.tolerance)

    def __iter_sparse_matrix(self):
        spmat = self.sparse_matrix
        if isinstance(spmat, sps.csc_matrix):
            warnings.warn("The sparse matrix format is CSC. For better "
                          "performance, specify the CSR format while "
                          "constructing a "
                          "BinnedSpikeTrain(sparse_format='csr')")
            spmat = spmat.tocsr()
        # taken from csr_matrix.__iter__()
        i0 = 0
        for i1 in spmat.indptr[1:]:
            indices = spmat.indices[i0:i1]
            data = spmat.data[i0:i1]
            yield indices, data
            i0 = i1

    def __getitem__(self, item):
        """
        Returns a binned slice view of itself; `t_start` and `t_stop` will be
        set accordingly to the second slicing argument, if any.

        Parameters
        ----------
        item : int or slice or tuple
            Spike train and bin index slicing, passed to
            ``self.sparse_matrix``.

        Returns
        -------
        BinnedSpikeTrainView
            A slice of itself that carry the original data. Any changes to
            the returned binned sparse matrix will affect the original data.
        """
        # taken from csr_matrix.__getitem__
        row, col = self.sparse_matrix._validate_indices(item)
        spmat = self.sparse_matrix[item]
        if np.isscalar(spmat):
            # data with one element
            spmat = sps.csr_matrix(([spmat], ([0], [0])), dtype=spmat.dtype)

        if isinstance(col, (int, np.integer)):
            start, stop, stride = col, col + 1, 1
        elif isinstance(col, slice):
            start, stop, stride = col.indices(self.n_bins)
        else:
            raise TypeError(f"The second slice argument ({col}), which "
                            "corresponds to bin indices, must be either int "
                            "or slice.")
        t_start = self._t_start + start * self._bin_size
        t_stop = self._t_start + stop * self._bin_size
        bin_size = stride * self._bin_size
        bst = BinnedSpikeTrainView(t_start=t_start,
                                   t_stop=t_stop,
                                   bin_size=bin_size,
                                   units=self.units,
                                   sparse_matrix=spmat,
                                   tolerance=self.tolerance)
        return bst

    def __setitem__(self, key, value):
        """
        Changes the values of ``self.sparse_matrix`` according to `key` and
        `value`. A shortcut to ``self.sparse_matrix[key] = value``.

        Parameters
        ----------
        key : int or list or tuple or slice
            The binned sparse matrix keys (axes slice) to change.
        value : int or list or tuple or slice
            New values of the sparse matrix selection.
        """
        self.sparse_matrix[key] = value

    def time_slice(self, t_start=None, t_stop=None, copy=False):
        """
        Returns a view or a copied view of currently binned spike trains with
        ``(t_start, t_stop)`` time slice. Only valid (fully overlapping) bins
        are sliced.

        Parameters
        ----------
        t_start, t_stop : pq.Quantity or None, optional
            Start and stop times or Nones.
            Default: None
        copy : bool, optional
            Copy the sparse matrix or not.
            Default: False

        Returns
        -------
        BinnedSpikeTrainView
            A time slice of itself.
        """
        if not is_time_quantity(t_start, t_stop, allow_none=True):
            raise TypeError("t_start and t_stop must be quantities")
        if t_start is None and t_stop is None and not copy:
            return self
        if t_start is None:
            start_index = 0
        else:
            t_start = t_start.rescale(self.units).item()
            start_index = (t_start - self._t_start) / self._bin_size
            start_index = math.ceil(start_index)
            start_index = max(start_index, 0)
        if t_stop is None:
            stop_index = self.n_bins
        else:
            t_stop = t_stop.rescale(self.units).item()
            stop_index = (t_stop - self._t_start) / self._bin_size
            stop_index = round_binning_errors(stop_index,
                                              tolerance=self.tolerance)
            stop_index = min(stop_index, self.n_bins)
        stop_index = max(stop_index, start_index)
        spmat = self.sparse_matrix[:, start_index: stop_index]
        if copy:
            spmat = spmat.copy()
        t_start = self._t_start + start_index * self._bin_size
        t_stop = self._t_start + stop_index * self._bin_size
        bst = BinnedSpikeTrainView(t_start=t_start,
                                   t_stop=t_stop,
                                   bin_size=self._bin_size,
                                   units=self.units,
                                   sparse_matrix=spmat,
                                   tolerance=self.tolerance)
        return bst

    def to_spike_trains(self, spikes="random", as_array=False,
                        annotate_bins=False):
        """
        Generate spike trains from the binned spike train object. This function
        is inverse to binning such that

        .. code-block:: python

            BinnedSpikeTrain(binned_st.to_spike_trains()) == binned_st

        The object bin size is stored in resulting
        ``spiketrain.annotations['bin_size']``.

        Parameters
        ----------
        spikes : {"left", "center", "random"}, optional
            Specifies how to generate spikes inside bins.

              * "left": align spikes from left to right to have equal inter-
              spike interval;

              * "center": align spikes around center to have equal inter-spike
              interval;

              * "random": generate spikes from a homogenous Poisson process;
              it's the fastest mode.
            Default: "random"
        as_array : bool, optional
            If True, numpy arrays are returned; otherwise, wrap the arrays in
            `neo.SpikeTrain`.
            Default: False
        annotate_bins : bool, optional
            If `as_array` is False, this flag allows to include the bin index
            in resulting ``spiketrain.array_annotations['bins']``.
            Default: False

        Returns
        -------
        spiketrains : list of neo.SpikeTrain
            A list of spike trains - one possible realisation of spiketrains
            that could have been used as the input to `BinnedSpikeTrain`.
        """
        description = f"generated from {self.__class__.__name__}"
        shift = 0
        if spikes == "center":
            shift = 1
            spikes = "left"
        spiketrains = []
        for indices, spike_count in self.__iter_sparse_matrix():
            bin_indices = np.repeat(indices, spike_count)
            t_starts = self._t_start + bin_indices * self._bin_size
            if spikes == "random":
                spiketrain = np.random.uniform(low=0, high=self._bin_size,
                                               size=spike_count.sum())
                spiketrain += t_starts
                spiketrain.sort()
            elif spikes == "left":
                spiketrain = [np.arange(shift, count + shift) / (count + shift)
                              for count in spike_count]
                spiketrain = np.hstack(spiketrain) * self._bin_size
                spiketrain += t_starts
            else:
                raise ValueError(f"Invalid 'spikes' mode: '{spikes}'")
            # account for the last bin
            spiketrain = spiketrain[spiketrain <= self._t_stop]
            if not as_array:
                array_ants = None
                if annotate_bins:
                    array_ants = dict(bins=bin_indices)
                spiketrain = neo.SpikeTrain(spiketrain, t_start=self._t_start,
                                            t_stop=self._t_stop,
                                            units=self.units, copy=False,
                                            description=description,
                                            array_annotations=array_ants,
                                            bin_size=self.bin_size)
            spiketrains.append(spiketrain)
        return spiketrains

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
            return self.sparse_matrix.sum(axis=axis)
        n_spikes_per_row = self.sparse_matrix.sum(axis=axis)
        n_spikes_per_row = np.ravel(n_spikes_per_row)
        return n_spikes_per_row

    @property
    def spike_indices(self):
        """
        A list of lists for each spike train (i.e., rows of the binned matrix),
        that in turn contains for each spike the index into the binned matrix
        where this spike enters.

        In contrast to `self.sparse_matrix.nonzero()`, this function will
        report two spikes falling in the same bin as two entries.

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
        >>> print(x.sparse_matrix.nonzero()[1])
        [0 1 3 4 5 6]
        >>> print(x.to_array())
        [[2, 1, 0, 1, 1, 1, 1, 0, 0, 0]]

        """
        spike_idx = []
        for indices, spike_count in self.__iter_sparse_matrix():
            # Extract each non-zeros column index and how often it exists,
            # i.e., how many spikes fall in this column
            n_cols = np.repeat(indices, spike_count)
            spike_idx.append(n_cols)
        return spike_idx

    @property
    def is_binary(self):
        """
        Returns True if the sparse matrix contains binary values only.
        Beware, that the function does not know if the input was binary
        because e.g `to_bool_array()` was used before or if the input is just
        sparse (i.e. only one spike per bin at maximum).

        Returns
        -------
        bool
            True for binary input, False otherwise.
        """

        return is_binary(self.sparse_matrix.data)

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
        return self.to_array(dtype=bool)

    def to_array(self, dtype=None):
        """
        Returns a dense matrix, calculated from the sparse matrix, with counted
        time points of spikes. The rows correspond to spike trains and the
        columns correspond to bins in a `BinnedSpikeTrain`.
        Entries contain the count of spikes that occurred in the given bin of
        the given spike train.

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
        array = self.sparse_matrix.toarray()
        if dtype is not None:
            array = array.astype(dtype)
        return array

    def binarize(self, copy=True):
        """
        Clip the internal array (no. of spikes in a bin) to `0` (no spikes) or
        `1` (at least one spike) values only.

        Parameters
        ----------
        copy : bool, optional
            If True, a **shallow** copy - a view of `BinnedSpikeTrain` - is
            returned with the data array filled with zeros and ones. Otherwise,
            the binarization (clipping) is done in-place. A shallow copy
            means that :attr:`indices` and :attr:`indptr` of a sparse matrix
            is shared with the original sparse matrix. Only the data is copied.
            If you want to perform a deep copy, call
            :func:`BinnedSpikeTrain.copy` prior to binarizing.
            Default: True

        Returns
        -------
        bst : BinnedSpikeTrain or BinnedSpikeTrainView
            A (view of) `BinnedSpikeTrain` with the sparse matrix data clipped
            to zeros and ones.

        """
        spmat = self.sparse_matrix
        if copy:
            data = np.ones(len(spmat.data), dtype=spmat.data.dtype)
            spmat = spmat.__class__(
                (data, spmat.indices, spmat.indptr),
                shape=spmat.shape, copy=False)
            bst = BinnedSpikeTrainView(t_start=self._t_start,
                                       t_stop=self._t_stop,
                                       bin_size=self._bin_size,
                                       units=self.units,
                                       sparse_matrix=spmat,
                                       tolerance=self.tolerance)
        else:
            spmat.data[:] = 1
            bst = self

        return bst

    @property
    def sparsity(self):
        """
        The sparsity of the sparse matrix computed as the no. of nonzero
        elements divided by the matrix size.

        Returns
        -------
        float
        """
        num_nonzero = self.sparse_matrix.data.shape[0]
        shape = self.sparse_matrix.shape
        size = shape[0] * shape[1]
        return num_nonzero / size

    def _create_sparse_matrix(self, spiketrains, sparse_format):
        """
        Converts `neo.SpikeTrain` objects to a scipy sparse matrix, which
        contains the binned spike times, and
        stores it in :attr:`sparse_matrix`.

        Parameters
        ----------
        spiketrains : neo.SpikeTrain or list of neo.SpikeTrain
            Spike trains to bin.

        """

        # The data type for numeric values
        data_dtype = np.int32

        if sparse_format == 'csr':
            sparse_format = sps.csr_matrix
        else:
            # csc
            sparse_format = sps.csc_matrix

        if not _check_neo_spiketrain(spiketrains):
            # a binned numpy array
            sparse_matrix = sparse_format(spiketrains, dtype=data_dtype)
            return sparse_matrix

        # Get index dtype that can accomodate the largest index
        # (this is the same dtype that will be used for the index arrays of the
        #  sparse matrix, so already using it here avoids array duplication)
        shape = (len(spiketrains), self.n_bins)
        numtype = np.int32
        if max(shape) > np.iinfo(numtype).max:
            numtype = np.int64

        row_ids, column_ids = [], []
        # data
        counts = []
        n_discarded = 0

        # all spiketrains carry the same units
        scale_units = 1 / self._bin_size
        for idx, st in enumerate(spiketrains):
            times = st.magnitude
            times = times[(times >= self._t_start) & (
                times <= self._t_stop)] - self._t_start
            bins = times * scale_units

            # shift spikes that are very close
            # to the right edge into the next bin
            bins = round_binning_errors(bins, tolerance=self.tolerance)
            valid_bins = bins[bins < self.n_bins]
            n_discarded += len(bins) - len(valid_bins)
            f, c = np.unique(valid_bins, return_counts=True)
            # f inherits the dtype np.int32 from bins, but c is created in
            # np.unique with the default int dtype (usually np.int64)
            c = c.astype(data_dtype)
            column_ids.append(f)
            counts.append(c)
            row_ids.append(np.repeat(idx, repeats=len(f)).astype(numtype))

        if n_discarded > 0:
            warnings.warn("Binning discarded {} last spike(s) of the "
                          "input spiketrain".format(n_discarded))

        # Stacking preserves the data type. In any case, while creating
        # the sparse matrix, a copy is performed even if we set 'copy' to False
        # explicitly (however, this might change in future scipy versions -
        # this depends on scipy csr matrix initialization implementation).
        counts = np.hstack(counts)
        column_ids = np.hstack(column_ids)
        row_ids = np.hstack(row_ids)

        sparse_matrix = sparse_format((counts, (row_ids, column_ids)),
                                      shape=shape, dtype=data_dtype,
                                      copy=False)

        return sparse_matrix


class BinnedSpikeTrainView(BinnedSpikeTrain):
    """
    A view of :class:`BinnedSpikeTrain`.

    This class is used to avoid deep copies in several functions of a binned
    spike train object like :meth:`BinnedSpikeTrain.binarize`,
    :meth:`BinnedSpikeTrain.time_slice`, etc.

    Parameters
    ----------
    t_start, t_stop : float
        Unit-less start and stop times that share the same units.
    bin_size : float
        Unit-less bin size that was used used in binning the `sparse_matrix`.
    units : pq.Quantity
        The units of input spike trains.
    sparse_matrix : scipy.sparse.csr_matrix
        Binned sparse matrix.
    tolerance : float or None, optional
        The tolerance property of the original `BinnedSpikeTrain`.
        Default: 1e-8

    Warnings
    --------
    This class is an experimental feature.
    """

    def __init__(self, t_start, t_stop, bin_size, units, sparse_matrix,
                 tolerance=1e-8):
        self._t_start = t_start
        self._t_stop = t_stop
        self._bin_size = bin_size
        self.n_bins = sparse_matrix.shape[1]
        self.units = units.copy()
        self.sparse_matrix = sparse_matrix
        self.tolerance = tolerance


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
