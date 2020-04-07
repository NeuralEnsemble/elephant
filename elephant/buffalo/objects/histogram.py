"""
This module implements a classes describing histogram analysis objects
in Elephant.

:copyright: Copyright 2014-2020 by the Elephant team, see `doc/authors.rst`.
:license: BSD, see LICENSE.txt for details.
"""

import numpy as np
import quantities as pq
import neo
from functools import wraps
from copy import deepcopy
from .base import AnalysisObject


class HistogramObject(AnalysisObject):
    """
    Base class for histograms.
    """

    @property
    def bin_size(self):
        raise NotImplementedError

    @property
    def edges(self):
        raise NotImplementedError

    @property
    def bins(self):
        raise NotImplementedError

    @property
    def bin_width(self):
        if isinstance(self.bin_size, pq.Quantity):
            return self.bin_size.magnitude
        return self.bin_size


class TimeHistogramObject(HistogramObject, neo.AnalogSignal):
    """
    Class to store outputs of the `elephant.statistics.time_histogram`
    function.

    Parameters
    ----------
    bins : np.ndarray or pq.Quantity
        Values of the histogram bins.
    bin_size : int or float or pq.Quantity
        Size of the bin.
    units : pq.Quantity
        Unit of `bins`.
        Default: pq.dimensionless.
    histogram_type : {'counts', 'mean', 'rate'}
        Type of histogram that bins represent.
        Default: 'counts'.
    t_start : pq.Quantity
        Starting time of the histogram.
        Default: None.
    t_stop : pq.Quantity
        End time of the histogram.
        Default: None.
    binary : boolean
        If source spike trains were binned.
        Default: None.
    warnings_raised : bool
        True if `elephant.statistics.time_histogram` raised warnings,
        False otherwise.
        Default: False.

    See Also
    --------
    elephant.statistics.time_histogram
    neo.AnalogSignal

    """

    _histogram_type = None
    _binary = None
    _t_stop = None

    _time_units = None

    def __new__(cls, bins, bin_size, units=pq.dimensionless,
                histogram_type='counts', t_start=None, t_stop=None,
                binary=None, warnings_raised=False):

        analysis_object = HistogramObject.__new__(HistogramObject)
        signal_object = neo.AnalogSignal.__new__(cls,
                                                 signal=bins,
                                                 sampling_period=bin_size,
                                                 units=units, t_start=t_start)
        signal_object.__dict__.update(deepcopy(analysis_object.__dict__))
        return signal_object

    def __init__(self, bins, bin_size, units=pq.dimensionless,
                 histogram_type='counts', t_start=None, t_stop=None,
                 binary=None, warnings_raised=False, **kwargs):

        self._histogram_type = histogram_type
        self._binary = binary
        self._t_stop = t_stop
        self._warn_raised = warnings_raised

        # Constructs the original `neo.AnalogSignal` that the `time_histogram`
        # function returns
        super().__init__(signal=bins, sampling_period=bin_size, units=units,
                         t_start=t_start)

    # Properties that depend on the function parameters/execution

    @property
    def histogram_type(self):
        return self._histogram_type

    @property
    def binary(self):
        return self._binary

    @property
    def bin_size(self):
        return self.sampling_period

    @property
    def time_start(self):
        return self.t_start

    @property
    def time_stop(self):
        return self._t_stop

    # Interface properties

    @property
    def time_units(self):
        if self._time_units is None:
            return self.times.units
        else:
            return self._time_units

    @time_units.setter
    def time_units(self, unit):
        self._time_units = unit

    @property
    def bin_width(self):
        if self._time_units is None:
            return self.bin_size.rescale(self.times.units)
        else:
            return self.bin_size.rescale(self._time_units)

    def _get_rescaled_edges(self):
        if self._time_units is None:
            return self.times.magnitude
        else:
            return self.times.rescale(self._time_units).magnitude

    @property
    def edges(self):
        return self._get_rescaled_edges()

    @property
    def bins(self):
        return self.squeeze().magnitude


class PSTHObject(TimeHistogramObject):
    """
    Class for storing peristimulus time histograms (PSTH) produced by
    `elephant.statistics.psth`.

    Parameters
    ----------
    bins : np.ndarray or pq.Quantity
        Values of the histogram bins.
    bin_size : int or float or pq.Quantity
        Size of the bin.
    units : pq.Quantity
        Unit of `bins`.
        Default: pq.dimensionless.
    histogram_type : {'counts', 'mean', 'rate'}
        Type of histogram that bins represent.
        Default: 'counts'.
    t_start : pq.Quantity
        Starting time of the histogram.
        Default: None.
    t_stop : pq.Quantity
        End time of the histogram.
        Default: None.
    binary : boolean
        If source spike trains were binned.
        Default: None.
    warnings_raised : bool
        True if `elephant.statistics.time_histogram` raised warnings,
        False otherwise.
        Default: False.
    """
    def __new__(cls, bins, bin_size, event_time, event_label=None,
                units=pq.dimensionless, histogram_type='counts',
                t_start=None, t_stop=None, binary=None,
                warnings_raised=False, **kwargs):

        return super().__new__(cls, bins, bin_size, units=units,
                               histogram_type=histogram_type,
                               t_start=t_start, t_stop=t_stop, binary=binary,
                               warnings_raised=warnings_raised)

    def __init__(self, bins, bin_size, event_time, event_label=None,
                 units=pq.dimensionless, histogram_type='counts',
                 t_start=None, t_stop=None, binary=None,
                 warnings_raised=False):

        self._event_time = event_time
        self._event_label = event_label

        super().__init__(bins, bin_size, units=units,
                         histogram_type=histogram_type, t_start=t_start,
                         t_stop=t_stop, binary=binary,
                         warnings_raised=warnings_raised)

    @classmethod
    def from_time_histogram(cls, histogram, event_time, event_label=None):
        if isinstance(histogram, TimeHistogramObject):
            obj = cls(histogram.bins, histogram.bin_size, event_time,
                      event_label=event_label, units=histogram.units,
                      histogram_type=histogram.histogram_type,
                      t_start=histogram.time_start, t_stop=histogram.time_stop,
                      binary=histogram.binary,
                      warnings_raised=histogram.warnings_raised)
            return obj

    @property
    def event_label(self):
        return self._event_label

    @property
    def event_time(self):
        return self._event_time.rescale(self.time_units)

    @property
    def edges(self):
        rescaled_event = self.event_time.magnitude
        rescaled_edges = self._get_rescaled_edges()
        return np.subtract(rescaled_edges, rescaled_event)
