# -*- coding: utf-8 -*-
"""
This module implements objects that performs analyses that produce histograms.

These classes support a standardized flow for processing a data input, producing standardized outputs and provenance
information.

:copyright: Copyright 2014-2019 by the Elephant team, see `doc/authors.rst`.
:license: BSD, see LICENSE.txt for details.
"""
from .base import Analysis
from .statistics import SingleISI
from .graphics import BuffaloGraphic
from .helpers import _check_ndarray_size_and_types, _check_quantities_size_and_unit, _check_list_size_and_types
import quantities as pq
import numpy as np
import matplotlib.pyplot as plt


class SingleISIHistogram(Analysis):
    """
    Class that creates the histogram of the interspike intervals of a single spiketrain.

    Paraneters
    ----------
    isi: SingleISI
        The Buffalo SingleISI object with interspike intervals

    params: dict
        Input parameters for the histogram. The following are REQUIRED:

            1. `bins`: int, float, pq.Quantity, list, np.ndarray
                If int, this is the number of bins
                If single item Quantity array, this is the width of the bin in time
                If float, this is the width of the bin in the same time unit as the data
                If Quantity array, list or np.ndarray, this are the bin edges

        The following parameters are optional:

            1. `max_isi`: int, float, pq.Quantity
                This is the maximum time to consider when computing the histogram (0, max_isi)
                If this is not specified, the histogram will be computed between (0, maximum time from the data)

            2. `graph_params`: dict
                Params for a BuffaloGraphic object

            3. `auto_label`: boolean
                If True, and `graph_params` does not specify labels, the defaul label will be created for the X axis.
                If False, do not add labels unless specified in `graph_params`.
                Default X axis label is "Interspike interval (time unit)", where time unit is the same as the input

    kwargs: dict
        Other input data

    Attributes
    ----------
    bin_width
    n_bins
    histogram
    max_isi
    bin_edges
    plot
    """

    _name = "SingleISI Histogram"
    _description = "Interspike interval time histogram (for a single spike train)"

    _required_params = ("bins",)
    _required_types = {"bins": (int, float, pq.Quantity, np.ndarray, list),
                       "max_isi": (int, float, pq.Quantity),
                       "graph_params": (dict,)}

    _bin_edges = None       # Edges of the histogram bins
    _histogram = None       # Values of each bin
    _plot = None            # BuffaloGraphic object with the generated plot

    _ALLOWED_NDARRAY_TYPES = (np.int, np.float)
    _ALLOWED_LIST_TYPES = (int, float)

    def __init__(self, isi, params={}, **kwargs):
        super().__init__(isi, params=params, **kwargs)

    def _validate_data(self, isi, **kwargs):
        if not isinstance(isi, SingleISI):
            raise ValueError(f"Input must be a SingleISI Buffalo object. Type {type(isi)} given.")

    def _validate_parameters(self):
        bins = self.get_input_parameter('bins')
        if isinstance(bins, pq.Quantity):
            _check_quantities_size_and_unit(bins, ndim=1, unit=pq.s)
        elif isinstance(bins, np.ndarray):
            _check_ndarray_size_and_types(bins, ndim=1, dtypes=self._ALLOWED_NDARRAY_TYPES)
        elif isinstance(bins, list):
            _check_list_size_and_types(bins, self._ALLOWED_LIST_TYPES)
        elif isinstance(bins, (float, int)):
            if bins <= 0:
                raise ValueError("Bins value must be positive")

    def _process(self, isi, **kwargs):
        self._data = isi

        # Force histogram to start at zero
        time_range = (0, np.max(self._data.intervals.magnitude))

        # If max_isi is set, check if quantity needs conversion, otherwise use that value
        max_isi = self.get_input_parameter('max_isi')
        if max_isi is not None:
            if isinstance(max_isi, pq.Quantity):
                if max_isi.units != self._data.time_unit:
                    max_isi = max_isi.rescale(self._data.time_unit)
                max_isi = max_isi.magnitude
            time_range = (0, max_isi)

        bins = self.get_input_parameter('bins')

        # If bins is float, convert to a Quantity using the input data time unit
        # This will be used later to calculate the number of bins to get the desired width
        if isinstance(bins, float):
            bins = pq.Quantity(bins, self._data.time_unit)

        # If bins is a quantity array, check if it needs conversion.
        # Then returns the magnitude or calculate bin edges based on the single value
        if isinstance(bins, pq.Quantity):
            # First, rescale to the same unit as the input data, if needed
            if bins.units != self._data.time_unit:
                bins = bins.rescale(self._data.time_unit)
            if len(bins) == 1:
                # If single element, then this represents the bin width in time units
                # We need to get the number of bins and adjust max_isi if not integer
                n_bins = np.ceil(time_range[1] / bins.magnitude)
                time_range = (0, n_bins * bins.magnitude)
                bins = np.linspace(0, n_bins * bins.magnitude, n_bins+1)
            else:
                # We have the exact edges already defined. Use the magnitude
                bins = bins.magnitude

        # Create empty dict if no parameters were given for the graph
        graph_params = self.get_input_parameter('graph_params')
        if graph_params is None:
            graph_params = {}

        # If specific label for x axis was not informed, and `auto_label` was not defined as False,
        # create default label "Interspike interval (time unit)"
        if not 'xlabel' in graph_params:
            auto_label = self.get_input_parameter('auto_label')
            if auto_label is None or auto_label:
                graph_params['xlabel'] = f"Interspike interval ({self._data.time_unit.dimensionality})"

        # DISCUSS: use matplotlib to generate histogram data and plot, or np.histogram and hard code plotting object?
        self._plot = BuffaloGraphic(params=graph_params)
        graph_params = self._clean_graph_params(graph_params)
        self._plot.activate()
        self._histogram, self._bin_edges, _ = plt.hist(isi.intervals, bins=bins, range=time_range, **graph_params)
        self._plot.set_text_and_axes()

        # Convert to Quantity
        self._bin_edges = pq.Quantity(self._bin_edges, self._data.time_unit)

    @staticmethod
    def _clean_graph_params(graph_params):
        """
        Removes any keys from the dict that are not keyed arguments for Matplotlib plotting functions

        Parameters
        ----------
        graph_params: dict
            Keys `name`, `bins`, `range`, `xlabel`, `ylabel`, `title`, `axes` will be removed.

        Returns
        -------
        dict
            Modified dictionary
        """
        for key in ['name', 'bins', 'range', 'xlabel', 'ylabel', 'title', 'axes']:
            if key in graph_params:
                graph_params.pop(key)
        return graph_params

    @property
    def bin_width(self):
        """
        The widths of each bin (time quantity).

        Returns
        -------
        Quantity array
        """
        return np.diff(self._bin_edges)

    @property
    def n_bins(self):
        """
        Number of bins in the produced histogram.

        Returns
        -------
        int
        """
        return len(self._histogram)

    @property
    def histogram(self):
        """
        Bins of the historam.

        Returns
        -------
        np.ndarray
        """
        return self._histogram

    @property
    def bin_edges(self):
        """
        Edges of histogram bins.

        Returns
        -------
        Quantity array
        """
        return self._bin_edges

    @property
    def max_isi(self):
        """
        Maximum ISI time of the produced histogram.

        Returns
        -------
        pq.Quantity
        """
        return self._bin_edges[-1]

    @property
    def plot(self):
        """
        Object with the histogram plot.

        Returns
        -------
        BuffaloGraphic
        """
        return self._plot
