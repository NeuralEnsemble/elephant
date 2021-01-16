"""
This module generates plots for analyses output by Elephant.
"""

import numpy as np
import quantities as pq
import matplotlib.pyplot as plt


def plot_time_histogram(histogram, time_unit=pq.s, title=""):
    """
    Function outside Buffalo/Elephant to visualize time histogram data.

    This function takes the result of `elephant.statistics.time_histogram`
    and plots the histogram, with time in `time_unit` units.

    Parameters
    ----------
    histogram : neo.AnalogSignal
        Object containing the histogram bins.
    time_unit : pq.Quantity, optional
        Desired unit for the plot time axis.
        If None, the current unit of `histogram` is used.
        Default: pq.s.
    title : str, optional
        Title for the plot object.
        Default: "".

    """

    # Rescale time axis if requested
    if time_unit is None:
        width = histogram.sampling_period.rescale(
            histogram.times.units).magnitude
        times = histogram.times.magnitude
        time_unit = histogram.times.units.dimensionality
    else:
        width = histogram.sampling_period.rescale(time_unit).magnitude
        times = histogram.times.rescale(time_unit).magnitude
        time_unit = time_unit.units.dimensionality

    # Create the plot (need to access `neo.AnalogSignal` properties according
    # to the information needed for the bar plot)
    plt.bar(times, histogram.squeeze().magnitude, align='edge', width=width)
    plt.xlabel("Time ({})".format(time_unit))
    plt.ylabel("Y Axis?")
    if len(title):
        plt.title(title)


def plot_time_histogram_object(histogram, time_unit=pq.s, title=""):
    """
    Function outside Buffalo/Elephant to visualize time histogram data,
    when input is a `buffalo.objects.AnalysisObject`.

    This function takes the result of refactored
    `elephant.statistics.time_histogram`, and plots the histogram,
    with time in `time_unit` units.

    Parameters
    ----------
    histogram : buffalo.objects.TimeHistogramObject
        Object containing the histogram.
    time_unit : pq.Quantity, optional
        Desired unit for the plot time axis.
        If None, the current unit of `histogram` is used.
        Default: pq.s.
    title : str, optional
        Title for the plot object.
        Default: "".

    """
    histogram.time_units = time_unit
    plt.bar(histogram.edges, histogram.bins, align='edge',
            width=histogram.bin_width)
    plt.ylabel(histogram.histogram_type)
    plt.xlabel("Time ({})".format(histogram.time_units.dimensionality))
    plt.title(title)


def plot_psth(histogram, event_time, time_unit=pq.s, title=""):
    """
    Function outside Buffalo/Elephant to generate a PSTH plot.

    The input is a `buffalo.objects.TimeHistogramObject`.
    The PSTH is built by offsetting the bars according to `event_time`.

    Event time is annotated into the object to store the value.
    A vertical line is added to the plot at the event time.

    Parameters
    ----------
    histogram : buffalo.objects.TimeHistogramObject
        Source time histogram from which the PSTH is built.
    event_time : pq.Quantity
        Time point that corresponds to the event.

    """
    histogram.time_units = time_unit
    event_time = event_time.rescale(histogram.time_units).magnitude
    histogram.annotate('PSTH', event_time)
    plt.bar(np.subtract(histogram.edges, event_time),
            histogram.bins, align='edge', width=histogram.bin_width)
    plt.ylabel(histogram.histogram_type)
    plt.xlabel("Time ({})".format(histogram.time_units.dimensionality))
    plt.axvline(0, linewidth=1, linestyle='dashed', color='black')
    plt.title(title)
