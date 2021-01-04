import os
import numpy as np
import matplotlib.pyplot as plt
import quantities as pq

from elephant.statistics import time_histogram, mean_firing_rate
from elephant.buffalo import provenance
from elephant.spike_train_generation import homogeneous_poisson_process

import warnings


SOURCE_DIR = "/home/koehler/PycharmProjects/multielectrode_grasp/datasets"


@provenance.Provenance(inputs=["histogram"])
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
    fig = plt.figure()
    plt.bar(times, histogram.squeeze().magnitude, align='edge', width=width)
    plt.xlabel("Time ({})".format(time_unit))
    plt.ylabel("Y Axis?")
    if len(title):
        plt.title(title)
    return fig


def main():
    provenance.activate()

    # Custom spike times
    generated_spike_times = homogeneous_poisson_process(10*pq.Hz)

    hist = time_histogram(generated_spike_times, bin_size=1*pq.ms,
                          output='mean')

    figure = plot_time_histogram(hist, title="Example histogram")
    plt.show()

    figure.savefig('isi.png')

    provenance.print_history()

    provenance.save_graph("time_histogram.html", show=True)


if __name__ == "__main__":
    main()
