"""
This example shows how Buffalo analysis objects store extended information on
the analysis, and how this would benefit a histogram plotting function.
"""

import matplotlib.pyplot as plt
import quantities as pq
import elephant.buffalo

from elephant.statistics import time_histogram

from elephant.buffalo.examples.utils import (get_spike_trains,
                                             plot_time_histogram,
                                             plot_time_histogram_object)


DEFAULT_TIME_UNIT = pq.s


def main(firing_rate, n_spiketrains, t_stop=2000*pq.ms, bin_size=2*pq.ms,
         time_unit=DEFAULT_TIME_UNIT, show_plot=False):

    # Generates spike data
    spiketrains = get_spike_trains(firing_rate, n_spiketrains, t_stop)

    # Time histogram parameters output
    print("Generating time histograms with bin size = {}".format(bin_size))
    print("Data is {} spike trains with rate {}".format(n_spiketrains,
                                                        firing_rate))
    print("Maximum spike time is {}\n\n".format(t_stop))

    # Using old `elephant.time_histogram` function, that returns
    # `neo.AnalogSignal`.

    time_hist_count = time_histogram(spiketrains, bin_size, output='counts')
    time_hist_mean = time_histogram(spiketrains, bin_size, output='mean')

    # Using new `elephant.statistics.time_histogram` function, that returns
    # `AnalysisObject` classes (`TimeHistogramObject` in the case of
    # `elephant.statistics.time_histogram`).
    # The `elephant.buffalo.USE_ANALYSIS_OBJECTS` flag is used to control
    # if Elephant functions output `AnalysisObject`s.

    elephant.buffalo.USE_ANALYSIS_OBJECTS = True

    time_hist_obj_count = time_histogram(spiketrains, bin_size,
                                         output='counts')
    time_hist_obj_mean = time_histogram(spiketrains, bin_size, output='mean')

    # Do plotting - use new code function that takes advantage of
    # `AnalysisObjects` properties.
    # The function will now know the Y axis of the histogram, without the need
    # to keep track of the function parameters.
    # Plots are compared to the output of the legacy function, which uses
    # `neo.AnalogSignal`, and that does not know the Y axis.

    figure_new_code = plt.figure()

    plt.subplot(2, 2, 1)
    plot_time_histogram(time_hist_count, title="neo.AnalogSignal - counts",
                        time_unit=time_unit)

    plt.subplot(2, 2, 2)
    plot_time_histogram(time_hist_mean, title="neo.AnalogSignal - mean",
                        time_unit=time_unit)

    plt.subplot(2, 2, 3)
    plot_time_histogram_object(time_hist_obj_count,
                               title="AnalysisObject - counts",
                               time_unit=time_unit)

    plt.subplot(2, 2, 4)
    plot_time_histogram_object(time_hist_obj_mean,
                               title="AnalysisObject - mean",
                               time_unit=time_unit)

    plt.tight_layout()

    figure_new_code.savefig('time_histogram_new_code.png')

    if show_plot:
        plt.show()


if __name__ == "__main__":
    firing_rate = 10 * pq.Hz
    n_spiketrains = 100
    bin_size = 5 * pq.ms
    show_plot = True
    main(firing_rate, n_spiketrains, bin_size=bin_size, show_plot=show_plot)
