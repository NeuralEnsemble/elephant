"""
This example shows how Buffalo analysis objects can be reused and how more
specific objects can be implemented building on more generic objects.
"""

import matplotlib.pyplot as plt
import quantities as pq
import elephant.buffalo

from elephant.statistics import time_histogram, psth
from elephant.statistics import PSTHObject

from elephant.buffalo.examples.utils import (get_spike_trains,
                                             plot_time_histogram_object,
                                             plot_psth)


DEFAULT_TIME_UNIT = pq.s


def main(firing_rate, n_spiketrains, t_stop=2000*pq.ms, bin_size=2*pq.ms,
         event_time=150*pq.ms, time_unit=DEFAULT_TIME_UNIT, show_plot=False):

    # Generates spike data
    spiketrains = get_spike_trains(firing_rate, n_spiketrains, t_stop)

    # PSTH parameters output
    print(f"Generating PSTH with bin size = {bin_size}")
    print(f"Event occurs at time = {event_time}")
    print(f"Data is {n_spiketrains} spike trains with rate {firing_rate}")
    print(f"Maximum spike time is {t_stop}\n\n")

    # Use new `elephant.statistics.time_histogram` function, that returns
    # `AnalysisObject` classes, to obtain a time histogram of the `spiketrains`

    elephant.buffalo.USE_ANALYSIS_OBJECTS = True

    time_hist_obj_count = time_histogram(spiketrains, bin_size,
                                         output="counts")

    # Creates PSTH using this previously calculated TimeHistogramObject
    psth_object = PSTHObject.from_time_histogram(time_hist_obj_count,
                                                 event_time)

    # Instead of using a previous TimeHistogram of the same segment,
    # calculate a new PSTH from the source spike trains.
    # A new function was implemented (`elephant.statistics.psth`)

    psth_object_from_spiketrains = psth(spiketrains, bin_size, event_time,
                                        output='counts')

    # Do plotting - both histograms should be similar

    figure_psth = plt.figure()

    # Plot a PSTH using a function that was implemented to use
    # `buffalo.objects.TimeHistogramObject`, but that has to implement the
    # offsets according to the event time
    plt.subplot(3, 1, 1)
    plot_psth(time_hist_obj_count, event_time,
              title="PSTH - from TimeHistogramObject",
              time_unit=time_unit)

    # Use a generic function used to plot
    # `buffalo.objects.TimeHistogramObject`, but now passing `PSTHObject` as
    # input.
    # A line is added at t = 0 to show the event time in the plot.
    # Since the function is plotting a PSTHObject, time t = 0 is automatically
    # set to the event time when accessing the edges of the histogram.
    plt.subplot(3, 1, 2)
    plot_time_histogram_object(psth_object,
                               title="PSTH - PSTHObject from previous time "
                                     "histogram",
                               time_unit=time_unit)
    plt.axvline(0, linewidth=1, linestyle="dashed", color="black")

    # Same as the second plot, but when using the PSTHObject calculated from
    # the spike trains. The plot must be the same.
    plt.subplot(3, 1, 3)
    plot_time_histogram_object(psth_object_from_spiketrains,
                               title="PSTH - PSTHObject from spiketrains",
                               time_unit=time_unit)
    plt.axvline(0, linewidth=1, linestyle="dashed", color="black")


    plt.tight_layout()

    figure_psth.savefig("psth.png")

    if show_plot:
        plt.show()


if __name__ == "__main__":
    firing_rate = 10 * pq.Hz
    n_spiketrains = 100
    bin_size = 10 * pq.ms
    show_plot = True
    main(firing_rate, n_spiketrains, bin_size=bin_size, show_plot=show_plot)
