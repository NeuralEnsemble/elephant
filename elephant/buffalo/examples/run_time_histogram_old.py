"""
This example shows basic functionality of the Buffalo analysis objects,
and compatibility with existing code that uses Elephant functions.
"""

import matplotlib.pyplot as plt
import quantities as pq
import neo

from elephant.statistics import time_histogram

from elephant.buffalo.examples.utils import (get_spike_trains,
                                             plot_time_histogram)


DEFAULT_TIME_UNIT = pq.s


def main(firing_rate, n_spiketrains, t_stop=2000*pq.ms,
         bin_size=50*pq.ms, time_unit=DEFAULT_TIME_UNIT):

    # Generates spike data
    spiketrains = get_spike_trains(firing_rate, n_spiketrains, t_stop)

    # Time histogram parameters output
    print(f"Generating time histograms with bin size = {bin_size}")
    print(f"Data is {n_spiketrains} spike trains with rate {firing_rate}")
    print(f"Maximum spike time is {t_stop}\n\n")

    # Using old `elephant.time_histogram` function, that returns
    # `neo.AnalogSignal`. `old` keyword parameter was added to the function
    # implementation to control the behavior

    time_hist_count = time_histogram(spiketrains, bin_size, output='counts',
                                     old=True)
    time_hist_mean = time_histogram(spiketrains, bin_size, output='mean',
                                    old=True)

    # Using new `elephant.statistics.time_histogram` function, that returns
    # `AnalysisObject` classes (`TimeHistogramObject` in the case of
    # `elephant.statistics.time_histogram`)

    time_hist_obj_count = time_histogram(spiketrains, bin_size,
                                         output='counts')
    time_hist_obj_mean = time_histogram(spiketrains, bin_size, output='mean')

    # Check compatibility with legacy code

    print("Checking if objects are compatible with `neo.AnalogSignal`")
    for obj in [time_hist_obj_count, time_hist_obj_mean]:
        print(f"Object {obj.pid}")    # AnalysisObjects base class has a PID
        print("Is neo.AnalogSignal? {}\n\n".format(isinstance(
            obj, neo.AnalogSignal)))

    print("Checking properties of the objects")
    print(f"Bin size script parameter: {bin_size}")
    print(f"Bin size object property value: {time_hist_obj_count.bin_size}")

    # Do plotting - use old code function that expects `neo.AnalogSignal`.
    # Legacy code should work with `AnalysisObject` types, and output
    # should be the same.

    figure_old_code = plt.figure()

    plt.subplot(2, 2, 1)
    plot_time_histogram(time_hist_count, title="neo.AnalogSignal - counts",
                        time_unit=time_unit)

    plt.subplot(2, 2, 2)
    plot_time_histogram(time_hist_mean, title="neo.AnalogSignal - mean",
                        time_unit=time_unit)

    plt.subplot(2, 2, 3)
    plot_time_histogram(time_hist_obj_count, title="AnalysisObject - counts",
                        time_unit=time_unit)

    plt.subplot(2, 2, 4)
    plot_time_histogram(time_hist_obj_mean, title="AnalysisObject - mean",
                        time_unit=time_unit)

    plt.tight_layout()

    figure_old_code.savefig('time_histogram_old_code.png')

    plt.show()


if __name__ == "__main__":
    firing_rate = 10 * pq.Hz
    n_spiketrains = 100
    main(firing_rate, n_spiketrains)
