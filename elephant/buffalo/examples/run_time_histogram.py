import os
from reachgraspio.reachgraspio import ReachGraspIO
import numpy as np
import matplotlib.pyplot as plt
import quantities as pq
from elephant.statistics import time_histogram, psth
from elephant.statistics import PSTHObject
import neo

SOURCE_DIR = "/home/koehler/PycharmProjects/multielectrode_grasp/datasets"

TIME_SCALE = pq.s


def plot_psth(time_histogram, event_time, time_unit=pq.s, title=""):
    """
    Function outside Buffalo/Elephant to visualize histogram data
    """
    time_histogram.time_units = time_unit
    event_time = event_time.rescale(time_histogram.time_units).magnitude
    time_histogram.annotate('PSTH', event_time)
    plt.bar(np.subtract(time_histogram.edges, event_time),
            time_histogram.bins, align='edge', width=time_histogram.bin_width)
    plt.ylabel(time_histogram.histogram_type)
    plt.xlabel("Time ({})".format(time_histogram.time_units.dimensionality))
    plt.axvline(0, linewidth=1, linestyle='dashed', color='black')
    plt.title(title)


def plot_time_histogram_object(time_histogram, time_unit=pq.s, title=""):
    """
    Function outside Buffalo/Elephant to visualize histogram data
    """
    time_histogram.time_units = time_unit
    plt.bar(time_histogram.edges, time_histogram.bins,
            align='edge', width=time_histogram.bin_width)
    plt.ylabel(time_histogram.histogram_type)
    plt.xlabel("Time ({})".format(time_histogram.time_units.dimensionality))
    plt.title(title)


def plot_time_histogram(bin_values, time_unit=pq.s, title=""):
    """
    Function outside Buffalo/Elephant to visualize histogram data
    """
    if time_unit is None:
        width = bin_values.sampling_period.rescale(
            bin_values.times.units).magnitude
        times = bin_values.times.magnitude
        time_unit = bin_values.times.units.dimensionality
    else:
        width = bin_values.sampling_period.rescale(time_unit).magnitude
        times = bin_values.times.rescale(time_unit).magnitude
        time_unit = time_unit.units.dimensionality
    plt.bar(times,
            bin_values.squeeze().magnitude,
            align='edge',
            width=width)
    plt.xlabel("Time ({})".format(time_unit))
    plt.ylabel("Y Axis?")
    plt.title(title)


def load_data(session_id, channels):
    """
    Loads R2G data using the custom BlackRockIO object ReachGraspIO.
    Sessions are stored in SOURCE_DIR
    """
    session_filename = os.path.join(SOURCE_DIR, session_id)
    session = ReachGraspIO(session_filename)

    block_parameters = dict()
    block_parameters['units'] = 'all'
    block_parameters['load_events'] = True
    block_parameters['load_waveforms'] = False
    block_parameters['scaling'] = 'voltage'
    block_parameters['correct_filter_shifts'] = True
    block_parameters['nsx_to_load'] = None
    block_parameters['channels'] = channels

    block = session.read_block(**block_parameters)
    block.create_relationship()
    assert len(block.segments) == 1
    return block


def main(session_id):

    # Load data using any custom function.
    # Should track data provenance at some point.

    channels = [10]
    block = load_data(session_id, channels)

    # Time histogram parameters

    source = block.segments[0].spiketrains
    bin_size = 50 * pq.ms
    t_stop = 2000 * pq.ms

    # Using old function (neo.AnalogSignal return)
    # `old` was added to the function implementation to control the behavior

    time_hist_count = time_histogram(source, bin_size, output='counts',
                                     t_stop=t_stop, old=True)
    time_hist_mean = time_histogram(source, bin_size, output='mean',
                                    t_stop=t_stop, old=True)

    # Using new function (return AnalysisObject class)

    time_hist_obj_count = time_histogram(source, bin_size, output='counts',
                                         t_stop=t_stop)
    time_hist_obj_mean = time_histogram(source, bin_size, output='mean',
                                        t_stop=t_stop)

    # Compatibility with legacy code

    print("Checking if objects are compatible with `neo.AnalogSignal`")
    for obj in [time_hist_obj_count, time_hist_obj_mean]:
        print(obj.pid)
        print("Is neo.AnalogSignal? {}".format(isinstance(
            obj, neo.AnalogSignal)))

    # Do plotting - use old code function that expects neo.AnalogSignal

    figure_old_code = plt.figure()
    plt.subplot(2, 2, 1)
    plot_time_histogram(time_hist_count, title="AnalogSignal - counts",
                        time_unit=TIME_SCALE)

    plt.subplot(2, 2, 2)
    plot_time_histogram(time_hist_mean, title="AnalogSignal - mean",
                        time_unit=TIME_SCALE)

    plt.subplot(2, 2, 3)
    plot_time_histogram(time_hist_obj_count, title="Object - counts",
                        time_unit=TIME_SCALE)

    plt.subplot(2, 2, 4)
    plot_time_histogram(time_hist_obj_mean, title="Object - mean",
                        time_unit=TIME_SCALE)

    plt.tight_layout()

    # Do plotting - use new code function that takes advantage of
    # AnalysisObjects

    figure_new_code = plt.figure()
    plt.subplot(2, 2, 1)
    plot_time_histogram(time_hist_count, title="AnalogSignal - counts",
                        time_unit=TIME_SCALE)

    plt.subplot(2, 2, 2)
    plot_time_histogram(time_hist_mean, title="AnalogSignal - mean",
                        time_unit=TIME_SCALE)

    plt.subplot(2, 2, 3)
    plot_time_histogram_object(time_hist_obj_count,
                               title="New function - Object - counts",
                               time_unit=TIME_SCALE)

    plt.subplot(2, 2, 4)
    plot_time_histogram_object(time_hist_obj_mean,
                               title="New function - Object - mean",
                               time_unit=TIME_SCALE)

    plt.tight_layout()

    # #### Test PSTH object ####
    # New function implemented in `elephant.statistics.psth`

    event_time = bin_size * 5

    # Calculate a PSTH using a TimeHistogram object of the segment of interest,
    # that was calculated previously

    psth_object = PSTHObject.from_time_histogram(time_hist_obj_count,
                                                 event_time)

    # Instead of using a previous TimeHistogram of the same segment,
    # calculate a new PSTH from the source spike trains
    scratch_psth = psth(source, bin_size, event_time, output='counts',
                        t_stop=t_stop)

    figure_psth = plt.figure()

    # Plot a PSTH "manually" using a function that was implemented
    # for PSTHs specifically
    plt.subplot(1, 3, 1)
    plot_psth(time_hist_obj_count, event_time, title="PSTH - manual",
              time_unit=TIME_SCALE)

    # Use a function that expects TimeHistogram objects, but add a line
    # at t = 0 (event time)
    plt.subplot(1, 3, 2)
    plot_time_histogram_object(psth_object, title="PSTH - from TH",
                               time_unit=TIME_SCALE)
    plt.axvline(0, linewidth=1, linestyle='dashed', color='black')

    # Same as the second plot, but when using the PSTHHistogram object
    # calculated from the spike trains
    plt.subplot(1, 3, 3)
    plot_time_histogram_object(scratch_psth, title="PSTH - from scratch",
                               time_unit = TIME_SCALE)
    plt.axvline(0, linewidth=1, linestyle='dashed', color='black')

    # Save and display plots

    figure_old_code.savefig('th_old_code.png')
    figure_new_code.savefig('th_new_code.png')
    figure_psth.savefig('psth.png')

    plt.show()


if __name__ == "__main__":
    session_id = "i140703-001"
    main(session_id)
