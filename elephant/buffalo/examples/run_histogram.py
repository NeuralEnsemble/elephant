import sys
sys.path.append("/home/koehler/PycharmProjects/reach_to_grasp/python")

import os
from reachgraspio.reachgraspio import ReachGraspIO
import numpy as np
import matplotlib.pyplot as plt
import quantities as pq
from elephant.statistics import isi, mean_firing_rate, fanofactor, time_histogram
from elephant.buffalo import provenance

import sys
import inspect
import ast


SOURCE_DIR = "/home/koehler/PycharmProjects/multielectrode_grasp/datasets"


def plot_histogram(bin_values, edges):
    """
    Function outside Buffalo/Elephant to visualize histogram data
    This replicates matplotlib's `hist` function if passing the original data
    Here we pass the bin values and the edges, as produced by `np.histogram`
    """
    bar_widths = np.diff(edges)
    bar_centers = edges[:-1] + (bar_widths / 2)
    plt.bar(bar_centers, height=bin_values, align='center', width=bar_widths)


#@provenance.Provenance(inputs=[])
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


#@provenance.Provenance(inputs=["spiketrains"])
def get_spike_train(spiketrains, order):
    return spiketrains[order]


def get_code():
    return inspect.getsource(inspect.getmodule(inspect.currentframe()))


def main(session_id):
    provenance.activate()
    #print(inspect.getsource(inspect.getmodule(inspect.currentframe())))

    # Load data using any custom function. Should track data provenance at some point
    channels = [10]
    block = load_data(session_id, channels)

    # ISI histograms using Buffalo (first 2 spiketrains of the segment)

    isi_times = isi(block.segments[0].spiketrains[0], axis=0)
    isi_times2 = isi(get_spike_train(block.segments[0].spiketrains, 0))

    firing_rate = mean_firing_rate(block.segments[0].spiketrains[0])
    fano_factor = fanofactor(block.segments[0].spiketrains)

    # Custom spike times
    generated_spike_times = np.array([1, 202, 405, 607, 904, 1100])
    isi_times3 = isi(generated_spike_times)

    # Following part is outside Elephant, not captured
    bins, edges = np.histogram(isi_times.magnitude, bins=10)
    bins2, edges2 = np.histogram(isi_times2.magnitude, bins=10)
    bins3, edges3 = np.histogram(isi_times3, bins=10)

    # Do plotting
    # figure = plt.figure()
    # plt.subplot(3, 1, 1)
    # plt.title(str(block.segments[0].spiketrains[0].annotations))
    # plot_histogram(bins, edges)
    #
    # plt.subplot(3, 1, 2)
    # plt.title(str(block.segments[0].spiketrains[1].annotations))
    # plot_histogram(bins2, edges2)
    #
    # plt.subplot(3, 1, 3)
    # plt.title('Custom spike train')
    # plot_histogram(bins3, edges3)
    # # plt.show()
    #
    # figure.savefig('isi.png')

    #provenance.print_graph()
    #provenance.save_prov_graph(show_nary=False, show_element_attributes=False)


if __name__ == "__main__":
    session_id = "i140703-001"
    main(session_id)
