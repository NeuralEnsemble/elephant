import sys
sys.path.remove('/home/koehler/PycharmProjects/reach_to_grasp/python')
print(sys.path)

import os
from reachgraspio.reachgraspio import ReachGraspIO
import numpy as np

from elephant.statistics import isi, mean_firing_rate, fanofactor
from elephant.buffalo import provenance
import networkx as nx
import matplotlib.pyplot as plt
import pickle

SOURCE_DIR = "/home/koehler/PycharmProjects/multielectrode_grasp/datasets"

np.array = provenance.Provenance(inputs=[0])(np.array)


@provenance.Provenance(inputs=[], file_input=['session_filename'])
def load_data(session_filename, channels):
    """
    Loads R2G data using the custom BlackRockIO object ReachGraspIO.
    Sessions are stored in SOURCE_DIR.
    """
    file, ext = os.path.splitext(session_filename)
    nsx_to_load = int(ext[-1])
    file_path = os.path.dirname(session_filename)

    session = ReachGraspIO(file,
                           odml_directory=file_path,
                           verbose=False)

    block = session.read_block(load_waveforms=False, load_events=True,
                               nsx_to_load=None)

    assert len(block.segments) == 1
    return block


@provenance.Provenance(inputs=['data'])
def test(data):
    print(data)
    print(data.annotations)


def main(session_id):
    provenance.activate()

    # Load data using any custom function.
    # Should track data provenance at some point

    channels = [10, 11, 12]
    session_filename = os.path.join(SOURCE_DIR, session_id)
    block = load_data(session_filename + ".ns6", channels)

    test(block)

    # ISI times (first spike train of the segment)
    # sts = block.segments[0].spiketrains
    # for spiketrain in sts:
    #     isi_times = isi(spiketrain, axis=0)
    #     firing_rate = mean_firing_rate(spiketrain)

    for index in range(len(block.segments[0].spiketrains)):
        isi_times = isi(block.segments[0].spiketrains[index], axis=0)
        firing_rate = mean_firing_rate(block.segments[0].spiketrains[index])

    fano_factor = fanofactor(block.segments[0].spiketrains)

    # Custom spike times
    generated_spike_times = np.array([1, 202, 405, 607, 904, 1100])
    isi_times2 = isi(generated_spike_times)

    # Analyze provenance track

    print("\n\n")
    # provenance.print_history()
    # graph = provenance.get_graph()
    # plot_bokeh(graph)
    # # nx.draw_networkx(graph)
    # # plt.show()
    # # print("Done")
    provenance.save_graph("basic.html", show=True)

    # provenance.dump_provenance("test.pkl")
    #
    # history = pickle.load(open("test.pkl", 'rb'))
    # graph = provenance.get_graph(history)
    # provenance.save_graph("basic2.html", source=graph, show=True)




if __name__ == "__main__":
    session_id = "i140703-001"
    main(session_id)
