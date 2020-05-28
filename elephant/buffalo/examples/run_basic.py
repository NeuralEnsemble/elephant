import sys
sys.path.append("/home/koehler/PycharmProjects/reach_to_grasp/python")

import os
from reachgraspio.reachgraspio import ReachGraspIO
import numpy as np

from elephant.statistics import isi, mean_firing_rate, fanofactor
from elephant.buffalo import provenance


SOURCE_DIR = "/home/koehler/PycharmProjects/multielectrode_grasp/datasets"


@provenance.Provenance(inputs=[])
def load_data(session_id, channels):
    """
    Loads R2G data using the custom BlackRockIO object ReachGraspIO.
    Sessions are stored in SOURCE_DIR.
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
    provenance.activate()

    # Load data using any custom function.
    # Should track data provenance at some point

    channels = [10]
    block = load_data(session_id, channels)

    # ISI times (first spike train of the segment)

    isi_times = isi(block.segments[0].spiketrains[0], axis=0)

    firing_rate = mean_firing_rate(block.segments[0].spiketrains[0])
    fano_factor = fanofactor(block.segments[0].spiketrains)

    # Custom spike times
    generated_spike_times = np.array([1, 202, 405, 607, 904, 1100])
    isi_times2 = isi(generated_spike_times)

    # Analyze provenance track

    print("\n\n")
    provenance.print_history()
    provenance.save_graph("graph.md")
    provenance.save_prov_graph("prov_graph.png",
                               show_nary=False, show_element_attributes=False)


if __name__ == "__main__":
    session_id = "i140703-001"
    main(session_id)