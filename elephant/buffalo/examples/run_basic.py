import os
import numpy as np

from elephant.statistics import isi, mean_firing_rate, fanofactor
from elephant.buffalo import provenance
from reachgraspio.reachgraspio import ReachGraspIO

SOURCE_DIR = "/home/koehler/datafiles/multielectrode_grasp/datasets"


np.array = provenance.Provenance(inputs=[0])(np.array)


@provenance.Provenance(inputs=[], file_input=['session_filename'])
def load_data(session_filename):
    """
    Loads R2G data using the custom BlackRockIO object ReachGraspIO.
    Sessions are stored in SOURCE_DIR.
    """
    file, ext = os.path.splitext(session_filename)
    nsx_to_load = int(ext[-1])
    file_path = os.path.dirname(session_filename)

    session = ReachGraspIO(file, nsx_to_load=nsx_to_load,
                           odml_directory=file_path,
                           verbose=False)

    block = session.read_block(load_waveforms=False, lazy=True)

    assert len(block.segments) == 1
    return block


def main(session_id):
    provenance.activate()

    session_filename = os.path.join(SOURCE_DIR, session_id)

    block = load_data(session_filename + ".ns2")

    # ISI times (first spike train of the segment)

    for index in range(len(block.segments[0].spiketrains)):
        isi_times = isi(block.segments[0].spiketrains[index], axis=0)
        firing_rate = mean_firing_rate(block.segments[0].spiketrains[index])

    fano_factor = fanofactor(block.segments[0].spiketrains)

    # Custom spike times

    generated_spike_times = np.array([1, 202, 405, 607, 904, 1100])

    isi_times2 = isi(generated_spike_times)

    # Analyze provenance track
    provenance.save_graph("basic.html", show=True)


if __name__ == "__main__":
    session_id = "i140703-001"
    main(session_id)
