import sys
from os.path import splitext

import neo
from elephant.statistics import isi, mean_firing_rate, fanofactor
from elephant.buffalo import provenance


@provenance.Provenance(inputs=[])
def load_data(session_file):
    """
    Loads data using BlackRockIO object.
    """
    _, file_ending = splitext(session_file)
    if file_ending == '.nev':
        nsx_to_load = None
    elif file_ending == '.ns2':
        nsx_to_load = 2
    elif file_ending == '.ns6':
        nsx_to_load = 6
    else:
        raise ValueError("File not supported!")

    session = neo.BlackrockIO(session_file, nsx_to_load=nsx_to_load)
    block = session.read_block(load_waveforms=True)
    block.create_relationship()

    return block


def main(session_file):
    provenance.activate()

    # Load data using any custom function

    block = load_data(session_file)

    # ISI times and mean firing rate (first spike train of the segment)

    isi_times = isi(block.segments[0].spiketrains[0], axis=0)

    firing_rate = mean_firing_rate(block.segments[0].spiketrains[0])

    # Fano factor of all spiketrains in the segment

    fano_factor = fanofactor(block.segments[0].spiketrains)

    provenance.save_graph('run_poster_demo.html', show=True)


if __name__ == "__main__":
    session_file = sys.argv[1]
    main(session_file)
