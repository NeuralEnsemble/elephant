"""
Example showing the basic usage of the provenance tracking module in Elephant.
This example is based on the publicly available Reach2Grasp dataset, that
must be downloaded from the repository together with the supporting code.

The path to the code where the `reachgraspio.py` interface is located must be
provided in the `PYTHONPATH` environment variable when running the script.

The session file to use is passed as a parameter when running the script:

run_basic.py [path_to_file.nsX]
"""

import os
import sys

import numpy as np

from reachgraspio import ReachGraspIO
from elephant.statistics import isi, mean_firing_rate, fanofactor

from elephant.buffalo import provenance
from elephant.buffalo.examples.utils.files import get_file_name


# We need to apply the decorator in functions of other modules
# The `np.array` function does not have named arguments. Therefore, the
# `inputs` argument to the constructor is the argument order that corresponds
# to the input.
np.array = provenance.Provenance(inputs=[0])(np.array)


@provenance.Provenance(inputs=[], file_input=['session_filename'])
def load_data(session_filename):
    """
    Loads Reach2Grasp data using the custom BlackRockIO object ReachGraspIO.

    Parameters
    ----------
    session_filename : str
        Full path to the dataset file.

    Returns
    -------
    neo.Block
        Block container with the session data. The block is lazy loaded.

    """
    file, ext = os.path.splitext(session_filename)
    file_path = os.path.dirname(session_filename)

    session = ReachGraspIO(file, odml_directory=file_path,
                           verbose=False)

    block = session.read_block(load_waveforms=False, nsx_to_load=None,
                               load_events=True, lazy=False, channels='all',
                               units='all')

    return block


def main(session_filename):
    provenance.activate()

    # Load the data

    block = load_data(session_filename)

    # Compute some statistics using the first spiketrain in the segment

    isi_times = isi(block.segments[0].spiketrains[0], axis=0)
    firing_rate = mean_firing_rate(block.segments[0].spiketrains[0])

    # Compute the Fano factor using all spiketrains in the segment

    fano_factor = fanofactor(block.segments[0].spiketrains)

    # Generate an array representing artificial spike times and compute
    # statistics

    generated_spike_times = np.array([0.001, 0.202, 0.405, 0.607, 0.904, 1.1])
    isi_times2 = isi(generated_spike_times)
    firing_rate2 = mean_firing_rate(generated_spike_times)

    # Save the provenance graph
    provenance.save_graph(get_file_name(__file__, extension=".html"),
                          show=True)


if __name__ == "__main__":

    if len(sys.argv) < 2:
        raise ValueError("You must specify the path to the data set file.")

    main(sys.argv[1])
