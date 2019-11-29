import os
import reachgraspio.reachgraspio as r2g
import numpy as np
import matplotlib.pyplot as plt
import quantities as pq
from elephant.buffalo.statistics import SingleISI
from elephant.buffalo.histogram import SingleISIHistogram
from elephant.statistics import isi


SOURCE_DIR = "/home/koehler/PycharmProjects/multielectrode_grasp/datasets"


def main(session_id):

    # Our parameters for ISI histogram
    channels = [10]
    bins = 10
    session_filename = os.path.join(SOURCE_DIR, session_id)

    # Read data (should be a Buffalo object later)
    # odml_dir = SOURCE_DIR  # FIXME: some error in the current r2g implementation
    session = r2g.ReachGraspIO(session_filename, odml_directory=None)

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

    # ISI using Buffalo

    channel10unit1 = SingleISI(block.segments[0].spiketrains[0], params={'time_unit': pq.s})
    hist_channel10unit1 = SingleISIHistogram(channel10unit1, params={'bins': pq.Quantity(np.array([0,0.2,0.4,0.8,1.2, 2.4, 4.8, 6]) * 1000, pq.ms)})

    plt.show()

    # Get some statistics
    print(f"Median ISI time: {str(channel10unit1.median())}")
    print(f"Histogram bin widths: {hist_channel10unit1.bin_width}")
    print(f"Histogram bin edges: {hist_channel10unit1.bin_edges}")
    print(f"Histogram bin count: {hist_channel10unit1.n_bins}")
    print(f"Histogram: {hist_channel10unit1.histogram}")


if __name__ == "__main__":
    session_id = "i140703-001"
    main(session_id)
