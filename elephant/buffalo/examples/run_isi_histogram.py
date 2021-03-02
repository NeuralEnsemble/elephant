import sys

import os
import numpy as np


# matplotlib.use('qt5agg')

import matplotlib.pyplot as plt
import quantities as pq

from reachgraspio import ReachGraspIO

from elephant.statistics import isi, mean_firing_rate, fanofactor
from elephant.spike_train_generation import homogeneous_poisson_process

from elephant.buffalo import provenance
from elephant.buffalo.examples.utils.files import get_file_name

import warnings

plt.Figure.savefig = \
    provenance.Provenance(inputs=['self'],
                          file_output=['fname'])(plt.Figure.savefig)


@provenance.Provenance(inputs=["isi_times"])
def plot_isi_histograms(grid, *isi_times, bin_size=2*pq.ms, max_time=500*pq.ms,
                        titles=None):
    """
    Function outside Buffalo/Elephant to generate multiple ISI histograms.
    This replicates matplotlib's `hist` function.

    Parameters
    ----------
    grid : tuple
        Subplot grid as used in :func:`matplotlib.subplots` function.
    isi_times : tuple of np.ndarray or pq.Quantity
        All the ISIs that are going to be plotted. They are the output of
        :func:`elephant.statistics.isi` function. If a `np.ndarray` is passed,
        ISIs should be in the same unit as `bin_size`.
    bin_size : pq.Quantity, optional
        The bin size of the ISI histogram. This will be the unit of the time
        axis. If `isi_times` has a `pq.Quantity` using another unit, it will be
        rescaled.
        Default: 2 * pq.ms
    max_time : pq.Quantity, optional
        Maximum time that will be displayed in the histogram. If unit is
        different than `bin_size`, it will be rescaled.
        Default: 250 * pq.ms
    titles : list, optional
        If not None, it must specify the title for each of the individual
        plots, in the same order as `isi_times`.
        Default: None

    Returns
    -------
    fig : plt.Figure
        Figure object with the plot.
    axes : plt.Axes or np.ndarray of plt.Axes
        If more than one subplot was created, this is an array with the axes of
        every individual subplot. If only one plot was created, this is the
        `plt.Axes` object of the plot.

    Raises
    ------
    TypeError
        If any item in `isi_times` is not `pq.Quantity` or `np.ndarray`.
    ValueError
        If `titles` is not None, and the length is different than `isi_times`.

    Warns
    -----
    UserWarning
        If an `np.ndarray` is passed in `isi_times`, warns that the values are
        assumed to be in the same unit as `bin_size`.

    See Also
    --------
    matplotlib.subplots   for details regarding subplots

    """
    if titles is not None:
        if len(titles) != len(isi_times):
            raise ValueError("The number of items in `titles` must be the"
                             "same as in `isi_times`!")
    fig, axes = plt.subplots(*grid)
    upper_bound = max_time.rescale(bin_size.units).magnitude.item()
    step = bin_size.magnitude.item()

    for index, isi_time in enumerate(isi_times):
        if isinstance(isi_time, pq.Quantity):
            times = isi_time.rescale(bin_size.units).magnitude
        elif isinstance(isi_time, np.ndarray):
            warnings.warn("`np.ndarray` assumed to be in units '{}'".format(
                bin_size.dimensionality))
            times = isi_time
        else:
            raise TypeError("ISI is not `pq.Quantity` or `np.ndarray`!")

        edges = np.arange(0, upper_bound, step)

        bins, _ = np.histogram(times, bins=edges)
        bar_widths = np.diff(edges)
        axes[index].bar(edges[:-1], height=bins, align='edge',
                        width=bar_widths)
        axes[index].set_xlabel("Inter-spike interval ({})".format(
            bin_size.dimensionality.string))
        axes[index].set_ylabel("Count")
        if titles is not None:
            axes[index].set_title(titles[index])

    fig.tight_layout()
    return fig, axes


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

    # ISI histograms using Buffalo (first 2 spiketrains of the segment)
    titles = []
    isis = []
    for idx in range(2):
        isi_times = isi(block.segments[0].spiketrains[idx])
        isis.append(isi_times)
        titles.append(str(block.segments[0].spiketrains[idx].annotations))

    fano_factor = fanofactor(block.segments[0].spiketrains)

    # Custom spike times
    generated_spike_times = homogeneous_poisson_process(50*pq.Hz,
                                                        as_array=True)
    isi_times3 = isi(generated_spike_times)
    isis.append(isi_times3)
    titles.append("Generated spike train")

    # Do plotting
    figure, axes = plot_isi_histograms((len(isis), 1), *isis, titles=titles)

    plt.show()

    figure.savefig("isi.png")

    # provenance.print_history()
    provenance.save_graph(get_file_name(__file__, extension=".html"),
                          show=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError("You must specify the path to the data set file.")

    main(sys.argv[1])
