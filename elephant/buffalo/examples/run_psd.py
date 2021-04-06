"""
This example shows the functionality of the data analysis object for power
spectrum density (PSDObject), and the integration with the provenance tracker.
"""

import numpy as np
import quantities as pq

from elephant.buffalo.examples.utils import get_analog_signal
from elephant.spectral import welch_psd

import elephant.buffalo as buffalo

import matplotlib.pyplot as plt

from numpy import mean
import logging

logging.basicConfig(level=logging.INFO)


mean = buffalo.Provenance(inputs=['a'])(mean)
get_analog_signal = buffalo.Provenance(inputs=[])(get_analog_signal)


@buffalo.Provenance(inputs=['axes', 'freqs', 'psd'])
def plot_lfp_psd(axes, freqs, psd, title, freq_range=None, **kwargs):
    if freq_range is None:
        freq_range = [0, np.max(freqs)]

    indexes = np.where((freqs >= freq_range[0]) & (freqs <= freq_range[1]))

    axes.semilogy(freqs[indexes], psd[indexes], **kwargs)
    axes.set_ylabel(f"Power [{psd.dimensionality}]")
    axes.set_xlabel(f"Frequency [{freqs.dimensionality}]")
    axes.set_title(title)


def main():

    buffalo.activate()
    buffalo.USE_ANALYSIS_OBJECTS = True

    fig, axes = plt.subplots()

    signal = get_analog_signal(frequency=30*pq.Hz, n_channels=5, t_stop=3*pq.s,
                               sampling_rate=30000*pq.Hz, amplitude=50*pq.uV)

    if buffalo.USE_ANALYSIS_OBJECTS:
        obj = welch_psd(signal)
        avg_psd = mean(obj.psd, axis=0)
        plot_lfp_psd(axes, obj.frequencies, avg_psd, 'AnalysisObject',
                     freq_range=[0, 49])
    else:
        freqs, psd = welch_psd(signal)
        avg_psd = mean(psd, axis=0)
        plot_lfp_psd(axes, freqs, avg_psd, 'Tuple', freq_range=[0, 49])

    buffalo.save_graph("psd_plot.html", show=True)

    plt.show()


if __name__ == "__main__":
    main()
