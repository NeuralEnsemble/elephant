import numpy as np
import quantities as pq

from elephant.buffalo.examples.utils import get_analog_signal
from elephant.buffalo.objects.base import AnalysisObject
from elephant.spectral import welch_psd

import elephant.buffalo

import matplotlib.pyplot as plt


def main():
    elephant.buffalo.USE_ANALYSIS_OBJECTS = True

    signal = get_analog_signal(frequency=30*pq.Hz, n_channels=5, t_stop=3*pq.s,
                               sampling_rate=30000*pq.Hz, amplitude=50*pq.uV)

    #freqs, psd = welch_psd(signal)
    obj = welch_psd(signal)
    if isinstance(obj, AnalysisObject):
        print(obj.params)

    freqs, psd = obj

    plot_freqs = np.where(freqs < 100)
    plt.plot(freqs[plot_freqs], psd[0, plot_freqs].flat)
    plt.show()


if __name__ == "__main__":
    main()
