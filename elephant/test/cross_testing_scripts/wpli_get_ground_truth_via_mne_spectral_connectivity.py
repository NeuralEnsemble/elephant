import os

import matplotlib.pyplot as plt
import numpy as np
import quantities as pq
import scipy
from mne.connectivity import spectral_connectivity
from scipy.io import savemat


def main():
    """
    This script calculates a WPLI ground-truth with
    MNE's spectral_connectivity(), which uses multitaper for FFT, so that
    just certain frequencies will be compared to the results of
    weighted_phase_lag_index(), which uses conventional FFT.
    """
    # Load first & second data file
    filename1 = os.path.sep.join(['artificial_LFPs_1.mat'])
    dataset1 = scipy.io.loadmat(filename1, squeeze_me=True)

    filename2 = os.path.sep.join(['artificial_LFPs_2.mat'])
    dataset2 = scipy.io.loadmat(filename2, squeeze_me=True)

    # get the relevant values
    lfps1 = dataset1['lfp_matrix'] * pq.uV
    sf1 = dataset1['sf'] * pq.Hz
    lfps2 = dataset2['lfp_matrix'] * pq.uV

    # data-shape: epochs/trial X signals X times
    lfp_both_signals = np.hstack((lfps1.magnitude, lfps2.magnitude))
    lfp_both_signals = lfp_both_signals.reshape(np.shape(lfps1)[0], 2,
                                                np.shape(lfps1)[1])

    wpli, freqs, times, n_epochs, n_tapers = spectral_connectivity(
        data=lfp_both_signals, method='wpli', sfreq=sf1.magnitude,
        verbose=True, mode='fourier', fmin=0)

    # get relevant relation: first signal compared to the second one
    wpli = wpli[1][0][0:]
    freqs = freqs[0:]

    fig, ax = plt.subplots(1, 1, figsize=(10, 8), num=1)
    fig.suptitle("Weighted Phase Lag Index - Ground Truth from MNE", size=20)
    ax.plot(freqs, wpli, label="WPLI")
    ax.set_xlabel('f (Hz)', size=16)
    ax.legend(fontsize=16, framealpha=0)
    plt.show()

    np.savetxt("ground_truth_WPLI_from_MNE_spectral_connectivity"
               "_with_artificial_LFPs_multitaped.csv",
               wpli, delimiter=",", fmt="%20.20e")


if __name__ == '__main__':
    main()
