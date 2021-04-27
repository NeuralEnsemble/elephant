import os

import numpy as np
from scipy.io import savemat


def _generate_datasets_for_ground_truth_testing(ntrial=40, tlength=2.5,
                                                srate=250):
    """
    Generates simple sinusoidal, artificial LFP-datasets.

    This function simulates the recording of two LFP-signals over several
    trials, for a certain duration of time and with a specified sampling
    rate. These datasets will be used to calculate WPLI-ground-truth
    with the MATlAB package FieldTrip and its function
    ft_connectivity_wpli(), its wrapper ft_connectivityanalysis() and
    with MNEs spectral_connectivity(). The last two use both multitaper
    for FFT, therefore just certain frequencies will be compared.

    Parameters
    ----------
    ntrial: int
        Number of trials in the datasets.
    tlength: float
        Time length of the datasets in seconds.
    srate: float
        Sampling rate used to 'record' the signals in Hz.

    Returns
    -------
    None

    Notes
    -----
    Used versions of MATLAB & FieldTrip:
    MATLAB          Version 9.9         (2020b)
    FieldTrip       fieldtrip-20210128

    Instead of using the ft_connectivityanalysis() wrapper function, which
    expects preprocessed data, the ft_connectivity_wpli() is called
    directly. That's because the preprocessing functions of FieldTrip have
    no 'conventional' Fourier-Transformation (FFT) method, but among
    others a multitaper-FFT. Because this python-version of WPLI doesn't
    use multitaper, but conventional FFT, the utilized MATLAB script also
    uses the conventional FFT to preprocess the data and calculate the
    cross-spectrum, which will then be passed to the ft_connectivity_wpli.

    """
    np.random.seed(73)

    times = np.arange(0, tlength, 1 / srate)

    # lfps_1 & 2 will be used for
    # 1) calculating ground-truth with FieldTrips' ft_connectivity_wpli
    # 2) comparison to mutlitaper-approaches like
    # FieldTrips' ft_connectivityanalysis()
    # and MNEs' spectral_connectivity at certain frequencies
    lfps_1 = [np.cos(2 * 16 * np.pi * times + np.pi / 2) +
              np.cos(2 * 36 * np.pi * times + np.pi / 2) +
              np.cos(2 * 52 * np.pi * times) +
              np.cos(2 * 100 * np.pi * times) +
              np.cos(2 * 70 * np.pi * times + np.pi / 2 + np.pi * (i % 2)) +
              np.random.normal(loc=0.0, scale=1, size=len(times))
              for i in range(ntrial)]
    lfps_1 = np.stack(lfps_1, axis=0)

    lfps_2 = [np.cos(2 * 16 * np.pi * times) +
              np.cos(2 * 36 * np.pi * times) +
              np.cos(2 * 52 * np.pi * times + np.pi / 2) +
              np.cos(2 * 100 * np.pi * times + np.pi / 2) +
              np.cos(2 * 70 * np.pi * times) +
              np.random.normal(loc=0.0, scale=1, size=len(times))
              for _ in range(ntrial)]
    lfps_2 = np.stack(lfps_2, axis=0)

    # save artificial LFP-dataset to .mat files
    mdic_1 = {"lfp_matrix": lfps_1, "time": times, "sf": srate}
    mdic_2 = {"lfp_matrix": lfps_2, "time": times, "sf": srate}
    filename1_artificial = "artificial_LFPs_1.mat"
    filename2_artificial = "artificial_LFPs_2.mat"
    savemat(filename1_artificial, mdic_1)
    savemat(filename2_artificial, mdic_2)

    # save cross-spectrum of those two LFP-matrices
    fft1 = np.fft.rfft(lfps_1)
    fft2 = np.fft.rfft(lfps_2)
    cs = fft1 * np.conjugate(fft2)
    cs_dict = {"cross_spectrum_matrix": cs}
    cs_fname = "i140703-001_cross_spectrum_of_artificial_LFPs_1_and_2.mat"
    savemat(cs_fname, cs_dict)


def main():
    """Run generation function."""
    _generate_datasets_for_ground_truth_testing()


if __name__ == '__main__':
    main()
