# -*- coding: utf-8 -*-
"""
Unit tests for the spectral module.

:copyright: Copyright 2014-2026 by the Elephant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
"""

import functools
import unittest
import warnings

import neo.core
import numpy as np
import pytest
import quantities as pq
import scipy
import scipy.fft
import scipy.signal as spsig
from neo import AnalogSignal
from numpy.testing import assert_array_equal, assert_array_almost_equal
from packaging import version

import elephant.spectral
from elephant.datasets import download_datasets


class BinnedSpikeTrainPSDMixin:
    """
    Checks on the normalization of a PSD estimated from a `BinnedSpikeTrain`.

    These properties hold for every PSD estimation function that accepts a
    `BinnedSpikeTrain`, so they are asserted identically for each of them.
    A test case mixing these tests in has to provide class attributes
    `psd_function` and `removes_mean`, and `extract_channel` if the spectrum
    of a single spike train is not obtained by indexing the first dimension of
    the returned array.
    """

    # The PSD function under test, must be overwritten by the test case
    psd_function = None

    # Whether the PSD function removes the mean of the signal with its default
    # parameters, must be overwritten by the test case
    removes_mean = None

    @staticmethod
    def extract_channel(psd, channel):
        """
        Extracts a specific channel from a given PSD object.
        """
        return psd[channel]

    def _mean_psd_in_band(self, binned_st, low, high, channel=0):
        """
        Compute the mean power spectral density within a specified frequency band.
        """
        freqs, psd = self.psd_function(binned_st)
        spectrum = self.extract_channel(psd, channel)
        mask = (freqs >= low) & (freqs <= high)
        return np.mean(spectrum[mask]).rescale('Hz').magnitude

    def test_binned_spiketrain_normalization(self):
        """
        The spike count of each bin is divided by the bin size, so that the
        bin size cancels out and the asymptote of 2*rate depends neither on
        how finely the spike train was binned nor on the rate itself. A
        regression in the normalization would scale the estimate with a power
        of the bin size (e.g., by factors of 4 and 25 between the bin sizes
        used here) or of the rate. The band of 20-80 Hz lies below the lowest
        Nyquist frequency (100 Hz for 5 ms bins) and is used to estimate the
        frequency at large f.
        """
        np.random.seed(123)
        bin_sizes = (1 * pq.ms, 2 * pq.ms, 5 * pq.ms)

        for rate in (10 * pq.Hz, 50 * pq.Hz, 200 * pq.Hz):
            expected = 2 * rate.rescale('Hz').magnitude

            # The same spike train is binned in different ways, so that the
            # comparison across bin sizes is not obscured by differences
            # between realizations.
            spiketrain = \
                elephant.spike_train_generation.StationaryPoissonProcess(
                    rate, t_stop=20 * pq.s).generate_spiketrain()

            estimates = []
            for bin_size in bin_sizes:
                binned_st = elephant.conversion.BinnedSpikeTrain(
                    spiketrain, bin_size=bin_size)
                avg_psd = self._mean_psd_in_band(
                    binned_st, 20 * pq.Hz, 80 * pq.Hz)
                estimates.append(avg_psd)

                self.assertAlmostEqual(
                    avg_psd, expected, delta=0.15 * expected,
                    msg=f"rate {rate}, bin size {bin_size} deviates from "
                        f"2*rate")

            # The estimates must also agree with each other
            self.assertLess(
                max(estimates) / min(estimates), 1.15,
                msg=f"bin sizes disagree with each other at rate {rate}")

    def test_binned_spiketrain_multiple_spiketrains(self):
        """
        A `BinnedSpikeTrain` holding several spike trains must yield one
        spectrum per spike train, each with the same scaling.
        """
        np.random.seed(123)
        rate = 50 * pq.Hz
        expected = 2 * rate.rescale('Hz').magnitude
        n_spiketrains = 3

        spiketrains = elephant.spike_train_generation.StationaryPoissonProcess(
            rate, t_stop=20 * pq.s).generate_n_spiketrains(n_spiketrains)
        binned_st = elephant.conversion.BinnedSpikeTrain(
            spiketrains, bin_size=2 * pq.ms)

        freqs, psd = self.psd_function(binned_st)
        self.assertEqual(psd.shape[0], n_spiketrains)

        for channel in range(n_spiketrains):
            avg_psd = self._mean_psd_in_band(
                binned_st, 100 * pq.Hz, 250 * pq.Hz, channel=channel)
            self.assertAlmostEqual(avg_psd, expected, delta=0.15 * expected,
                                   msg=f"spike train {channel}")

    def test_binned_spiketrain_regular_peak(self):
        """
        A perfectly regular spike train of frequency f0 has a line spectrum
        with peaks at f0 and its harmonics. All harmonics carry the same
        power, so the maximum is searched in a band around f0 only.
        """
        f0 = 25.
        t_stop = 20.
        # spikes in the middle of the bins, so that binning is unambiguous
        spike_times = np.arange(0, t_stop, 1 / f0) + 0.0005
        spiketrain = neo.SpikeTrain(spike_times * pq.s,
                                    t_stop=t_stop * pq.s)

        for bin_size in (1 * pq.ms, 5 * pq.ms):
            binned_st = elephant.conversion.BinnedSpikeTrain(
                spiketrain, bin_size=bin_size)
            freqs, psd = self.psd_function(binned_st)
            freqs = freqs.rescale('Hz').magnitude
            spectrum = self.extract_channel(psd, 0).magnitude

            # Take a band acround the peak frequency
            band = (freqs > f0 / 2) & (freqs < 3 * f0 / 2)
            # Find empirical peak frequency in the band
            peak_frequency = freqs[band][np.argmax(spectrum[band])]
            # The empirical peak should be +- one bin of the real f0
            self.assertLessEqual(abs(peak_frequency - f0),
                                 freqs[1] - freqs[0],
                                 msg=f"bin size {bin_size}")

            # Off-center power in the band should drop significantly
            off_peak = band & (np.abs(freqs - f0) > 2)
            self.assertGreater(np.max(spectrum[band]),
                               1000 * np.median(spectrum[off_peak]),
                               msg=f"bin size {bin_size}")

    def test_binned_spiketrain_total_power(self):
        """
        The area under the PSD equals the power of the analyzed rate signal
        r_k = n_k / b. For a Poisson process of rate lambda, this rate signal
        has the variance lambda / b and the mean square lambda / b + lambda**2.
        The area is expected to match the variance if the PSD function removes
        the mean, and the mean square otherwise. Since the PSD is flat at
        2 * lambda independently of b, but extends up to the Nyquist frequency
        1 / (2 b), the area grows with 1 / b. With the default parameters, the
        windows and tapers weight the samples unequally, so that the area and
        the plain sample power agree only up to statistical fluctuations of
        the order of 1 / sqrt(number of spikes).
        """
        np.random.seed(123)
        rate = 50 * pq.Hz
        spiketrain = elephant.spike_train_generation.StationaryPoissonProcess(
            rate, t_stop=100 * pq.s).generate_spiketrain()

        for bin_size in (1 * pq.ms, 5 * pq.ms, 10 * pq.ms):
            binned_st = elephant.conversion.BinnedSpikeTrain(
                spiketrain, bin_size=bin_size)
            freqs, psd = self.psd_function(binned_st)
            freqs = freqs.rescale('Hz').magnitude
            spectrum = self.extract_channel(psd, 0).rescale('Hz').magnitude
            area = np.sum(spectrum) * (freqs[1] - freqs[0])

            signal = binned_st.to_analog_signal(
                scaling="normalized").rescale('Hz').magnitude.ravel()
            lam = rate.rescale('Hz').magnitude
            b = bin_size.rescale('s').magnitude
            if self.removes_mean:
                sample_power = np.var(signal)
                expected_power = lam / b
            else:
                sample_power = np.mean(signal ** 2)
                expected_power = lam / b + lam ** 2

            self.assertAlmostEqual(area / sample_power, 1, delta=0.05,
                                   msg=f"bin size {bin_size}")
            self.assertAlmostEqual(area / expected_power, 1, delta=0.05,
                                   msg=f"bin size {bin_size}")

    def test_binned_spiketrain_units(self):
        """
        The frequencies are returned in Hz, and the PSD of the rate signal in
        Hz^2/Hz, i.e., in Hz.
        """
        np.random.seed(123)
        spiketrain = elephant.spike_train_generation.StationaryPoissonProcess(
            50 * pq.Hz, t_stop=10 * pq.s).generate_spiketrain()
        binned_st = elephant.conversion.BinnedSpikeTrain(
            spiketrain, bin_size=2 * pq.ms)
        freqs, psd = self.psd_function(binned_st)
        self.assertEqual(freqs.units, pq.Hz)
        self.assertEqual(psd.units, pq.Hz)

    def test_binned_spiketrain_empty(self):
        """
        An empty spike train yields a PSD that vanishes at all frequencies.
        """
        empty_st = neo.SpikeTrain([] * pq.s, t_start=0 * pq.s,
                                  t_stop=10 * pq.s)
        binned_empty = elephant.conversion.BinnedSpikeTrain(
            empty_st, bin_size=2 * pq.ms)
        _, psd = self.psd_function(binned_empty)
        self.assertEqual(np.max(psd), 0 * pq.Hz)


class WelchPSDTestCase(BinnedSpikeTrainPSDMixin, unittest.TestCase):
    psd_function = staticmethod(elephant.spectral.welch_psd)
    removes_mean = True

    def test_welch_psd_errors(self):
        # generate a dummy data
        data = AnalogSignal(np.zeros(5000), sampling_period=0.001 * pq.s,
                            units='mV')

        # check for invalid parameter values
        # - length of segments
        self.assertRaises(ValueError, elephant.spectral.welch_psd, data,
                          len_segment=0)
        self.assertRaises(ValueError, elephant.spectral.welch_psd, data,
                          len_segment=data.shape[0] * 2)
        # - number of segments
        self.assertRaises(ValueError, elephant.spectral.welch_psd, data,
                          n_segments=0)
        self.assertRaises(ValueError, elephant.spectral.welch_psd, data,
                          n_segments=data.shape[0] * 2)
        # - frequency resolution
        self.assertRaises(ValueError, elephant.spectral.welch_psd, data,
                          frequency_resolution=-1)
        self.assertRaises(ValueError, elephant.spectral.welch_psd, data,
                          frequency_resolution=data.sampling_rate /
                          (data.shape[0] + 1))
        # - overlap
        self.assertRaises(ValueError, elephant.spectral.welch_psd, data,
                          overlap=-1.0)
        self.assertRaises(ValueError, elephant.spectral.welch_psd, data,
                          overlap=1.1)

    def test_welch_psd_warnings(self):
        # generate a dummy data
        data = AnalogSignal(np.zeros(5000), sampling_period=0.001 * pq.s,
                            units='mV')
        # Test deprecation warning for 'hanning' window
        self.assertWarns(DeprecationWarning, elephant.spectral.welch_psd,
                         data, window='hanning')

    def test_welch_psd_behavior(self):
        # generate data by adding white noise and a sinusoid
        data_length = 5000
        sampling_period = 0.001
        signal_freq = 100.0
        noise = np.random.normal(size=data_length)
        signal = [np.sin(2 * np.pi * signal_freq * t)
                  for t in np.arange(0, data_length * sampling_period,
                                     sampling_period)]
        data = AnalogSignal(np.array(signal + noise),
                            sampling_period=sampling_period * pq.s,
                            units='mV')

        # consistency between different ways of specifying segment length
        freqs1, psd1 = elephant.spectral.welch_psd(
            data, len_segment=data_length // 5, overlap=0)
        freqs2, psd2 = elephant.spectral.welch_psd(
            data, n_segments=5, overlap=0)
        self.assertTrue((psd1 == psd2).all() and (freqs1 == freqs2).all())

        # frequency resolution and consistency with data
        freq_res = 1.0 * pq.Hz
        freqs, psd = elephant.spectral.welch_psd(
            data, frequency_resolution=freq_res)
        self.assertAlmostEqual(freq_res, freqs[1] - freqs[0])
        self.assertEqual(freqs[psd.argmax()], signal_freq)
        freqs_np, psd_np = elephant.spectral.welch_psd(
            data.magnitude.flatten(), fs=1 / sampling_period,
            frequency_resolution=freq_res)
        self.assertTrue((freqs == freqs_np).all() and (psd == psd_np).all())

        # check of scipy.signal.welch() parameters
        params = {'window': 'hamming', 'nfft': 1024, 'detrend': 'linear',
                  'return_onesided': False, 'scaling': 'spectrum'}
        for key, val in params.items():
            freqs, psd = elephant.spectral.welch_psd(
                data, len_segment=1000, overlap=0, **{key: val})
            freqs_spsig, psd_spsig = spsig.welch(
                np.rollaxis(data.magnitude, 0, len(data.shape)),
                fs=1 / sampling_period,
                nperseg=1000,
                noverlap=0,
                **{key: val}
            )
            self.assertTrue(
                (freqs == freqs_spsig).all() and (
                        psd == psd_spsig).all())

        # - generate multidimensional data for check of parameter `axis`
        num_channel = 4
        data_length = 5000
        data_multidim = np.random.normal(size=(num_channel, data_length))
        freqs, psd = elephant.spectral.welch_psd(data_multidim)
        freqs_T, psd_T = elephant.spectral.welch_psd(data_multidim.T, axis=0)
        self.assertTrue(np.all(freqs == freqs_T))
        self.assertTrue(np.all(psd == psd_T.T))

    def test_welch_psd_input_types(self):
        # generate a test data
        sampling_period = 0.001
        data = AnalogSignal(np.array(np.random.normal(size=5000)),
                            sampling_period=sampling_period * pq.s,
                            units='mV')

        # outputs from AnalogSignal input are of Quantity type (standard usage)
        freqs_neo, psd_neo = elephant.spectral.welch_psd(data)
        self.assertTrue(isinstance(freqs_neo, pq.quantity.Quantity))
        self.assertTrue(isinstance(psd_neo, pq.quantity.Quantity))

        # outputs from Quantity array input are of Quantity type
        freqs_pq, psd_pq = elephant.spectral.welch_psd(
            data.magnitude.flatten() * data.units, fs=1 / sampling_period)
        self.assertTrue(isinstance(freqs_pq, pq.quantity.Quantity))
        self.assertTrue(isinstance(psd_pq, pq.quantity.Quantity))

        # outputs from Numpy ndarray input are NOT of Quantity type
        freqs_np, psd_np = elephant.spectral.welch_psd(
            data.magnitude.flatten(), fs=1 / sampling_period)
        self.assertFalse(isinstance(freqs_np, pq.quantity.Quantity))
        self.assertFalse(isinstance(psd_np, pq.quantity.Quantity))

        # check if the results from different input types are identical
        self.assertTrue(
            (freqs_neo == freqs_pq).all() and (
                    psd_neo == psd_pq).all())
        self.assertTrue(
            (freqs_neo == freqs_np).all() and (
                    psd_neo == psd_np).all())

    def test_welch_psd_binned_spiketrain_two_sided(self):
        """
        The 2*rate asymptote is a consequence of the one-sided convention, in
        which the negative frequencies are folded onto the positive ones. The
        two-sided spectrum of a Poisson process levels off at rate instead.
        """
        np.random.seed(123)
        rate = 50 * pq.Hz
        t_stop = 20 * pq.s
        spiketrain = elephant.spike_train_generation.StationaryPoissonProcess(
            rate, t_stop=t_stop).generate_spiketrain()
        binned_st = elephant.conversion.BinnedSpikeTrain(
            spiketrain, bin_size=2 * pq.ms)

        freqs, psd = elephant.spectral.welch_psd(
            binned_st, return_onesided=False)

        # Check scaling: The two-sided spectrum of a Poisson process levels off
        # at the Poisson rate.
        mask = np.abs(freqs) > 100 * pq.Hz
        avg_psd = np.mean(psd[0, mask]).rescale('Hz').magnitude
        self.assertAlmostEqual(
            avg_psd, rate.rescale('Hz').magnitude,
            delta=0.15 * rate.rescale('Hz').magnitude)

    def test_welch_psd_multidim_input(self):
        # generate multidimensional data
        num_channel = 4
        data_length = 5000
        sampling_period = 0.001
        noise = np.random.normal(size=(num_channel, data_length))
        data_np = np.array(noise)
        # Since row-column order in AnalogSignal is different from the
        # conventional one, `data_np` needs to be transposed when it's used to
        # define an AnalogSignal
        data_neo = AnalogSignal(data_np.T,
                                sampling_period=sampling_period * pq.s,
                                units='mV')
        data_neo_1dim = AnalogSignal(data_np[0],
                                     sampling_period=sampling_period * pq.s,
                                     units='mV')

        # check if the results from different input types are identical
        freqs_np, psd_np = elephant.spectral.welch_psd(data_np,
                                                       fs=1 / sampling_period)
        freqs_neo, psd_neo = elephant.spectral.welch_psd(data_neo)
        freqs_neo_1dim, psd_neo_1dim = elephant.spectral.welch_psd(
            data_neo_1dim)
        self.assertTrue(np.all(freqs_np == freqs_neo))
        self.assertTrue(np.all(psd_np == psd_neo))
        self.assertTrue(np.all(psd_neo_1dim == psd_neo[0]))


class MultitaperPSDTestCase(BinnedSpikeTrainPSDMixin, unittest.TestCase):
    psd_function = staticmethod(functools.partial(
        elephant.spectral.multitaper_psd, detrend=False))
    removes_mean = False

    def test_multitaper_psd_errors(self):
        # generate dummy data
        data_length = 5000
        signal = AnalogSignal(np.zeros(data_length),
                              sampling_period=0.001 * pq.s,
                              units='mV')
        fs = signal.sampling_rate
        self.assertIsInstance(fs, pq.Quantity)

        # check for invalid parameter values
        # - number of tapers
        self.assertRaises(ValueError, elephant.spectral.multitaper_psd, signal,
                          num_tapers=-5, detrend=False)
        self.assertRaises(TypeError, elephant.spectral.multitaper_psd, signal,
                          num_tapers=-5.0, detrend=False)
        # - peak resolution
        self.assertRaises(ValueError, elephant.spectral.multitaper_psd, signal,
                          peak_resolution=-1, detrend=False)

    def test_multitaper_psd_behavior(self):
        # generate data (frequency domain to time domain)
        r = np.ones(2501) * 0.2
        r[0], r[500] = 0, 10  # Zero DC, peak at 100 Hz
        phi = np.random.uniform(-np.pi, np.pi, len(r))
        fake_coeffs = r*np.exp(1j * phi)
        fake_ts = scipy.fft.irfft(fake_coeffs)
        sampling_period = 0.001
        freqs = scipy.fft.rfftfreq(len(fake_ts), d=sampling_period)
        signal_freq = freqs[r.argmax()]

        data = AnalogSignal(fake_ts, sampling_period=sampling_period * pq.s,
                            units='mV')

        # consistency between different ways of specifying number of tapers
        freqs1, psd1 = elephant.spectral.multitaper_psd(data,
                                                        fs=data.sampling_rate,
                                                        nw=3.5, detrend=False)
        freqs2, psd2 = elephant.spectral.multitaper_psd(data,
                                                        fs=data.sampling_rate,
                                                        nw=3.5,
                                                        num_tapers=6,
                                                        detrend=False)
        self.assertTrue((psd1 == psd2).all() and (freqs1 == freqs2).all())

        # peak resolution and consistency with data
        peak_res = 1.0 * pq.Hz
        freqs, psd = elephant.spectral.multitaper_psd(
            data, peak_resolution=peak_res, detrend=False)
        self.assertEqual(freqs[psd.argmax()], signal_freq)
        freqs_np, psd_np = elephant.spectral.multitaper_psd(
            data.magnitude.flatten(), fs=1 / sampling_period,
            peak_resolution=peak_res, detrend=False)
        self.assertTrue((freqs == freqs_np).all() and (psd == psd_np).all())

    def test_multitaper_psd_parameter_hierarchy(self):
        # generate data by adding white noise and a sinusoid
        data_length = 5000
        sampling_period = 0.001
        signal_freq = 100.0
        noise = np.random.normal(size=data_length)
        signal = [np.sin(2 * np.pi * signal_freq * t)
                  for t in np.arange(0, data_length * sampling_period,
                                     sampling_period)]
        data = AnalogSignal(np.array(signal + noise),
                            sampling_period=sampling_period * pq.s,
                            units='mV')

        # Test num_tapers vs nw
        freqs1, psd1 = elephant.spectral.multitaper_psd(data,
                                                        fs=data.sampling_rate,
                                                        nw=3,
                                                        num_tapers=9,
                                                        detrend=False)
        freqs2, psd2 = elephant.spectral.multitaper_psd(data,
                                                        fs=data.sampling_rate,
                                                        nw=3, detrend=False)
        self.assertTrue((freqs1 == freqs2).all() and (psd1 != psd2).all())

        # Test peak_resolution vs nw
        freqs1, psd1 = elephant.spectral.multitaper_psd(data,
                                                        fs=data.sampling_rate,
                                                        nw=3,
                                                        num_tapers=9,
                                                        peak_resolution=1,
                                                        detrend=False)
        freqs2, psd2 = elephant.spectral.multitaper_psd(data,
                                                        fs=data.sampling_rate,
                                                        nw=3,
                                                        num_tapers=9,
                                                        detrend=False)
        self.assertTrue((freqs1 == freqs2).all() and (psd1 != psd2).all())

    def test_multitaper_psd_against_nitime(self):
        """
        This test assesses the match between this implementation of
        multitaper against nitime (0.8) using a predefined time series
        generated by an autoregressive model.

        Please follow the link below for more details:
        https://gin.g-node.org/INM-6/elephant-data/src/master/unittest/spectral/multitaper_psd
        """
        repo_path = r"unittest/spectral/multitaper_psd/data"

        files_to_download = [
            ("time_series.npy", "ff43797e2ac94613f510b20a31e2e80e"),
            ("psd_nitime.npy", "89d1f53957e66c786049ea425b53c0e8")
        ]

        downloaded_files = {}
        for filename, checksum in files_to_download:
            downloaded_files[filename] = {
                'filename': filename,
                'path': download_datasets(repo_path=f"{repo_path}/{filename}",
                                          checksum=checksum)}

        time_series = np.load(downloaded_files['time_series.npy']['path'])
        psd_nitime = np.load(downloaded_files['psd_nitime.npy']['path'])

        freqs, psd_multitaper = elephant.spectral.multitaper_psd(
            signal=time_series, fs=0.1, nw=4, num_tapers=8, detrend=False)

        np.testing.assert_allclose(np.squeeze(psd_multitaper), psd_nitime,
                                   rtol=0.3, atol=0.1)

    def test_multitaper_psd_input_types(self):
        # generate a test data
        sampling_period = 0.001
        data = AnalogSignal(np.array(np.random.normal(size=5000)),
                            sampling_period=sampling_period * pq.s,
                            units='mV')

        # outputs from AnalogSignal input are of Quantity type (standard usage)
        freqs_neo, psd_neo = elephant.spectral.multitaper_psd(
            data, detrend=False)
        self.assertTrue(isinstance(freqs_neo, pq.quantity.Quantity))
        self.assertTrue(isinstance(psd_neo, pq.quantity.Quantity))

        # outputs from Quantity array input are of Quantity type
        freqs_pq, psd_pq = elephant.spectral.multitaper_psd(
            data.magnitude.flatten() * data.units, fs=1 / sampling_period,
            detrend=False)
        self.assertTrue(isinstance(freqs_pq, pq.quantity.Quantity))
        self.assertTrue(isinstance(psd_pq, pq.quantity.Quantity))

        # outputs from Numpy ndarray input are NOT of Quantity type
        freqs_np, psd_np = elephant.spectral.multitaper_psd(
            data.magnitude.flatten(), fs=1 / sampling_period, detrend=False)
        self.assertFalse(isinstance(freqs_np, pq.quantity.Quantity))
        self.assertFalse(isinstance(psd_np, pq.quantity.Quantity))

        # fs with and without units
        fs_hz = 1 * pq.Hz
        fs_int = 1
        freqs_fs_hz, psd_fs_hz = elephant.spectral.multitaper_psd(
            data.magnitude.T, fs=fs_hz, detrend=False)
        freqs_fs_int, psd_fs_int = elephant.spectral.multitaper_psd(
            data.magnitude.T, fs=fs_int, detrend=False)

        np.testing.assert_array_equal(freqs_fs_hz, freqs_fs_int)
        np.testing.assert_array_equal(psd_fs_hz, psd_fs_int)

        # check if the results from different input types are identical
        self.assertTrue(
            (freqs_neo == freqs_pq).all() and (
                    psd_neo == psd_pq).all())
        self.assertTrue(
            (freqs_neo == freqs_np).all() and (
                    psd_neo == psd_np).all())


    def test_multitaper_psd_detrend(self):
        rng = np.random.default_rng(123)
        fs = 1000.
        noise = rng.normal(size=(2, 5000))
        data = noise + np.array([[50.], [-20.]])

        # the default None behaves like False, but announces the change of
        # the default to 'constant'
        with self.assertWarns(FutureWarning):
            _, psd_default = elephant.spectral.multitaper_psd(data, fs=fs)
        _, psd_false = elephant.spectral.multitaper_psd(
            data, fs=fs, detrend=False)
        assert_array_equal(psd_default, psd_false)

        # 'constant' subtracts the mean of each channel before tapering
        _, psd_constant = elephant.spectral.multitaper_psd(
            data, fs=fs, detrend='constant')
        _, psd_manual = elephant.spectral.multitaper_psd(
            data - data.mean(axis=-1, keepdims=True), fs=fs, detrend=False)
        np.testing.assert_allclose(psd_constant, psd_manual, rtol=1e-10)

        # 'linear' subtracts the least-squares linear fit of each channel
        ramp = data + np.linspace(0., 30., data.shape[-1])
        _, psd_linear = elephant.spectral.multitaper_psd(
            ramp, fs=fs, detrend='linear')
        _, psd_manual = elephant.spectral.multitaper_psd(
            spsig.detrend(ramp, axis=-1, type='linear'), fs=fs,
            detrend=False)
        np.testing.assert_allclose(psd_linear, psd_manual, rtol=1e-10)

        # a function is applied to the data of shape (n_channels, n_samples)
        _, psd_function = elephant.spectral.multitaper_psd(
            data, fs=fs,
            detrend=lambda x: x - x.mean(axis=-1, keepdims=True))
        np.testing.assert_allclose(psd_function, psd_constant, rtol=1e-10)

        # without detrending, the power of a constant leaks beyond the
        # bandwidth of the tapers (here nw = 4 bins); detrending removes it
        constant = np.full(5000, 10.)
        _, psd_raw = elephant.spectral.multitaper_psd(
            constant, fs=fs, nw=4, detrend=False)
        _, psd_detrended = elephant.spectral.multitaper_psd(
            constant, fs=fs, nw=4, detrend='constant')
        self.assertTrue(np.all(psd_raw[0, 5:20] > 1e-3))
        np.testing.assert_allclose(psd_detrended, 0, atol=1e-20)

        for invalid in ('quadratic', True, lambda x: x[:, 1:]):
            self.assertRaises(ValueError, elephant.spectral.multitaper_psd,
                              data, fs=fs, detrend=invalid)

    def test_multitaper_detrend_explicit_no_warning(self):
        data = np.random.default_rng(1).normal(size=(2, 1000))
        for detrend in (False, 'constant', 'linear'):
            with warnings.catch_warnings():
                warnings.simplefilter('error', FutureWarning)
                elephant.spectral.multitaper_psd(data, detrend=detrend)
                elephant.spectral.segmented_multitaper_psd(
                    data, n_segments=4, detrend=detrend)
                elephant.spectral.multitaper_cross_spectrum(
                    data, detrend=detrend)
                elephant.spectral.segmented_multitaper_cross_spectrum(
                    data, n_segments=4, detrend=detrend)
                elephant.spectral.multitaper_coherence(
                    data[0], data[1], n_segments=4, detrend=detrend)

    def test_multitaper_detrend_default_warns_once(self):
        # the segmented functions issue the warning once per call, and not
        # once per segment
        data = np.random.default_rng(1).normal(size=(2, 1000))
        for func, args in (
                (elephant.spectral.segmented_multitaper_psd, (data,)),
                (elephant.spectral.segmented_multitaper_cross_spectrum,
                 (data,)),
                (elephant.spectral.multitaper_coherence,
                 (data[0], data[1]))):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                func(*args, n_segments=4)
            future = [w for w in caught
                      if issubclass(w.category, FutureWarning)]
            self.assertEqual(len(future), 1)
            # the warning points to the caller of the public function
            self.assertEqual(future[0].filename, __file__)

    def test_multitaper_psd_detrend_binned_spiketrain(self):
        """
        Detrending removes the mean firing rate, but leaves the level of
        2 * rate at high frequencies unchanged.
        """
        np.random.seed(123)
        rate = 20 * pq.Hz
        spiketrain = elephant.spike_train_generation.StationaryPoissonProcess(
            rate, t_stop=100 * pq.s).generate_spiketrain()
        binned_st = elephant.conversion.BinnedSpikeTrain(
            spiketrain, bin_size=5 * pq.ms)
        freqs, psd_raw = elephant.spectral.multitaper_psd(
            binned_st, detrend=False)
        _, psd_detrended = elephant.spectral.multitaper_psd(
            binned_st, detrend='constant')
        # the DC component is of the order of rate**2 * T and is removed
        self.assertGreater(psd_raw[0, 0].magnitude,
                           100 * 2 * rate.magnitude)
        self.assertLess(psd_detrended[0, 0].magnitude, 2 * rate.magnitude)
        # the leakage of the DC component interferes with the signal in each
        # bin, which changes individual bins by about 3% at 1 Hz, but by
        # less than 0.5% above 10 Hz
        high = freqs > 10 * pq.Hz
        np.testing.assert_allclose(psd_detrended[0, high].magnitude,
                                   psd_raw[0, high].magnitude, rtol=0.01)


class SegmentedMultitaperPSDTestCase(BinnedSpikeTrainPSDMixin,
                                     unittest.TestCase):
    psd_function = staticmethod(functools.partial(
        elephant.spectral.segmented_multitaper_psd, detrend=False))
    removes_mean = False

    # The following assertions test _segmented_apply_func in the context
    # of segmented_multitaper_psd. In other words, only the segmentation is
    # addressed. The inner workings of the multitaper function are tested
    # in the MultitaperPSDTestCase.
    def test_segmented_multitaper_psd_errors(self):
        # generate dummy data
        data_length = 5000
        signal = AnalogSignal(np.zeros(data_length),
                              sampling_period=0.001 * pq.s,
                              units='mV')
        fs = signal.sampling_rate

        # check for invalid parameter values
        # - frequency resolution
        self.assertRaises(ValueError,
                          elephant.spectral.segmented_multitaper_psd, signal,
                          frequency_resolution=-10, detrend=False)

        # - n per segment
        # n_per_seg = int(fs / dF), where dF is the frequency_resolution
        broken_freq_resolution = fs / (data_length+1)
        self.assertRaises(ValueError,
                          elephant.spectral.segmented_multitaper_psd, signal,
                          frequency_resolution=broken_freq_resolution,
                          detrend=False)

        # - length of segment (negative)
        self.assertRaises(ValueError,
                          elephant.spectral.segmented_multitaper_psd, signal,
                          len_segment=-10, detrend=False)

        # - length of segment (larger than data length)
        self.assertRaises(ValueError,
                          elephant.spectral.segmented_multitaper_psd, signal,
                          len_segment=data_length+1, detrend=False)

        # - number of segments (negative)
        self.assertRaises(ValueError,
                          elephant.spectral.segmented_multitaper_psd, signal,
                          n_segments=-10, detrend=False)

        # - number of segments (larger than data length)
        self.assertRaises(ValueError,
                          elephant.spectral.segmented_multitaper_psd, signal,
                          n_segments=data_length+1, detrend=False)

    def test_segmented_multitaper_psd_behavior(self):
        # generate data (frequency domain to time domain)
        r = np.ones(2501) * 0.2
        r[0], r[500] = 0, 10  # Zero DC, peak at 100 Hz
        phi = np.random.uniform(-np.pi, np.pi, len(r))
        fake_coeffs = r*np.exp(1j * phi)
        fake_ts = scipy.fft.irfft(fake_coeffs)
        sampling_period = 0.001
        freqs = scipy.fft.rfftfreq(len(fake_ts), d=sampling_period)
        signal_freq = freqs[r.argmax()]

        data = AnalogSignal(fake_ts, sampling_period=sampling_period * pq.s,
                            units='mV')

        # consistency between different ways of specifying n_per_seg
        # n_per_seg = int(fs/dF) and n_per_seg = len_segment
        frequency_resolution = 1 * pq.Hz
        len_segment = int(data.sampling_rate / frequency_resolution)

        freqs_fr, psd_fr = elephant.spectral.segmented_multitaper_psd(
            data, frequency_resolution=frequency_resolution, detrend=False)

        freqs_ls, psd_ls = elephant.spectral.segmented_multitaper_psd(
            data, len_segment=len_segment, detrend=False)

        np.testing.assert_array_equal(freqs_fr, freqs_ls)
        np.testing.assert_array_equal(psd_fr, psd_ls)

    def test_segmented_multitaper_psd_parameter_hierarchy(self):
        # generate data by adding white noise and a sinusoid
        data_length = 5000
        sampling_period = 0.001
        signal_freq = 100.0
        noise = np.random.normal(size=data_length)
        signal = [np.sin(2 * np.pi * signal_freq * t)
                  for t in np.arange(0, data_length * sampling_period,
                                     sampling_period)]
        data = AnalogSignal(np.array(signal + noise),
                            sampling_period=sampling_period * pq.s,
                            units='mV')

        # test frequency_resolution vs len_segment vs n_segments
        n_segments = 5
        len_segment = 2000
        frequency_resolution = 0.25 * pq.Hz

        freqs_ns, psd_ns = \
            elephant.spectral.segmented_multitaper_psd(
                data, n_segments=n_segments, detrend=False)

        freqs_ls, psd_ls = \
            elephant.spectral.segmented_multitaper_psd(
                data, n_segments=n_segments, len_segment=len_segment,
                detrend=False)

        freqs_fr, psd_fr = \
            elephant.spectral.segmented_multitaper_psd(
                data, n_segments=n_segments, len_segment=len_segment,
                frequency_resolution=frequency_resolution, detrend=False)

        self.assertTrue(freqs_ns.shape < freqs_ls.shape < freqs_fr.shape)
        self.assertTrue(psd_ns.shape < psd_ls.shape < psd_fr.shape)

    def test_segmented_multitaper_psd_input_types(self):
        # generate a test data
        sampling_period = 0.001
        data = AnalogSignal(np.array(np.random.normal(size=5000)),
                            sampling_period=sampling_period * pq.s,
                            units='mV')

        # outputs from AnalogSignal input are of Quantity type (standard usage)
        freqs_neo, psd_neo = elephant.spectral.segmented_multitaper_psd(
            data, detrend=False)
        self.assertTrue(isinstance(freqs_neo, pq.quantity.Quantity))
        self.assertTrue(isinstance(psd_neo, pq.quantity.Quantity))

        # outputs from Quantity array input are of Quantity type
        freqs_pq, psd_pq = elephant.spectral.segmented_multitaper_psd(
            data.magnitude.flatten() * data.units, fs=1 / sampling_period,
            detrend=False)
        self.assertTrue(isinstance(freqs_pq, pq.quantity.Quantity))
        self.assertTrue(isinstance(psd_pq, pq.quantity.Quantity))

        # outputs from Numpy ndarray input are NOT of Quantity type
        freqs_np, psd_np = elephant.spectral.segmented_multitaper_psd(
            data.magnitude.flatten(), fs=1 / sampling_period, detrend=False)
        self.assertFalse(isinstance(freqs_np, pq.quantity.Quantity))
        self.assertFalse(isinstance(psd_np, pq.quantity.Quantity))

        # frequency resolution with and without units
        freq_res_hz = 1 * pq.Hz
        freq_res_int = 1

        freqs_int, psd_int = elephant.spectral.segmented_multitaper_psd(
            data, frequency_resolution=freq_res_int, detrend=False)

        freqs_hz, psd_hz = elephant.spectral.segmented_multitaper_psd(
            data, frequency_resolution=freq_res_hz, detrend=False)

        np.testing.assert_array_equal(freqs_int, freqs_hz)
        np.testing.assert_array_equal(psd_int, psd_hz)

        # fs with and without units
        fs_hz = 1 * pq.Hz
        fs_int = 1
        freqs_fs_hz, psd_fs_hz = elephant.spectral.multitaper_psd(
            data.magnitude.T, fs=fs_hz, detrend=False)
        freqs_fs_int, psd_fs_int = elephant.spectral.multitaper_psd(
            data.magnitude.T, fs=fs_int, detrend=False)

        np.testing.assert_array_equal(freqs_fs_hz, freqs_fs_int)
        np.testing.assert_array_equal(psd_fs_hz, psd_fs_int)

        # check if the results from different input types are identical
        self.assertTrue(
            (freqs_neo == freqs_pq).all() and (
                    psd_neo == psd_pq).all())
        self.assertTrue(
            (freqs_neo == freqs_np).all() and (
                    psd_neo == psd_np).all())

    def test_segmented_multitaper_psd_multichannel_shape_and_values(self):
        """
        Regression test for a bug where `segmented_multitaper_psd` returned
        a (n_channels, n_channels, n_freqs) array whose off-diagonal entries
        were `channel[:, np.newaxis, :]` broadcast across the row (i.e. a
        duplicate of the channel's own spectrum), rather than the genuine
        per-channel spectrum this function computes. `_segmented_apply_func`
        allocated that shape unconditionally, assuming every wrapped
        function computes a real cross-spectrum like
        `multitaper_cross_spectrum` does; `multitaper_psd` does not.

        The fixed return shape must be (n_channels, n_freqs), matching
        `multitaper_psd` and `welch_psd`, and its values must equal the
        (correctly computed) diagonal of `segmented_multitaper_cross_spectrum`,
        up to the one-sided-vs-two-sided factor of 2.
        """
        data_length = 5000
        sampling_period = 0.001
        signal_freq = 100.0
        noise = np.random.normal(size=(3, data_length))
        time_points = np.arange(0, data_length * sampling_period,
                                sampling_period)
        signal_x = np.sin(2 * np.pi * signal_freq * time_points) + noise[0]
        signal_y = np.cos(2 * np.pi * signal_freq * time_points) + noise[1]
        signal_z = 2 * noise[2]
        data = AnalogSignal(np.vstack([signal_x, signal_y, signal_z]).T,
                            sampling_period=sampling_period * pq.s,
                            units='mV')

        freqs_psd, psd = elephant.spectral.segmented_multitaper_psd(
            data, n_segments=4, nw=4, num_tapers=8, detrend=False)

        self.assertEqual(psd.shape, (3, len(freqs_psd)))

        freqs_csd, cross_spec = \
            elephant.spectral.segmented_multitaper_cross_spectrum(
                data, n_segments=4, nw=4, num_tapers=8,
                return_onesided=True, detrend=False)

        self.assertTrue((freqs_psd == freqs_csd).all())

        auto_spectra = 2 * np.diagonal(cross_spec).T.real.magnitude
        np.testing.assert_allclose(psd.magnitude, auto_spectra,
                                   rtol=0.01, atol=0.01)

        # The three channels have different power, so a broadcast bug
        # (every row equal to the channel's own value repeated) would show
        # up as the per-channel spectra all being equal to one another.
        self.assertFalse(np.allclose(psd[0].magnitude, psd[2].magnitude,
                                     rtol=0.1))


    def test_segmented_multitaper_psd_detrend(self):
        rng = np.random.default_rng(123)
        fs = 1000.
        noise = rng.normal(size=(2, 4000))
        # offsets that are constant within each of 4 non-overlapping segments
        offsets = np.repeat([10., -5., 30., 0.], 1000)
        data = noise + offsets

        kwargs = dict(fs=fs, n_segments=4, overlap=0)
        with self.assertWarns(FutureWarning):
            _, psd_default = elephant.spectral.segmented_multitaper_psd(
                data, **kwargs)
        _, psd_false = elephant.spectral.segmented_multitaper_psd(
            data, detrend=False, **kwargs)
        assert_array_equal(psd_default, psd_false)

        # 'constant' subtracts the mean of each segment, which removes the
        # offsets entirely
        _, psd_constant = elephant.spectral.segmented_multitaper_psd(
            data, detrend='constant', **kwargs)
        _, psd_noise = elephant.spectral.segmented_multitaper_psd(
            noise, detrend='constant', **kwargs)
        np.testing.assert_allclose(psd_constant, psd_noise, rtol=1e-8)
        self.assertGreater(psd_default[0, 0], 100 * psd_constant[0, 0])

        # 'linear' subtracts the linear fit of each segment, which removes
        # ramps that differ between the segments entirely
        ramps = np.concatenate([np.linspace(0., slope, 1000)
                                for slope in (10., -5., 30., 0.)])
        _, psd_linear = elephant.spectral.segmented_multitaper_psd(
            noise + offsets + ramps, detrend='linear', **kwargs)
        _, psd_noise_linear = elephant.spectral.segmented_multitaper_psd(
            noise, detrend='linear', **kwargs)
        np.testing.assert_allclose(psd_linear, psd_noise_linear, rtol=1e-8)

        self.assertRaises(ValueError,
                          elephant.spectral.segmented_multitaper_psd,
                          data, detrend='quadratic', **kwargs)


class MultitaperCrossSpectrumTestCase(unittest.TestCase):
    def test_multitaper_cross_spectrum_errors(self):
        # generate dummy data
        data_length = 5000
        signal = AnalogSignal(np.zeros(data_length),
                              sampling_period=0.001 * pq.s,
                              units='mV')
        fs = signal.sampling_rate

        # check for invalid parameter values
        # - number of tapers
        self.assertRaises(ValueError,
                          elephant.spectral.multitaper_cross_spectrum, signal,
                          fs=fs, num_tapers=-5, detrend=False)
        self.assertRaises(TypeError,
                          elephant.spectral.multitaper_cross_spectrum, signal,
                          fs=fs, num_tapers=-5.0, detrend=False)

        # - peak resolution
        self.assertRaises(ValueError,
                          elephant.spectral.multitaper_cross_spectrum, signal,
                          fs=fs, peak_resolution=-1, detrend=False)

    def test_multitaper_cross_spectrum_behavior(self):
        # generate data (frequency domain to time domain)
        r = np.ones(2501) * 0.2
        r[0], r[500] = 0, 10  # Zero DC, peak at 100 Hz
        phi_x = np.random.uniform(-np.pi, np.pi, len(r))
        phi_y = np.random.uniform(-np.pi, np.pi, len(r))
        fake_coeffs_x = r*np.exp(1j * phi_x)
        fake_coeffs_y = r*np.exp(1j * phi_y)
        signal_x = scipy.fft.irfft(fake_coeffs_x)
        signal_y = scipy.fft.irfft(fake_coeffs_y)
        sampling_period = 0.001
        freqs = scipy.fft.rfftfreq(len(signal_x), d=sampling_period)
        signal_freq = freqs[r.argmax()]
        data = AnalogSignal(np.vstack([signal_x, signal_y]).T,
                            sampling_period=sampling_period * pq.s,
                            units='mV')

        # consistency between different ways of specifying number of tapers
        freqs1, cross_spec1 = \
            elephant.spectral.multitaper_cross_spectrum(
                    data,
                    fs=data.sampling_rate,
                    nw=3.5, detrend=False)
        freqs2, cross_spec2 = \
            elephant.spectral.multitaper_cross_spectrum(
                    data,
                    fs=data.sampling_rate,
                    nw=3.5,
                    num_tapers=6, detrend=False)
        self.assertTrue((cross_spec1 == cross_spec2).all()
                        and (freqs1 == freqs2).all())

        # peak resolution and consistency with data
        peak_res = 1.0 * pq.Hz
        freqs, cross_spec = \
            elephant.spectral.multitaper_cross_spectrum(
                    data, peak_resolution=peak_res, detrend=False)

        self.assertEqual(freqs[cross_spec[0, 0].argmax()], signal_freq)
        freqs_np, cross_spec_np = \
            elephant.spectral.multitaper_cross_spectrum(
                    data.magnitude.T, fs=1 / sampling_period,
                    peak_resolution=peak_res, detrend=False)
        assert_array_equal(freqs.magnitude, freqs_np)
        assert_array_almost_equal(cross_spec.magnitude, cross_spec_np)

        # one-sided vs two-sided spectrum
        freqs_os, cross_spec_os = \
            elephant.spectral.multitaper_cross_spectrum(
                data, return_onesided=True, detrend=False)

        freqs_ts, cross_spec_ts = \
            elephant.spectral.multitaper_cross_spectrum(
                data, return_onesided=False, detrend=False)

        # Nyquist frequency is negative when using onesided=False (fftfreq)
        # See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.rfftfreq.html#scipy.fft.rfftfreq  # noqa
        nonnegative_freqs_indices = np.nonzero(freqs_ts >= 0)[0]
        nyquist_freq_idx = np.abs(freqs_ts).argmax()
        ts_freq_indices = np.append(nonnegative_freqs_indices,
                                    nyquist_freq_idx)
        ts_overlap_freqs = np.append(
            freqs_ts[nonnegative_freqs_indices].rescale('Hz').magnitude,
            np.abs(freqs_ts[nyquist_freq_idx].rescale('Hz').magnitude)) * pq.Hz

        np.testing.assert_array_equal(freqs_os, ts_overlap_freqs)

        np.testing.assert_allclose(
            cross_spec_os.magnitude,
            cross_spec_ts[:, :, ts_freq_indices].magnitude, rtol=1e-12, atol=0)

    def test_multitaper_cross_spectrum_parameter_hierarchy(self):
        # generate data by adding white noise and a sinusoid
        data_length = 5000
        sampling_period = 0.001
        signal_freq = 100.0
        noise = np.random.normal(size=(2, data_length))
        time_points = np.arange(0, data_length * sampling_period,
                                sampling_period)
        signal_x = np.sin(2 * np.pi * signal_freq * time_points) + noise[0]
        signal_y = np.cos(2 * np.pi * signal_freq * time_points) + noise[1]
        data = AnalogSignal(np.vstack([signal_x, signal_y]).T,
                            sampling_period=sampling_period * pq.s,
                            units='mV')

        # Test num_tapers vs nw
        freqs1, cross_spec1 = elephant.spectral.multitaper_cross_spectrum(
            data, fs=data.sampling_rate, nw=3, num_tapers=9, detrend=False)
        freqs2, cross_spec2 = elephant.spectral.multitaper_cross_spectrum(
            data, fs=data.sampling_rate, nw=3, detrend=False)

        self.assertTrue((freqs1 == freqs2).all()
                        and (cross_spec1 != cross_spec2).all())

        # Test peak_resolution vs nw
        freqs1, cross_spec1 = elephant.spectral.multitaper_cross_spectrum(
            data, fs=data.sampling_rate, nw=3, num_tapers=9, peak_resolution=1,
            detrend=False)
        freqs2, cross_spec2 = elephant.spectral.multitaper_cross_spectrum(
            data, fs=data.sampling_rate, nw=3, num_tapers=9, detrend=False)

        self.assertTrue((freqs1 == freqs2).all()
                        and (cross_spec1 != cross_spec2).all())

    def test_multitaper_cross_spectrum_input_types(self):
        # generate a test data
        data_length = 5000
        sampling_period = 0.001
        signal_freq = 100.0
        noise = np.random.normal(size=(2, data_length))
        time_points = np.arange(0, data_length * sampling_period,
                                sampling_period)
        signal_x = np.sin(2 * np.pi * signal_freq * time_points) + noise[0]
        signal_y = np.cos(2 * np.pi * signal_freq * time_points) + noise[1]
        data = AnalogSignal(np.vstack([signal_x, signal_y]).T,
                            sampling_period=sampling_period * pq.s,
                            units='mV')

        # outputs from AnalogSignal input are of Quantity type (standard usage)
        freqs_neo, cross_spec_neo \
            = elephant.spectral.multitaper_cross_spectrum(data, detrend=False)
        self.assertTrue(isinstance(freqs_neo, pq.quantity.Quantity))
        self.assertTrue(isinstance(cross_spec_neo, pq.quantity.Quantity))

        # outputs from Quantity array input are of Quantity type
        freqs_pq, cross_spec_pq \
            = elephant.spectral.multitaper_cross_spectrum(
                    data.magnitude.T * data.units,
                    fs=1 / (sampling_period * pq.s), detrend=False)
        self.assertTrue(isinstance(freqs_pq, pq.quantity.Quantity))
        self.assertTrue(isinstance(cross_spec_pq, pq.quantity.Quantity))

        # outputs from Numpy ndarray input are NOT of Quantity type
        freqs_np, cross_spec_np \
            = elephant.spectral.multitaper_cross_spectrum(
                    data.magnitude.T,
                    fs=1 / (sampling_period * pq.s), detrend=False)
        self.assertFalse(isinstance(freqs_np, pq.quantity.Quantity))
        self.assertFalse(isinstance(cross_spec_np, pq.quantity.Quantity))

        # check if the results from different input types are identical
        self.assertTrue(
            (freqs_neo == freqs_pq).all() and
            (cross_spec_neo == cross_spec_pq).all())
        self.assertTrue(
            (freqs_neo == freqs_np).all() and
            (cross_spec_neo == cross_spec_np).all())


    def test_multitaper_cross_spectrum_detrend(self):
        rng = np.random.default_rng(123)
        fs = 1000.
        noise = rng.normal(size=(2, 5000))
        data = noise + np.array([[50.], [-20.]])

        for onesided in (True, False):
            with self.assertWarns(FutureWarning):
                _, cs_default = elephant.spectral.multitaper_cross_spectrum(
                    data, fs=fs, return_onesided=onesided)
            _, cs_false = elephant.spectral.multitaper_cross_spectrum(
                data, fs=fs, return_onesided=onesided, detrend=False)
            assert_array_equal(cs_default, cs_false)

            # 'constant' subtracts the mean of each channel before tapering
            _, cs_constant = elephant.spectral.multitaper_cross_spectrum(
                data, fs=fs, return_onesided=onesided, detrend='constant')
            _, cs_manual = elephant.spectral.multitaper_cross_spectrum(
                data - data.mean(axis=-1, keepdims=True), fs=fs,
                return_onesided=onesided, detrend=False)
            np.testing.assert_allclose(cs_constant, cs_manual, rtol=1e-5)

        # without detrending, the cross spectrum of two constants leaks
        # beyond the bandwidth of the tapers; detrending removes it
        constants = np.full((2, 5000), [[10.], [3.]])
        _, cs_raw = elephant.spectral.multitaper_cross_spectrum(
            constants, fs=fs, nw=4, detrend=False)
        _, cs_detrended = elephant.spectral.multitaper_cross_spectrum(
            constants, fs=fs, nw=4, detrend='constant')
        self.assertTrue(np.all(np.abs(cs_raw[0, 1, 5:20]) > 1e-4))
        np.testing.assert_allclose(np.abs(cs_detrended), 0, atol=1e-20)

        # 'linear' subtracts the least-squares linear fit of each channel
        ramp = data + np.linspace(0., 30., data.shape[-1])
        _, cs_linear = elephant.spectral.multitaper_cross_spectrum(
            ramp, fs=fs, detrend='linear')
        _, cs_manual = elephant.spectral.multitaper_cross_spectrum(
            spsig.detrend(ramp, axis=-1, type='linear'), fs=fs,
            detrend=False)
        np.testing.assert_allclose(cs_linear, cs_manual, rtol=1e-5)

        self.assertRaises(ValueError,
                          elephant.spectral.multitaper_cross_spectrum,
                          data, fs=fs, detrend='quadratic')


class SegmentedMultitaperCrossSpectrumTestCase(unittest.TestCase):
    def test_segmented_multitaper_cross_spectrum_errors(self):
        # generate dummy data
        data_length = 5000
        signal = AnalogSignal(np.zeros(data_length),
                              sampling_period=0.001 * pq.s,
                              units='mV')
        fs = signal.sampling_rate

        # - frequency resolution
        self.assertRaises(
            ValueError, elephant.spectral.segmented_multitaper_cross_spectrum,
            signal, fs=fs, frequency_resolution=-10, detrend=False)

        # - n per segment
        # n_per_seg = int(fs / dF), where dF is the frequency_resolution
        broken_freq_resolution = fs / (data_length+1)
        self.assertRaises(
            ValueError, elephant.spectral.segmented_multitaper_cross_spectrum,
            signal, fs=fs, frequency_resolution=broken_freq_resolution,
            detrend=False)

        # - length of segment (negative)
        self.assertRaises(
            ValueError, elephant.spectral.segmented_multitaper_cross_spectrum,
            signal, fs=fs, len_segment=-10, detrend=False)

        # - length of segment (larger than data length)
        self.assertRaises(
            ValueError, elephant.spectral.segmented_multitaper_cross_spectrum,
            signal, fs=fs, len_segment=data_length+1, detrend=False)

        # - number of segments (negative)
        self.assertRaises(
            ValueError, elephant.spectral.segmented_multitaper_cross_spectrum,
            signal, fs=fs, n_segments=-10, detrend=False)

        # - number of segments (larger than data length)
        self.assertRaises(
            ValueError, elephant.spectral.segmented_multitaper_cross_spectrum,
            signal, fs=fs, n_segments=data_length+1, detrend=False)

    def test_segmented_multitaper_cross_spectrum_behavior(self):
        # generate data (frequency domain to time domain)
        r = np.ones(2501) * 0.2
        r[0], r[500] = 0, 10  # Zero DC, peak at 100 Hz
        phi_x = np.random.uniform(-np.pi, np.pi, len(r))
        phi_y = np.random.uniform(-np.pi, np.pi, len(r))
        fake_coeffs_x = r*np.exp(1j * phi_x)
        fake_coeffs_y = r*np.exp(1j * phi_y)
        signal_x = scipy.fft.irfft(fake_coeffs_x)
        signal_y = scipy.fft.irfft(fake_coeffs_y)
        sampling_period = 0.001
        freqs = scipy.fft.rfftfreq(len(signal_x), d=sampling_period)
        signal_freq = freqs[r.argmax()]
        data = AnalogSignal(np.vstack([signal_x, signal_y]).T,
                            sampling_period=sampling_period * pq.s,
                            units='mV')

        # consistency between different ways of specifying n_per_seg
        # n_per_seg = int(fs/dF) and n_per_seg = len_segment
        frequency_resolution = 1 * pq.Hz
        len_segment = int(data.sampling_rate / frequency_resolution)

        freqs_fr, cross_spec_fr = \
            elephant.spectral.segmented_multitaper_cross_spectrum(
                data, frequency_resolution=frequency_resolution, detrend=False)

        freqs_ls, cross_spec_ls = \
            elephant.spectral.segmented_multitaper_cross_spectrum(
                data, len_segment=len_segment, detrend=False)

        np.testing.assert_array_equal(freqs_fr, freqs_ls)
        np.testing.assert_array_equal(cross_spec_fr, cross_spec_ls)

        # one-sided vs two-sided spectrum
        freqs_os, cross_spec_os = \
            elephant.spectral.segmented_multitaper_cross_spectrum(
                data, return_onesided=True, detrend=False)

        freqs_ts, cross_spec_ts = \
            elephant.spectral.segmented_multitaper_cross_spectrum(
                data, return_onesided=False, detrend=False)

        # test overlap parameter
        no_overlap = 0
        half_overlap = 0.5
        large_overlap = 0.99
        n_segments = 10

        freqs_no, cross_spec_no = \
            elephant.spectral.segmented_multitaper_cross_spectrum(
                data, n_segments=n_segments, overlap=no_overlap, detrend=False)

        freqs_ho, cross_spec_ho = \
            elephant.spectral.segmented_multitaper_cross_spectrum(
                data, n_segments=n_segments, overlap=half_overlap,
                detrend=False)

        freqs_lo, cross_spec_lo = \
            elephant.spectral.segmented_multitaper_cross_spectrum(
                data, n_segments=n_segments, overlap=large_overlap,
                detrend=False)

        self.assertTrue(freqs_no.shape < freqs_ho.shape < freqs_lo.shape)
        self.assertTrue(
            cross_spec_no.shape < cross_spec_ho.shape < cross_spec_lo.shape)

        # Nyquist frequency is negative when using onesided=False (fftfreq)
        # See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.rfftfreq.html#scipy.fft.rfftfreq  # noqa
        nonnegative_freqs_indices = np.nonzero(freqs_ts >= 0)[0]
        nyquist_freq_idx = np.abs(freqs_ts).argmax()
        ts_freq_indices = np.append(nonnegative_freqs_indices,
                                    nyquist_freq_idx)
        ts_overlap_freqs = np.append(
            freqs_ts[nonnegative_freqs_indices].rescale('Hz').magnitude,
            np.abs(freqs_ts[nyquist_freq_idx].rescale('Hz').magnitude)) * pq.Hz

        np.testing.assert_array_equal(freqs_os, ts_overlap_freqs)

        np.testing.assert_allclose(
            cross_spec_os.magnitude,
            cross_spec_ts[:, :, ts_freq_indices].magnitude, rtol=1e-12, atol=0)

    def test_segmented_multitaper_cross_spectrum_parameter_hierarchy(self):
        # test frequency_resolution vs len_segment vs n_segments
        # generate data (frequency domain to time domain)
        r = np.ones(2501) * 0.2
        r[0], r[500] = 0, 10  # Zero DC, peak at 100 Hz
        phi_x = np.random.uniform(-np.pi, np.pi, len(r))
        phi_y = np.random.uniform(-np.pi, np.pi, len(r))
        fake_coeffs_x = r*np.exp(1j * phi_x)
        fake_coeffs_y = r*np.exp(1j * phi_y)
        signal_x = scipy.fft.irfft(fake_coeffs_x)
        signal_y = scipy.fft.irfft(fake_coeffs_y)
        sampling_period = 0.001
        freqs = scipy.fft.rfftfreq(len(signal_x), d=sampling_period)
        signal_freq = freqs[r.argmax()]
        data = AnalogSignal(np.vstack([signal_x, signal_y]).T,
                            sampling_period=sampling_period * pq.s,
                            units='mV')

        n_segments = 5
        len_segment = 2000
        frequency_resolution = 1 * pq.Hz

        freqs_ns, cross_spec_ns = \
            elephant.spectral.segmented_multitaper_cross_spectrum(
                data, n_segments=n_segments, detrend=False)

        freqs_ls, cross_spec_ls = \
            elephant.spectral.segmented_multitaper_cross_spectrum(
                data, n_segments=n_segments, len_segment=len_segment,
                detrend=False)

        freqs_fr, cross_spec_fr = \
            elephant.spectral.segmented_multitaper_cross_spectrum(
                data, n_segments=n_segments, len_segment=len_segment,
                frequency_resolution=frequency_resolution, detrend=False)

        self.assertNotEqual(freqs_ns.shape, freqs_ls.shape)
        self.assertNotEqual(freqs_ls.shape, freqs_fr.shape)
        self.assertNotEqual(freqs_fr.shape, freqs_ns.shape)
        self.assertNotEqual(cross_spec_ns.shape, cross_spec_ls.shape)
        self.assertNotEqual(cross_spec_ls.shape, cross_spec_fr.shape)
        self.assertNotEqual(cross_spec_fr.shape, cross_spec_ns.shape)

    def test_segmented_multitaper_cross_spectrum_against_multitaper_psd(self):
        data_length = 5000
        sampling_period = 0.001
        signal_freq = 100.0
        noise = np.random.normal(size=(2, data_length))
        time_points = np.arange(0, data_length * sampling_period,
                                sampling_period)
        signal_x = np.sin(2 * np.pi * signal_freq * time_points) + noise[0]
        signal_y = np.cos(2 * np.pi * signal_freq * time_points) + noise[1]
        data = AnalogSignal(np.vstack([signal_x, signal_y]).T,
                            sampling_period=sampling_period * pq.s,
                            units='mV')

        freqs1, psd_multitaper = elephant.spectral.multitaper_psd(
            signal=data, fs=data.sampling_rate, nw=4, num_tapers=8,
            detrend=False)

        psd_multitaper[:, 1:] /= 2  # since comparing rfft and fft results

        freqs2, cross_spec = \
            elephant.spectral.segmented_multitaper_cross_spectrum(
                data,
                fs=data.sampling_rate,
                nw=4,
                num_tapers=8,
                return_onesided=True, detrend=False)

        self.assertTrue((freqs1 == freqs2).all())

        np.testing.assert_allclose(psd_multitaper.magnitude,
                                   np.diagonal(cross_spec).T.real.magnitude,
                                   rtol=0.01,
                                   atol=0.01)

    def test_segmented_multitaper_cross_spectrum_input_types(self):
        # generate a test data
        data_length = 5000
        sampling_period = 0.001
        signal_freq = 100.0
        noise = np.random.normal(size=(2, data_length))
        time_points = np.arange(0, data_length * sampling_period,
                                sampling_period)
        signal_x = np.sin(2 * np.pi * signal_freq * time_points) + noise[0]
        signal_y = np.cos(2 * np.pi * signal_freq * time_points) + noise[1]
        data = AnalogSignal(np.vstack([signal_x, signal_y]).T,
                            sampling_period=sampling_period * pq.s,
                            units='mV')

        # frequency resolution as an integer
        freq_res_int = 1
        freq_res_hz = 1 * pq.Hz

        freqs_int, cross_spec_int = \
            elephant.spectral.segmented_multitaper_cross_spectrum(
                data, frequency_resolution=freq_res_int, detrend=False)

        freqs_hz, cross_spec_hz = \
            elephant.spectral.segmented_multitaper_cross_spectrum(
                data, frequency_resolution=freq_res_hz, detrend=False)

        np.testing.assert_array_equal(freqs_int, freqs_hz)
        np.testing.assert_array_equal(cross_spec_int, cross_spec_hz)


    def test_segmented_multitaper_cross_spectrum_detrend(self):
        rng = np.random.default_rng(123)
        fs = 1000.
        noise = rng.normal(size=(2, 4000))
        # offsets that are constant within each of 4 non-overlapping segments
        offsets = np.repeat([10., -5., 30., 0.], 1000)
        data = noise + offsets

        kwargs = dict(fs=fs, n_segments=4, overlap=0)
        with self.assertWarns(FutureWarning):
            _, cs_default = \
                elephant.spectral.segmented_multitaper_cross_spectrum(
                    data, **kwargs)
        _, cs_false = elephant.spectral.segmented_multitaper_cross_spectrum(
            data, detrend=False, **kwargs)
        assert_array_equal(cs_default, cs_false)

        # 'constant' subtracts the mean of each segment, which removes the
        # offsets entirely
        _, cs_constant = \
            elephant.spectral.segmented_multitaper_cross_spectrum(
                data, detrend='constant', **kwargs)
        _, cs_noise = elephant.spectral.segmented_multitaper_cross_spectrum(
            noise, detrend='constant', **kwargs)
        np.testing.assert_allclose(cs_constant, cs_noise, rtol=1e-4,
                                   atol=1e-9)
        self.assertGreater(np.abs(cs_default[0, 1, 0]),
                           100 * np.abs(cs_constant[0, 1, 0]))

        self.assertRaises(
            ValueError, elephant.spectral.segmented_multitaper_cross_spectrum,
            data, detrend='quadratic', **kwargs)


class MultitaperCoherenceTestCase(unittest.TestCase):
    def test_multitaper_coherence_input_types(self):
        # Generate dummy data
        data_length = 10000
        sampling_period = 0.001
        signal_freq = 100.0
        np.random.seed(123)
        noise = np.random.normal(size=(2, data_length))
        time_points = np.arange(0, data_length * sampling_period,
                                sampling_period)
        # Signals are designed to have coherence peak at `signal_freq`
        arr_signal_i = np.sin(2 * np.pi * signal_freq * time_points) + noise[0]
        arr_signal_j = np.cos(2 * np.pi * signal_freq * time_points) + noise[1]

        fs = 1000 * pq.Hz
        anasig_signal_i = neo.core.AnalogSignal(arr_signal_i,
                                                sampling_rate=fs,
                                                units=pq.mV)
        anasig_signal_j = neo.core.AnalogSignal(arr_signal_j,
                                                sampling_rate=fs,
                                                units=pq.mV)

        arr_f, arr_coh, arr_phi = elephant.spectral.multitaper_coherence(
            arr_signal_i, arr_signal_j, fs=fs, detrend=False)
        anasig_f, anasig_coh, anasig_phi = \
            elephant.spectral.multitaper_coherence(anasig_signal_i,
                                                   anasig_signal_j,
                                                   detrend=False)

        np.testing.assert_array_equal(arr_f, anasig_f)
        np.testing.assert_allclose(arr_coh, anasig_coh.magnitude, atol=1e-6)
        np.testing.assert_array_almost_equal(arr_phi, anasig_phi.magnitude)

    def test_multitaper_cohere_peak(self):
        # Generate dummy data
        data_length = 10000
        sampling_period = 0.001
        signal_freq = 100.0
        noise = np.random.normal(size=(2, data_length))
        time_points = np.arange(0, data_length * sampling_period,
                                sampling_period)
        # Signals are designed to have coherence peak at `signal_freq`
        signal_i = np.sin(2 * np.pi * signal_freq * time_points) + noise[0]
        signal_j = np.cos(2 * np.pi * signal_freq * time_points) + noise[1]

        # Estimate coherence and phase lag with the multitaper method
        freq1, coh1, phase_lag1 = elephant.spectral.multitaper_coherence(
            signal_i,
            signal_j,
            fs=1/sampling_period,
            n_segments=16, detrend=False)

        indices, vals = scipy.signal.find_peaks(coh1, height=0.8, distance=10)

        peak_freqs = freq1[indices]

        np.testing.assert_allclose(peak_freqs,
                                   signal_freq*np.ones(len(peak_freqs)),
                                   rtol=0.05)

    @pytest.mark.skipif(version.parse(np.__version__)>version.parse("1.25.0"),
                        reason="This test will fail with numpy version"
                               "1.25.0 - 1.25.2,  see issue #24000"
                               "https://github.com/numpy/numpy/issues/24000 ")
    def test_multitaper_cohere_perfect_cohere(self):
        # Generate dummy data
        data_length = 10000
        sampling_period = 0.001
        signal_freq = 100.0
        noise = np.random.normal(size=(1, data_length))
        time_points = np.arange(0, data_length * sampling_period,
                                sampling_period)
        signal = np.cos(2 * np.pi * signal_freq * time_points) + noise

        # Estimate coherence and phase lag with the multitaper method
        freq1, coh, phase_lag = elephant.spectral.multitaper_coherence(
            signal,
            signal,
            fs=1/sampling_period,
            n_segments=16, detrend=False)

        np.testing.assert_array_equal(phase_lag, np.zeros(phase_lag.size))
        np.testing.assert_array_equal(coh, np.ones(coh.size))

    def test_multitaper_cohere_no_cohere(self):
        # Generate dummy data
        data_length = 10000
        sampling_period = 0.001
        time_points = np.arange(0, data_length * sampling_period,
                                sampling_period)

        signal_i = np.sin(2 * np.pi * 2.5 * time_points)
        signal_j = np.sin(2 * np.pi * 5 * time_points)

        # Estimate coherence and phase lag with the multitaper method
        freq, coh, phase_lag = elephant.spectral.multitaper_coherence(
            signal_i,
            signal_j,
            fs=1/sampling_period,
            n_segments=16, detrend=False)

        np.testing.assert_allclose(coh, np.zeros(coh.size), atol=0.002)

    def test_multitaper_cohere_phase_lag(self):
        # Generate dummy data
        data_length = 10000
        sampling_period = 0.001
        signal_freq = 100.0
        time_points = np.arange(0, data_length * sampling_period,
                                sampling_period)

        # Signals are designed to have maximal phase lag at 100 with value pi/4
        signal_i = np.sin(2 * np.pi * signal_freq * time_points + np.pi / 4)
        signal_j = np.cos(2 * np.pi * signal_freq * time_points)

        # Estimate coherence and phase lag with the multitaper method
        freq, coh, phase_lag = elephant.spectral.multitaper_coherence(
            signal_i,
            signal_j,
            fs=1/sampling_period,
            n_segments=16,
            num_tapers=8, detrend=False)

        indices, vals = scipy.signal.find_peaks(phase_lag,
                                                height=0.8 * np.pi / 4,
                                                distance=10)

        # Get peak frequencies and peak heights
        peak_freqs = freq[indices]
        peak_heights = vals['peak_heights']

        np.testing.assert_allclose(peak_freqs,
                                   signal_freq*np.ones(len(peak_freqs)),
                                   rtol=0.05)
        np.testing.assert_allclose(peak_heights,
                                   np.pi / 4 * np.ones(len(peak_heights)),
                                   rtol=0.05)


    def test_multitaper_coherence_detrend(self):
        """
        A common offset of two independent signals produces spurious
        coherence at low frequencies, unless the signals are detrended.
        """
        rng = np.random.default_rng(123)
        fs = 1000.
        signal_i, signal_j = rng.normal(size=(2, 8000))
        offset = 20.

        kwargs = dict(fs=fs, n_segments=4)
        with self.assertWarns(FutureWarning):
            _, coh_raw, _ = elephant.spectral.multitaper_coherence(
                signal_i + offset, signal_j + offset, **kwargs)
        _, coh_false, _ = elephant.spectral.multitaper_coherence(
            signal_i + offset, signal_j + offset, detrend=False, **kwargs)
        assert_array_equal(coh_raw, coh_false)
        _, coh_detrended, _ = elephant.spectral.multitaper_coherence(
            signal_i + offset, signal_j + offset, detrend='constant',
            **kwargs)
        _, coh_reference, _ = elephant.spectral.multitaper_coherence(
            signal_i, signal_j, detrend='constant', **kwargs)

        # the offset makes the lowest bins almost perfectly coherent
        self.assertTrue(np.all(coh_raw[:3] > 0.99))
        np.testing.assert_allclose(coh_detrended, coh_reference, rtol=1e-3,
                                   atol=1e-6)
        self.assertTrue(np.all(coh_detrended[:3] < 0.9))

        self.assertRaises(ValueError, elephant.spectral.multitaper_coherence,
                          signal_i, signal_j, detrend='quadratic', **kwargs)


class WelchCohereTestCase(unittest.TestCase):
    def test_welch_cohere_errors(self):
        # generate a dummy data
        x = AnalogSignal(np.zeros(5000), sampling_period=0.001 * pq.s,
                         units='mV')
        y = AnalogSignal(np.zeros(5000), sampling_period=0.001 * pq.s,
                         units='mV')

        # check for invalid parameter values
        # - length of segments
        self.assertRaises(ValueError, elephant.spectral.welch_coherence, x, y,
                          len_segment=0)
        self.assertRaises(ValueError, elephant.spectral.welch_coherence, x, y,
                          len_segment=x.shape[0] * 2)
        # - number of segments
        self.assertRaises(ValueError, elephant.spectral.welch_coherence, x, y,
                          n_segments=0)
        self.assertRaises(ValueError, elephant.spectral.welch_coherence, x, y,
                          n_segments=x.shape[0] * 2)
        # - frequency resolution
        self.assertRaises(ValueError, elephant.spectral.welch_coherence, x, y,
                          frequency_resolution=-1)
        self.assertRaises(ValueError, elephant.spectral.welch_coherence, x, y,
                          frequency_resolution=x.sampling_rate /
                          (x.shape[0] + 1))
        # - overlap
        self.assertRaises(ValueError, elephant.spectral.welch_coherence, x, y,
                          overlap=-1.0)
        self.assertRaises(ValueError, elephant.spectral.welch_coherence, x, y,
                          overlap=1.1)

    def test_welch_cohere_warnings(self):
        # generate a dummy data
        x = AnalogSignal(np.zeros(5000), sampling_period=0.001 * pq.s,
                         units='mV')
        y = AnalogSignal(np.zeros(5000), sampling_period=0.001 * pq.s,
                         units='mV')
        # Test deprecation warning for 'hanning' window
        self.assertWarns(DeprecationWarning, elephant.spectral.welch_coherence,
                         x, y, window='hanning')

    def test_welch_cohere_behavior(self):
        # generate data by adding white noise and a sinusoid
        data_length = 5000
        sampling_period = 0.001
        signal_freq = 100.0
        noise1 = np.random.normal(size=data_length) * 0.01
        noise2 = np.random.normal(size=data_length) * 0.01
        signal1 = [np.cos(2 * np.pi * signal_freq * t)
                   for t in np.arange(0, data_length * sampling_period,
                                      sampling_period)]
        signal2 = [np.sin(2 * np.pi * signal_freq * t)
                   for t in np.arange(0, data_length * sampling_period,
                                      sampling_period)]
        x = AnalogSignal(np.array(signal1 + noise1), units='mV',
                         sampling_period=sampling_period * pq.s)
        y = AnalogSignal(np.array(signal2 + noise2), units='mV',
                         sampling_period=sampling_period * pq.s)

        # consistency between different ways of specifying segment length
        freqs1, coherency1, phase_lag1 = elephant.spectral.welch_coherence(
            x, y, len_segment=data_length // 5, overlap=0)
        freqs2, coherency2, phase_lag2 = elephant.spectral.welch_coherence(
            x, y, n_segments=5, overlap=0)
        self.assertTrue((coherency1 == coherency2).all() and
                        (phase_lag1 == phase_lag2).all() and
                        (freqs1 == freqs2).all())

        # frequency resolution and consistency with data
        freq_res = 1.0 * pq.Hz
        freqs, coherency, phase_lag = elephant.spectral.welch_coherence(
            x, y, frequency_resolution=freq_res)
        self.assertAlmostEqual(freq_res, freqs[1] - freqs[0])
        self.assertAlmostEqual(freqs[coherency.argmax()], signal_freq,
                               places=2)
        self.assertAlmostEqual(phase_lag[coherency.argmax()], -np.pi / 2,
                               places=2)
        freqs_np, coherency_np, phase_lag_np = \
            elephant.spectral.welch_coherence(x.magnitude.flatten(),
                                              y.magnitude.flatten(),
                                              fs=1 / sampling_period,
                                              frequency_resolution=freq_res)
        assert_array_equal(freqs.simplified.magnitude, freqs_np)
        assert_array_equal(coherency[:, 0], coherency_np)
        assert_array_equal(phase_lag[:, 0], phase_lag_np)

        # - check the behavior of parameter `axis` using multidimensional data
        num_channel = 4
        data_length = 5000
        x_multidim = np.random.normal(size=(num_channel, data_length))
        y_multidim = np.random.normal(size=(num_channel, data_length))
        freqs, coherency, phase_lag = \
            elephant.spectral.welch_coherence(x_multidim, y_multidim)
        freqs_T, coherency_T, phase_lag_T = elephant.spectral.welch_coherence(
            x_multidim.T, y_multidim.T, axis=0)
        assert_array_equal(freqs, freqs_T)
        assert_array_equal(coherency, coherency_T.T)
        assert_array_equal(phase_lag, phase_lag_T.T)

    def test_welch_cohere_input_types(self):
        # generate a test data
        sampling_period = 0.001
        x = AnalogSignal(np.array(np.random.normal(size=5000)),
                         sampling_period=sampling_period * pq.s,
                         units='mV')
        y = AnalogSignal(np.array(np.random.normal(size=5000)),
                         sampling_period=sampling_period * pq.s,
                         units='mV')

        # outputs from AnalogSignal input are of Quantity type
        # (standard usage)
        freqs_neo, coherency_neo, phase_lag_neo = \
            elephant.spectral.welch_coherence(x, y)
        self.assertTrue(isinstance(freqs_neo, pq.quantity.Quantity))
        self.assertTrue(isinstance(phase_lag_neo, pq.quantity.Quantity))

        # outputs from Quantity array input are of Quantity type
        freqs_pq, coherency_pq, phase_lag_pq = elephant.spectral \
            .welch_coherence(x.magnitude.flatten() * x.units,
                             y.magnitude.flatten() * y.units,
                             fs=1 / sampling_period)
        self.assertTrue(isinstance(freqs_pq, pq.quantity.Quantity))
        self.assertTrue(isinstance(phase_lag_pq, pq.quantity.Quantity))

        # outputs from Numpy ndarray input are NOT of Quantity type
        freqs_np, coherency_np, phase_lag_np = elephant.spectral \
            .welch_coherence(x.magnitude.flatten(),
                             y.magnitude.flatten(),
                             fs=1 / sampling_period)
        self.assertFalse(isinstance(freqs_np, pq.quantity.Quantity))
        self.assertFalse(isinstance(phase_lag_np, pq.quantity.Quantity))

        # check if the results from different input types are identical
        self.assertTrue((freqs_neo == freqs_pq).all() and
                        (coherency_neo[:, 0] == coherency_pq).all() and
                        (phase_lag_neo[:, 0] == phase_lag_pq).all())
        self.assertTrue((freqs_neo == freqs_np).all() and
                        (coherency_neo[:, 0] == coherency_np).all() and
                        (phase_lag_neo[:, 0] == phase_lag_np).all())

    def test_welch_cohere_multidim_input(self):
        # generate multidimensional data
        num_channel = 4
        data_length = 5000
        sampling_period = 0.001
        x_np = np.array(np.random.normal(size=(num_channel, data_length)))
        y_np = np.array(np.random.normal(size=(num_channel, data_length)))
        # Since row-column order in AnalogSignal is different from the
        # convention in NumPy/SciPy, `data_np` needs to be transposed when it's
        # used to define an AnalogSignal
        x_neo = AnalogSignal(x_np.T, units='mV',
                             sampling_period=sampling_period * pq.s)
        y_neo = AnalogSignal(y_np.T, units='mV',
                             sampling_period=sampling_period * pq.s)
        x_neo_1dim = AnalogSignal(x_np[0], units='mV',
                                  sampling_period=sampling_period * pq.s)
        y_neo_1dim = AnalogSignal(y_np[0], units='mV',
                                  sampling_period=sampling_period * pq.s)

        # check if the results from different input types are identical
        freqs_np, coherency_np, phase_lag_np = elephant.spectral \
            .welch_coherence(x_np, y_np, fs=1 / sampling_period)
        freqs_neo, coherency_neo, phase_lag_neo = \
            elephant.spectral.welch_coherence(x_neo, y_neo)
        freqs_neo_1dim, coherency_neo_1dim, phase_lag_neo_1dim = \
            elephant.spectral.welch_coherence(x_neo_1dim, y_neo_1dim)
        self.assertTrue(np.all(freqs_np == freqs_neo))
        self.assertTrue(np.all(coherency_np.T == coherency_neo))
        self.assertTrue(np.all(phase_lag_np.T == phase_lag_neo))
        self.assertTrue(
            np.all(coherency_neo_1dim[:, 0] == coherency_neo[:, 0]))
        self.assertTrue(
            np.all(phase_lag_neo_1dim[:, 0] == phase_lag_neo[:, 0]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
