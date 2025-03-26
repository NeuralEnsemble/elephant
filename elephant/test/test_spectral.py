# -*- coding: utf-8 -*-
"""
Unit tests for the spectral module.

:copyright: Copyright 2014-2024 by the Elephant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
"""

import unittest

import neo.core
import numpy as np
import pytest
import quantities as pq
import scipy
import scipy.fft
import scipy.signal as spsig
from neo import AnalogSignal
from numpy.testing import assert_array_equal
from packaging import version

import elephant.spectral
from elephant.datasets import download_datasets


class WelchPSDTestCase(unittest.TestCase):
    def test_welch_psd_errors(self):
        # generate a dummy data
        data = AnalogSignal(np.zeros(5000), sampling_period=0.001 * pq.s, units="mV")

        # check for invalid parameter values
        # - length of segments
        self.assertRaises(ValueError, elephant.spectral.welch_psd, data, len_segment=0)
        self.assertRaises(
            ValueError, elephant.spectral.welch_psd, data, len_segment=data.shape[0] * 2
        )
        # - number of segments
        self.assertRaises(ValueError, elephant.spectral.welch_psd, data, n_segments=0)
        self.assertRaises(
            ValueError, elephant.spectral.welch_psd, data, n_segments=data.shape[0] * 2
        )
        # - frequency resolution
        self.assertRaises(
            ValueError, elephant.spectral.welch_psd, data, frequency_resolution=-1
        )
        self.assertRaises(
            ValueError,
            elephant.spectral.welch_psd,
            data,
            frequency_resolution=data.sampling_rate / (data.shape[0] + 1),
        )
        # - overlap
        self.assertRaises(ValueError, elephant.spectral.welch_psd, data, overlap=-1.0)
        self.assertRaises(ValueError, elephant.spectral.welch_psd, data, overlap=1.1)

    def test_welch_psd_warnings(self):
        # generate a dummy data
        data = AnalogSignal(np.zeros(5000), sampling_period=0.001 * pq.s, units="mV")
        # Test deprecation warning for 'hanning' window
        self.assertWarns(
            DeprecationWarning, elephant.spectral.welch_psd, data, window="hanning"
        )

    def test_welch_psd_behavior(self):
        # generate data by adding white noise and a sinusoid
        data_length = 5000
        sampling_period = 0.001
        signal_freq = 100.0
        noise = np.random.normal(size=data_length)
        signal = [
            np.sin(2 * np.pi * signal_freq * t)
            for t in np.arange(0, data_length * sampling_period, sampling_period)
        ]
        data = AnalogSignal(
            np.array(signal + noise), sampling_period=sampling_period * pq.s, units="mV"
        )

        # consistency between different ways of specifying segment length
        freqs1, psd1 = elephant.spectral.welch_psd(
            data, len_segment=data_length // 5, overlap=0
        )
        freqs2, psd2 = elephant.spectral.welch_psd(data, n_segments=5, overlap=0)
        self.assertTrue((psd1 == psd2).all() and (freqs1 == freqs2).all())

        # frequency resolution and consistency with data
        freq_res = 1.0 * pq.Hz
        freqs, psd = elephant.spectral.welch_psd(data, frequency_resolution=freq_res)
        self.assertAlmostEqual(freq_res, freqs[1] - freqs[0])
        self.assertEqual(freqs[psd.argmax()], signal_freq)
        freqs_np, psd_np = elephant.spectral.welch_psd(
            data.magnitude.flatten(),
            fs=1 / sampling_period,
            frequency_resolution=freq_res,
        )
        self.assertTrue((freqs == freqs_np).all() and (psd == psd_np).all())

        # check of scipy.signal.welch() parameters
        params = {
            "window": "hamming",
            "nfft": 1024,
            "detrend": "linear",
            "return_onesided": False,
            "scaling": "spectrum",
        }
        for key, val in params.items():
            freqs, psd = elephant.spectral.welch_psd(
                data, len_segment=1000, overlap=0, **{key: val}
            )
            freqs_spsig, psd_spsig = spsig.welch(
                np.rollaxis(data, 0, len(data.shape)),
                fs=1 / sampling_period,
                nperseg=1000,
                noverlap=0,
                **{key: val},
            )
            self.assertTrue((freqs == freqs_spsig).all() and (psd == psd_spsig).all())

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
        data = AnalogSignal(
            np.array(np.random.normal(size=5000)),
            sampling_period=sampling_period * pq.s,
            units="mV",
        )

        # outputs from AnalogSignal input are of Quantity type (standard usage)
        freqs_neo, psd_neo = elephant.spectral.welch_psd(data)
        self.assertTrue(isinstance(freqs_neo, pq.quantity.Quantity))
        self.assertTrue(isinstance(psd_neo, pq.quantity.Quantity))

        # outputs from Quantity array input are of Quantity type
        freqs_pq, psd_pq = elephant.spectral.welch_psd(
            data.magnitude.flatten() * data.units, fs=1 / sampling_period
        )
        self.assertTrue(isinstance(freqs_pq, pq.quantity.Quantity))
        self.assertTrue(isinstance(psd_pq, pq.quantity.Quantity))

        # outputs from Numpy ndarray input are NOT of Quantity type
        freqs_np, psd_np = elephant.spectral.welch_psd(
            data.magnitude.flatten(), fs=1 / sampling_period
        )
        self.assertFalse(isinstance(freqs_np, pq.quantity.Quantity))
        self.assertFalse(isinstance(psd_np, pq.quantity.Quantity))

        # check if the results from different input types are identical
        self.assertTrue((freqs_neo == freqs_pq).all() and (psd_neo == psd_pq).all())
        self.assertTrue((freqs_neo == freqs_np).all() and (psd_neo == psd_np).all())

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
        data_neo = AnalogSignal(
            data_np.T, sampling_period=sampling_period * pq.s, units="mV"
        )
        data_neo_1dim = AnalogSignal(
            data_np[0], sampling_period=sampling_period * pq.s, units="mV"
        )

        # check if the results from different input types are identical
        freqs_np, psd_np = elephant.spectral.welch_psd(data_np, fs=1 / sampling_period)
        freqs_neo, psd_neo = elephant.spectral.welch_psd(data_neo)
        freqs_neo_1dim, psd_neo_1dim = elephant.spectral.welch_psd(data_neo_1dim)
        self.assertTrue(np.all(freqs_np == freqs_neo))
        self.assertTrue(np.all(psd_np == psd_neo))
        self.assertTrue(np.all(psd_neo_1dim == psd_neo[0]))


class MultitaperPSDTestCase(unittest.TestCase):
    def test_multitaper_psd_errors(self):
        # generate dummy data
        data_length = 5000
        signal = AnalogSignal(
            np.zeros(data_length), sampling_period=0.001 * pq.s, units="mV"
        )
        fs = signal.sampling_rate
        self.assertIsInstance(fs, pq.Quantity)

        # check for invalid parameter values
        # - number of tapers
        self.assertRaises(
            ValueError, elephant.spectral.multitaper_psd, signal, num_tapers=-5
        )
        self.assertRaises(
            TypeError, elephant.spectral.multitaper_psd, signal, num_tapers=-5.0
        )
        # - peak resolution
        self.assertRaises(
            ValueError, elephant.spectral.multitaper_psd, signal, peak_resolution=-1
        )

    def test_multitaper_psd_behavior(self):
        # generate data (frequency domain to time domain)
        r = np.ones(2501) * 0.2
        r[0], r[500] = 0, 10  # Zero DC, peak at 100 Hz
        phi = np.random.uniform(-np.pi, np.pi, len(r))
        fake_coeffs = r * np.exp(1j * phi)
        fake_ts = scipy.fft.irfft(fake_coeffs)
        sampling_period = 0.001
        freqs = scipy.fft.rfftfreq(len(fake_ts), d=sampling_period)
        signal_freq = freqs[r.argmax()]

        data = AnalogSignal(fake_ts, sampling_period=sampling_period * pq.s, units="mV")

        # consistency between different ways of specifying number of tapers
        freqs1, psd1 = elephant.spectral.multitaper_psd(
            data, fs=data.sampling_rate, nw=3.5
        )
        freqs2, psd2 = elephant.spectral.multitaper_psd(
            data, fs=data.sampling_rate, nw=3.5, num_tapers=6
        )
        self.assertTrue((psd1 == psd2).all() and (freqs1 == freqs2).all())

        # peak resolution and consistency with data
        peak_res = 1.0 * pq.Hz
        freqs, psd = elephant.spectral.multitaper_psd(data, peak_resolution=peak_res)
        self.assertEqual(freqs[psd.argmax()], signal_freq)
        freqs_np, psd_np = elephant.spectral.multitaper_psd(
            data.magnitude.flatten(), fs=1 / sampling_period, peak_resolution=peak_res
        )
        self.assertTrue((freqs == freqs_np).all() and (psd == psd_np).all())

    def test_multitaper_psd_parameter_hierarchy(self):
        # generate data by adding white noise and a sinusoid
        data_length = 5000
        sampling_period = 0.001
        signal_freq = 100.0
        noise = np.random.normal(size=data_length)
        signal = [
            np.sin(2 * np.pi * signal_freq * t)
            for t in np.arange(0, data_length * sampling_period, sampling_period)
        ]
        data = AnalogSignal(
            np.array(signal + noise), sampling_period=sampling_period * pq.s, units="mV"
        )

        # Test num_tapers vs nw
        freqs1, psd1 = elephant.spectral.multitaper_psd(
            data, fs=data.sampling_rate, nw=3, num_tapers=9
        )
        freqs2, psd2 = elephant.spectral.multitaper_psd(
            data, fs=data.sampling_rate, nw=3
        )
        self.assertTrue((freqs1 == freqs2).all() and (psd1 != psd2).all())

        # Test peak_resolution vs nw
        freqs1, psd1 = elephant.spectral.multitaper_psd(
            data, fs=data.sampling_rate, nw=3, num_tapers=9, peak_resolution=1
        )
        freqs2, psd2 = elephant.spectral.multitaper_psd(
            data, fs=data.sampling_rate, nw=3, num_tapers=9
        )
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
            ("psd_nitime.npy", "89d1f53957e66c786049ea425b53c0e8"),
        ]

        downloaded_files = {}
        for filename, checksum in files_to_download:
            downloaded_files[filename] = {
                "filename": filename,
                "path": download_datasets(
                    repo_path=f"{repo_path}/{filename}", checksum=checksum
                ),
            }

        time_series = np.load(downloaded_files["time_series.npy"]["path"])
        psd_nitime = np.load(downloaded_files["psd_nitime.npy"]["path"])

        freqs, psd_multitaper = elephant.spectral.multitaper_psd(
            signal=time_series, fs=0.1, nw=4, num_tapers=8
        )

        np.testing.assert_allclose(
            np.squeeze(psd_multitaper), psd_nitime, rtol=0.3, atol=0.1
        )

    def test_multitaper_psd_input_types(self):
        # generate a test data
        sampling_period = 0.001
        data = AnalogSignal(
            np.array(np.random.normal(size=5000)),
            sampling_period=sampling_period * pq.s,
            units="mV",
        )

        # outputs from AnalogSignal input are of Quantity type (standard usage)
        freqs_neo, psd_neo = elephant.spectral.multitaper_psd(data)
        self.assertTrue(isinstance(freqs_neo, pq.quantity.Quantity))
        self.assertTrue(isinstance(psd_neo, pq.quantity.Quantity))

        # outputs from Quantity array input are of Quantity type
        freqs_pq, psd_pq = elephant.spectral.multitaper_psd(
            data.magnitude.flatten() * data.units, fs=1 / sampling_period
        )
        self.assertTrue(isinstance(freqs_pq, pq.quantity.Quantity))
        self.assertTrue(isinstance(psd_pq, pq.quantity.Quantity))

        # outputs from Numpy ndarray input are NOT of Quantity type
        freqs_np, psd_np = elephant.spectral.multitaper_psd(
            data.magnitude.flatten(), fs=1 / sampling_period
        )
        self.assertFalse(isinstance(freqs_np, pq.quantity.Quantity))
        self.assertFalse(isinstance(psd_np, pq.quantity.Quantity))

        # fs with and without units
        fs_hz = 1 * pq.Hz
        fs_int = 1
        freqs_fs_hz, psd_fs_hz = elephant.spectral.multitaper_psd(
            data.magnitude.T, fs=fs_hz
        )
        freqs_fs_int, psd_fs_int = elephant.spectral.multitaper_psd(
            data.magnitude.T, fs=fs_int
        )

        np.testing.assert_array_equal(freqs_fs_hz, freqs_fs_int)
        np.testing.assert_array_equal(psd_fs_hz, psd_fs_int)

        # check if the results from different input types are identical
        self.assertTrue((freqs_neo == freqs_pq).all() and (psd_neo == psd_pq).all())
        self.assertTrue((freqs_neo == freqs_np).all() and (psd_neo == psd_np).all())


class SegmentedMultitaperPSDTestCase(unittest.TestCase):
    # The following assertions test _segmented_apply_func in the context
    # of segmented_multitaper_psd. In other words, only the segmentation is
    # addressed. The inner workings of the multitaper function are tested
    # in the MultitaperPSDTestCase.
    def test_segmented_multitaper_psd_errors(self):
        # generate dummy data
        data_length = 5000
        signal = AnalogSignal(
            np.zeros(data_length), sampling_period=0.001 * pq.s, units="mV"
        )
        fs = signal.sampling_rate

        # check for invalid parameter values
        # - frequency resolution
        self.assertRaises(
            ValueError,
            elephant.spectral.segmented_multitaper_psd,
            signal,
            frequency_resolution=-10,
        )

        # - n per segment
        # n_per_seg = int(fs / dF), where dF is the frequency_resolution
        broken_freq_resolution = fs / (data_length + 1)
        self.assertRaises(
            ValueError,
            elephant.spectral.segmented_multitaper_psd,
            signal,
            frequency_resolution=broken_freq_resolution,
        )

        # - length of segment (negative)
        self.assertRaises(
            ValueError,
            elephant.spectral.segmented_multitaper_psd,
            signal,
            len_segment=-10,
        )

        # - length of segment (larger than data length)
        self.assertRaises(
            ValueError,
            elephant.spectral.segmented_multitaper_psd,
            signal,
            len_segment=data_length + 1,
        )

        # - number of segments (negative)
        self.assertRaises(
            ValueError,
            elephant.spectral.segmented_multitaper_psd,
            signal,
            n_segments=-10,
        )

        # - number of segments (larger than data length)
        self.assertRaises(
            ValueError,
            elephant.spectral.segmented_multitaper_psd,
            signal,
            n_segments=data_length + 1,
        )

    def test_segmented_multitaper_psd_behavior(self):
        # generate data (frequency domain to time domain)
        r = np.ones(2501) * 0.2
        r[0], r[500] = 0, 10  # Zero DC, peak at 100 Hz
        phi = np.random.uniform(-np.pi, np.pi, len(r))
        fake_coeffs = r * np.exp(1j * phi)
        fake_ts = scipy.fft.irfft(fake_coeffs)
        sampling_period = 0.001
        freqs = scipy.fft.rfftfreq(len(fake_ts), d=sampling_period)
        signal_freq = freqs[r.argmax()]

        data = AnalogSignal(fake_ts, sampling_period=sampling_period * pq.s, units="mV")

        # consistency between different ways of specifying n_per_seg
        # n_per_seg = int(fs/dF) and n_per_seg = len_segment
        frequency_resolution = 1 * pq.Hz
        len_segment = int(data.sampling_rate / frequency_resolution)

        freqs_fr, psd_fr = elephant.spectral.segmented_multitaper_psd(
            data, frequency_resolution=frequency_resolution
        )

        freqs_ls, psd_ls = elephant.spectral.segmented_multitaper_psd(
            data, len_segment=len_segment
        )

        np.testing.assert_array_equal(freqs_fr, freqs_ls)
        np.testing.assert_array_equal(psd_fr, psd_ls)

    def test_segmented_multitaper_psd_parameter_hierarchy(self):
        # generate data by adding white noise and a sinusoid
        data_length = 5000
        sampling_period = 0.001
        signal_freq = 100.0
        noise = np.random.normal(size=data_length)
        signal = [
            np.sin(2 * np.pi * signal_freq * t)
            for t in np.arange(0, data_length * sampling_period, sampling_period)
        ]
        data = AnalogSignal(
            np.array(signal + noise), sampling_period=sampling_period * pq.s, units="mV"
        )

        # test frequency_resolution vs len_segment vs n_segments
        n_segments = 5
        len_segment = 2000
        frequency_resolution = 0.25 * pq.Hz

        freqs_ns, psd_ns = elephant.spectral.segmented_multitaper_psd(
            data, n_segments=n_segments
        )

        freqs_ls, psd_ls = elephant.spectral.segmented_multitaper_psd(
            data, n_segments=n_segments, len_segment=len_segment
        )

        freqs_fr, psd_fr = elephant.spectral.segmented_multitaper_psd(
            data,
            n_segments=n_segments,
            len_segment=len_segment,
            frequency_resolution=frequency_resolution,
        )

        self.assertTrue(freqs_ns.shape < freqs_ls.shape < freqs_fr.shape)
        self.assertTrue(psd_ns.shape < psd_ls.shape < psd_fr.shape)

    def test_segmented_multitaper_psd_input_types(self):
        # generate a test data
        sampling_period = 0.001
        data = AnalogSignal(
            np.array(np.random.normal(size=5000)),
            sampling_period=sampling_period * pq.s,
            units="mV",
        )

        # outputs from AnalogSignal input are of Quantity type (standard usage)
        freqs_neo, psd_neo = elephant.spectral.segmented_multitaper_psd(data)
        self.assertTrue(isinstance(freqs_neo, pq.quantity.Quantity))
        self.assertTrue(isinstance(psd_neo, pq.quantity.Quantity))

        # outputs from Quantity array input are of Quantity type
        freqs_pq, psd_pq = elephant.spectral.segmented_multitaper_psd(
            data.magnitude.flatten() * data.units, fs=1 / sampling_period
        )
        self.assertTrue(isinstance(freqs_pq, pq.quantity.Quantity))
        self.assertTrue(isinstance(psd_pq, pq.quantity.Quantity))

        # outputs from Numpy ndarray input are NOT of Quantity type
        freqs_np, psd_np = elephant.spectral.segmented_multitaper_psd(
            data.magnitude.flatten(), fs=1 / sampling_period
        )
        self.assertFalse(isinstance(freqs_np, pq.quantity.Quantity))
        self.assertFalse(isinstance(psd_np, pq.quantity.Quantity))

        # frequency resolution with and without units
        freq_res_hz = 1 * pq.Hz
        freq_res_int = 1

        freqs_int, psd_int = elephant.spectral.segmented_multitaper_psd(
            data, frequency_resolution=freq_res_int
        )

        freqs_hz, psd_hz = elephant.spectral.segmented_multitaper_psd(
            data, frequency_resolution=freq_res_hz
        )

        np.testing.assert_array_equal(freqs_int, freqs_hz)
        np.testing.assert_array_equal(psd_int, psd_hz)

        # fs with and without units
        fs_hz = 1 * pq.Hz
        fs_int = 1
        freqs_fs_hz, psd_fs_hz = elephant.spectral.multitaper_psd(
            data.magnitude.T, fs=fs_hz
        )
        freqs_fs_int, psd_fs_int = elephant.spectral.multitaper_psd(
            data.magnitude.T, fs=fs_int
        )

        np.testing.assert_array_equal(freqs_fs_hz, freqs_fs_int)
        np.testing.assert_array_equal(psd_fs_hz, psd_fs_int)

        # check if the results from different input types are identical
        self.assertTrue((freqs_neo == freqs_pq).all() and (psd_neo == psd_pq).all())
        self.assertTrue((freqs_neo == freqs_np).all() and (psd_neo == psd_np).all())


class MultitaperCrossSpectrumTestCase(unittest.TestCase):
    def test_multitaper_cross_spectrum_errors(self):
        # generate dummy data
        data_length = 5000
        signal = AnalogSignal(
            np.zeros(data_length), sampling_period=0.001 * pq.s, units="mV"
        )
        fs = signal.sampling_rate

        # check for invalid parameter values
        # - number of tapers
        self.assertRaises(
            ValueError,
            elephant.spectral.multitaper_cross_spectrum,
            signal,
            fs=fs,
            num_tapers=-5,
        )
        self.assertRaises(
            TypeError,
            elephant.spectral.multitaper_cross_spectrum,
            signal,
            fs=fs,
            num_tapers=-5.0,
        )

        # - peak resolution
        self.assertRaises(
            ValueError,
            elephant.spectral.multitaper_cross_spectrum,
            signal,
            fs=fs,
            peak_resolution=-1,
        )

    def test_multitaper_cross_spectrum_behavior(self):
        # generate data (frequency domain to time domain)
        r = np.ones(2501) * 0.2
        r[0], r[500] = 0, 10  # Zero DC, peak at 100 Hz
        phi_x = np.random.uniform(-np.pi, np.pi, len(r))
        phi_y = np.random.uniform(-np.pi, np.pi, len(r))
        fake_coeffs_x = r * np.exp(1j * phi_x)
        fake_coeffs_y = r * np.exp(1j * phi_y)
        signal_x = scipy.fft.irfft(fake_coeffs_x)
        signal_y = scipy.fft.irfft(fake_coeffs_y)
        sampling_period = 0.001
        freqs = scipy.fft.rfftfreq(len(signal_x), d=sampling_period)
        signal_freq = freqs[r.argmax()]
        data = AnalogSignal(
            np.vstack([signal_x, signal_y]).T,
            sampling_period=sampling_period * pq.s,
            units="mV",
        )

        # consistency between different ways of specifying number of tapers
        freqs1, cross_spec1 = elephant.spectral.multitaper_cross_spectrum(
            data, fs=data.sampling_rate, nw=3.5
        )
        freqs2, cross_spec2 = elephant.spectral.multitaper_cross_spectrum(
            data, fs=data.sampling_rate, nw=3.5, num_tapers=6
        )
        self.assertTrue((cross_spec1 == cross_spec2).all() and (freqs1 == freqs2).all())

        # peak resolution and consistency with data
        peak_res = 1.0 * pq.Hz
        freqs, cross_spec = elephant.spectral.multitaper_cross_spectrum(
            data, peak_resolution=peak_res
        )

        self.assertEqual(freqs[cross_spec[0, 0].argmax()], signal_freq)
        freqs_np, cross_spec_np = elephant.spectral.multitaper_cross_spectrum(
            data.magnitude.T, fs=1 / sampling_period, peak_resolution=peak_res
        )
        self.assertTrue(
            (freqs == freqs_np).all() and (cross_spec == cross_spec_np).all()
        )

        # one-sided vs two-sided spectrum
        freqs_os, cross_spec_os = elephant.spectral.multitaper_cross_spectrum(
            data, return_onesided=True
        )

        freqs_ts, cross_spec_ts = elephant.spectral.multitaper_cross_spectrum(
            data, return_onesided=False
        )

        # Nyquist frequency is negative when using onesided=False (fftfreq)
        # See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.rfftfreq.html#scipy.fft.rfftfreq  # noqa
        nonnegative_freqs_indices = np.nonzero(freqs_ts >= 0)[0]
        nyquist_freq_idx = np.abs(freqs_ts).argmax()
        ts_freq_indices = np.append(nonnegative_freqs_indices, nyquist_freq_idx)
        ts_overlap_freqs = (
            np.append(
                freqs_ts[nonnegative_freqs_indices].rescale("Hz").magnitude,
                np.abs(freqs_ts[nyquist_freq_idx].rescale("Hz").magnitude),
            )
            * pq.Hz
        )

        np.testing.assert_array_equal(freqs_os, ts_overlap_freqs)

        np.testing.assert_allclose(
            cross_spec_os.magnitude,
            cross_spec_ts[:, :, ts_freq_indices].magnitude,
            rtol=1e-12,
            atol=0,
        )

    def test_multitaper_cross_spectrum_parameter_hierarchy(self):
        # generate data by adding white noise and a sinusoid
        data_length = 5000
        sampling_period = 0.001
        signal_freq = 100.0
        noise = np.random.normal(size=(2, data_length))
        time_points = np.arange(0, data_length * sampling_period, sampling_period)
        signal_x = np.sin(2 * np.pi * signal_freq * time_points) + noise[0]
        signal_y = np.cos(2 * np.pi * signal_freq * time_points) + noise[1]
        data = AnalogSignal(
            np.vstack([signal_x, signal_y]).T,
            sampling_period=sampling_period * pq.s,
            units="mV",
        )

        # Test num_tapers vs nw
        freqs1, cross_spec1 = elephant.spectral.multitaper_cross_spectrum(
            data, fs=data.sampling_rate, nw=3, num_tapers=9
        )
        freqs2, cross_spec2 = elephant.spectral.multitaper_cross_spectrum(
            data, fs=data.sampling_rate, nw=3
        )

        self.assertTrue((freqs1 == freqs2).all() and (cross_spec1 != cross_spec2).all())

        # Test peak_resolution vs nw
        freqs1, cross_spec1 = elephant.spectral.multitaper_cross_spectrum(
            data, fs=data.sampling_rate, nw=3, num_tapers=9, peak_resolution=1
        )
        freqs2, cross_spec2 = elephant.spectral.multitaper_cross_spectrum(
            data, fs=data.sampling_rate, nw=3, num_tapers=9
        )

        self.assertTrue((freqs1 == freqs2).all() and (cross_spec1 != cross_spec2).all())

    def test_multitaper_cross_spectrum_input_types(self):
        # generate a test data
        data_length = 5000
        sampling_period = 0.001
        signal_freq = 100.0
        noise = np.random.normal(size=(2, data_length))
        time_points = np.arange(0, data_length * sampling_period, sampling_period)
        signal_x = np.sin(2 * np.pi * signal_freq * time_points) + noise[0]
        signal_y = np.cos(2 * np.pi * signal_freq * time_points) + noise[1]
        data = AnalogSignal(
            np.vstack([signal_x, signal_y]).T,
            sampling_period=sampling_period * pq.s,
            units="mV",
        )

        # outputs from AnalogSignal input are of Quantity type (standard usage)
        freqs_neo, cross_spec_neo = elephant.spectral.multitaper_cross_spectrum(data)
        self.assertTrue(isinstance(freqs_neo, pq.quantity.Quantity))
        self.assertTrue(isinstance(cross_spec_neo, pq.quantity.Quantity))

        # outputs from Quantity array input are of Quantity type
        freqs_pq, cross_spec_pq = elephant.spectral.multitaper_cross_spectrum(
            data.magnitude.T * data.units, fs=1 / (sampling_period * pq.s)
        )
        self.assertTrue(isinstance(freqs_pq, pq.quantity.Quantity))
        self.assertTrue(isinstance(cross_spec_pq, pq.quantity.Quantity))

        # outputs from Numpy ndarray input are NOT of Quantity type
        freqs_np, cross_spec_np = elephant.spectral.multitaper_cross_spectrum(
            data.magnitude.T, fs=1 / (sampling_period * pq.s)
        )
        self.assertFalse(isinstance(freqs_np, pq.quantity.Quantity))
        self.assertFalse(isinstance(cross_spec_np, pq.quantity.Quantity))

        # check if the results from different input types are identical
        self.assertTrue(
            (freqs_neo == freqs_pq).all() and (cross_spec_neo == cross_spec_pq).all()
        )
        self.assertTrue(
            (freqs_neo == freqs_np).all() and (cross_spec_neo == cross_spec_np).all()
        )


class SegmentedMultitaperCrossSpectrumTestCase(unittest.TestCase):
    def test_segmented_multitaper_cross_spectrum_errors(self):
        # generate dummy data
        data_length = 5000
        signal = AnalogSignal(
            np.zeros(data_length), sampling_period=0.001 * pq.s, units="mV"
        )
        fs = signal.sampling_rate

        # - frequency resolution
        self.assertRaises(
            ValueError,
            elephant.spectral.segmented_multitaper_cross_spectrum,
            signal,
            fs=fs,
            frequency_resolution=-10,
        )

        # - n per segment
        # n_per_seg = int(fs / dF), where dF is the frequency_resolution
        broken_freq_resolution = fs / (data_length + 1)
        self.assertRaises(
            ValueError,
            elephant.spectral.segmented_multitaper_cross_spectrum,
            signal,
            fs=fs,
            frequency_resolution=broken_freq_resolution,
        )

        # - length of segment (negative)
        self.assertRaises(
            ValueError,
            elephant.spectral.segmented_multitaper_cross_spectrum,
            signal,
            fs=fs,
            len_segment=-10,
        )

        # - length of segment (larger than data length)
        self.assertRaises(
            ValueError,
            elephant.spectral.segmented_multitaper_cross_spectrum,
            signal,
            fs=fs,
            len_segment=data_length + 1,
        )

        # - number of segments (negative)
        self.assertRaises(
            ValueError,
            elephant.spectral.segmented_multitaper_cross_spectrum,
            signal,
            fs=fs,
            n_segments=-10,
        )

        # - number of segments (larger than data length)
        self.assertRaises(
            ValueError,
            elephant.spectral.segmented_multitaper_cross_spectrum,
            signal,
            fs=fs,
            n_segments=data_length + 1,
        )

    def test_segmented_multitaper_cross_spectrum_behavior(self):
        # generate data (frequency domain to time domain)
        r = np.ones(2501) * 0.2
        r[0], r[500] = 0, 10  # Zero DC, peak at 100 Hz
        phi_x = np.random.uniform(-np.pi, np.pi, len(r))
        phi_y = np.random.uniform(-np.pi, np.pi, len(r))
        fake_coeffs_x = r * np.exp(1j * phi_x)
        fake_coeffs_y = r * np.exp(1j * phi_y)
        signal_x = scipy.fft.irfft(fake_coeffs_x)
        signal_y = scipy.fft.irfft(fake_coeffs_y)
        sampling_period = 0.001
        freqs = scipy.fft.rfftfreq(len(signal_x), d=sampling_period)
        signal_freq = freqs[r.argmax()]
        data = AnalogSignal(
            np.vstack([signal_x, signal_y]).T,
            sampling_period=sampling_period * pq.s,
            units="mV",
        )

        # consistency between different ways of specifying n_per_seg
        # n_per_seg = int(fs/dF) and n_per_seg = len_segment
        frequency_resolution = 1 * pq.Hz
        len_segment = int(data.sampling_rate / frequency_resolution)

        freqs_fr, cross_spec_fr = elephant.spectral.segmented_multitaper_cross_spectrum(
            data, frequency_resolution=frequency_resolution
        )

        freqs_ls, cross_spec_ls = elephant.spectral.segmented_multitaper_cross_spectrum(
            data, len_segment=len_segment
        )

        np.testing.assert_array_equal(freqs_fr, freqs_ls)
        np.testing.assert_array_equal(cross_spec_fr, cross_spec_ls)

        # one-sided vs two-sided spectrum
        freqs_os, cross_spec_os = elephant.spectral.segmented_multitaper_cross_spectrum(
            data, return_onesided=True
        )

        freqs_ts, cross_spec_ts = elephant.spectral.segmented_multitaper_cross_spectrum(
            data, return_onesided=False
        )

        # test overlap parameter
        no_overlap = 0
        half_overlap = 0.5
        large_overlap = 0.99
        n_segments = 10

        freqs_no, cross_spec_no = elephant.spectral.segmented_multitaper_cross_spectrum(
            data, n_segments=n_segments, overlap=no_overlap
        )

        freqs_ho, cross_spec_ho = elephant.spectral.segmented_multitaper_cross_spectrum(
            data, n_segments=n_segments, overlap=half_overlap
        )

        freqs_lo, cross_spec_lo = elephant.spectral.segmented_multitaper_cross_spectrum(
            data, n_segments=n_segments, overlap=large_overlap
        )

        self.assertTrue(freqs_no.shape < freqs_ho.shape < freqs_lo.shape)
        self.assertTrue(cross_spec_no.shape < cross_spec_ho.shape < cross_spec_lo.shape)

        # Nyquist frequency is negative when using onesided=False (fftfreq)
        # See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.rfftfreq.html#scipy.fft.rfftfreq  # noqa
        nonnegative_freqs_indices = np.nonzero(freqs_ts >= 0)[0]
        nyquist_freq_idx = np.abs(freqs_ts).argmax()
        ts_freq_indices = np.append(nonnegative_freqs_indices, nyquist_freq_idx)
        ts_overlap_freqs = (
            np.append(
                freqs_ts[nonnegative_freqs_indices].rescale("Hz").magnitude,
                np.abs(freqs_ts[nyquist_freq_idx].rescale("Hz").magnitude),
            )
            * pq.Hz
        )

        np.testing.assert_array_equal(freqs_os, ts_overlap_freqs)

        np.testing.assert_allclose(
            cross_spec_os.magnitude,
            cross_spec_ts[:, :, ts_freq_indices].magnitude,
            rtol=1e-12,
            atol=0,
        )

    def test_segmented_multitaper_cross_spectrum_parameter_hierarchy(self):
        # test frequency_resolution vs len_segment vs n_segments
        # generate data (frequency domain to time domain)
        r = np.ones(2501) * 0.2
        r[0], r[500] = 0, 10  # Zero DC, peak at 100 Hz
        phi_x = np.random.uniform(-np.pi, np.pi, len(r))
        phi_y = np.random.uniform(-np.pi, np.pi, len(r))
        fake_coeffs_x = r * np.exp(1j * phi_x)
        fake_coeffs_y = r * np.exp(1j * phi_y)
        signal_x = scipy.fft.irfft(fake_coeffs_x)
        signal_y = scipy.fft.irfft(fake_coeffs_y)
        sampling_period = 0.001
        freqs = scipy.fft.rfftfreq(len(signal_x), d=sampling_period)
        signal_freq = freqs[r.argmax()]
        data = AnalogSignal(
            np.vstack([signal_x, signal_y]).T,
            sampling_period=sampling_period * pq.s,
            units="mV",
        )

        n_segments = 5
        len_segment = 2000
        frequency_resolution = 1 * pq.Hz

        freqs_ns, cross_spec_ns = elephant.spectral.segmented_multitaper_cross_spectrum(
            data, n_segments=n_segments
        )

        freqs_ls, cross_spec_ls = elephant.spectral.segmented_multitaper_cross_spectrum(
            data, n_segments=n_segments, len_segment=len_segment
        )

        freqs_fr, cross_spec_fr = elephant.spectral.segmented_multitaper_cross_spectrum(
            data,
            n_segments=n_segments,
            len_segment=len_segment,
            frequency_resolution=frequency_resolution,
        )

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
        time_points = np.arange(0, data_length * sampling_period, sampling_period)
        signal_x = np.sin(2 * np.pi * signal_freq * time_points) + noise[0]
        signal_y = np.cos(2 * np.pi * signal_freq * time_points) + noise[1]
        data = AnalogSignal(
            np.vstack([signal_x, signal_y]).T,
            sampling_period=sampling_period * pq.s,
            units="mV",
        )

        freqs1, psd_multitaper = elephant.spectral.multitaper_psd(
            signal=data, fs=data.sampling_rate, nw=4, num_tapers=8
        )

        psd_multitaper[:, 1:] /= 2  # since comparing rfft and fft results

        freqs2, cross_spec = elephant.spectral.segmented_multitaper_cross_spectrum(
            data, fs=data.sampling_rate, nw=4, num_tapers=8, return_onesided=True
        )

        self.assertTrue((freqs1 == freqs2).all())

        np.testing.assert_allclose(
            psd_multitaper.magnitude,
            np.diagonal(cross_spec).T.real.magnitude,
            rtol=0.01,
            atol=0.01,
        )

    def test_segmented_multitaper_cross_spectrum_input_types(self):
        # generate a test data
        data_length = 5000
        sampling_period = 0.001
        signal_freq = 100.0
        noise = np.random.normal(size=(2, data_length))
        time_points = np.arange(0, data_length * sampling_period, sampling_period)
        signal_x = np.sin(2 * np.pi * signal_freq * time_points) + noise[0]
        signal_y = np.cos(2 * np.pi * signal_freq * time_points) + noise[1]
        data = AnalogSignal(
            np.vstack([signal_x, signal_y]).T,
            sampling_period=sampling_period * pq.s,
            units="mV",
        )

        # frequency resolution as an integer
        freq_res_int = 1
        freq_res_hz = 1 * pq.Hz

        freqs_int, cross_spec_int = (
            elephant.spectral.segmented_multitaper_cross_spectrum(
                data, frequency_resolution=freq_res_int
            )
        )

        freqs_hz, cross_spec_hz = elephant.spectral.segmented_multitaper_cross_spectrum(
            data, frequency_resolution=freq_res_hz
        )

        np.testing.assert_array_equal(freqs_int, freqs_hz)
        np.testing.assert_array_equal(cross_spec_int, cross_spec_hz)


class MultitaperCoherenceTestCase(unittest.TestCase):
    def test_multitaper_coherence_input_types(self):
        # Generate dummy data
        data_length = 10000
        sampling_period = 0.001
        signal_freq = 100.0
        np.random.seed(123)
        noise = np.random.normal(size=(2, data_length))
        time_points = np.arange(0, data_length * sampling_period, sampling_period)
        # Signals are designed to have coherence peak at `signal_freq`
        arr_signal_i = np.sin(2 * np.pi * signal_freq * time_points) + noise[0]
        arr_signal_j = np.cos(2 * np.pi * signal_freq * time_points) + noise[1]

        fs = 1000 * pq.Hz
        anasig_signal_i = neo.core.AnalogSignal(
            arr_signal_i, sampling_rate=fs, units=pq.mV
        )
        anasig_signal_j = neo.core.AnalogSignal(
            arr_signal_j, sampling_rate=fs, units=pq.mV
        )

        arr_f, arr_coh, arr_phi = elephant.spectral.multitaper_coherence(
            arr_signal_i, arr_signal_j, fs=fs
        )
        anasig_f, anasig_coh, anasig_phi = elephant.spectral.multitaper_coherence(
            anasig_signal_i, anasig_signal_j
        )

        np.testing.assert_array_equal(arr_f, anasig_f)
        np.testing.assert_allclose(arr_coh, anasig_coh, atol=1e-6)
        np.testing.assert_array_equal(arr_phi, anasig_phi)

    def test_multitaper_cohere_peak(self):
        # Generate dummy data
        data_length = 10000
        sampling_period = 0.001
        signal_freq = 100.0
        noise = np.random.normal(size=(2, data_length))
        time_points = np.arange(0, data_length * sampling_period, sampling_period)
        # Signals are designed to have coherence peak at `signal_freq`
        signal_i = np.sin(2 * np.pi * signal_freq * time_points) + noise[0]
        signal_j = np.cos(2 * np.pi * signal_freq * time_points) + noise[1]

        # Estimate coherence and phase lag with the multitaper method
        freq1, coh1, phase_lag1 = elephant.spectral.multitaper_coherence(
            signal_i, signal_j, fs=1 / sampling_period, n_segments=16
        )

        indices, vals = scipy.signal.find_peaks(coh1, height=0.8, distance=10)

        peak_freqs = freq1[indices]

        np.testing.assert_allclose(
            peak_freqs, signal_freq * np.ones(len(peak_freqs)), rtol=0.05
        )

    @pytest.mark.skipif(
        version.parse(np.__version__) > version.parse("1.25.0"),
        reason="This test will fail with numpy version"
        "1.25.0 - 1.25.2,  see issue #24000"
        "https://github.com/numpy/numpy/issues/24000 ",
    )
    def test_multitaper_cohere_perfect_cohere(self):
        # Generate dummy data
        data_length = 10000
        sampling_period = 0.001
        signal_freq = 100.0
        noise = np.random.normal(size=(1, data_length))
        time_points = np.arange(0, data_length * sampling_period, sampling_period)
        signal = np.cos(2 * np.pi * signal_freq * time_points) + noise

        # Estimate coherence and phase lag with the multitaper method
        freq1, coh, phase_lag = elephant.spectral.multitaper_coherence(
            signal, signal, fs=1 / sampling_period, n_segments=16
        )

        np.testing.assert_array_equal(phase_lag, np.zeros(phase_lag.size))
        np.testing.assert_array_equal(coh, np.ones(coh.size))

    def test_multitaper_cohere_no_cohere(self):
        # Generate dummy data
        data_length = 10000
        sampling_period = 0.001
        time_points = np.arange(0, data_length * sampling_period, sampling_period)

        signal_i = np.sin(2 * np.pi * 2.5 * time_points)
        signal_j = np.sin(2 * np.pi * 5 * time_points)

        # Estimate coherence and phase lag with the multitaper method
        freq, coh, phase_lag = elephant.spectral.multitaper_coherence(
            signal_i, signal_j, fs=1 / sampling_period, n_segments=16
        )

        np.testing.assert_allclose(coh, np.zeros(coh.size), atol=0.002)

    def test_multitaper_cohere_phase_lag(self):
        # Generate dummy data
        data_length = 10000
        sampling_period = 0.001
        signal_freq = 100.0
        time_points = np.arange(0, data_length * sampling_period, sampling_period)

        # Signals are designed to have maximal phase lag at 100 with value pi/4
        signal_i = np.sin(2 * np.pi * signal_freq * time_points + np.pi / 4)
        signal_j = np.cos(2 * np.pi * signal_freq * time_points)

        # Estimate coherence and phase lag with the multitaper method
        freq, coh, phase_lag = elephant.spectral.multitaper_coherence(
            signal_i, signal_j, fs=1 / sampling_period, n_segments=16, num_tapers=8
        )

        indices, vals = scipy.signal.find_peaks(
            phase_lag, height=0.8 * np.pi / 4, distance=10
        )

        # Get peak frequencies and peak heights
        peak_freqs = freq[indices]
        peak_heights = vals["peak_heights"]

        np.testing.assert_allclose(
            peak_freqs, signal_freq * np.ones(len(peak_freqs)), rtol=0.05
        )
        np.testing.assert_allclose(
            peak_heights, np.pi / 4 * np.ones(len(peak_heights)), rtol=0.05
        )


class WelchCohereTestCase(unittest.TestCase):
    def test_welch_cohere_errors(self):
        # generate a dummy data
        x = AnalogSignal(np.zeros(5000), sampling_period=0.001 * pq.s, units="mV")
        y = AnalogSignal(np.zeros(5000), sampling_period=0.001 * pq.s, units="mV")

        # check for invalid parameter values
        # - length of segments
        self.assertRaises(
            ValueError, elephant.spectral.welch_coherence, x, y, len_segment=0
        )
        self.assertRaises(
            ValueError,
            elephant.spectral.welch_coherence,
            x,
            y,
            len_segment=x.shape[0] * 2,
        )
        # - number of segments
        self.assertRaises(
            ValueError, elephant.spectral.welch_coherence, x, y, n_segments=0
        )
        self.assertRaises(
            ValueError,
            elephant.spectral.welch_coherence,
            x,
            y,
            n_segments=x.shape[0] * 2,
        )
        # - frequency resolution
        self.assertRaises(
            ValueError, elephant.spectral.welch_coherence, x, y, frequency_resolution=-1
        )
        self.assertRaises(
            ValueError,
            elephant.spectral.welch_coherence,
            x,
            y,
            frequency_resolution=x.sampling_rate / (x.shape[0] + 1),
        )
        # - overlap
        self.assertRaises(
            ValueError, elephant.spectral.welch_coherence, x, y, overlap=-1.0
        )
        self.assertRaises(
            ValueError, elephant.spectral.welch_coherence, x, y, overlap=1.1
        )

    def test_welch_cohere_warnings(self):
        # generate a dummy data
        x = AnalogSignal(np.zeros(5000), sampling_period=0.001 * pq.s, units="mV")
        y = AnalogSignal(np.zeros(5000), sampling_period=0.001 * pq.s, units="mV")
        # Test deprecation warning for 'hanning' window
        self.assertWarns(
            DeprecationWarning,
            elephant.spectral.welch_coherence,
            x,
            y,
            window="hanning",
        )

    def test_welch_cohere_behavior(self):
        # generate data by adding white noise and a sinusoid
        data_length = 5000
        sampling_period = 0.001
        signal_freq = 100.0
        noise1 = np.random.normal(size=data_length) * 0.01
        noise2 = np.random.normal(size=data_length) * 0.01
        signal1 = [
            np.cos(2 * np.pi * signal_freq * t)
            for t in np.arange(0, data_length * sampling_period, sampling_period)
        ]
        signal2 = [
            np.sin(2 * np.pi * signal_freq * t)
            for t in np.arange(0, data_length * sampling_period, sampling_period)
        ]
        x = AnalogSignal(
            np.array(signal1 + noise1),
            units="mV",
            sampling_period=sampling_period * pq.s,
        )
        y = AnalogSignal(
            np.array(signal2 + noise2),
            units="mV",
            sampling_period=sampling_period * pq.s,
        )

        # consistency between different ways of specifying segment length
        freqs1, coherency1, phase_lag1 = elephant.spectral.welch_coherence(
            x, y, len_segment=data_length // 5, overlap=0
        )
        freqs2, coherency2, phase_lag2 = elephant.spectral.welch_coherence(
            x, y, n_segments=5, overlap=0
        )
        self.assertTrue(
            (coherency1 == coherency2).all()
            and (phase_lag1 == phase_lag2).all()
            and (freqs1 == freqs2).all()
        )

        # frequency resolution and consistency with data
        freq_res = 1.0 * pq.Hz
        freqs, coherency, phase_lag = elephant.spectral.welch_coherence(
            x, y, frequency_resolution=freq_res
        )
        self.assertAlmostEqual(freq_res, freqs[1] - freqs[0])
        self.assertAlmostEqual(freqs[coherency.argmax()], signal_freq, places=2)
        self.assertAlmostEqual(phase_lag[coherency.argmax()], -np.pi / 2, places=2)
        freqs_np, coherency_np, phase_lag_np = elephant.spectral.welch_coherence(
            x.magnitude.flatten(),
            y.magnitude.flatten(),
            fs=1 / sampling_period,
            frequency_resolution=freq_res,
        )
        assert_array_equal(freqs.simplified.magnitude, freqs_np)
        assert_array_equal(coherency[:, 0], coherency_np)
        assert_array_equal(phase_lag[:, 0], phase_lag_np)

        # - check the behavior of parameter `axis` using multidimensional data
        num_channel = 4
        data_length = 5000
        x_multidim = np.random.normal(size=(num_channel, data_length))
        y_multidim = np.random.normal(size=(num_channel, data_length))
        freqs, coherency, phase_lag = elephant.spectral.welch_coherence(
            x_multidim, y_multidim
        )
        freqs_T, coherency_T, phase_lag_T = elephant.spectral.welch_coherence(
            x_multidim.T, y_multidim.T, axis=0
        )
        assert_array_equal(freqs, freqs_T)
        assert_array_equal(coherency, coherency_T.T)
        assert_array_equal(phase_lag, phase_lag_T.T)

    def test_welch_cohere_input_types(self):
        # generate a test data
        sampling_period = 0.001
        x = AnalogSignal(
            np.array(np.random.normal(size=5000)),
            sampling_period=sampling_period * pq.s,
            units="mV",
        )
        y = AnalogSignal(
            np.array(np.random.normal(size=5000)),
            sampling_period=sampling_period * pq.s,
            units="mV",
        )

        # outputs from AnalogSignal input are of Quantity type
        # (standard usage)
        freqs_neo, coherency_neo, phase_lag_neo = elephant.spectral.welch_coherence(
            x, y
        )
        self.assertTrue(isinstance(freqs_neo, pq.quantity.Quantity))
        self.assertTrue(isinstance(phase_lag_neo, pq.quantity.Quantity))

        # outputs from Quantity array input are of Quantity type
        freqs_pq, coherency_pq, phase_lag_pq = elephant.spectral.welch_coherence(
            x.magnitude.flatten() * x.units,
            y.magnitude.flatten() * y.units,
            fs=1 / sampling_period,
        )
        self.assertTrue(isinstance(freqs_pq, pq.quantity.Quantity))
        self.assertTrue(isinstance(phase_lag_pq, pq.quantity.Quantity))

        # outputs from Numpy ndarray input are NOT of Quantity type
        freqs_np, coherency_np, phase_lag_np = elephant.spectral.welch_coherence(
            x.magnitude.flatten(), y.magnitude.flatten(), fs=1 / sampling_period
        )
        self.assertFalse(isinstance(freqs_np, pq.quantity.Quantity))
        self.assertFalse(isinstance(phase_lag_np, pq.quantity.Quantity))

        # check if the results from different input types are identical
        self.assertTrue(
            (freqs_neo == freqs_pq).all()
            and (coherency_neo[:, 0] == coherency_pq).all()
            and (phase_lag_neo[:, 0] == phase_lag_pq).all()
        )
        self.assertTrue(
            (freqs_neo == freqs_np).all()
            and (coherency_neo[:, 0] == coherency_np).all()
            and (phase_lag_neo[:, 0] == phase_lag_np).all()
        )

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
        x_neo = AnalogSignal(x_np.T, units="mV", sampling_period=sampling_period * pq.s)
        y_neo = AnalogSignal(y_np.T, units="mV", sampling_period=sampling_period * pq.s)
        x_neo_1dim = AnalogSignal(
            x_np[0], units="mV", sampling_period=sampling_period * pq.s
        )
        y_neo_1dim = AnalogSignal(
            y_np[0], units="mV", sampling_period=sampling_period * pq.s
        )

        # check if the results from different input types are identical
        freqs_np, coherency_np, phase_lag_np = elephant.spectral.welch_coherence(
            x_np, y_np, fs=1 / sampling_period
        )
        freqs_neo, coherency_neo, phase_lag_neo = elephant.spectral.welch_coherence(
            x_neo, y_neo
        )
        freqs_neo_1dim, coherency_neo_1dim, phase_lag_neo_1dim = (
            elephant.spectral.welch_coherence(x_neo_1dim, y_neo_1dim)
        )
        self.assertTrue(np.all(freqs_np == freqs_neo))
        self.assertTrue(np.all(coherency_np.T == coherency_neo))
        self.assertTrue(np.all(phase_lag_np.T == phase_lag_neo))
        self.assertTrue(np.all(coherency_neo_1dim[:, 0] == coherency_neo[:, 0]))
        self.assertTrue(np.all(phase_lag_neo_1dim[:, 0] == phase_lag_neo[:, 0]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
