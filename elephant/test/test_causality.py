# -*- coding: utf-8 -*-
"""
Unit tests for the causality module.

:copyright: Copyright 2014-2024 by the Elephant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
"""

from __future__ import division, print_function

import unittest

import numpy as np
import quantities as pq
from neo.core import AnalogSignal
from numpy.testing import assert_array_almost_equal

from elephant.spectral import multitaper_cross_spectrum, multitaper_coherence
import elephant.causality.granger
from elephant.datasets import download_datasets


class PairwiseGrangerTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(1)
        cls.ground_truth = cls._generate_ground_truth()

    @staticmethod
    def _generate_ground_truth(length_2d=30000):
        order = 2
        signal = np.zeros((2, length_2d + order))

        weights_1 = np.array([[0.9, 0], [0.9, -0.8]])
        weights_2 = np.array([[-0.5, 0], [-0.2, -0.5]])

        weights = np.stack((weights_1, weights_2))

        noise_covariance = np.array([[1.0, 0.0], [0.0, 1.0]])

        for i in range(length_2d):
            for lag in range(order):
                signal[:, i + order] += np.dot(weights[lag], signal[:, i + 1 - lag])
            rnd_var = np.random.multivariate_normal([0, 0], noise_covariance)
            signal[:, i + order] += rnd_var

        signal = signal[:, 2:]

        # Return signals as Nx2
        return signal.T

    def setUp(self):
        # Generate a smaller random dataset for tests other than ground truth,
        # using a different seed than in the ground truth - the convergence
        # should not depend on the seed.
        np.random.seed(10)
        self.signal = self._generate_ground_truth(length_2d=1000)

        # Estimate Granger causality
        self.causality = elephant.causality.granger.pairwise_granger(
            self.signal, max_order=10, information_criterion="bic"
        )

    def test_analog_signal_input(self):
        """
        Check if analog signal input result matches an otherwise identical 2D
        numpy array input result.
        """
        analog_signal = AnalogSignal(self.signal, units="V", sampling_rate=1 * pq.Hz)
        analog_signal_causality = elephant.causality.granger.pairwise_granger(
            analog_signal, max_order=10, information_criterion="bic"
        )
        self.assertEqual(
            analog_signal_causality.directional_causality_x_y,
            self.causality.directional_causality_x_y,
        )
        self.assertEqual(
            analog_signal_causality.directional_causality_y_x,
            self.causality.directional_causality_y_x,
        )
        self.assertEqual(
            analog_signal_causality.instantaneous_causality,
            self.causality.instantaneous_causality,
        )
        self.assertEqual(
            analog_signal_causality.total_interdependence,
            self.causality.total_interdependence,
        )

    def test_aic(self):
        identity_matrix = np.eye(2, 2)
        self.assertEqual(
            elephant.causality.granger._aic(
                identity_matrix, order=2, dimension=2, length=2
            ),
            8.0,
        )

    def test_bic(self):
        identity_matrix = np.eye(2, 2)
        assert_array_almost_equal(
            elephant.causality.granger._bic(
                identity_matrix, order=2, dimension=2, length=2
            ),
            5.54517744,
            decimal=8,
        )

    def test_lag_covariances_error(self):
        """
        Check that if a signal length is shorter than the set max_lag, a
        ValueError is raised.
        """
        short_signals = np.array([[1, 2], [3, 4]])
        self.assertRaises(
            ValueError,
            elephant.causality.granger._lag_covariances,
            short_signals,
            dimension=2,
            max_lag=3,
        )

    def test_pairwise_granger_error_null_signals(self):
        null_signals = np.array([[0, 0], [0, 0]])
        self.assertRaises(
            ValueError,
            elephant.causality.granger.pairwise_granger,
            null_signals,
            max_order=2,
        )

    def test_pairwise_granger_identical_signal(self):
        same_signal = np.hstack(
            [self.signal[:, 0, np.newaxis], self.signal[:, 0, np.newaxis]]
        )
        self.assertRaises(
            ValueError,
            elephant.causality.granger.pairwise_granger,
            signals=same_signal,
            max_order=2,
        )

    def test_pairwise_granger_error_1d_array(self):
        array_1d = np.ones(10, dtype=np.float32)
        self.assertRaises(
            ValueError,
            elephant.causality.granger.pairwise_granger,
            array_1d,
            max_order=2,
        )

    def test_result_namedtuple(self):
        """
        Check if the result of pairwise_granger is in the form of namedtuple.
        """
        # Import the namedtuple class for the result formatting
        from elephant.causality.granger import Causality

        # Check that the output matches the class
        self.assertIsInstance(self.causality, Causality)

    def test_result_directional_causalities_not_negative(self):
        """
        The directional causalities should never be negative.
        """
        self.assertTrue(self.causality.directional_causality_x_y >= 0)
        self.assertTrue(self.causality.directional_causality_y_x >= 0)

    def test_result_instantaneous_causality_not_negative(self):
        """
        The time-series granger instantaneous causality should never assume
        negative values.
        """
        self.assertTrue(self.causality.instantaneous_causality >= 0)

    def test_total_channel_interdependence_equals_sum_of_other_three(self):
        """
        Test if total interdependence is equal to the sum of the other three
        measures. It should be equal. In this test, however, almost equality
        is asserted due to a loss of significance with larger datasets.
        """
        causality_sum = (
            self.causality.directional_causality_x_y
            + self.causality.directional_causality_y_x
            + self.causality.instantaneous_causality
        )
        assert_array_almost_equal(
            self.causality.total_interdependence, causality_sum, decimal=2
        )

    def test_all_four_result_values_are_floats(self):
        self.assertIsInstance(self.causality.directional_causality_x_y, float)
        self.assertIsInstance(self.causality.directional_causality_y_x, float)
        self.assertIsInstance(self.causality.instantaneous_causality, float)
        self.assertIsInstance(self.causality.total_interdependence, float)

    def test_ground_truth_vector_autoregressive_model(self):
        """
        Test the output of _optimal_vector_arm against the output of R vars
        generated using VAR(t(signal), lag.max=10, ic='AIC').
        """
        # First equation coefficients from R vars
        first_y1_l1 = 0.8947573989
        first_y2_l1 = -0.0003449514
        first_y1_l2 = -0.4934377020
        first_y2_l2 = -0.0018548490

        # Second equation coefficients from R vars
        second_y1_l1 = 9.009503e-01
        second_y2_l1 = -8.124731e-01
        second_y1_l2 = -1.871460e-01
        second_y2_l2 = -5.012730e-01

        coefficients, _, _ = elephant.causality.granger._optimal_vector_arm(
            self.ground_truth.T, dimension=2, max_order=10, information_criterion="aic"
        )

        # Arrange the ground truth values in the same shape as coefficients
        ground_truth_coefficients = np.asarray(
            [
                [[first_y1_l1, first_y2_l1], [second_y1_l1, second_y2_l1]],
                [[first_y1_l2, first_y2_l2], [second_y1_l2, second_y2_l2]],
            ]
        )

        assert_array_almost_equal(coefficients, ground_truth_coefficients, decimal=4)

    def test_wrong_kwarg_optimal_vector_arm(self):
        wrong_ic_criterion = "cic"

        self.assertRaises(
            ValueError,
            elephant.causality.granger._optimal_vector_arm,
            self.ground_truth.T,
            2,
            10,
            wrong_ic_criterion,
        )


class ConditionalGrangerTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(1)
        cls.ground_truth = cls._generate_ground_truth()

    @staticmethod
    def _generate_ground_truth(length_2d=30000, causality_type="indirect"):
        """
        Recreated from Example 2 section 5.2 of :cite:'granger-Ding06-0608035'.
        The following should generate three signals in one of the two ways:
         1. "indirect" would generate data which contains no direct
        causal influence from Y to X, but mediated through Z
        (i.e. Y -> Z -> X).
        2. "both" would generate data which contains both direct and indirect
        causal influences from Y to X.

        """
        if causality_type == "indirect":
            y_t_lag_2 = 0
        elif causality_type == "both":
            y_t_lag_2 = 0.2
        else:
            raise ValueError("causality_type should be either 'indirect' or 'both'")

        order = 2
        signal = np.zeros((3, length_2d + order))

        weights_1 = np.array([[0.8, 0, 0.4], [0, 0.9, 0], [0.0, 0.5, 0.5]])

        weights_2 = np.array([[-0.5, y_t_lag_2, 0.0], [0.0, -0.8, 0], [0, 0, -0.2]])

        weights = np.stack((weights_1, weights_2))

        noise_covariance = np.array([[0.3, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.2]])

        for i in range(length_2d):
            for lag in range(order):
                signal[:, i + order] += np.dot(weights[lag], signal[:, i + 1 - lag])
            rnd_var = np.random.multivariate_normal([0, 0, 0], noise_covariance)
            signal[:, i + order] += rnd_var

        signal = signal[:, 2:]

        # Return signals as Nx3
        return signal.T

    def setUp(self):
        # Generate a smaller random dataset for tests other than ground truth,
        # using a different seed than in the ground truth - the convergence
        # should not depend on the seed.
        np.random.seed(10)
        self.signal = self._generate_ground_truth(length_2d=1000)

        # Generate a small dataset for containing both direct and indirect
        # causality.
        self.non_zero_signal = self._generate_ground_truth(
            length_2d=1000, causality_type="both"
        )
        # Estimate Granger causality
        self.conditional_causality = elephant.causality.granger.conditional_granger(
            self.signal, max_order=10, information_criterion="bic"
        )

    def test_result_is_float(self):
        self.assertIsInstance(self.conditional_causality, float)

    def test_ground_truth_zero_value_conditional_causality(self):
        self.assertEqual(
            elephant.causality.granger.conditional_granger(
                self.ground_truth, 10, "bic"
            ),
            0.0,
        )

    def test_ground_truth_zero_value_conditional_causality_anasig(self):
        signals = AnalogSignal(self.ground_truth, sampling_rate=1 * pq.Hz, units="V")
        self.assertEqual(
            elephant.causality.granger.conditional_granger(signals, 10, "bic"), 0.0
        )

    def test_non_zero_conditional_causality(self):
        self.assertGreater(
            elephant.causality.granger.conditional_granger(
                self.non_zero_signal, 10, "bic"
            ),
            0.0,
        )

    def test_conditional_causality_wrong_input_shape(self):
        signals = np.random.normal(0, 1, (4, 10, 1))

        self.assertRaises(
            ValueError,
            elephant.causality.granger.conditional_granger,
            signals,
            10,
            "bic",
        )


class PairwiseSpectralGrangerTestCase(unittest.TestCase):
    def test_bracket_operator_one_signal(self):
        # Generate a spectrum from random dataset and test bracket operator
        np.random.seed(10)
        n = 10
        spectrum = np.random.normal(0, 1, n)

        # Generate causal part according to The Factorization of Matricial
        # Spectral Densities', Wilson 1972, SiAM J Appl Math, Definition 1.2
        # (ii).

        spectrum_causal = np.fft.ifft(spectrum, axis=0)
        spectrum_causal[(n + 1) // 2 :] = 0
        spectrum_causal[0] /= 2

        spectrum_causal_ground_truth = np.fft.fft(spectrum_causal, axis=0)

        spectrum_causal_est = elephant.causality.granger._bracket_operator(
            spectrum=spectrum, num_freqs=n, num_signals=1
        )

        np.testing.assert_array_almost_equal(
            spectrum_causal_est, spectrum_causal_ground_truth
        )

    def test_bracket_operator_mult_signal(self):
        # Generate a spectrum from random dataset and test bracket operator
        np.random.seed(10)
        n = 10
        num_signals = 3
        spectrum = np.random.normal(0, 1, (n, num_signals, num_signals))

        # Generate causal part according to The Factorization of Matricial
        # Spectral Densities', Wilson 1972, SiAM J Appl Math, Definition 1.2
        # (ii).

        spectrum_causal = np.fft.ifft(spectrum, axis=0)
        spectrum_causal[(n + 1) // 2 :] = 0
        spectrum_causal[0] /= 2

        spectrum_causal_ground_truth = np.fft.fft(spectrum_causal, axis=0)

        # Set element below diagonal at zero frequency to zero
        spectrum_causal_ground_truth[0, 1, 0] = 0
        spectrum_causal_ground_truth[0, 2, 0] = 0
        spectrum_causal_ground_truth[0, 2, 1] = 0

        spectrum_causal_est = elephant.causality.granger._bracket_operator(
            spectrum=spectrum, num_freqs=n, num_signals=num_signals
        )

        np.testing.assert_array_almost_equal(
            spectrum_causal_est, spectrum_causal_ground_truth
        )

    def test_spectral_factorization(self):
        np.random.seed(11)
        n = 100
        num_signals = 2
        signals = np.random.normal(0, 1, (num_signals, n))

        _, cross_spec = multitaper_cross_spectrum(signals, return_onesided=True)

        cross_spec = np.transpose(cross_spec, (2, 0, 1))

        cov_matrix, transfer_function = (
            elephant.causality.granger._spectral_factorization(
                cross_spec, num_iterations=100
            )
        )

        cross_spec_est = np.matmul(
            np.matmul(transfer_function, cov_matrix),
            elephant.causality.granger._dagger(transfer_function),
        )

        np.testing.assert_array_almost_equal(cross_spec, cross_spec_est)

    def test_spectral_factorization_non_conv_exception(self):
        np.random.seed(11)
        n = 10
        num_signals = 2
        signals = np.random.normal(0, 1, (num_signals, n))

        _, cross_spec = multitaper_cross_spectrum(signals, return_onesided=True)

        cross_spec = np.transpose(cross_spec, (2, 0, 1))

        self.assertRaises(
            Exception,
            elephant.causality.granger._spectral_factorization,
            cross_spec,
            num_iterations=1,
        )

    def test_spectral_factorization_initial_cond(self):
        # Cross spectrum at zero frequency must always be symmetric
        wrong_cross_spec = np.array([[[1, 2], [-1, 1]], [[1, 1], [1, 1]]])
        self.assertRaises(
            ValueError,
            elephant.causality.granger._spectral_factorization,
            wrong_cross_spec,
            num_iterations=10,
        )

    def test_dagger_2d(self):
        matrix_array = np.array([[1j, 0], [2, 3]], dtype=complex)

        true_dagger = np.array([[-1j, 2], [0, 3]], dtype=complex)

        dagger_matrix_array = elephant.causality.granger._dagger(matrix_array)

        np.testing.assert_array_equal(true_dagger, dagger_matrix_array)

    def test_total_channel_interdependence_equals_transformed_coherence(self):
        np.random.seed(11)
        n = int(2**8)
        num_signals = 2
        signals = np.random.normal(0, 1, (num_signals, n))

        freqs, coh, phase_lag = multitaper_coherence(
            signals[0], signals[1], len_segment=2**7, num_tapers=2
        )
        f, spectral_causality = elephant.causality.granger.pairwise_spectral_granger(
            signals[0], signals[1], len_segment=2**7, num_tapers=2
        )

        total_interdependence = spectral_causality[3]
        # Cut last frequency due to length of segment being even and
        # multitaper_coherence using the real FFT in contrast to
        # pairwise_spectral_granger which has to use the full FFT.
        true_total_interdependence = -np.log(1 - coh)[:-1]
        np.testing.assert_allclose(
            total_interdependence, true_total_interdependence, atol=1e-5
        )

    def test_pairwise_spectral_granger_against_ground_truth(self):
        """
        Test pairwise_spectral_granger using an example from Ding et al. 2006

        Please follow the link below for more details:
        https://gin.g-node.org/NeuralEnsemble/elephant-data/src/master/unittest/causality/granger/pairwise_spectral_granger  # noqa

        """

        repo_path = r"unittest/causality/granger/pairwise_spectral_granger/data"

        files_to_download = [
            ("time_series.npy", "54e0b3fbd904ccb48c75228c070a1a2a"),
            ("weights.npy", "eb1fc5590da5507293c63b25b1e3a7fc"),
            ("noise_covariance.npy", "6f80ccff2b2aa9485dc9c01d81570bf5"),
        ]

        downloaded_files = {}
        for filename, checksum in files_to_download:
            downloaded_files[filename] = {
                "filename": filename,
                "path": download_datasets(
                    repo_path=f"{repo_path}/{filename}", checksum=checksum
                ),
            }

        signals = np.load(downloaded_files["time_series.npy"]["path"])
        weights = np.load(downloaded_files["weights.npy"]["path"])
        cov = np.load(downloaded_files["noise_covariance.npy"]["path"])

        # Estimate spectral Granger Causality
        f, spectral_causality = elephant.causality.granger.pairwise_spectral_granger(
            signals[0], signals[1], len_segment=2**7, num_tapers=3
        )

        # Calculate ground truth spectral Granger Causality
        # Formulae taken from Ding et al., Granger Causality: Basic Theory and
        # Application to Neuroscience, 2006
        fn = np.linspace(0, np.pi, len(f))
        freqs_for_theo = np.array([1, 2])[:, np.newaxis] * fn
        A_theo = np.identity(2)[np.newaxis] - weights[0] * np.exp(
            -1j * freqs_for_theo[0][:, np.newaxis, np.newaxis]
        )
        A_theo -= weights[1] * np.exp(
            -1j * freqs_for_theo[1][:, np.newaxis, np.newaxis]
        )

        H_theo = np.array(
            [[A_theo[:, 1, 1], -A_theo[:, 0, 1]], [-A_theo[:, 1, 0], A_theo[:, 0, 0]]]
        )
        H_theo /= np.linalg.det(A_theo)
        H_theo = np.moveaxis(H_theo, 2, 0)

        S_theo = np.matmul(
            np.matmul(H_theo, cov), elephant.causality.granger._dagger(H_theo)
        )

        H_tilde_xx = H_theo[:, 0, 0] + (cov[0, 1] / cov[0, 0] * H_theo[:, 0, 1])
        H_tilde_yy = H_theo[:, 1, 1] + (cov[0, 1] / cov[1, 1] * H_theo[:, 1, 0])

        true_directional_causality_y_x = np.log(
            S_theo[:, 0, 0].real / (H_tilde_xx * cov[0, 0] * H_tilde_xx.conj()).real
        )

        true_directional_causality_x_y = np.log(
            S_theo[:, 1, 1].real / (H_tilde_yy * cov[1, 1] * H_tilde_yy.conj()).real
        )

        true_instantaneous_causality = np.log(
            (H_tilde_xx * cov[0, 0] * H_tilde_xx.conj()).real
            * (H_tilde_yy * cov[1, 1] * H_tilde_yy.conj()).real
        )
        true_instantaneous_causality -= np.linalg.slogdet(S_theo)[1]

        np.testing.assert_allclose(
            spectral_causality.directional_causality_x_y,
            true_directional_causality_x_y,
            atol=0.06,
        )

        np.testing.assert_allclose(
            spectral_causality.directional_causality_y_x,
            true_directional_causality_y_x,
            atol=0.06,
        )

        np.testing.assert_allclose(
            spectral_causality.instantaneous_causality,
            true_instantaneous_causality,
            atol=0.06,
        )

    def test_pairwise_spectral_granger_against_r_grangers(self):
        """
        Test pairwise_spectral_granger against R grangers implementation

        Please follow the link below for more details:
        https://gin.g-node.org/NeuralEnsemble/elephant-data/src/master/unittest/causality/granger/pairwise_spectral_granger  # noqa

        """

        repo_path = r"unittest/causality/granger/pairwise_spectral_granger/data"

        files_to_download = [
            ("time_series_small.npy", "b33dc12d4291db7c2087dd8429f15ab4"),
            ("gc_matrix.npy", "c57262145e74a178588ff0a1004879e2"),
        ]

        downloaded_files = {}
        for filename, checksum in files_to_download:
            downloaded_files[filename] = {
                "filename": filename,
                "path": download_datasets(
                    repo_path=f"{repo_path}/{filename}", checksum=checksum
                ),
            }

        signal = np.load(downloaded_files["time_series_small.npy"]["path"])
        gc_matrix = np.load(downloaded_files["gc_matrix.npy"]["path"])

        denom = 20
        f, spectral_causality = elephant.causality.granger.pairwise_spectral_granger(
            signal[0],
            signal[1],
            len_segment=int(len(signal[0]) / denom),
            num_tapers=15,
            fs=1,
            num_iterations=50,
        )

        np.testing.assert_allclose(gc_matrix[::denom, 0], f, atol=4e-5)
        np.testing.assert_allclose(
            gc_matrix[::denom, 1], spectral_causality[0], atol=0.085
        )
        np.testing.assert_allclose(
            gc_matrix[::denom, 2], spectral_causality[1], atol=0.035
        )


if __name__ == "__main__":
    unittest.main()
