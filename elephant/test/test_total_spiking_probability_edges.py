import unittest
from pathlib import Path
from typing import Tuple, Union

from neo import SpikeTrain
import numpy as np
from quantities import millisecond as ms
from scipy.io import loadmat

from elephant.conversion import BinnedSpikeTrain
from elephant.functional_connectivity_src.total_spiking_probability_edges import (
    generate_filter_pairs,
    normalized_cross_correlation,
    TspeFilterPair,
    total_spiking_probability_edges,
)

from elephant.datasets import download_datasets


class TotalSpikingProbabilityEdgesTestCase(unittest.TestCase):
    def test_generate_filter_pairs(self):
        a = [1]
        b = [1]
        c = [1]
        test_output = [
            TspeFilterPair(
                edge_filter=np.array([-1.0, 0.0, 2.0, 0.0, -1.0]),
                running_total_filter=np.array([1.0]),
                needed_padding=2,
                surrounding_window_size=1,
                observed_window_size=1,
                crossover_window_size=1,
            )
        ]

        function_output = generate_filter_pairs(a, b, c)

        for filter_pair_function, filter_pair_test in zip(function_output, test_output):
            np.testing.assert_array_equal(
                filter_pair_function.edge_filter, filter_pair_test.edge_filter
            )

            np.testing.assert_array_equal(
                filter_pair_function.running_total_filter,
                filter_pair_test.running_total_filter,
            )

            self.assertEqual(
                filter_pair_function.needed_padding, filter_pair_test.needed_padding
            )

            self.assertEqual(
                filter_pair_function.surrounding_window_size,
                filter_pair_test.surrounding_window_size,
            )

            self.assertEqual(
                filter_pair_function.observed_window_size,
                filter_pair_test.observed_window_size,
            )

            self.assertEqual(
                filter_pair_function.crossover_window_size,
                filter_pair_test.crossover_window_size,
            )

    def test_normalized_cross_correlation(self):
        # Generate Spiketrains
        delay_time = 5
        spike_times = [3, 4, 5] * ms
        spike_times_delayed = spike_times + delay_time * ms

        spiketrains = BinnedSpikeTrain(
            [
                SpikeTrain(spike_times, t_stop=20.0 * ms),
                SpikeTrain(spike_times_delayed, t_stop=20.0 * ms),
            ],
            bin_size=1 * ms,
        )

        test_output = np.array([[[0.0, 0.0], [1.1, 0.0]], [[0.0, 1.1], [0.0, 0.0]]])

        function_output = normalized_cross_correlation(
            spiketrains, [-delay_time, delay_time]
        )

        assert np.allclose(function_output, test_output, 0.1)

    def test_total_spiking_probability_edges(self):
        files = [
            "SW/new_sim0_100.mat",
            "BA/new_sim0_100.mat",
            "CA/new_sim0_100.mat",
            "ER05/new_sim0_100.mat",
            "ER10/new_sim0_100.mat",
            "ER15/new_sim0_100.mat",
        ]

        for datafile in files:
            repo_base_path = (
                "unittest/functional_connectivity/total_spiking_probability_edges/data/"
            )
            downloaded_dataset_path = download_datasets(repo_base_path + datafile)

            spiketrains, original_data = load_spike_train_simulated(
                downloaded_dataset_path
            )

            connectivity_matrix, delay_matrix = total_spiking_probability_edges(
                spiketrains
            )

            # Remove self-connections
            np.fill_diagonal(connectivity_matrix, 0)

            _, _, _, auc = roc_curve(connectivity_matrix, original_data)

            self.assertGreater(auc, 0.95)


# ====== HELPER FUNCTIONS ======


def classify_connections(connectivity_matrix: np.ndarray, threshold: int):
    connectivity_matrix_binarized = connectivity_matrix.copy()

    mask_excitatory = connectivity_matrix_binarized > threshold
    mask_inhibitory = connectivity_matrix_binarized < -threshold

    mask_left = ~(mask_excitatory + mask_inhibitory)

    connectivity_matrix_binarized[mask_excitatory] = 1
    connectivity_matrix_binarized[mask_inhibitory] = -1
    connectivity_matrix_binarized[mask_left] = 0

    return connectivity_matrix_binarized


def confusion_matrix(estimate, original, threshold: int = 1):
    """
    Definition:
        - TP: Matches of connections are True Positive
        - FP: Mismatches are False Positive,
        - TN: Matches for non-existing synapses are True Negative
        - FN: mismatches are False Negative.
    """
    if not np.all(np.isin([-1, 0, 1], np.unique(estimate))):
        estimate = classify_connections(estimate, threshold)
    if not np.all(np.isin([-1, 0, 1], np.unique(original))):
        original = classify_connections(original, threshold)

    TP = (np.not_equal(estimate, 0) & np.not_equal(original, 0)).sum()

    TN = (np.equal(estimate, 0) & np.equal(original, 0)).sum()

    FP = (np.not_equal(estimate, 0) & np.equal(original, 0)).sum()

    FN = (np.equal(estimate, 0) & np.not_equal(original, 0)).sum()

    return TP, TN, FP, FN


def fall_out(TP: int, TN: int, FP: int, FN: int):
    FPR = FP / (FP + TN)
    return FPR


def sensitivity(TP: int, TN: int, FP: int, FN: int):
    TPR = TP / (TP + FN)
    return TPR


def roc_curve(estimate, original):
    tpr_list = []
    fpr_list = []

    max_threshold = max(np.max(np.abs(estimate)), 1)

    thresholds = np.linspace(max_threshold, 0, 30)

    for t in thresholds:
        conf_matrix = confusion_matrix(estimate, original, threshold=t)

        tpr_list.append(sensitivity(*conf_matrix))
        fpr_list.append(fall_out(*conf_matrix))

    auc = np.trapz(tpr_list, fpr_list)

    return tpr_list, fpr_list, thresholds, auc


def load_spike_train_simulated(
    path: Union[Path, str],
    bin_size=None,
    t_stop=None,
) -> Tuple[BinnedSpikeTrain, np.ndarray]:
    if isinstance(path, str):
        path = Path(path)

    if not bin_size:
        bin_size = 1 * ms

    data = loadmat(path, simplify_cells=True)["data"]

    if "asdf" not in data:
        raise ValueError('Incorrect Dataformat: Missing spiketrain_data in"asdf"')

    spiketrain_data = data["asdf"]

    # Get number of electrodesa and recording_duration from last element of
    # data array
    n_electrodes, recording_duration_ms = spiketrain_data[-1]
    recording_duration_ms = recording_duration_ms * ms

    # Create spiketrains
    spiketrains = []
    for spiketrain_raw in spiketrain_data[0:n_electrodes]:
        spiketrains.append(
            SpikeTrain(
                spiketrain_raw * ms,
                t_stop=recording_duration_ms,
            )
        )

    spiketrains = BinnedSpikeTrain(
        spiketrains, bin_size=bin_size, t_stop=t_stop or recording_duration_ms
    )

    # Load original_data
    original_data = data["SWM"].T

    return spiketrains, original_data
