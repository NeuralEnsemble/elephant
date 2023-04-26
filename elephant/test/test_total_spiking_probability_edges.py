from typing import List

from neo import SpikeTrain
import numpy as np
import pytest
from quantities import millisecond as ms

from elephant.conversion import BinnedSpikeTrain
from elephant.functional_connectivity_src.total_spiking_probability_edges import (
    generate_filter_pairs,
    normalized_cross_correlation,
    tspe_filter_pair,
)


def test_generate_filter_pairs():
    a = [1]
    b = [1]
    c = [1]
    test_output = [
        tspe_filter_pair(
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
        assert np.array_equal(
            filter_pair_function.edge_filter, filter_pair_test.edge_filter
        )
        assert np.array_equal(
            filter_pair_function.running_total_filter,
            filter_pair_test.running_total_filter,
        )
        assert filter_pair_function.needed_padding == filter_pair_test.needed_padding
        assert (
            filter_pair_function.surrounding_window_size
            == filter_pair_test.surrounding_window_size
        )
        assert (
            filter_pair_function.observed_window_size
            == filter_pair_test.observed_window_size
        )
        assert (
            filter_pair_function.crossover_window_size
            == filter_pair_test.crossover_window_size
        )


def test_normalized_cross_correlation():
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
