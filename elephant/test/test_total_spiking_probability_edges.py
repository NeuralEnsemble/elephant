from typing import List

import numpy as np
import pytest

from elephant.functional_connectivity_src.total_spiking_probability_edges import (
    generate_filter_pairs,
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
            a=1,
            b=1,
            c=1,
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
        assert filter_pair_function.a == filter_pair_test.a
        assert filter_pair_function.b == filter_pair_test.b
        assert filter_pair_function.c == filter_pair_test.c
