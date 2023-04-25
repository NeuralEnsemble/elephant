import itertools
from typing import List, NamedTuple

import numpy as np


def total_spiking_probability_edges():
    """
    Performs the Total spiking probability edges (TSPE) :cite:`tspe-de_blasi2007_???` on spiketrains ...

    Parameters
    ----------
    spiketrains: neo.spiketrains
    Returns
    -------
    tspe_matrix
    """

    tspe_matrix = filter_edges()

    return tspe_matrix


def normalized_cross_correlation():
    pass


def filter_edges():
    pass


def generate_edge_filter(
    a: int,
    b: int,
    c: int,
) -> np.ndarray:
    r"""Generate an edge filter

    The edge filter is generated using following piecewise defined function:

    .. math::
        g_{(i)} = \begin{cases}
            - \frac{1}{a} & 0 \lt i \leq a \\
            \frac{2}{b} & a+c \lt i \leq a + b + c \\
            - \frac{1}{a} & a+b+2c \lt i \leq 2a + b + 2c
            \end{cases}

    """
    filter_length = (2 * a) + b + (2 * c)
    i = np.arange(1, filter_length + 1, dtype=np.float64)

    conditions = [
        (i > 0) & (i <= a),
        (i > (a + c)) & (i <= a + b + c),
        (i > a + b + (2 * c)) & (i <= (2 * a) + b + (2 * c)),
    ]

    values = [-(1 / a), 2 / b, -(1 / a), 0]  # Default Value

    filter = np.piecewise(i, conditions, values)

    return filter


def generate_running_total_filter(b: int) -> np.ndarray:
    return np.ones(b)


class tspe_filter_pair(NamedTuple):
    edge_filter: np.ndarray
    running_total_filter: np.ndarray
    needed_padding: int
    a: int
    b: int
    c: int


def generate_filter_pairs(
    a: List[int],
    b: List[int],
    c: List[int],
) -> List[tspe_filter_pair]:
    """Generates filter pairs of edge and running total filter using all
    permutations of given parameters
    """
    filter_pairs = []

    for _a, _b, _c in itertools.product(a, b, c):
        edge_filter = generate_edge_filter(_a, _b, _c)
        running_total_filter = generate_running_total_filter(_b)
        needed_padding = _a + _c
        filter_pair = tspe_filter_pair(
            edge_filter, running_total_filter, needed_padding, _a, _b, _c
        )
        filter_pairs.append(filter_pair)

    return filter_pairs
