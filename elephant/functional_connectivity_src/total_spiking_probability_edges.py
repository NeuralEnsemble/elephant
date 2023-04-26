import itertools
from typing import Iterable, List, NamedTuple, Union, Optional

import numpy as np
from scipy.signal import oaconvolve

from elephant.conversion import BinnedSpikeTrain


def total_spiking_probability_edges(
    spike_trains: BinnedSpikeTrain,
    a: Optional[List[int]] = None,
    b: Optional[List[int]] = None,
    c: Optional[List[int]] = None,
    max_delay: int = 25,
    normalize: bool = False
        ):
    """
    Performs the Total spiking probability edges (TSPE) :cite:`tspe-de_blasi2007_???` on spiketrains ...

    Parameters
    ----------
    spiketrains: neo.spiketrains

    Returns
    -------
    tspe_matrix
    """

    if not a:
        a = [3, 4, 5, 6, 7, 8]

    if not b:
        b = [2, 3, 4, 5, 6]

    if not c:
        c = [0]

    n_neurons, n_bins = spike_trains.shape

    filter_pairs = generate_filter_pairs(a, b, c)

    # Calculate normalized cross corelation for different delays
    # The delay range ranges from 0 to max-delay and includes
    # padding for the filter convolution
    max_padding = max(a) + max(c)
    delay_times = list(range(-max_padding,max_delay + max_padding))
    NCC_d = normalized_cross_correlation(spike_trains, delay_times=delay_times)

    # Normalize to counter network-bursts
    if normalize:
        for delay_time in delay_times:
            NCC_d[:,:,delay_time] /= np.sum(NCC_d[:,:,delay_time][~np.identity(NCC_d.shape[0],dtype=bool)])

    # Apply edge and running total filter
    tspe_matrix = np.zeros((n_neurons, n_neurons, max_delay))
    for filter in filter_pairs:
        # Select ncc_window based on needed filter padding
        NCC_window = NCC_d[:,:,max_padding-filter.needed_padding:max_delay+max_padding+filter.needed_padding]

        # Compute two convolutions with edge- and running total filter
        x1 = oaconvolve(NCC_window, np.expand_dims(filter.edge_filter,(0,1)), mode="valid",axes=2)
        x2 = oaconvolve(x1, np.expand_dims(filter.running_total_filter,(0,1)), mode="full",axes=2)

        tspe_matrix += x2

    return tspe_matrix


def normalized_cross_correlation(
    spike_trains: BinnedSpikeTrain,
    delay_times: Union[int, List[int], Iterable[int]] = 0,
) -> np.ndarray:
    r"""normalized cross correlation using std deviation

    Computes the normalized_cross_correlation between all
    Spiketrains inside a BinnedSpikeTrain-Object at a given delay_time

    The underlying formula is:

    .. math::
        NCC_{X\arrY(d)} = \frac{1}{N_{bins}}\sum_{i=-\inf}^{\inf}{\frac{(y_{(i)} - \bar{y}) \cdot (x_{(i-d) - \bar{x})}{\sigma_x \cdot \sigma_y}}}

    """

    n_neurons, n_bins = spike_trains.shape

    # Get sparse array of BinnedSpikeTrain
    spike_trains_array = spike_trains.sparse_matrix

    # Get std deviation of spike trains
    spike_trains_zeroed = spike_trains_array - spike_trains_array.mean(axis=1)
    spike_trains_std = np.std(spike_trains_zeroed, ddof=1, axis=1)
    std_factors = spike_trains_std @ spike_trains_std.T

    # Loop over delay times
    if isinstance(delay_times, int):
        delay_times = [delay_times]
    elif isinstance(delay_times, list):
        pass
    elif isinstance(delay_times, Iterable):
        delay_times = list(delay_times)

    NCC_d = np.zeros((len(delay_times), n_neurons, n_neurons))

    for index, delay_time in enumerate(delay_times):
        # Uses theoretical zero-padding for shifted values,
        # but since $0 \cdot x = 0$ values can simply be omitted
        if delay_time == 0:
            CC = spike_trains_array[:, :] @ spike_trains_array[:, :].transpose()

        elif delay_time > 0:
            CC = (
                spike_trains_array[:, delay_time:]
                @ spike_trains_array[:, :-delay_time].transpose()
            )

        else:
            CC = (
                spike_trains_array[:, :delay_time]
                @ spike_trains_array[:, -delay_time:].transpose()
            )

        # Normalize using std deviation
        NCC = CC / std_factors / n_bins

        # Compute cross correlation at given delay time
        NCC_d[index, :, :] = NCC

    # Move delay_time axis to back of array
    # Makes index using neurons more intuitive → (n_neuron, n_neuron, delay_times)
    NCC_d = np.moveaxis(NCC_d, 0, -1)

    return NCC_d


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
