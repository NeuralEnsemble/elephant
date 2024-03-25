import itertools
from typing import Iterable, List, NamedTuple, Union, Optional

import numpy as np
from scipy.signal import oaconvolve

from elephant.conversion import BinnedSpikeTrain


def total_spiking_probability_edges(
    spike_trains: BinnedSpikeTrain,
    surrounding_window_sizes: Optional[List[int]] = None,
    observed_window_sizes: Optional[List[int]] = None,
    crossover_window_sizes: Optional[List[int]] = None,
    max_delay: int = 25,
    normalize: bool = False,
):
    r"""
    Use total spiking probability edges (TSPE) to estimate
    the functional connectivity and delay-times of a neural-network.

    This algorithm uses a normalized cross correlation between pairs of
    spiketrains at different delay-times to get a cross-correlogram.
    Afterwards a series of convolutions with multiple edge-filters
    on the cross-correlogram are preformed, in order to estimate the
    connectivity between neurons and thus allowing the discrimination
    between inhibitory and excitatory effects.

    The default window-sizes and max-delay were optimized using
    in-silico generated spiketrains.

    *Background:*

    - On an excitatory connection the spikerate increases and decreases again
     due to the refractory period which results in local maxima in the
     cross-correlogram followed by downwards slope

    - On an inhibitory connection the spikerate decreases and after refractory
     period, increases again which results in lokal minima surrounded by high
     values in the cross-correlogram.

    - An Edge-Filter can be used to interpret the cross-correlogram and
    accentuate the lokal Maxima and Minima

    *Procedure:*

    1) Compute normalized cross-correlation :math:`NCC` of spiketrains of all
    Neuronpairs
    2) Convolve :math:`NCC` with Edge-Filter :math:`g_{i}` to compute
    :math:`SPE`
    3) Convolve :math:`SPE` with corresponding Running-Total-Filter
    :math:`h_{i}` to account for different lengths after convolution with
    Edge-Filter
    4) Compute :math:`TSPE` using the sum of all :math:`SPE` for all different
    filterpairs
    5) Compute connectivitymatrix by using the index of the tspe-values with
    the highest absolute values

    *Normalized Cross-Correlation:*

    .. math ::

        NCC_{XY}(d) = \frac{1}{N} \sum_{i=-\infty}^{\infty}{ \frac{ (y_{(i)} -
        \bar{y}) \cdot (x_{(i-d)} - \bar{x}) }{ \sigma_x \cdot \sigma_y }}

    *Spiking Probability Edges*

    .. math ::
        SPE_{X \rightarrow Y(d)} = NCC_{XY}(d) * g(i)

    *Total Spiking Probability Edges:*

    .. math ::
        TSPE_{X \rightarrow Y}(d) = \sum_{n=1}^{N_a \cdot N_b \cdot N_c}
        {SPE_{X \rightarrow Y}^{(n)}(d) * h(i)^{(n)} }

    :cite:`functional_connectivity-de_blasi19_169`

    Parameters
    ----------
    spike_trains : (N, ) elephant.conversion.BinnedSpikeTrain
        A binned spike train containing all neurons for connectivity estimation
    surrounding_window_sizes : List[int], default = [3, 4, 5, 6, 7, 8]
        Array of window-sizes for the surrounding area of the point of
        interest.
    observed_window_sizes : List[int], default = [2, 3, 4, 5, 6]
        Array of window-sizes for the observed area
    crossover_window_sizes : List[int], default = [0]
        Array of window-sizes for the crossover between surrounding and
        observed window.
    max_delay : int, default = 25
        Defines the max delay when performing the normalized crosscorrelations.
        Value depends on the bin-size of the BinnedSpikeTrain.
        On a bin-size of *1ms* a value of *25* corresponds to *25ms*
    normalize : bool, optional
        Normalize the output [experimental]

    Returns
    -------
    connectivity_matrix : (N, N) np.ndarray
        Square Matrix of the connectivity estimation between neurons.
        Positive values describe an excitatory connection while
        negative values describe an inhibitory connection.
    delay_matrix : (N, N) np.ndarray
        Square Matrix of the delay_times between neuron-activity.
    """

    if not surrounding_window_sizes:
        surrounding_window_sizes = [3, 4, 5, 6, 7, 8]

    if not observed_window_sizes:
        observed_window_sizes = [2, 3, 4, 5, 6]

    if not crossover_window_sizes:
        crossover_window_sizes = [0]

    n_neurons, n_bins = spike_trains.shape

    filter_pairs = generate_filter_pairs(
        surrounding_window_sizes, observed_window_sizes, crossover_window_sizes
    )

    # Calculate normalized cross corelation for different delays
    # The delay range ranges from 0 to max-delay and includes
    # padding for the filter convolution
    max_padding = max(surrounding_window_sizes) + max(crossover_window_sizes)
    delay_times = list(range(-max_padding, max_delay + max_padding))
    NCC_d = normalized_cross_correlation(spike_trains, delay_times=delay_times)

    # Normalize to counter network-bursts
    if normalize:
        for delay_time in delay_times:
            NCC_d[:, :, delay_time] /= np.sum(
                NCC_d[:, :, delay_time][~np.identity(NCC_d.shape[0],
                                                     dtype=bool)]
            )

    # Apply edge and running total filter
    tspe_matrix = np.zeros((n_neurons, n_neurons, max_delay))
    for filter in filter_pairs:
        # Select ncc_window based on needed filter padding
        NCC_window = NCC_d[
            :,
            :,
            max_padding
            - filter.needed_padding: max_delay
            + max_padding
            + filter.needed_padding,
        ]

        # Compute two convolutions with edge- and running total filter
        x1 = oaconvolve(
            NCC_window, np.expand_dims(filter.edge_filter, (0, 1)),
            mode="valid", axes=2
        )
        x2 = oaconvolve(
            x1, np.expand_dims(filter.running_total_filter, (0, 1)),
            mode="full", axes=2
        )

        tspe_matrix += x2

    # Take maxima of absolute of delays to get estimation for connectivity
    connectivity_matrix_index = np.argmax(np.abs(tspe_matrix),
                                          axis=2, keepdims=True)
    connectivity_matrix = np.take_along_axis(tspe_matrix,
                                             connectivity_matrix_index, axis=2
                                             ).squeeze(axis=2)
    delay_matrix = connectivity_matrix_index.squeeze()

    return connectivity_matrix, delay_matrix


def normalized_cross_correlation(
    spike_trains: BinnedSpikeTrain,
    delay_times: Union[int, List[int], Iterable[int]] = 0,
) -> np.ndarray:
    r"""
    Normalized cross correlation using std deviation

    Computes the normalized_cross_correlation between all
    Spiketrains inside a BinnedSpikeTrain-Object at a given delay_time

    The underlying formula is:

    .. math::
        NCC_{X\arrY(d)} = \frac{1}{N_{bins}}\sum_{i=-\inf}^{\inf}{
        \frac{(y_{(i)} - \bar{y}) \cdot (x_{(i-d) - \bar{x})}{\sigma_x
        \cdot \sigma_y}}}

    The subtraction of mean-values is omitted, since it offers little added
    accuracy but increases the compute-time immensely.
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
            CC = spike_trains_array[:, :] @ spike_trains_array[:, :
                                                               ].transpose()

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

        # Convert CC to dense matrix before performing the division
        CC = CC.toarray()
        # Normalize using std deviation
        NCC = CC / std_factors / n_bins

        # Compute cross correlation at given delay time
        NCC_d[index, :, :] = NCC

    # Move delay_time axis to back of array
    # Makes index using neurons more intuitive → (n_neuron, n_neuron,
    #                                             delay_times)
    NCC_d = np.moveaxis(NCC_d, 0, -1)

    return NCC_d


def generate_edge_filter(
    surrounding_window_size: int,
    observed_window_size: int,
    crossover_window_size: int,
) -> np.ndarray:
    r"""Generate an edge filter

    The edge filter is generated using following piecewise defined function:

    a = surrounding_window_size
    b = observed_window_size
    c = crossover_window_size

    .. math::
        g_{(i)} = \begin{cases}
            - \frac{1}{a} & 0 \lt i \leq a \\
            \frac{2}{b} & a+c \lt i \leq a + b + c \\
            - \frac{1}{a} & a+b+2c \lt i \leq 2a + b + 2c
            \end{cases}

    """
    filter_length = (
        (2 * surrounding_window_size)
        + observed_window_size
        + (2 * crossover_window_size)
    )
    i = np.arange(1, filter_length + 1, dtype=np.float64)

    conditions = [
        (i > 0) & (i <= surrounding_window_size),
        (i > (surrounding_window_size + crossover_window_size))
        & (i <= surrounding_window_size + observed_window_size +
           crossover_window_size),
        (
            i
            > surrounding_window_size
            + observed_window_size
            + (2 * crossover_window_size)
        )
        & (
            i
            <= (2 * surrounding_window_size)
            + observed_window_size
            + (2 * crossover_window_size)
        ),
    ]

    values = [
        -(1 / surrounding_window_size),
        2 / observed_window_size,
        -(1 / surrounding_window_size),
        0,
    ]  # Default Value

    edge_filter = np.piecewise(i, conditions, values)

    return edge_filter


def generate_running_total_filter(observed_window_size: int) -> np.ndarray:
    return np.ones(observed_window_size)


class TspeFilterPair(NamedTuple):
    edge_filter: np.ndarray
    running_total_filter: np.ndarray
    needed_padding: int
    surrounding_window_size: int
    observed_window_size: int
    crossover_window_size: int


def generate_filter_pairs(
    surrounding_window_sizes: List[int],
    observed_window_sizes: List[int],
    crossover_window_sizes: List[int],
) -> List[TspeFilterPair]:
    """Generates filter pairs of edge and running total filter using all
    permutations of given parameters
    """
    filter_pairs = []

    for _a, _b, _c in itertools.product(
        surrounding_window_sizes, observed_window_sizes, crossover_window_sizes
    ):
        edge_filter = generate_edge_filter(_a, _b, _c)
        running_total_filter = generate_running_total_filter(_b)
        needed_padding = _a + _c
        filter_pair = TspeFilterPair(
            edge_filter, running_total_filter, needed_padding, _a, _b, _c
        )
        filter_pairs.append(filter_pair)

    return filter_pairs
